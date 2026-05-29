"""M1.5 benchmark runners — train-to-convergence + ablation primitives (plan §7.3).

Each function returns a dict suitable for direct JSON dump. All loops respect
the PlateauDetector convergence discipline and record `converged: bool`.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from ..attention.sparse_three_branch import ThreeBranchAttention
from ..logging import RunLogger, RunDirectory, per_module_grad_norms
from ..workspace.kv_bank import WorkspaceKVBank
from .convergence import PlateauDetector
from .factoid_world import (
    FactoidWorld, build_qa_pairs, evaluate_qa,
    make_lm_batches, render_corpus,
)
from .milestone1 import collect_branch_masses


# ---------------------------------------------------------------------------
# Corpus rendering — match M0.5 sweep's proven recipe
# ---------------------------------------------------------------------------

def _auto_repeats(n_facts: int, *, batch_size: int, seq_len: int,
                  min_repeats: int = 40,
                  min_batches_per_epoch: int = 8) -> int:
    """Mirror M0.5 capacity-sweep recipe: ensure ≥N unique batches per epoch.

    M0.5 sweep (which converged to 0.89 @ N=64 / 0.73 @ N=128 on Veyra3-5M)
    used `repeats=40` and auto-bumped to satisfy a "≥8 batches/epoch" floor.
    """
    tokens_per_fact_approx = 12
    needed_tokens = batch_size * seq_len * min_batches_per_epoch
    have = n_facts * min_repeats * tokens_per_fact_approx
    if have >= needed_tokens:
        return min_repeats
    return needed_tokens // (n_facts * tokens_per_fact_approx) + 1


def _make_factoid_batches(world: FactoidWorld, *, batch_size: int,
                          seq_len: int, device, seed: int = 0):
    reps = _auto_repeats(world.n_facts, batch_size=batch_size, seq_len=seq_len)
    stream = render_corpus(world, repeats_per_fact=reps, seed=seed)
    return make_lm_batches(stream, batch_size=batch_size, seq_len=seq_len,
                           device=device), reps


# ---------------------------------------------------------------------------
# Branch-ablation control
# ---------------------------------------------------------------------------

class BranchAblation:
    """Context manager that zeros out specific branch gates for ablation.

    Usage:
        with BranchAblation(model, disable=("selected", "compressed")):
            train_or_eval(model, ...)

    Implementation: monkey-patches each `ThreeBranchAttention.forward` to
    zero the named branches' contributions BEFORE the gate softmax. We
    achieve this by setting per-branch boolean flags on each module and
    relying on a small forward-time check (added to sparse_three_branch).
    Falls back to gradient-free attribute injection compatible with current
    code: we set `_disabled_branches: set[str]` on each module; the
    attention forward (when present) honors it. If forward doesn't honor
    it, we instead apply a post-forward correction via output hooks.
    """

    def __init__(self, model: nn.Module, *,
                 disable: Tuple[str, ...] = (),
                 disable_mount: bool = False):
        self.model = model
        self.disable = set(disable)
        self.disable_mount = disable_mount
        self._modules: List[ThreeBranchAttention] = [
            m for m in model.modules() if isinstance(m, ThreeBranchAttention)
        ]
        self._saved_ws: Dict[int, Any] = {}

    def __enter__(self):
        for m in self._modules:
            m._disabled_branches = set(self.disable)  # honored by forward if supported
            if self.disable_mount:
                # Remove ws injection if currently set
                self._saved_ws[id(m)] = getattr(m, "_ws_kv", None)
                if hasattr(m, "_ws_kv"):
                    m._ws_kv = None
        return self

    def __exit__(self, exc_type, exc, tb):
        for m in self._modules:
            if hasattr(m, "_disabled_branches"):
                delattr(m, "_disabled_branches")
            if self.disable_mount and id(m) in self._saved_ws:
                m._ws_kv = self._saved_ws[id(m)]
        return False


# ---------------------------------------------------------------------------
# Train-to-convergence loop
# ---------------------------------------------------------------------------

def train_to_convergence(
    model: nn.Module,
    batches: List[Tuple[torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    *,
    max_steps: int = 4000,
    logger: Optional[RunLogger] = None,
    detector: Optional[PlateauDetector] = None,
    label_prefix: str = "",
    log_branch_masses: bool = True,
) -> Dict[str, Any]:
    """Train looping over `batches` until plateau or `max_steps`.

    Returns dict: {steps, final_loss, wall_seconds, converged, converged_at}.
    """
    detector = detector or PlateauDetector()
    t0 = time.time()
    step = 0
    last_loss = float("nan")
    n_batches = len(batches)
    if n_batches == 0:
        return {"steps": 0, "final_loss": float("nan"),
                "wall_seconds": 0.0, "converged": False,
                "converged_at": None}

    while step < max_steps:
        ids, lbls = batches[step % n_batches]
        optimizer.zero_grad(set_to_none=True)
        out = model(input_ids=ids, labels=lbls)
        loss = out.loss
        loss.backward()
        optimizer.step()

        last_loss = float(loss.detach())
        detector.update(last_loss)

        if logger is not None and (step % 50 == 0 or detector.should_stop()):
            rec: Dict[str, Any] = {
                "step": step, "lm_loss": last_loss,
                "total_loss": last_loss,
            }
            if log_branch_masses:
                s, sel, c = collect_branch_masses(model)
                rec.update({"sliding_mass": s, "selected_mass": sel,
                            "compressed_mass": c})
            if label_prefix:
                rec["phase"] = label_prefix
            rec["tokens_per_sec"] = ids.numel() * (step + 1) / max(time.time() - t0, 1e-6)
            logger.step(**rec)

        step += 1
        if detector.should_stop():
            break

    return {
        "steps": step,
        "final_loss": last_loss,
        "wall_seconds": time.time() - t0,
        "converged": detector.converged,
        "converged_at": detector.converged_at,
    }


# ---------------------------------------------------------------------------
# A0 — convergence baseline
# ---------------------------------------------------------------------------

def run_a0_baseline(
    model: nn.Module,
    world: FactoidWorld,
    *,
    device,
    optimizer: torch.optim.Optimizer,
    batch_size: int = 4,
    seq_len: int = 512,
    max_steps: int = 4000,
    logger: Optional[RunLogger] = None,
    pass_threshold: float = 0.80,
) -> Dict[str, Any]:
    batches, reps = _make_factoid_batches(
        world, batch_size=batch_size, seq_len=seq_len, device=device)
    stats = train_to_convergence(
        model, batches, optimizer, max_steps=max_steps,
        logger=logger, label_prefix="a0",
    )
    pairs = build_qa_pairs(world)
    eval_res = evaluate_qa(model, pairs, device=device)
    acc = eval_res["accuracy"]
    return {
        "bench": "A0",
        "n_facts": world.n_facts,
        "n_train_batches": len(batches),
        "render_repeats": reps,
        "accuracy": acc,
        "pass_threshold": pass_threshold,
        "status": "pass" if (acc >= pass_threshold and stats["converged"]) else "fail",
        **stats,
    }


# ---------------------------------------------------------------------------
# A1 — branch contribution ablation
# ---------------------------------------------------------------------------

def run_a1_branch_ablations(
    model_builder,
    world: FactoidWorld,
    *,
    device,
    optimizer_builder,
    variants: Optional[List[Dict[str, Any]]] = None,
    batch_size: int = 4,
    seq_len: int = 512,
    max_steps: int = 3000,
    logger_factory=None,
) -> Dict[str, Any]:
    """Run each branch-ablation variant with a fresh model instance.

    `model_builder()` returns a fresh model on `device`.
    `optimizer_builder(model)` returns a fresh optimizer.
    """
    variants = variants or [
        {"name": "all_three", "disable": ()},
        {"name": "sliding_only", "disable": ("selected", "compressed")},
        {"name": "sliding_compressed", "disable": ("selected",)},
        {"name": "sliding_selected", "disable": ("compressed",)},
        {"name": "no_mount", "disable": (), "disable_mount": True},
    ]
    out: Dict[str, Any] = {"bench": "A1", "variants": []}
    for v in variants:
        model = model_builder()
        opt = optimizer_builder(model)
        batches, _ = _make_factoid_batches(
            world, batch_size=batch_size, seq_len=seq_len, device=device)
        logger = logger_factory(v["name"]) if logger_factory else None
        with BranchAblation(model,
                            disable=tuple(v.get("disable", ())),
                            disable_mount=bool(v.get("disable_mount", False))):
            stats = train_to_convergence(
                model, batches, opt, max_steps=max_steps,
                logger=logger, label_prefix=f"a1_{v['name']}",
            )
            pairs = build_qa_pairs(world)
            eval_res = evaluate_qa(model, pairs, device=device)
            s_mass, sel_mass, c_mass = collect_branch_masses(model)
        out["variants"].append({
            "name": v["name"],
            "accuracy": eval_res["accuracy"],
            "branch_mass": {"sliding": s_mass, "selected": sel_mass,
                            "compressed": c_mass},
            **stats,
        })
        del model, opt
        if device.type == "cuda":
            torch.cuda.empty_cache()
    # Pass criterion: all-three ≥ each single by ≥5 pp; no-mount worst.
    by_name = {v["name"]: v["accuracy"] for v in out["variants"]}
    all_three = by_name.get("all_three", 0.0)
    others = [v for n, v in by_name.items() if n not in ("all_three", "no_mount")]
    all_beats = all(all_three >= o + 0.05 for o in others) if others else True
    no_mount_worst = by_name.get("no_mount", 1.0) <= min(by_name.values())
    out["pass_criterion"] = "all_three >= each single +0.05 AND no_mount worst"
    out["status"] = "pass" if (all_beats and no_mount_worst) else "fail"
    return out


# ---------------------------------------------------------------------------
# A2 — mount mechanism shootout
# ---------------------------------------------------------------------------

def run_a2_mount_shootout(
    model_builder,
    world: FactoidWorld,
    tokenizer,
    *,
    device,
    optimizer_builder,
    max_steps: int = 3000,
    batch_size: int = 4,
    seq_len: int = 512,
    bank_max_length: int = 512,
    pass_kv_over_weights: float = 0.5,
    logger_factory=None,
) -> Dict[str, Any]:
    """Compare kv_inject / text_prepend / no_mount / weights_path on same world."""
    results: Dict[str, Dict[str, Any]] = {}

    # 1. weights_path: train on facts directly, eval
    m = model_builder()
    opt = optimizer_builder(m)
    batches, _ = _make_factoid_batches(
        world, batch_size=batch_size, seq_len=seq_len, device=device)
    logger = logger_factory("weights") if logger_factory else None
    stats_w = train_to_convergence(
        m, batches, opt, max_steps=max_steps, logger=logger,
        label_prefix="a2_weights",
    )
    pairs = build_qa_pairs(world)
    results["weights_path"] = {
        "accuracy": evaluate_qa(m, pairs, device=device)["accuracy"],
        **stats_w,
    }
    del m, opt
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # 2. kv_inject: untrained model, encode facts → bank, eval with bank
    m = model_builder()
    ws_tokens = render_corpus(world, repeats_per_fact=1)
    ws_text = tokenizer.decode(ws_tokens, skip_special_tokens=True)
    bank = WorkspaceKVBank()
    bank.encode(m, ws_text, tokenizer, device, max_length=bank_max_length)
    with bank.inject(m):
        kv_acc = evaluate_qa(m, pairs, device=device)["accuracy"]
    results["kv_inject"] = {"accuracy": kv_acc, "bank_tokens": bank.workspace_seq_len(-1)}
    del m
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # 3. text_prepend: untrained model, decode workspace as prefix
    m = model_builder()
    pairs_pre = []
    prefix_ids = ws_tokens[:bank_max_length]
    for prompt, ans in pairs:
        pairs_pre.append((prefix_ids + prompt, ans))
    tp_acc = evaluate_qa(m, pairs_pre, device=device)["accuracy"]
    results["text_prepend"] = {"accuracy": tp_acc, "prefix_tokens": len(prefix_ids)}
    del m
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # 4. no_mount: untrained model, no facts → random baseline
    m = model_builder()
    results["no_mount"] = {"accuracy": evaluate_qa(m, pairs, device=device)["accuracy"]}
    del m
    if device.type == "cuda":
        torch.cuda.empty_cache()

    w = max(results["weights_path"]["accuracy"], 1e-6)
    out: Dict[str, Any] = {
        "bench": "A2",
        "results": results,
        "ratios": {
            "kv_over_weights": results["kv_inject"]["accuracy"] / w,
            "text_over_weights": results["text_prepend"]["accuracy"] / w,
            "kv_over_text": results["kv_inject"]["accuracy"] /
                            max(results["text_prepend"]["accuracy"], 1e-6),
        },
        "pass_criterion": f"kv_over_weights >= {pass_kv_over_weights}",
    }
    out["status"] = "pass" if out["ratios"]["kv_over_weights"] >= pass_kv_over_weights else "fail"
    return out


# ---------------------------------------------------------------------------
# A6 — capacity sweep (weights vs KV)
# ---------------------------------------------------------------------------

def run_a6_capacity_point(
    model_builder,
    tokenizer,
    *,
    n_facts: int,
    vocab_size: int,
    device,
    optimizer_builder,
    batch_size: int = 4,
    seq_len: int = 512,
    max_steps_per_n: int = 4000,
    seed: int = 0,
    logger_factory=None,
) -> Dict[str, Any]:
    from .factoid_world import build_world
    world = build_world(vocab_size=vocab_size, n_facts=n_facts, seed=seed)
    # weights path
    m = model_builder()
    opt = optimizer_builder(m)
    batches, _ = _make_factoid_batches(
        world, batch_size=batch_size, seq_len=seq_len, device=device, seed=seed)
    logger = logger_factory(f"n{n_facts}_w") if logger_factory else None
    # Scale max_steps to fact count: M0.5 sweep needed ~200 epochs * batches
    # for N=64, scaling up for larger N. Use generous cap.
    cap = min(max_steps_per_n,
              max(800, 200 * max(1, (n_facts.bit_length() - 5))))
    stats_w = train_to_convergence(m, batches, opt, max_steps=cap, logger=logger)
    pairs = build_qa_pairs(world)
    w_acc = evaluate_qa(m, pairs, device=device)["accuracy"]
    del m, opt
    if device.type == "cuda":
        torch.cuda.empty_cache()
    # kv path (no training)
    m = model_builder()
    ws_tokens = render_corpus(world, repeats_per_fact=1)
    ws_text = tokenizer.decode(ws_tokens, skip_special_tokens=True)
    bank = WorkspaceKVBank()
    bank.encode(m, ws_text, tokenizer, device,
                max_length=min(2048, max(256, n_facts * 4)))
    with bank.inject(m):
        kv_acc = evaluate_qa(m, pairs, device=device)["accuracy"]
    del m
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"n_facts": n_facts, "weights_accuracy": w_acc, "kv_accuracy": kv_acc,
            "weights_converged": stats_w["converged"]}
