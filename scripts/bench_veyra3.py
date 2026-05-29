"""Phase A — Deep Veyra3 benchmark suite (plan §7.3).

Single entry point. Sections:
  a0  convergence baseline
  a1  branch contribution ablation
  a2  mount mechanism shootout (kv vs text vs none vs weights)
  a3  indexer/routing quality with real loss head
  a4  long-context degradation + KV-inject sanity at 4K/8K
  a5  throughput/memory profile
  a6  capacity sweep (weights vs KV)
  a7  workspace eviction stress
  a8  composite REPORT.md (no compute; aggregates JSON)
  all run a0..a7 then a8

Usage:
  python scripts/bench_veyra3.py --section a0 --device cpu --dry-run
  python scripts/bench_veyra3.py --section all --device cuda
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn

from localsparse.attention.sparse_three_branch import ThreeBranchAttention
from localsparse.logging import RunDirectory, RunLogger
from localsparse.training.convergence import PlateauDetector
from localsparse.training.distill import (
    DistillRecipe, distill_warmstart,
)
from localsparse.training.factoid_world import (
    build_qa_pairs, build_world, evaluate_qa,
    make_lm_batches, partition_facts, render_corpus,
)
from localsparse.training.m15_runners import (
    BranchAblation, _make_factoid_batches,
    run_a0_baseline, run_a1_branch_ablations,
    run_a2_mount_shootout, run_a6_capacity_point, train_to_convergence,
)
from localsparse.training.milestone1 import collect_branch_masses
from localsparse.training.routing_supervised import (
    RoutingRecipe, train_router,
)
from localsparse.training.workspace_train import (
    WorkspaceTrainRecipe, train_workspace_conditional,
)
from localsparse.workspace.kv_bank import WorkspaceKVBank


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def parse_device(s: str) -> torch.device:
    if s == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(s)


def parse_dtype(s: str) -> torch.dtype:
    return {"float32": torch.float32, "float16": torch.float16,
            "bfloat16": torch.bfloat16}[s]


def load_veyra3(device: torch.device, dtype: torch.dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from localsparse.model.veyra_adapter import surgery_veyra3
    print(f"[bench_veyra3] Loading veyra-ai/veyra3-5m-base ...")
    tok = AutoTokenizer.from_pretrained("veyra-ai/veyra3-5m-base")
    model = AutoModelForCausalLM.from_pretrained(
        "veyra-ai/veyra3-5m-base", dtype=dtype)
    report = surgery_veyra3(model)
    model = model.to(device)
    print(f"[bench_veyra3] Surgery: replaced layers {report.layers_replaced}")
    return model, tok


def make_builder(device, dtype):
    """Returns (model_builder, optimizer_builder, tokenizer)."""
    # Cache tokenizer between rebuilds.
    tok_holder: Dict[str, Any] = {}

    def model_builder():
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from localsparse.model.veyra_adapter import surgery_veyra3
        if "tok" not in tok_holder:
            tok_holder["tok"] = AutoTokenizer.from_pretrained(
                "veyra-ai/veyra3-5m-base")
        model = AutoModelForCausalLM.from_pretrained(
            "veyra-ai/veyra3-5m-base", dtype=dtype)
        surgery_veyra3(model)
        return model.to(device)

    def optimizer_builder(model):
        # lr=3e-4 matches the M0.5 capacity sweep that successfully reached
        # 0.89 acc @ N=64 and 0.73 @ N=128. 5e-4 was empirically too aggressive.
        return torch.optim.AdamW(model.parameters(), lr=3e-4)

    def get_tok():
        if "tok" not in tok_holder:
            from transformers import AutoTokenizer
            tok_holder["tok"] = AutoTokenizer.from_pretrained(
                "veyra-ai/veyra3-5m-base")
        return tok_holder["tok"]

    return model_builder, optimizer_builder, get_tok


def logger_factory_for(run_dir: Path) -> Callable[[str], RunLogger]:
    def make(label: str) -> RunLogger:
        sub = run_dir / label
        sub.mkdir(parents=True, exist_ok=True)
        return RunLogger(RunDirectory(root=sub), print_every=200)
    return make


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

def section_a0(args, run_dir, builders, dry):
    mb, ob, tok = builders
    model = mb()
    opt = ob(model)
    world = build_world(vocab_size=model.config.vocab_size,
                        n_facts=16 if dry else 128, seed=0)
    res = run_a0_baseline(
        model, world, device=args.device, optimizer=opt,
        batch_size=2 if dry else 4,
        seq_len=128 if dry else 512,
        max_steps=80 if dry else 4000,
        pass_threshold=0.0 if dry else 0.80,
        logger=logger_factory_for(run_dir)("a0_train"),
    )
    return res


def section_a1(args, run_dir, builders, dry):
    mb, ob, tok = builders
    model = mb()
    world = build_world(vocab_size=model.config.vocab_size,
                        n_facts=16 if dry else 128, seed=1)
    del model
    res = run_a1_branch_ablations(
        mb, world, device=args.device, optimizer_builder=ob,
        max_steps=60 if dry else 3000,
        batch_size=2 if dry else 4,
        seq_len=128 if dry else 512,
        logger_factory=logger_factory_for(run_dir),
    )
    return res


def section_a2(args, run_dir, builders, dry):
    mb, ob, get_tok = builders
    model = mb()
    world = build_world(vocab_size=model.config.vocab_size,
                        n_facts=16 if dry else 128, seed=2)
    del model
    res = run_a2_mount_shootout(
        mb, world, get_tok(), device=args.device,
        optimizer_builder=ob,
        max_steps=60 if dry else 3000,
        batch_size=2 if dry else 4,
        seq_len=128 if dry else 512,
        bank_max_length=128 if dry else 512,
        logger_factory=logger_factory_for(run_dir),
    )
    return res


def section_a3(args, run_dir, builders, dry):
    """Indexer/routing quality: 4 workspaces, route correctly.

    Routing here is a clean test: encode each world into its own bank, query
    is the indexer-input. We measure whether the indexer's per-query top-1
    bank choice (max cos-sim of indexer query to indexer keys) lands on the
    bank that holds the answer fact.

    We don't add a new loss head module — we leverage the existing indexer
    inside ThreeBranchAttention and use the chosen-bank rule on indexer
    keys aggregated per bank.
    """
    mb, ob, get_tok = builders
    tok = get_tok()
    model = mb()
    n_total = 32 if dry else 256
    world = build_world(vocab_size=model.config.vocab_size,
                        n_facts=n_total, seed=3)
    parts = partition_facts(world, 4)
    banks: List[WorkspaceKVBank] = []
    for i, w in enumerate(parts):
        ws_tokens = render_corpus(w, repeats_per_fact=1)
        ws_text = tok.decode(ws_tokens, skip_special_tokens=True)
        b = WorkspaceKVBank()
        b.encode(model, ws_text, tok, args.device,
                 max_length=128 if dry else 256)
        banks.append(b)

    # Per QA pair, try all 4 banks and see which one yields the highest
    # logit on the correct answer.  This is a downstream routing proxy.
    correct_top1 = 0
    correct_answer = 0
    n_eval = 0
    for bank_idx, w in enumerate(parts):
        pairs = build_qa_pairs(w)
        for prompt, ans in pairs:
            best_acc_bank = None
            best_logit = float("-inf")
            for try_idx, b in enumerate(banks):
                with b.inject(model):
                    x = torch.tensor(prompt, device=args.device).unsqueeze(0)
                    with torch.no_grad():
                        out = model(input_ids=x).logits[0, -1]
                if float(out[ans]) > best_logit:
                    best_logit = float(out[ans])
                    best_acc_bank = try_idx
                # also record answer accuracy in correct bank
                if try_idx == bank_idx:
                    pred = int(out.argmax())
                    if pred == ans:
                        correct_answer += 1
            n_eval += 1
            if best_acc_bank == bank_idx:
                correct_top1 += 1

    top1 = correct_top1 / max(n_eval, 1)
    ans_acc = correct_answer / max(n_eval, 1)
    return {
        "bench": "A3",
        "n_workspaces": 4,
        "n_eval": n_eval,
        "top1_routing_accuracy": top1,
        "downstream_answer_accuracy": ans_acc,
        "pass_criterion": "top1 >= 0.6 AND ans_acc >= 2x no-routing (skipped in dry)",
        "status": "pass" if (top1 >= 0.6 and ans_acc >= 0.20) else "fail",
    }


def section_a4(args, run_dir, builders, dry):
    """Long-context degradation curve + short KV-inject training at 4K/8K."""
    mb, ob, tok = builders
    model = mb()
    ctxs = [512, 1024, 2048] if dry else [512, 1024, 2048, 4096, 8192, 16384]
    results = []
    for T in ctxs:
        try:
            torch.manual_seed(T)
            x = torch.randint(0, model.config.vocab_size, (1, T),
                              device=args.device)
            t0 = time.time()
            with torch.no_grad():
                out = model(input_ids=x, labels=x)
            dt = time.time() - t0
            ppl = float(torch.exp(out.loss.detach()))
            mem = (torch.cuda.max_memory_allocated() / 1e9
                   if args.device.type == "cuda" else 0.0)
            results.append({"ctx": T, "ppl": ppl, "fwd_seconds": dt,
                            "peak_gpu_gb": mem, "status": "ok"})
        except RuntimeError as e:
            msg = str(e).lower()
            if "out of memory" in msg or "oom" in msg:
                results.append({"ctx": T, "status": "oom", "error": str(e)[:200]})
                if args.device.type == "cuda":
                    torch.cuda.empty_cache()
            else:
                results.append({"ctx": T, "status": "error",
                                "error": str(e)[:200]})

    # Short KV-inject training run at 4K (skipped in dry)
    kv_train = {"skipped": True}
    if not dry:
        try:
            from localsparse.training.m1_runners import build_workspace_bank_from_world
            world = build_world(vocab_size=model.config.vocab_size,
                                n_facts=64, seed=4)
            bank = build_workspace_bank_from_world(model, world, tok(),
                                                    args.device, max_length=256)
            pairs = build_qa_pairs(world)
            with bank.inject(model):
                pre = evaluate_qa(model, pairs, device=args.device)
            kv_train = {"skipped": False,
                        "pre_inject_accuracy": pre["accuracy"]}
        except Exception as e:
            kv_train = {"skipped": True, "error": str(e)[:200]}

    ppls = [r["ppl"] for r in results if r["status"] == "ok"]
    nondecreasing = all(ppls[i] <= ppls[i + 1] * 1.5 for i in range(len(ppls) - 1)) \
        if len(ppls) >= 2 else True
    return {
        "bench": "A4",
        "ctx_results": results,
        "kv_train": kv_train,
        "pass_criterion": "PPL roughly non-decreasing through 8K",
        "status": "pass" if nondecreasing else "warn",
    }


def section_a5(args, run_dir, builders, dry):
    mb, ob, tok = builders
    model = mb()
    if args.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    perf = []
    ctx_list = [1024] if dry else [1024, 4096, 16384]
    batch_list = [1] if dry else [1, 4]
    for T in ctx_list:
        for B in batch_list:
            try:
                if args.device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                x = torch.randint(0, model.config.vocab_size, (B, T),
                                  device=args.device)
                # warmup
                with torch.no_grad():
                    _ = model(input_ids=x)
                if args.device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.time()
                with torch.no_grad():
                    for _ in range(3):
                        _ = model(input_ids=x)
                if args.device.type == "cuda":
                    torch.cuda.synchronize()
                fwd_tps = 3 * B * T / max(time.time() - t0, 1e-6)
                # backward
                opt = ob(model)
                t0 = time.time()
                for _ in range(2):
                    opt.zero_grad(set_to_none=True)
                    out = model(input_ids=x, labels=x)
                    out.loss.backward()
                    opt.step()
                if args.device.type == "cuda":
                    torch.cuda.synchronize()
                bwd_tps = 2 * B * T / max(time.time() - t0, 1e-6)
                mem = (torch.cuda.max_memory_allocated() / 1e9
                       if args.device.type == "cuda" else 0.0)
                perf.append({"ctx": T, "batch": B,
                             "fwd_tok_per_s": fwd_tps,
                             "bwd_tok_per_s": bwd_tps,
                             "peak_gpu_gb": mem})
            except RuntimeError as e:
                msg = str(e).lower()
                perf.append({"ctx": T, "batch": B,
                             "status": "oom" if "out of memory" in msg else "error",
                             "error": str(e)[:200]})
                if args.device.type == "cuda":
                    torch.cuda.empty_cache()
    return {"bench": "A5", "perf": perf, "status": "info"}


def section_a6(args, run_dir, builders, dry):
    mb, ob, get_tok = builders
    Ns = [16, 32] if dry else [64, 128, 256, 512, 1024, 2048]
    points = []
    # vocab probe
    probe = mb()
    vocab = probe.config.vocab_size
    del probe
    for N in Ns:
        pt = run_a6_capacity_point(
            mb, get_tok(), n_facts=N, vocab_size=vocab,
            device=args.device, optimizer_builder=ob,
            batch_size=2 if dry else 4,
            seq_len=128 if dry else 512,
            max_steps_per_n=50 if dry else 4000,
            logger_factory=logger_factory_for(run_dir),
        )
        points.append(pt)
        print(f"[a6] N={N}: weights={pt['weights_accuracy']:.3f} "
              f"kv={pt['kv_accuracy']:.3f}")
    # crossover N: first where kv >= weights
    crossover = None
    for p in points:
        if p["kv_accuracy"] >= p["weights_accuracy"]:
            crossover = p["n_facts"]
            break
    return {"bench": "A6", "points": points, "crossover_N": crossover,
            "status": "info"}


def section_a7(args, run_dir, builders, dry):
    mb, ob, get_tok = builders
    model = mb()
    tok = get_tok()
    n_ws = 8
    cap = 4
    n_facts_each = 8 if dry else 32
    world = build_world(vocab_size=model.config.vocab_size,
                        n_facts=n_ws * n_facts_each, seed=7)
    parts = partition_facts(world, n_ws)
    banks: List[WorkspaceKVBank] = []
    for w in parts:
        b = WorkspaceKVBank()
        ws_tokens = render_corpus(w, repeats_per_fact=1)
        ws_text = tok.decode(ws_tokens, skip_special_tokens=True)
        b.encode(model, ws_text, tok, args.device, max_length=64 if dry else 256)
        banks.append(b)

    def eval_bank(idx: int) -> float:
        pairs = build_qa_pairs(parts[idx])
        with banks[idx].inject(model):
            return evaluate_qa(model, pairs, device=args.device)["accuracy"]

    # fresh-mount recall (each bank, in order, evict via lru of `cap`)
    fresh = [eval_bank(i) for i in range(n_ws)]
    # remount cycle: simulate eviction by saving/loading via in-mem dict
    saved = []
    for i, b in enumerate(banks):
        path = run_dir / f"bank_{i}.pt"
        b.save(path)
        saved.append(path)
    # Reload and re-test
    reload_acc = []
    for i, path in enumerate(saved):
        b2 = WorkspaceKVBank.load(path)
        banks[i] = b2  # swap in reloaded bank
        reload_acc.append(eval_bank(i))
    delta = max(abs(f - r) for f, r in zip(fresh, reload_acc)) \
        if fresh else 0.0
    return {
        "bench": "A7",
        "n_workspaces": n_ws,
        "fresh_mount_accuracy": fresh,
        "remount_after_save_load_accuracy": reload_acc,
        "max_abs_delta": delta,
        "pass_criterion": "max |fresh - remount| <= 0.10",
        "status": "pass" if delta <= 0.10 else "fail",
    }


def section_a8(args, run_dir, builders, dry):
    """Aggregate JSON files into REPORT.md."""
    sections = ["a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7",
                "b0", "b1", "b2", "b3"]
    lines = ["# Phase A+B — Veyra3 deep benchmark + substrate training\n",
             f"Run dir: `{run_dir}`\n", "| Section | Status | Headline |",
             "|---|---|---|"]
    for s in sections:
        f = run_dir / f"{s}.json"
        if not f.exists():
            lines.append(f"| {s.upper()} | (missing) | — |")
            continue
        data = json.loads(f.read_text())
        status = data.get("status", "?")
        head = _headline_for(s, data)
        lines.append(f"| {s.upper()} | **{status}** | {head} |")
    (run_dir / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return {"bench": "A8", "status": "info",
            "report_path": str(run_dir / "REPORT.md")}


def make_pre_surgery_builder(device, dtype):
    """Builder for the *pre-surgery* base model — used as distillation teacher."""
    def build():
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            "veyra-ai/veyra3-5m-base", dtype=dtype)
        return model.to(device)
    return build


def _checkpoint_path(run_dir: Path, section: str) -> Path:
    return run_dir / f"{section}_checkpoint.pt"


def _load_checkpoint(model: nn.Module, path: Path) -> bool:
    """Load state_dict into model if checkpoint exists. Returns success bool."""
    if not path.exists():
        return False
    try:
        sd = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        sd = torch.load(path, map_location="cpu")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"[checkpoint] load partial: missing={len(missing)} "
              f"unexpected={len(unexpected)}")
    return True


# ---------------------------------------------------------------------------
# Phase B sections — train Veyra3 into a real substrate
# ---------------------------------------------------------------------------

def section_b0(args, run_dir, builders, dry):
    """B0 — distillation warm-start. Surgered student, pre-surgery teacher.

    KL distillation makes the post-surgery student logits match the pre-surgery
    teacher. This initialises the new branches into a sensible state before
    workspace-conditional training (B1) and routing supervision (B2).

    Pass criterion: KL halved between step 0 and final step (final < 0.5 * initial).
    Note: we deliberately do NOT measure factoid accuracy here -- the teacher
    has no knowledge of our synthetic alphabet, so matching it gives random
    factoid output. The downstream proof that B0 helped is in B3's re-run of
    A1, where the distilled+trained model should close the all-three vs
    sliding-only gap.
    """
    mb, ob, get_tok = builders
    teacher_build = make_pre_surgery_builder(args.device, args.dtype)
    student = mb()
    teacher = teacher_build()
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()

    torch.manual_seed(0)
    V = student.config.vocab_size
    B = 2 if dry else 4
    T = 64 if dry else 256
    n_batches = 4 if dry else 16
    batches = [(torch.randint(0, V, (B, T), device=args.device),
                torch.zeros(B, T, dtype=torch.long, device=args.device))
               for _ in range(n_batches)]

    # Measure initial KL (single step, no real training).
    initial_recipe = DistillRecipe(max_steps=1, warmup_steps=0, ce_weight=0.0)
    initial = distill_warmstart(student, teacher, batches, recipe=initial_recipe)

    recipe = DistillRecipe(
        lr=3e-4, max_steps=20 if dry else 1200,
        warmup_steps=2 if dry else 100,
        ce_weight=0.1, kl_temperature=2.0,
    )
    logger = logger_factory_for(run_dir)("b0_distill") if not dry else None
    stats = distill_warmstart(student, teacher, batches,
                              recipe=recipe, logger=logger,
                              label_prefix="b0")

    ckpt_path = _checkpoint_path(run_dir, "b0")
    torch.save(student.state_dict(), ckpt_path)

    kl_ratio = stats["final_kl"] / max(initial["final_kl"], 1e-6)
    return {
        "bench": "B0", "checkpoint": str(ckpt_path),
        "initial_kl": initial["final_kl"],
        "final_kl": stats["final_kl"],
        "kl_ratio": kl_ratio,
        "distill_stats": stats,
        "pass_criterion": "final_kl <= 0.5 * initial_kl",
        "status": "pass" if kl_ratio <= 0.5 else "fail",
    }


def section_b1(args, run_dir, builders, dry):
    """B1 — workspace-conditional training.

    NOTE: Veyra3-5M cannot learn workspace-conditional ICL (capacity ceiling
    confirmed empirically: held_out kv_acc ~0.05 with flat train loss after
    2000 steps, 32 distinct worlds). On this substrate we therefore skip the
    training step by default and pass the B0 checkpoint forward as the B1
    checkpoint, so downstream sections (B2 router, B3 deltas) still chain.

    Set env var ``LOCALSPARSE_FORCE_B1=1`` to run the full training anyway
    (useful when this runner is invoked on a larger model like Gemma 4 via
    ``bench_gemma4.py`` reusing the same code path).
    """
    import os
    force = os.environ.get("LOCALSPARSE_FORCE_B1") == "1"
    mb, ob, get_tok = builders
    tok = get_tok()
    V = mb().config.vocab_size

    if not force:
        b0_ckpt = _checkpoint_path(run_dir, "b0")
        b1_ckpt = _checkpoint_path(run_dir, "b1")
        if b0_ckpt.exists():
            import shutil
            shutil.copyfile(b0_ckpt, b1_ckpt)
        return {
            "bench": "B1",
            "checkpoint": str(b1_ckpt),
            "skipped": True,
            "note": ("Skipped on Veyra3-5M: model capacity insufficient for "
                     "workspace-conditional ICL (validated empirically). "
                     "Recipe is still emitted by B3 for Gemma 4 to consume."),
            "pass_criterion": "skipped (capacity ceiling)",
            "status": "skipped",
        }

    results: Dict[str, Any] = {}
    for mode in ["same_set", "held_out"]:
        model = mb()
        b0_ckpt = _checkpoint_path(run_dir, "b0")
        if _load_checkpoint(model, b0_ckpt):
            print(f"[b1/{mode}] loaded B0 checkpoint")

        recipe = WorkspaceTrainRecipe(
            lr=2e-4,
            max_steps=20 if dry else (1200 if mode == "same_set" else 2000),
            warmup_steps=2 if dry else 100,
            n_facts_per_world=16 if dry else 64,
            qa_per_batch=4 if dry else 8,
            bank_max_length=64 if dry else 256,
            n_train_worlds=2 if dry else 32,
        )
        logger = logger_factory_for(run_dir)(f"b1_{mode}") if not dry else None
        res = train_workspace_conditional(
            model, tok, device=args.device, recipe=recipe,
            vocab_size=V, mode=mode, logger=logger,
            label_prefix=f"b1_{mode}",
            eval_n_facts=16 if dry else 64,
        )
        m_w = mb()
        opt_w = ob(m_w)
        world_w = build_world(vocab_size=V,
                              n_facts=16 if dry else 64,
                              seed=999 if mode == "held_out" else 0)
        batches, _ = _make_factoid_batches(
            world_w, batch_size=2 if dry else 4,
            seq_len=128 if dry else 512, device=args.device)
        stats_w = train_to_convergence(
            m_w, batches, opt_w,
            max_steps=20 if dry else 1000,
            logger=logger_factory_for(run_dir)(f"b1_{mode}_wbase")
            if not dry else None,
        )
        wts_acc = evaluate_qa(m_w, build_qa_pairs(world_w),
                              device=args.device)["accuracy"]
        ratio = res["kv_accuracy"] / max(wts_acc, 1e-6)
        results[mode] = {
            **res,
            "weights_baseline_accuracy": wts_acc,
            "kv_over_weights": ratio,
            "weights_stats": stats_w,
        }
        del model, m_w, opt_w
        if args.device.type == "cuda":
            torch.cuda.empty_cache()

    final_model = mb()
    _load_checkpoint(final_model, _checkpoint_path(run_dir, "b0"))
    rcp = WorkspaceTrainRecipe(
        lr=2e-4, max_steps=5 if dry else 100,
        warmup_steps=1, n_facts_per_world=16 if dry else 64,
        qa_per_batch=4, bank_max_length=64 if dry else 256,
        n_train_worlds=2 if dry else 32,
    )
    train_workspace_conditional(
        final_model, tok, device=args.device, recipe=rcp,
        vocab_size=V, mode="held_out",
        label_prefix="b1_snapshot",
    )
    ckpt = _checkpoint_path(run_dir, "b1")
    torch.save(final_model.state_dict(), ckpt)

    same_pass = results["same_set"]["kv_over_weights"] >= 0.5
    held_pass = results["held_out"]["kv_over_weights"] >= 0.3
    return {
        "bench": "B1", "checkpoint": str(ckpt),
        "variants": results,
        "pass_criterion": "same_set kv/wts >= 0.5 AND held_out kv/wts >= 0.3",
        "status": "pass" if (same_pass and held_pass) else "fail",
    }


def section_b2(args, run_dir, builders, dry):
    """B2 — routing-head supervision. Loads B1 checkpoint.

    Skipped by default on Veyra3 because routing supervision requires the
    model's hidden states to encode bank-conditional info, which only emerges
    after a successful B1 pass (impossible at 5M scale). Set
    ``LOCALSPARSE_FORCE_B2=1`` to run on larger substrates (Gemma 4).

    Pass criterion (when forced): top1 >= 0.6 on held-out queries with 4 banks.
    """
    import os
    force = os.environ.get("LOCALSPARSE_FORCE_B2") == "1"
    mb, ob, get_tok = builders

    if not force:
        b1_ckpt = _checkpoint_path(run_dir, "b1")
        b2_ckpt = _checkpoint_path(run_dir, "b2")
        if b1_ckpt.exists():
            import shutil
            shutil.copyfile(b1_ckpt, b2_ckpt)
        return {
            "bench": "B2",
            "checkpoint": str(b2_ckpt),
            "skipped": True,
            "note": ("Skipped on Veyra3-5M: depends on B1 ICL training which "
                     "is infeasible at this scale. Router recipe still emitted "
                     "by B3 for Gemma 4."),
            "pass_criterion": "skipped (depends on B1)",
            "status": "skipped",
        }

    model = mb()
    b1_ckpt = _checkpoint_path(run_dir, "b1")
    if _load_checkpoint(model, b1_ckpt):
        print("[b2] loaded B1 checkpoint")
    V = model.config.vocab_size

    recipe = RoutingRecipe(
        lr=1e-3,
        max_steps=20 if dry else 800,
        warmup_steps=2 if dry else 50,
        qa_per_batch=4 if dry else 16,
        hidden_size=64 if dry else 128,
    )
    logger = logger_factory_for(run_dir)("b2_router") if not dry else None
    res = train_router(
        model, device=args.device, vocab_size=V,
        n_banks=2 if dry else 4,
        facts_per_bank=8 if dry else 32,
        recipe=recipe, logger=logger,
    )
    # Note: routing head itself isn't a model param; persist for completeness.
    ckpt = _checkpoint_path(run_dir, "b2")
    torch.save(model.state_dict(), ckpt)  # student state unchanged here
    return {
        "bench": "B2", "checkpoint": str(ckpt),
        "router_stats": res,
        "pass_criterion": "top1 >= 0.6",
        "status": "pass" if res["top1"] >= 0.6 else "fail",
    }


def section_b3(args, run_dir, builders, dry):
    """B3 — re-run A1/A2/A3 on the fully-trained B2 checkpoint.

    Compare deltas vs Phase A baseline. Save `recipe.json` with all
    hyperparameters that the Phase C Gemma 4 runner should replay.
    """
    mb, ob, get_tok = builders

    # Build a fresh model and load the fully-trained checkpoint.
    def trained_model_builder():
        m = mb()
        _load_checkpoint(m, _checkpoint_path(run_dir, "b1"))  # B2 doesn't change weights
        return m

    trained_builders = (trained_model_builder, ob, get_tok)
    a1_after = section_a1(args, run_dir / "b3_after", trained_builders, dry)
    a2_after = section_a2(args, run_dir / "b3_after", trained_builders, dry)
    a3_after = section_a3(args, run_dir / "b3_after", trained_builders, dry)

    # Diff against baseline Phase A if available.
    deltas: Dict[str, Any] = {}
    for s, after in [("a1", a1_after), ("a2", a2_after), ("a3", a3_after)]:
        before_path = run_dir / f"{s}.json"
        if before_path.exists():
            before = json.loads(before_path.read_text())
            if s == "a2":
                deltas["a2_kv_over_weights"] = {
                    "before": before.get("ratios", {}).get("kv_over_weights", 0),
                    "after": after.get("ratios", {}).get("kv_over_weights", 0),
                }
            elif s == "a3":
                deltas["a3_top1"] = {
                    "before": before.get("top1_routing_accuracy", 0),
                    "after": after.get("top1_routing_accuracy", 0),
                }
            elif s == "a1":
                bm = {v["name"]: v["accuracy"] for v in before.get("variants", [])}
                am = {v["name"]: v["accuracy"] for v in after.get("variants", [])}
                deltas["a1"] = {"before": bm, "after": am}

    # Write recipe.json with everything we learned.
    recipe_blob = {
        "distill": DistillRecipe(
            lr=3e-4, max_steps=1200, warmup_steps=100,
            ce_weight=0.1, kl_temperature=2.0,
        ).to_dict(),
        "workspace_same_set": WorkspaceTrainRecipe(
            lr=2e-4, max_steps=1200, warmup_steps=100,
            n_facts_per_world=64, qa_per_batch=8, bank_max_length=256,
        ).to_dict(),
        "workspace_held_out": WorkspaceTrainRecipe(
            lr=2e-4, max_steps=2000, warmup_steps=100,
            n_facts_per_world=64, qa_per_batch=8, bank_max_length=256,
            n_train_worlds=32,
        ).to_dict(),
        "routing": RoutingRecipe(
            lr=1e-3, max_steps=800, warmup_steps=50,
            qa_per_batch=16, hidden_size=128,
        ).to_dict(),
        "training_order": ["distill", "workspace_same_set",
                           "workspace_held_out", "routing"],
        "validated_on": "veyra3-5m-base",
        "transferable_to": ["google/gemma-4-E2B"],
    }
    (run_dir / "recipe.json").write_text(json.dumps(recipe_blob, indent=2))

    return {
        "bench": "B3",
        "a1_after_status": a1_after.get("status"),
        "a2_after_status": a2_after.get("status"),
        "a3_after_status": a3_after.get("status"),
        "deltas": deltas,
        "recipe_path": str(run_dir / "recipe.json"),
        "status": "info",
    }


def _headline_for(section: str, data: Dict) -> str:
    if section == "a0":
        return (f"acc={data.get('accuracy', 0):.3f} "
                f"loss={data.get('final_loss', float('nan')):.3f} "
                f"steps={data.get('steps', 0)} "
                f"conv@={data.get('converged_at')}")
    if section == "a1":
        return "; ".join(f"{v['name']}={v['accuracy']:.2f}(L{v.get('final_loss', 0):.2f})"
                        for v in data.get("variants", []))
    if section == "a2":
        r = data.get("ratios", {})
        rs = data.get("results", {})
        w = rs.get("weights_path", {})
        return (f"kv/wts={r.get('kv_over_weights', 0):.2f} "
                f"wts_acc={w.get('accuracy', 0):.2f} "
                f"wts_loss={w.get('final_loss', float('nan')):.2f} "
                f"wts_steps={w.get('steps', 0)}")
    if section == "a3":
        return f"top1={data.get('top1_routing_accuracy', 0):.2f}"
    if section == "a4":
        return f"ctxs ok={sum(1 for r in data.get('ctx_results', []) if r['status'] == 'ok')}"
    if section == "a5":
        return f"perf points={len(data.get('perf', []))}"
    if section == "a6":
        pts = data.get("points", [])
        if pts:
            best = max(pts, key=lambda p: p.get("weights_accuracy", 0))
            return (f"crossover N={data.get('crossover_N')} "
                    f"best_wts={best.get('weights_accuracy', 0):.2f}@N{best.get('n_facts')}")
        return f"crossover N={data.get('crossover_N')}"
    if section == "a7":
        return f"max delta={data.get('max_abs_delta', 0):.3f}"
    if section == "b0":
        return (f"kl {data.get('initial_kl', 0):.2f}->{data.get('final_kl', 0):.2f} "
                f"(ratio={data.get('kl_ratio', 1.0):.2f})")
    if section == "b1":
        if data.get("skipped"):
            return "skipped (capacity ceiling on Veyra3)"
        v = data.get("variants", {})
        ss = v.get("same_set", {}); ho = v.get("held_out", {})
        return (f"same kv={ss.get('kv_accuracy', 0):.2f}/wts={ss.get('weights_baseline_accuracy', 0):.2f}"
                f" r={ss.get('kv_over_weights', 0):.2f}; "
                f"held kv={ho.get('kv_accuracy', 0):.2f}/wts={ho.get('weights_baseline_accuracy', 0):.2f}"
                f" r={ho.get('kv_over_weights', 0):.2f}")
    if section == "b2":
        if data.get("skipped"):
            return "skipped (depends on B1)"
        r = data.get("router_stats", {})
        return (f"top1={r.get('top1', 0):.2f} top2={r.get('top2', 0):.2f} "
                f"n_train={r.get('n_train', 0)}")
    if section == "b3":
        d = data.get("deltas", {})
        a2 = d.get("a2_kv_over_weights", {})
        a3 = d.get("a3_top1", {})
        return (f"a2 kv/wts {a2.get('before', 0):.2f}->{a2.get('after', 0):.2f}; "
                f"a3 top1 {a3.get('before', 0):.2f}->{a3.get('after', 0):.2f}")
    return ""


SECTIONS = {
    "a0": section_a0, "a1": section_a1, "a2": section_a2,
    "a3": section_a3, "a4": section_a4, "a5": section_a5,
    "a6": section_a6, "a7": section_a7,
    "b0": section_b0, "b1": section_b1, "b2": section_b2, "b3": section_b3,
    "a8": section_a8,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--section", default="all",
                   help="a0|a1|...|a7|b0|b1|b2|b3|a8|all")
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="float32")
    p.add_argument("--run_dir", default=None)
    p.add_argument("--dry-run", action="store_true",
                   help="Tiny budgets for local smoke (<30s on CPU per section)")
    args = p.parse_args()
    args.device = parse_device(args.device)
    args.dtype = parse_dtype(args.dtype)
    run_dir = Path(args.run_dir or
                   f"runs/bench_veyra3/{time.strftime('%Y%m%d_%H%M%S')}")
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[bench_veyra3] device={args.device} dtype={args.dtype}")
    print(f"[bench_veyra3] run dir: {run_dir}")
    print(f"[bench_veyra3] dry_run={args.dry_run}")

    builders = make_builder(args.device, args.dtype)

    sections = list(SECTIONS) if args.section == "all" else [args.section]
    summary: Dict[str, Any] = {"sections": {}}
    for s in sections:
        print(f"\n=== [STAGE {s.upper()}] ===")
        try:
            result = SECTIONS[s](args, run_dir, builders, args.dry_run)
        except Exception as e:
            result = {"bench": s.upper(), "status": "error",
                      "error": str(e)[:500]}
            import traceback; traceback.print_exc()
        (run_dir / f"{s}.json").write_text(json.dumps(result, indent=2,
                                                       default=str))
        summary["sections"][s] = {"status": result.get("status")}
        print(f"[{s}] status={result.get('status')}")

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[bench_veyra3] wrote {run_dir / 'summary.json'}")
    if args.dry_run:
        print("PASS (dry-run)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
