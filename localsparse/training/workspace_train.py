"""Workspace-conditional training (Phase B1).

After distillation warm-start (B0), the surgered model behaves like the
original at the logit level but still does not *use* injected KVs from
the workspace bank — the kv_acc column in Phase A's A2 was 0.000 because
the model never received a training signal that taught it to attend to
injected K/V.

This module fixes that. Each training step:
  1. Sample a fact set (the "workspace world")
  2. Encode its rendered corpus into a WorkspaceKVBank
  3. Inject the bank → run the model on a QA prompt → CE loss on answer
  4. Backprop. The bank tensors are kept frozen (they came from a
     no_grad encode pass); only the live model parameters update.

Two variants:
  - `same_set`: train + eval on the same fact set (memorisation-OK,
    cheapest signal that the mount path works at all).
  - `held_out`: train on world A facts → eval on world B facts with B's
    bank injected. True generalisation: the model must learn the
    *protocol* of using injected KVs, not the specific facts.

The training itself is fast because the bank.encode pass is no-grad and
the forward+backward is on a short QA prompt (~10 tokens). On Veyra3-5M
MPS we expect <5 min for 1000 steps with N=64 facts.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..logging import RunLogger
from ..workspace.kv_bank import WorkspaceKVBank
from .factoid_world import (
    FactoidWorld, build_qa_pairs, build_world, evaluate_qa, render_corpus,
)


@dataclass
class WorkspaceTrainRecipe:
    """Transferable hyperparameters for workspace-conditional training."""
    lr: float = 2e-4
    weight_decay: float = 0.0
    max_steps: int = 1500
    warmup_steps: int = 100
    grad_clip: float = 1.0
    n_facts_per_world: int = 64
    qa_per_batch: int = 8
    bank_max_length: int = 512
    n_train_worlds: int = 32
    """Number of distinct factoid worlds rotated during training.

    With 1 world, weight-memorisation wins (model ignores the bank).
    With 32+ worlds, weight-memorisation is infeasible -> gradient
    descent is forced to learn to attend to the injected bank.
    """
    n_object_label_smoothing: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "WorkspaceTrainRecipe":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def _build_qa_batch(qa_pairs: List[Tuple[List[int], int]], *, indices: List[int],
                    device, pad_id: int = 0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack a list of (prompt, ans) into a left-padded batch.

    Returns (input_ids, attention_mask, answer_ids).
    answer_ids has shape (B,) — we score on the last non-pad position.
    """
    selected = [qa_pairs[i] for i in indices]
    max_len = max(len(p) for p, _ in selected)
    B = len(selected)
    input_ids = torch.full((B, max_len), pad_id, dtype=torch.long, device=device)
    attn_mask = torch.zeros((B, max_len), dtype=torch.long, device=device)
    answers = torch.zeros(B, dtype=torch.long, device=device)
    for i, (prompt, ans) in enumerate(selected):
        L = len(prompt)
        # Left-pad so the last token position is the prompt's last token.
        input_ids[i, max_len - L:] = torch.tensor(prompt, device=device)
        attn_mask[i, max_len - L:] = 1
        answers[i] = ans
    return input_ids, attn_mask, answers


def _encode_world_bank(model: nn.Module, world: FactoidWorld, tokenizer, device,
                       *, max_length: int) -> WorkspaceKVBank:
    """Encode a world's rendered corpus into a fresh bank (no_grad)."""
    bank = WorkspaceKVBank()
    ws_tokens = render_corpus(world, repeats_per_fact=1)
    ws_text = tokenizer.decode(ws_tokens, skip_special_tokens=True)
    bank.encode(model, ws_text, tokenizer, device, max_length=max_length)
    return bank


def _lr_schedule(step: int, recipe: WorkspaceTrainRecipe) -> float:
    base = recipe.lr
    if step < recipe.warmup_steps:
        return base * (step + 1) / max(1, recipe.warmup_steps)
    remain = max(1, recipe.max_steps - recipe.warmup_steps)
    progress = (step - recipe.warmup_steps) / remain
    return base * (1.0 - 0.9 * min(1.0, progress))


def train_workspace_conditional(
    model: nn.Module,
    tokenizer,
    *,
    device,
    recipe: Optional[WorkspaceTrainRecipe] = None,
    vocab_size: int,
    mode: str = "same_set",  # "same_set" | "held_out"
    eval_n_facts: Optional[int] = None,
    logger: Optional[RunLogger] = None,
    label_prefix: str = "wstrain",
    seed: int = 0,
) -> Dict[str, Any]:
    """Train the model to use injected KV banks for QA.

    Returns dict with kv_acc and weights-path baseline for diff.
    """
    recipe = recipe or WorkspaceTrainRecipe()
    t0 = time.time()

    if mode not in {"same_set", "held_out"}:
        raise ValueError(f"unknown mode={mode}")

    opt = torch.optim.AdamW(model.parameters(), lr=recipe.lr,
                            weight_decay=recipe.weight_decay)

    # Pre-build training world(s). For "same_set", just 1 world.
    # For "held_out", many worlds so the model can't memorise -> must use bank.
    train_worlds: List[FactoidWorld] = []
    if mode == "same_set":
        train_worlds = [build_world(vocab_size=vocab_size,
                                    n_facts=recipe.n_facts_per_world, seed=seed)]
    else:
        for i in range(recipe.n_train_worlds):
            train_worlds.append(build_world(vocab_size=vocab_size,
                                            n_facts=recipe.n_facts_per_world,
                                            seed=seed + 1 + i))

    # Pre-encode banks (no_grad) and prepare QA index pools per world.
    bank_and_qa: List[Tuple[WorkspaceKVBank, List[Tuple[List[int], int]]]] = []
    for w in train_worlds:
        b = _encode_world_bank(model, w, tokenizer, device,
                               max_length=recipe.bank_max_length)
        qa = build_qa_pairs(w)
        bank_and_qa.append((b, qa))

    last_loss = float("nan")
    last_acc = float("nan")
    rng = torch.Generator(device="cpu").manual_seed(seed)

    for step in range(recipe.max_steps):
        # Randomly sample a (bank, qa) pair this step (instead of round-robin).
        # With many worlds + random sampling, weight-memorisation becomes
        # infeasible — gradient descent has to use the bank to maintain loss.
        w_idx = int(torch.randint(0, len(bank_and_qa), (1,),
                                  generator=rng).item())
        bank, qa_pairs = bank_and_qa[w_idx]

        # Sample qa_per_batch QA pairs.
        idx = torch.randint(0, len(qa_pairs), (recipe.qa_per_batch,),
                            generator=rng).tolist()
        input_ids, attn_mask, answers = _build_qa_batch(
            qa_pairs, indices=idx, device=device)

        lr_now = _lr_schedule(step, recipe)
        for g in opt.param_groups:
            g["lr"] = lr_now

        opt.zero_grad(set_to_none=True)
        with bank.inject(model):
            out = model(input_ids=input_ids, attention_mask=attn_mask)
        logits = out.logits[:, -1, :]  # (B, V)

        if recipe.n_object_label_smoothing > 0:
            loss = F.cross_entropy(
                logits, answers,
                label_smoothing=recipe.n_object_label_smoothing,
            )
        else:
            loss = F.cross_entropy(logits, answers)
        loss.backward()
        if recipe.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), recipe.grad_clip)
        opt.step()

        last_loss = float(loss.detach())
        with torch.no_grad():
            pred = logits.argmax(dim=-1)
            last_acc = float((pred == answers).float().mean())

        if logger is not None and (step % 50 == 0 or step == recipe.max_steps - 1):
            logger.step(step=step, lm_loss=last_loss, total_loss=last_loss,
                        qa_acc=last_acc, lr=lr_now, phase=label_prefix,
                        tokens_per_sec=input_ids.numel() * (step + 1)
                                       / max(time.time() - t0, 1e-6))

    # Evaluation.
    if mode == "same_set":
        eval_world = train_worlds[0]
    else:
        # Truly held-out world for held_out mode.
        eval_world = build_world(vocab_size=vocab_size,
                                 n_facts=eval_n_facts or recipe.n_facts_per_world,
                                 seed=seed + 999)

    eval_bank = _encode_world_bank(model, eval_world, tokenizer, device,
                                   max_length=recipe.bank_max_length)
    eval_qa = build_qa_pairs(eval_world)

    with eval_bank.inject(model):
        kv_res = evaluate_qa(model, eval_qa, device=device)
    no_bank_res = evaluate_qa(model, eval_qa, device=device)

    return {
        "mode": mode,
        "kv_accuracy": kv_res["accuracy"],
        "no_mount_accuracy": no_bank_res["accuracy"],
        "kv_minus_nomount": kv_res["accuracy"] - no_bank_res["accuracy"],
        "steps": recipe.max_steps,
        "final_loss": last_loss,
        "final_train_acc": last_acc,
        "wall_seconds": time.time() - t0,
        "recipe": recipe.to_dict(),
        "n_eval_facts": len(eval_qa),
    }
