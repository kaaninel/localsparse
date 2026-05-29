"""Distillation warm-start for post-surgery models (Phase B0).

After we surgery a base LM (Veyra3 / Gemma 4 / ...) by swapping selected
attention layers with `Gemma4ThreeBranchAttention`, the newly inserted
selected/compressed branches start from a random/identity-ish state. The
branch gate has not learned to weight them, so they actively *hurt*
accuracy at small scale (Phase A A1 finding: sliding-only > all-three by
+8pp on Veyra3-5M).

This module performs a brief KL-distillation warm-start where:
  - teacher = pre-surgery clone of the same checkpoint (frozen)
  - student = surgered model (all params trainable)
  - loss   = KL(teacher_logits || student_logits) with optional CE on
             argmax token labels for stability when teacher is sharp.

The recipe (LR schedule, KL temperature, freeze schedule, total steps)
is the *transferable artefact*: weight values do not transfer between
Veyra3 and Gemma 4 (different shapes), but a recipe that works for
Veyra3 typically scales to Gemma 4 with at most LR re-tuning.

Returns a `DistillRecipe` dataclass that bench_gemma4 can replay.
"""
from __future__ import annotations

import time
from copy import deepcopy
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..logging import RunLogger


@dataclass
class DistillRecipe:
    """Hyperparameters used in a distillation run. Transferable to Gemma."""
    lr: float = 3e-4
    weight_decay: float = 0.0
    kl_temperature: float = 2.0
    kl_weight: float = 1.0
    ce_weight: float = 0.1
    max_steps: int = 1500
    warmup_steps: int = 100
    freeze_parent_steps: int = 0  # if >0, freeze original (non-branch) params for N steps
    grad_clip: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DistillRecipe":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def make_teacher_clone(student_pre_surgery_factory) -> nn.Module:
    """Build a frozen teacher from a fresh pre-surgery checkpoint.

    `student_pre_surgery_factory()` must return a *fresh* base model load
    (NO surgery applied). We freeze and return it.
    """
    teacher = student_pre_surgery_factory()
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()
    return teacher


def _new_param_names(student: nn.Module) -> List[str]:
    """Names of parameters that were *added* by surgery (i.e. new branches).

    Heuristic: any parameter under a ThreeBranchAttention module path.
    """
    from ..attention.sparse_three_branch import ThreeBranchAttention
    names: List[str] = []
    for mod_name, mod in student.named_modules():
        if isinstance(mod, ThreeBranchAttention):
            for p_name, _ in mod.named_parameters():
                names.append(f"{mod_name}.{p_name}" if mod_name else p_name)
    return names


def _set_requires_grad(model: nn.Module, names: List[str], requires: bool) -> None:
    nameset = set(names)
    for n, p in model.named_parameters():
        if n in nameset:
            p.requires_grad = requires


def _lr_schedule(step: int, recipe: DistillRecipe) -> float:
    """Linear warmup then linear decay to 10% of base lr."""
    base = recipe.lr
    if step < recipe.warmup_steps:
        return base * (step + 1) / max(1, recipe.warmup_steps)
    remain = max(1, recipe.max_steps - recipe.warmup_steps)
    progress = (step - recipe.warmup_steps) / remain
    return base * (1.0 - 0.9 * min(1.0, progress))


def distill_warmstart(
    student: nn.Module,
    teacher: nn.Module,
    batches: List[Tuple[torch.Tensor, torch.Tensor]],
    *,
    recipe: Optional[DistillRecipe] = None,
    logger: Optional[RunLogger] = None,
    label_prefix: str = "distill",
) -> Dict[str, Any]:
    """Run KL distillation. Teacher provides soft targets; student matches them.

    Returns dict {steps, final_loss, final_kl, final_ce, wall_seconds, recipe}.
    """
    recipe = recipe or DistillRecipe()
    t0 = time.time()
    device = next(student.parameters()).device
    teacher.to(device)

    # Freeze schedule: optionally freeze non-branch params at start.
    branch_names = _new_param_names(student)
    if recipe.freeze_parent_steps > 0:
        for n, p in student.named_parameters():
            p.requires_grad = (n in set(branch_names))

    opt = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=recipe.lr, weight_decay=recipe.weight_decay,
    )

    n_batches = len(batches)
    last = {"kl": float("nan"), "ce": float("nan"), "total": float("nan")}
    for step in range(recipe.max_steps):
        # Unfreeze parent once freeze window ends.
        if step == recipe.freeze_parent_steps and recipe.freeze_parent_steps > 0:
            for p in student.parameters():
                p.requires_grad = True
            opt = torch.optim.AdamW(student.parameters(), lr=recipe.lr,
                                    weight_decay=recipe.weight_decay)

        # Manual LR override (so we don't need a Scheduler object).
        lr_now = _lr_schedule(step, recipe)
        for g in opt.param_groups:
            g["lr"] = lr_now

        ids, _ = batches[step % n_batches]
        opt.zero_grad(set_to_none=True)

        with torch.no_grad():
            t_out = teacher(input_ids=ids)
            t_logits = t_out.logits

        s_out = student(input_ids=ids)
        s_logits = s_out.logits

        T = recipe.kl_temperature
        # KL(teacher_soft || student_soft) — use KLDivLoss with log_softmax input.
        kl = F.kl_div(
            F.log_softmax(s_logits / T, dim=-1),
            F.softmax(t_logits / T, dim=-1),
            reduction="batchmean",
        ) * (T * T)

        # Optional CE on teacher argmax (stabilises when teacher is very sharp).
        ce = torch.tensor(0.0, device=device)
        if recipe.ce_weight > 0:
            with torch.no_grad():
                hard = t_logits.argmax(dim=-1)
            ce = F.cross_entropy(
                s_logits.view(-1, s_logits.size(-1)),
                hard.view(-1),
            )

        loss = recipe.kl_weight * kl + recipe.ce_weight * ce
        loss.backward()
        if recipe.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(student.parameters(), recipe.grad_clip)
        opt.step()

        last = {
            "kl": float(kl.detach()),
            "ce": float(ce.detach()),
            "total": float(loss.detach()),
        }
        if logger is not None and (step % 50 == 0 or step == recipe.max_steps - 1):
            logger.step(
                step=step, lm_loss=last["total"], total_loss=last["total"],
                kl_loss=last["kl"], ce_loss=last["ce"], lr=lr_now,
                phase=label_prefix,
                tokens_per_sec=ids.numel() * (step + 1) / max(time.time() - t0, 1e-6),
            )

    return {
        "steps": recipe.max_steps,
        "final_loss": last["total"],
        "final_kl": last["kl"],
        "final_ce": last["ce"],
        "wall_seconds": time.time() - t0,
        "recipe": recipe.to_dict(),
    }
