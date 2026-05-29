"""Milestone 1 trainer: surgery sanity check.

Goal: after replacing attention layers with `ThreeBranchAttention`, the
model's PPL on a held-out FineWeb subset should be within 1.3× the base
model's PPL. The training loop here is intentionally minimal — it
collects a few hundred steps of LM loss to "heal" the new layers'
randomly-initialized components without disturbing the inherited Q/K/V/O
weights too much.

The script is the single source of truth for what M1 does. The Colab
notebook just calls into it.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Callable, Iterable

import torch
import torch.nn as nn

from .data import Batch, synthetic_lm_batch
from .losses import branch_balance_loss, surgery_regression_loss


@dataclass
class M1Config:
    steps: int = 200
    batch_size: int = 1
    seq_len: int = 1024
    lr: float = 5e-5
    branch_balance_weight: float = 0.01
    surgery_kl_weight: float = 0.5
    log_every: int = 10
    grad_accum: int = 1


@dataclass
class M1Stats:
    step: int = 0
    losses: list[float] = field(default_factory=list)
    lm_losses: list[float] = field(default_factory=list)
    branch_losses: list[float] = field(default_factory=list)
    kl_losses: list[float] = field(default_factory=list)
    branch_mass_history: list[tuple[float, float, float]] = field(default_factory=list)


def collect_branch_masses(model: nn.Module) -> tuple[float, float, float]:
    """Aggregate the last forward's branch masses across all ThreeBranchAttention layers.

    We rely on the convention that ThreeBranchAttention stashes the latest
    BranchOutputs on `self._last_branch_outputs` after each forward (we
    add that below as needed).
    """
    s = sel = c = 0.0
    n = 0
    for mod in model.modules():
        bo = getattr(mod, "_last_branch_outputs", None)
        if bo is None:
            continue
        s += float(bo.sliding_mass.detach())
        sel += float(bo.selected_mass.detach())
        c += float(bo.compressed_mass.detach())
        n += 1
    if n == 0:
        return 0.0, 0.0, 0.0
    return s / n, sel / n, c / n


def train_step(
    student: nn.Module,
    teacher: Optional[nn.Module],
    batch: Batch,
    optimizer: torch.optim.Optimizer,
    cfg: M1Config,
) -> dict:
    device = next(student.parameters()).device
    ids = batch.input_ids.to(device)
    labels = batch.labels.to(device)

    out = student(input_ids=ids, labels=labels)
    lm_loss = out.loss

    branch_loss = torch.zeros((), device=device, dtype=lm_loss.dtype)
    s, sel, c = collect_branch_masses(student)
    if s + sel + c > 0:
        branch_loss = cfg.branch_balance_weight * branch_balance_loss(
            torch.tensor(s, device=device),
            torch.tensor(sel, device=device),
            torch.tensor(c, device=device),
        )

    kl_loss = torch.zeros((), device=device, dtype=lm_loss.dtype)
    if teacher is not None:
        with torch.no_grad():
            t_out = teacher(input_ids=ids)
        kl_loss = surgery_regression_loss(
            out.logits, t_out.logits, weight=cfg.surgery_kl_weight)

    total = lm_loss + branch_loss + kl_loss
    (total / cfg.grad_accum).backward()
    return {
        "total": float(total.detach()),
        "lm": float(lm_loss.detach()),
        "branch": float(branch_loss.detach()),
        "kl": float(kl_loss.detach()),
        "branch_mass": (s, sel, c),
    }


def run_m1(
    student: nn.Module,
    teacher: Optional[nn.Module],
    batch_iter: Iterable[Batch],
    cfg: M1Config,
    *,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> M1Stats:
    optimizer = optimizer or torch.optim.AdamW(student.parameters(), lr=cfg.lr)
    stats = M1Stats()
    batch_iter = iter(batch_iter)
    t0 = time.time()
    for step in range(cfg.steps):
        try:
            batch = next(batch_iter)
        except StopIteration:
            break
        log = train_step(student, teacher, batch, optimizer, cfg)
        if (step + 1) % cfg.grad_accum == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        stats.step = step
        stats.losses.append(log["total"])
        stats.lm_losses.append(log["lm"])
        stats.branch_losses.append(log["branch"])
        stats.kl_losses.append(log["kl"])
        stats.branch_mass_history.append(log["branch_mass"])
        if (step + 1) % cfg.log_every == 0:
            elapsed = time.time() - t0
            print(
                f"[m1] step={step+1}/{cfg.steps} "
                f"total={log['total']:.4f} lm={log['lm']:.4f} "
                f"branch={log['branch']:.4f} kl={log['kl']:.4f} "
                f"mass={log['branch_mass']} t={elapsed:.1f}s"
            )
    return stats
