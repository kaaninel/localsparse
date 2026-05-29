"""Shared helpers for Phase B/C trainers (M0.5).

These tiny utilities exist to keep the gate-runner scripts in
`scripts/` thin and uniform.
"""
from __future__ import annotations

import time
from typing import Iterable, List, Tuple, Optional, Dict, Any

import torch
import torch.nn as nn

from ..logging import RunLogger, FailureDetector, per_module_grad_norms
from .factoid_world import FactoidWorld, build_qa_pairs, evaluate_qa
from .milestone1 import collect_branch_masses


def train_facts(
    model: nn.Module, *, batches: List[Tuple[torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer, epochs: int, logger: RunLogger,
    detector: Optional[FailureDetector] = None,
    log_branch_masses: bool = True, label_prefix: str = "",
) -> Dict[str, float]:
    """Generic train loop on (input_ids, labels) batches. Returns final stats."""
    t0 = time.time()
    step = 0
    last_loss = float("nan")
    for ep in range(epochs):
        for (ids, lbls) in batches:
            optimizer.zero_grad(set_to_none=True)
            out = model(input_ids=ids, labels=lbls)
            loss = out.loss
            loss.backward()
            gn = per_module_grad_norms(model)
            optimizer.step()
            rec: Dict[str, Any] = {
                "step": step, "epoch": ep, "lm_loss": float(loss.detach()),
                "total_loss": float(loss.detach()), **gn,
            }
            if log_branch_masses:
                s, sel, c = collect_branch_masses(model)
                rec.update({"sliding_mass": s, "selected_mass": sel,
                            "compressed_mass": c})
            if label_prefix:
                rec["phase"] = label_prefix
            rec["tokens_per_sec"] = ids.numel() * (step + 1) / max(time.time() - t0, 1e-6)
            logger.step(**rec)
            if detector is not None:
                detector.check(rec)
            last_loss = rec["lm_loss"]
            step += 1
    return {"steps": step, "final_loss": last_loss,
            "wall_seconds": time.time() - t0}


def eval_world(model: nn.Module, world: FactoidWorld, *, device,
               k_eval: int = 0) -> Dict[str, float]:
    pairs = build_qa_pairs(world)
    return evaluate_qa(model, pairs, device=device, k_eval=k_eval)
