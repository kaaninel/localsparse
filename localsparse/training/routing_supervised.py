"""Routing-head supervision (Phase B2).

Phase A's A3 measured *unsupervised* routing — for each query, it brute-
forced which of N banks gave the highest correct-answer logit and called
that "routing top-1". On Veyra3-5M this scored 0.26 (random for N=4).

This module supervises a tiny routing classifier so the model learns to
predict *which workspace bank to mount* given a query. The head takes
the model's last-token hidden state (computed without any bank injected)
and produces a softmax over bank indices.

Supervision: each (query, ground-truth-bank-index) pair from the
factoid worlds. Standard CE.

In production this head sits in front of the agent loop:
  query → router.predict(bank_id) → bank.inject() → answer.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..logging import RunLogger
from .factoid_world import (
    FactoidWorld, build_qa_pairs, build_world, partition_facts,
)


@dataclass
class RoutingRecipe:
    """Transferable hyperparameters for the bank routing head."""
    lr: float = 1e-3
    weight_decay: float = 0.0
    max_steps: int = 800
    warmup_steps: int = 50
    hidden_size: int = 128
    qa_per_batch: int = 16
    grad_clip: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RoutingRecipe":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class RouterHead(nn.Module):
    """Tiny MLP that maps a query embedding to bank-index logits."""

    def __init__(self, *, model_hidden: int, n_banks: int, hidden_size: int = 128):
        super().__init__()
        self.n_banks = n_banks
        self.net = nn.Sequential(
            nn.Linear(model_hidden, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, n_banks),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _model_hidden_size(model: nn.Module) -> int:
    """Best-effort: look at config.hidden_size, or peek a parameter."""
    cfg = getattr(model, "config", None)
    if cfg is not None and hasattr(cfg, "hidden_size"):
        return int(cfg.hidden_size)
    emb = next((m for m in model.modules() if isinstance(m, nn.Embedding)), None)
    if emb is not None:
        return emb.embedding_dim
    raise RuntimeError("Cannot infer model hidden size for RouterHead")


def _last_hidden_for_prompt(model: nn.Module, prompt_ids: List[int],
                            device, *, layer: int = -1) -> torch.Tensor:
    """Run the model forward and grab the last-token hidden state."""
    x = torch.tensor(prompt_ids, device=device).unsqueeze(0)
    with torch.no_grad():
        out = model(input_ids=x, output_hidden_states=True)
    hs = out.hidden_states[layer]  # (1, T, H)
    return hs[0, -1]


def _gather_router_dataset(
    model: nn.Module,
    worlds: List[FactoidWorld],
    *,
    device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build (X, y): X = (N, hidden); y = (N,) bank-index labels.

    Hidden states are computed without any bank injected — the router
    decides which bank to mount BEFORE injection happens.
    """
    feats: List[torch.Tensor] = []
    labels: List[int] = []
    for bank_idx, w in enumerate(worlds):
        pairs = build_qa_pairs(w)
        for prompt, _ans in pairs:
            h = _last_hidden_for_prompt(model, prompt, device)
            feats.append(h.detach().float().cpu())
            labels.append(bank_idx)
    X = torch.stack(feats, dim=0)
    y = torch.tensor(labels, dtype=torch.long)
    return X, y


def _lr_schedule(step: int, recipe: RoutingRecipe) -> float:
    base = recipe.lr
    if step < recipe.warmup_steps:
        return base * (step + 1) / max(1, recipe.warmup_steps)
    remain = max(1, recipe.max_steps - recipe.warmup_steps)
    progress = (step - recipe.warmup_steps) / remain
    return base * (1.0 - 0.9 * min(1.0, progress))


def train_router(
    model: nn.Module,
    *,
    device,
    vocab_size: int,
    n_banks: int = 4,
    facts_per_bank: int = 64,
    holdout_fraction: float = 0.2,
    recipe: Optional[RoutingRecipe] = None,
    seed: int = 0,
    logger: Optional[RunLogger] = None,
    label_prefix: str = "router",
) -> Dict[str, Any]:
    """Train RouterHead on (query → bank-index) supervision.

    Build N partitioned worlds, gather hidden-state features for each
    QA prompt, split into train/holdout, train the head, eval top-1.
    """
    recipe = recipe or RoutingRecipe()
    t0 = time.time()

    world = build_world(vocab_size=vocab_size,
                        n_facts=n_banks * facts_per_bank, seed=seed)
    worlds = partition_facts(world, n_banks, seed=seed)

    X, y = _gather_router_dataset(model, worlds, device=device)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(X.size(0), generator=g)
    X, y = X[perm], y[perm]
    n_hold = max(1, int(holdout_fraction * X.size(0)))
    X_tr, X_te = X[n_hold:], X[:n_hold]
    y_tr, y_te = y[n_hold:], y[:n_hold]

    hidden = _model_hidden_size(model)
    head = RouterHead(model_hidden=hidden, n_banks=n_banks,
                      hidden_size=recipe.hidden_size).to(device).float()
    X_tr = X_tr.to(device); y_tr = y_tr.to(device)
    X_te = X_te.to(device); y_te = y_te.to(device)

    opt = torch.optim.AdamW(head.parameters(), lr=recipe.lr,
                            weight_decay=recipe.weight_decay)

    last_loss = float("nan")
    last_acc = float("nan")
    rng = torch.Generator(device="cpu").manual_seed(seed)

    for step in range(recipe.max_steps):
        idx = torch.randint(0, X_tr.size(0), (recipe.qa_per_batch,),
                            generator=rng).to(device)
        xb = X_tr[idx]
        yb = y_tr[idx]

        lr_now = _lr_schedule(step, recipe)
        for g_ in opt.param_groups:
            g_["lr"] = lr_now

        opt.zero_grad(set_to_none=True)
        logits = head(xb)
        loss = F.cross_entropy(logits, yb)
        loss.backward()
        if recipe.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(head.parameters(), recipe.grad_clip)
        opt.step()

        last_loss = float(loss.detach())
        with torch.no_grad():
            last_acc = float((logits.argmax(-1) == yb).float().mean())

        if logger is not None and (step % 50 == 0 or step == recipe.max_steps - 1):
            logger.step(step=step, lm_loss=last_loss, total_loss=last_loss,
                        train_acc=last_acc, lr=lr_now, phase=label_prefix)

    head.eval()
    with torch.no_grad():
        te_logits = head(X_te)
        te_pred = te_logits.argmax(-1)
        top1 = float((te_pred == y_te).float().mean())
        top2 = float(((te_logits.topk(min(2, n_banks), dim=-1).indices
                       == y_te.unsqueeze(-1)).any(-1)).float().mean())

    return {
        "n_banks": n_banks,
        "facts_per_bank": facts_per_bank,
        "n_train": int(X_tr.size(0)),
        "n_holdout": int(X_te.size(0)),
        "top1": top1,
        "top2": top2,
        "final_train_loss": last_loss,
        "final_train_acc": last_acc,
        "steps": recipe.max_steps,
        "wall_seconds": time.time() - t0,
        "recipe": recipe.to_dict(),
    }
