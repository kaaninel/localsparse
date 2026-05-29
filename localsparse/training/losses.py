"""Training losses.

Three auxiliary losses are defined alongside the standard LM cross-entropy:

  - `branch_balance_loss`: encourages each of (sliding, selected, compressed)
    to carry non-trivial attention mass (≥5% by default). Used in M2.
  - `selection_consistency_loss`: EMA of indexer selection across steps;
    penalises per-step flipping without forbidding legitimate switches.
    Used in M8.
  - `surgery_regression_loss`: KL between surgery'd model logits and the
    base model's. Keeps the surgery'd attention from drifting too far in
    M1 before compressed/selected branches are trained.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def branch_balance_loss(
    sliding_mass: torch.Tensor,
    selected_mass: torch.Tensor,
    compressed_mass: torch.Tensor,
    *,
    floor: float = 0.05,
) -> torch.Tensor:
    """Hinge penalty on each branch's mass falling below `floor`.
    Each input is a scalar tensor (average attention mass over heads/positions)."""
    masses = torch.stack([sliding_mass, selected_mass, compressed_mass])
    deficit = torch.clamp(floor - masses, min=0.0)
    return deficit.pow(2).sum()


class SelectionConsistencyLoss(nn.Module):
    """Per-layer EMA of indexer selection logits → MSE penalty.

    The EMA is updated each forward call. `alpha` controls decay (high =
    smoother). The loss term measures `||logits_t - EMA||²`.

    Usage:
        loss_fn = SelectionConsistencyLoss(num_layers=24)
        for step in range(...):
            logits = collect_indexer_logits()  # (L, B, H, T_q, n_blocks)
            aux = loss_fn(logits)
    """

    def __init__(self, num_layers: int, alpha: float = 0.9, weight: float = 0.05):
        super().__init__()
        self.alpha = alpha
        self.weight = weight
        self._ema: dict[int, torch.Tensor] = {}
        self.num_layers = num_layers

    def reset(self) -> None:
        self._ema.clear()

    def forward(self, layer_idx: int, logits: torch.Tensor) -> torch.Tensor:
        prev = self._ema.get(layer_idx)
        if prev is None or prev.shape != logits.shape:
            self._ema[layer_idx] = logits.detach()
            return torch.zeros((), device=logits.device, dtype=logits.dtype)
        new_ema = self.alpha * prev + (1.0 - self.alpha) * logits.detach()
        self._ema[layer_idx] = new_ema
        return self.weight * F.mse_loss(logits, prev)


def surgery_regression_loss(
    student_logits: torch.Tensor,    # (B, T, V)
    teacher_logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    weight: float = 1.0,
) -> torch.Tensor:
    """KL(teacher || student) at given temperature; teacher is detached."""
    t = teacher_logits.detach() / temperature
    s = student_logits / temperature
    return weight * F.kl_div(
        F.log_softmax(s, dim=-1),
        F.softmax(t, dim=-1),
        reduction="batchmean",
    ) * (temperature ** 2)


def routing_ce_loss(routing_logits: torch.Tensor,
                    target_workspace_ids: torch.Tensor) -> torch.Tensor:
    """Cross-entropy on indexer routing target for the M8 synthetic recipe.

    routing_logits: (B, num_workspaces)
    target_workspace_ids: (B,)
    """
    return F.cross_entropy(routing_logits, target_workspace_ids)
