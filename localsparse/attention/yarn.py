"""YaRN rotary-position-embedding extension.

Implements the standard YaRN formulation for extending a model trained at
`original_max_position_embeddings` to a longer target context. Used in
Milestone 5 (256K ctx extension).

References:
  - YaRN: Efficient Context Window Extension of Large Language Models
    https://arxiv.org/abs/2309.00071
"""
from __future__ import annotations

import math
import torch


def yarn_inv_freq(
    head_dim: int,
    base: float = 10_000.0,
    scaling_factor: float = 1.0,
    original_max_position: int = 4096,
    beta_fast: int = 32,
    beta_slow: int = 1,
    extrapolation_factor: float = 1.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return YaRN-corrected inverse frequencies of shape (head_dim // 2,)."""
    if scaling_factor <= 1.0:
        # Standard RoPE
        idx = torch.arange(0, head_dim, 2, device=device, dtype=dtype)
        return 1.0 / (base ** (idx / head_dim))

    # NTK-by-parts + ramp interpolation
    def find_correction_dim(num_rotations: int) -> float:
        return (head_dim * math.log(original_max_position / (num_rotations * 2 * math.pi))
                / (2 * math.log(base)))

    low = max(math.floor(find_correction_dim(beta_fast)), 0)
    high = min(math.ceil(find_correction_dim(beta_slow)), head_dim - 1)

    idx = torch.arange(0, head_dim, 2, device=device, dtype=dtype)
    inv_freq_extrap = 1.0 / (base ** (idx / head_dim))
    inv_freq_interp = 1.0 / (scaling_factor * (base ** (idx / head_dim)))

    # ramp from interpolation (low) to extrapolation (high)
    linear = (idx - low) / max(high - low, 1)
    ramp = linear.clamp(0, 1)
    mask = 1.0 - ramp * extrapolation_factor

    return inv_freq_interp * mask + inv_freq_extrap * (1.0 - mask)


def build_rope_cache(
    seq_len: int,
    head_dim: int,
    *,
    base: float = 10_000.0,
    scaling_factor: float = 1.0,
    original_max_position: int = 4096,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    workspace_offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (cos, sin) of shape (seq_len, head_dim).

    `workspace_offset` shifts the position grid so that per-workspace
    RoPE (Scheme B4 from §O1) can be expressed by passing the workspace's
    independent position range here.
    """
    inv_freq = yarn_inv_freq(
        head_dim, base=base, scaling_factor=scaling_factor,
        original_max_position=original_max_position,
        device=device, dtype=dtype,
    )
    t = torch.arange(workspace_offset, workspace_offset + seq_len,
                     device=device, dtype=dtype)
    freqs = torch.einsum("i,j->ij", t, inv_freq)        # (T, head_dim/2)
    emb = torch.cat([freqs, freqs], dim=-1)             # (T, head_dim)
    return emb.cos(), emb.sin()


def apply_rope(q: torch.Tensor, k: torch.Tensor,
               cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE.

    q, k: (..., T, head_dim)
    cos, sin: (T, head_dim)
    """
    def rotate_half(x):
        h = x.shape[-1] // 2
        return torch.cat([-x[..., h:], x[..., :h]], dim=-1)
    # Broadcast cos/sin across batch + heads
    while cos.dim() < q.dim():
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    q_out = (q * cos) + (rotate_half(q) * sin)
    k_out = (k * cos) + (rotate_half(k) * sin)
    return q_out, k_out
