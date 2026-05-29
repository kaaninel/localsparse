"""Lightning indexer: scores queries against compressed/super KV summaries
to drive top-k block selection for the `selected` branch of 3-branch attention.

Architecture (plan.md §2.2):
  - Separate small head per layer with d_idx (default 64), independent
    from the main K-head.
  - Operates in INT4 at inference; FP16/BF16 during training.
  - Hierarchical 2-level: super-summaries (block 4096) → compressed
    summaries (block 64) → fine-grained blocks.

This module is a *PyTorch reference impl*. A fused Triton kernel is
deferred to Milestone 4 (training-time perf gates).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional


class LightningIndexer(nn.Module):
    """Per-layer indexer head.

    Produces a small `d_idx`-dim projection of Q and K, then dot-products
    them to obtain selection scores.

    Input shapes:
      q:    (B, H_q, T_q, head_dim)
      k_compressed: (B, H_kv, N_compressed, head_dim)
      k_super:      (B, H_kv, N_super,      head_dim)

    Output:
      compressed_scores: (B, H_q, T_q, N_compressed)
      super_scores:      (B, H_q, T_q, N_super)
    """

    def __init__(self, hidden_size: int, num_heads: int, head_dim: int,
                 d_idx: int = 64):
        super().__init__()
        self.head_dim = head_dim
        self.d_idx = d_idx
        self.num_heads = num_heads
        # Small projections (head_dim → d_idx) shared across heads to keep
        # parameters tiny (4096 → 64 saves a lot vs full QK projections).
        self.q_proj = nn.Linear(head_dim, d_idx, bias=False)
        self.k_proj = nn.Linear(head_dim, d_idx, bias=False)
        # Scale chosen to keep dot-product variance ~1.
        self.scale = 1.0 / (d_idx ** 0.5)

    def score(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """q: (B, H_q, T_q, head_dim)
           k: (B, H_kv, N, head_dim)  (may be 1 KV head per group)
           returns: (B, H_q, T_q, N)
        """
        q_idx = self.q_proj(q) * self.scale           # (B, H_q, T_q, d_idx)
        k_idx = self.k_proj(k)                         # (B, H_kv, N, d_idx)
        # Broadcast GQA: expand H_kv → H_q
        if k_idx.shape[1] != q_idx.shape[1]:
            group = q_idx.shape[1] // k_idx.shape[1]
            k_idx = k_idx.repeat_interleave(group, dim=1)
        # (B, H_q, T_q, N)
        return torch.einsum("bhtd,bhnd->bhtn", q_idx, k_idx)


def hierarchical_topk(
    super_scores: torch.Tensor,        # (B, H, T, N_super)
    compressed_scores: torch.Tensor,   # (B, H, T, N_comp)
    *,
    super_to_comp_ratio: int,
    top_k_super: int,
    top_k_comp: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Two-level selection: pick top super-blocks, then top compressed
    inside the selected supers, then return their flat compressed indices.

    Returns:
      sel_comp_idx:  (B, H, T, top_k_comp)  flat compressed-block indices
      sel_comp_mask: (B, H, T, N_comp)      0/1 mask of selected blocks
    """
    B, H, T, N_super = super_scores.shape
    _, _, _, N_comp = compressed_scores.shape

    # Step 1: top-K super-blocks per query
    K_s = min(top_k_super, N_super)
    _, super_idx = super_scores.topk(K_s, dim=-1)       # (B, H, T, K_s)

    # Step 2: mask out compressed blocks NOT inside selected supers
    # Each super block covers `super_to_comp_ratio` compressed blocks.
    comp_block_super = (torch.arange(N_comp, device=compressed_scores.device)
                        // super_to_comp_ratio)         # (N_comp,)
    # Build mask: for each (B,H,T) check membership of super_idx
    # (B, H, T, N_comp, K_s) compare → reduce over K_s
    super_idx_b = super_idx.unsqueeze(-2)               # (B, H, T, 1, K_s)
    comp_block_b = comp_block_super.view(1, 1, 1, N_comp, 1)
    in_selected_super = (comp_block_b == super_idx_b).any(dim=-1)  # (B,H,T,N_comp)

    masked_comp_scores = compressed_scores.masked_fill(
        ~in_selected_super, float("-inf"))

    # Step 3: top-k compressed blocks among the survivors
    K_c = min(top_k_comp, N_comp)
    sel_scores, sel_comp_idx = masked_comp_scores.topk(K_c, dim=-1)
    # Replace any -inf positions (when fewer survivors than K_c) with 0
    sel_comp_mask = torch.zeros_like(compressed_scores)
    sel_comp_mask.scatter_(-1, sel_comp_idx, 1.0)
    return sel_comp_idx, sel_comp_mask
