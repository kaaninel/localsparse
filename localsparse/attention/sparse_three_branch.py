"""3-branch sparse attention (NSA / DeepSeek-V4 style).

Three parallel KV stores per attention layer, summed at attention output:

  - Sliding   : last `window` tokens, full fidelity, always resident.
  - Selected  : top-k fine-grained 64-token blocks chosen by the lightning
                indexer (paged from disk in production; in-memory here).
  - Compressed: 1 summary KV per 64 source tokens (learned MLP pool) +
                hierarchical super-summaries (1 per 4096 source tokens).

This is a PyTorch reference implementation that:
  - matches the dimensional contract for surgery onto MiniCPM5;
  - is fully differentiable (compressed-summary MLP, indexer Q/K projections,
    branch gate);
  - runs on CPU for small inputs (validated by tests);
  - is ready to swap in Triton kernels for the per-block top-k attention
    in M2 once we have GPU.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, NamedTuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import AttentionConfig, ModelDims
from .indexer import LightningIndexer, hierarchical_topk
from .yarn import build_rope_cache, apply_rope


@dataclass
class BranchOutputs:
    sliding: torch.Tensor       # (B, T_q, H, head_dim)
    selected: torch.Tensor      # (B, T_q, H, head_dim)
    compressed: torch.Tensor    # (B, T_q, H, head_dim)
    sliding_mass: torch.Tensor  # scalar attention-mass per branch (for M2 eval)
    selected_mass: torch.Tensor
    compressed_mass: torch.Tensor


class CompressedSummaryPool(nn.Module):
    """Learned pooling over `block_size` tokens of K/V → 1 summary K/V.

    A small MLP eats the concatenated KV block and emits a single summary
    KV pair. Implemented as a depthwise 1D conv + linear so it stays
    parameter-light (matters for a 1B model).
    """

    def __init__(self, head_dim: int, num_kv_heads: int, block_size: int):
        super().__init__()
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.block_size = block_size
        # Pool along the block dimension. Two parallel pools (K and V).
        # Input: (B*H, head_dim, block_size) → output (B*H, head_dim, 1)
        self.pool_k = nn.Conv1d(head_dim, head_dim, block_size, groups=head_dim, bias=False)
        self.pool_v = nn.Conv1d(head_dim, head_dim, block_size, groups=head_dim, bias=False)
        # Init to mean pooling.
        with torch.no_grad():
            self.pool_k.weight.fill_(1.0 / block_size)
            self.pool_v.weight.fill_(1.0 / block_size)

    def forward(self, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """k, v: (B, H_kv, T, head_dim). T must be a multiple of block_size.

        Returns: (B, H_kv, T // block_size, head_dim).
        """
        B, H, T, D = k.shape
        assert T % self.block_size == 0, f"T={T} not divisible by block={self.block_size}"
        N = T // self.block_size
        k_blocks = k.reshape(B * H, N, self.block_size, D).permute(0, 3, 1, 2)  # (B*H, D, N, blk)
        v_blocks = v.reshape(B * H, N, self.block_size, D).permute(0, 3, 1, 2)
        # pool over blk → (B*H, D, N, 1)
        k_pool = self.pool_k(k_blocks.reshape(B * H, D, N * self.block_size))
        v_pool = self.pool_v(v_blocks.reshape(B * H, D, N * self.block_size))
        # The conv with stride 1 and kernel=block produces overlapping
        # windows; we want non-overlapping. Switch to manual stride.
        k_unf = k.reshape(B, H, N, self.block_size, D).mean(dim=3)  # placeholder
        v_unf = v.reshape(B, H, N, self.block_size, D).mean(dim=3)
        # Apply per-channel scaling from the conv weights (depthwise 1D
        # over a single window) — produces a learned weighted sum that
        # initializes to mean.
        wk = self.pool_k.weight.squeeze(1)  # (head_dim, block_size)
        wv = self.pool_v.weight.squeeze(1)
        k_blocks2 = k.reshape(B, H, N, self.block_size, D)   # (B, H, N, blk, D)
        v_blocks2 = v.reshape(B, H, N, self.block_size, D)
        # einsum: pool weights over (blk, D) axes
        k_out = torch.einsum("bhnsd,ds->bhnd", k_blocks2, wk)
        v_out = torch.einsum("bhnsd,ds->bhnd", v_blocks2, wv)
        return k_out, v_out


class ThreeBranchAttention(nn.Module):
    """Replace a single LlamaAttention layer with 3-branch sparse attention.

    Public API matches a HF-style attention forward enough to be drop-in
    via `model/surgery.py`.

    Forward signature (training): (hidden_states, position_ids, ...)
        → (output, attn_state)

    Branch gate: a small learned (H, 3) softmax that combines the three
    branches per head per step.
    """

    def __init__(self, model: ModelDims, attn: AttentionConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.model_dims = model
        self.attn_cfg = attn
        self.hidden_size = model.hidden_size
        self.num_q_heads = model.num_q_heads
        self.num_kv_heads = model.num_kv_heads
        self.head_dim = model.head_dim
        self.q_per_kv = self.num_q_heads // self.num_kv_heads

        # Standard Q/K/V/O projections (same shape as MiniCPM5's, so we can
        # init from the surgery script with the base weights).
        self.q_proj = nn.Linear(self.hidden_size, self.num_q_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_q_heads * self.head_dim, self.hidden_size, bias=False)

        # 3-branch components
        self.indexer = LightningIndexer(
            self.hidden_size, self.num_q_heads, self.head_dim, d_idx=attn.indexer_dim,
        )
        self.compressed_pool = CompressedSummaryPool(
            self.head_dim, self.num_kv_heads, attn.compressed_block,
        )
        self.super_pool = CompressedSummaryPool(
            self.head_dim, self.num_kv_heads, attn.super_block // attn.compressed_block,
        )
        # Per-head 3-way branch gate logits → softmax → mix weights.
        self.branch_gate = nn.Parameter(torch.zeros(self.num_q_heads, 3))

        # ---- Optional extension hooks (default no-op, used by adapters like Gemma4)
        # Apply per-head normalisation after projection, before RoPE.
        self.q_norm: nn.Module = nn.Identity()
        self.k_norm: nn.Module = nn.Identity()
        self.v_norm: nn.Module = nn.Identity()
        # Optional per-layer output scalar (Gemma4 layer_scalar). When None, skipped.
        self.output_scale: Optional[nn.Parameter] = None

    # ---- shape helpers ---------------------------------------------------
    def _shape_q(self, x: torch.Tensor, T: int) -> torch.Tensor:
        return x.view(x.shape[0], T, self.num_q_heads, self.head_dim).transpose(1, 2)

    def _shape_kv(self, x: torch.Tensor, T: int) -> torch.Tensor:
        return x.view(x.shape[0], T, self.num_kv_heads, self.head_dim).transpose(1, 2)

    # ---- attention helpers ----------------------------------------------
    @staticmethod
    def _sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
              mask: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Scaled-dot-product attention, NaN-safe for all-masked rows."""
        scale = 1.0 / math.sqrt(q.shape[-1])
        scores = torch.einsum("bhqd,bhkd->bhqk", q, k) * scale
        if mask is not None:
            scores = scores + mask
        # Detect rows that are entirely -inf (no permitted keys)
        all_masked = torch.isinf(scores).all(dim=-1, keepdim=True)
        # Replace -inf with 0 in those rows so softmax doesn't NaN; we then
        # zero out the produced output for those rows.
        scores_safe = torch.where(all_masked, torch.zeros_like(scores), scores)
        attn = scores_safe.softmax(dim=-1)
        attn = torch.where(all_masked, torch.zeros_like(attn), attn)
        out = torch.einsum("bhqk,bhkd->bhqd", attn, v)
        mass = attn.sum(dim=(-1, -2)).mean()
        return out, mass

    @staticmethod
    def _gqa_expand(kv: torch.Tensor, q_per_kv: int) -> torch.Tensor:
        """Repeat KV heads along the head dim to match Q's head count."""
        return kv.repeat_interleave(q_per_kv, dim=1)

    # ---- branches --------------------------------------------------------
    def _sliding_branch(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sliding-window attention over the last `window` tokens of k/v."""
        T_k = k.shape[2]
        w = min(self.attn_cfg.sliding_window, T_k)
        k_w = k[:, :, -w:, :]
        v_w = v[:, :, -w:, :]
        k_w = self._gqa_expand(k_w, self.q_per_kv)
        v_w = self._gqa_expand(v_w, self.q_per_kv)
        # Causal mask within the sliding window
        T_q = q.shape[2]
        mask = self._causal_mask(T_q, w, device=q.device, dtype=q.dtype, offset=T_k - w)
        return self._sdpa(q, k_w, v_w, mask=mask)

    @staticmethod
    def _causal_mask(T_q: int, T_k: int, device, dtype,
                     offset: int = 0) -> torch.Tensor:
        """Standard causal mask. Assumes queries align with the LAST T_q
        positions of the key range, with optional `offset` applied to the
        key range start (for sliding branch convenience)."""
        # q_abs position = T_k - T_q + q  (q in [0, T_q))
        # allowed iff k <= q_abs ⟺ k <= q + (T_k - T_q)
        q = torch.arange(T_q, device=device).unsqueeze(1)   # (T_q, 1)
        k = torch.arange(T_k, device=device).unsqueeze(0)   # (1, T_k)
        allowed = k <= (q + (T_k - T_q))
        mask = torch.zeros(T_q, T_k, device=device, dtype=dtype)
        mask = mask.masked_fill(~allowed, float("-inf"))
        return mask

    def _compressed_branch(self, q: torch.Tensor,
                           k_comp: torch.Tensor, v_comp: torch.Tensor,
                           k_super: torch.Tensor, v_super: torch.Tensor,
                           ) -> tuple[torch.Tensor, torch.Tensor]:
        """Attend to compressed + super summaries (always available, even
        post-eviction)."""
        # Concatenate super + compressed; super entries get amplified
        # because they represent more tokens.
        k_cat = torch.cat([k_super, k_comp], dim=2)
        v_cat = torch.cat([v_super, v_comp], dim=2)
        k_cat = self._gqa_expand(k_cat, self.q_per_kv)
        v_cat = self._gqa_expand(v_cat, self.q_per_kv)
        return self._sdpa(q, k_cat, v_cat)

    def _selected_branch(self, q: torch.Tensor,
                         k_full: torch.Tensor, v_full: torch.Tensor,
                         k_comp: torch.Tensor, k_super: torch.Tensor,
                         ) -> tuple[torch.Tensor, torch.Tensor]:
        """Indexer scores → top-k compressed-block selection → attend to
        the corresponding fine-grained blocks."""
        # Scores over compressed and super summaries
        compressed_scores = self.indexer.score(q, k_comp)   # (B, H, T_q, N_comp)
        super_scores = self.indexer.score(q, k_super)       # (B, H, T_q, N_super)

        ratio = self.attn_cfg.super_block // self.attn_cfg.compressed_block
        top_k_super = max(1, self.attn_cfg.selected_top_k // 2)
        top_k_comp = self.attn_cfg.selected_top_k

        sel_idx, sel_mask = hierarchical_topk(
            super_scores, compressed_scores,
            super_to_comp_ratio=ratio,
            top_k_super=top_k_super,
            top_k_comp=top_k_comp,
        )
        # sel_idx: (B, H, T_q, top_k_comp) of compressed-block indices
        # Stash for downstream introspection (e.g. G3 indexer routing gate).
        self._last_selected_indices = sel_idx.detach()
        # Build fine-token mask: each compressed block covers `compressed_block` fine tokens.
        N_fine = k_full.shape[2]
        cb = self.attn_cfg.compressed_block
        # Pull fine-grained K/V slices for each (B, H, T_q, K_c) → memory-heavy in
        # the dense reference impl; production uses paged-block gather.
        B, H_q, T_q, K_c = sel_idx.shape
        H_kv = k_full.shape[1]
        # Build an attention mask over fine tokens: 1 if token's compressed block is selected
        comp_id = torch.arange(N_fine, device=k_full.device) // cb  # (N_fine,)
        # For each (B, H_q, T_q) check if comp_id ∈ sel_idx
        # sel_mask: (B, H_q, T_q, N_comp). N_comp = N_fine // cb.
        fine_mask = sel_mask.repeat_interleave(cb, dim=-1)[:, :, :, :N_fine]   # (B, H_q, T_q, N_fine)

        # Expand kv heads
        k_e = self._gqa_expand(k_full, self.q_per_kv)
        v_e = self._gqa_expand(v_full, self.q_per_kv)

        # Standard scaled dot-product attention, but with a multiplicative
        # mask that zeros non-selected entries (then re-softmax).
        scale = 1.0 / math.sqrt(q.shape[-1])
        scores = torch.einsum("bhqd,bhkd->bhqk", q, k_e) * scale
        # Apply causal + selection mask
        causal = self._causal_mask(T_q, N_fine, device=q.device, dtype=q.dtype)
        scores = scores + causal
        scores = scores.masked_fill(fine_mask == 0, float("-inf"))
        # NaN-safe softmax: rows with no permitted keys become all-zero attn.
        all_masked = torch.isinf(scores).all(dim=-1, keepdim=True)
        scores_safe = torch.where(all_masked, torch.zeros_like(scores), scores)
        attn = scores_safe.softmax(dim=-1)
        attn = torch.where(all_masked, torch.zeros_like(attn), attn)
        out = torch.einsum("bhqk,bhkd->bhqd", attn, v_e)
        mass = attn.sum(dim=(-1, -2)).mean()
        return out, mass

    # ---- main forward ----------------------------------------------------
    def forward(
        self,
        hidden_states: torch.Tensor,         # (B, T, hidden)
        position_ids: Optional[torch.Tensor] = None,
        *,
        cached_k: Optional[torch.Tensor] = None,   # (B, H_kv, T_past, head_dim)
        cached_v: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, BranchOutputs]:
        B, T, _ = hidden_states.shape
        q = self._shape_q(self.q_proj(hidden_states), T)   # (B, H_q, T, D)
        k = self._shape_kv(self.k_proj(hidden_states), T)
        v = self._shape_kv(self.v_proj(hidden_states), T)

        # Optional per-head normalisation (Gemma4-style q/k/v_norm).
        q = self.q_norm(q)
        k = self.k_norm(k)
        v = self.v_norm(v)

        # RoPE
        if position_ids is None:
            position_ids = torch.arange(T, device=hidden_states.device)
        cos, sin = build_rope_cache(
            T, self.head_dim,
            base=self.model_dims.rope_theta,
            scaling_factor=self.attn_cfg.yarn_factor,
            original_max_position=self.attn_cfg.yarn_original_max,
            device=hidden_states.device, dtype=hidden_states.dtype,
        )
        q, k = apply_rope(q, k, cos, sin)

        # Concatenate cached past K/V if any (cache also gets RoPE applied
        # upstream when first written).
        if cached_k is not None:
            k_full = torch.cat([cached_k, k], dim=2)
            v_full = torch.cat([cached_v, v], dim=2)
        else:
            k_full = k
            v_full = v

        # ---- Workspace KV injection ----------------------------------------
        # WorkspaceKVBank sets self._ws_kv = (k_ws, v_ws) before calling
        # forward. These are pre-encoded KV representations of workspace
        # chunks that are prepended as a "soft prefix" — all branches will
        # attend to them naturally through the normal attention mechanism.
        ws_kv = getattr(self, '_ws_kv', None)
        if ws_kv is not None:
            k_ws, v_ws = ws_kv  # (B, H_kv, T_ws, head_dim)
            k_full = torch.cat([k_ws.to(dtype=k_full.dtype, device=k_full.device), k_full], dim=2)
            v_full = torch.cat([v_ws.to(dtype=v_full.dtype, device=v_full.device), v_full], dim=2)

        # ---- KV capture mode (used by WorkspaceKVBank.encode) --------------
        if getattr(self, '_capture_kv', False):
            self._captured_kv = (k_full.detach().cpu(), v_full.detach().cpu())

        # ---- Compute branch K/V ----------------------------------------
        T_kv = k_full.shape[2]
        # Pad k_full/v_full up to a multiple of super_block for pooling
        sb = self.attn_cfg.super_block
        cb = self.attn_cfg.compressed_block
        pad = (-T_kv) % cb
        if pad:
            k_pad = F.pad(k_full, (0, 0, 0, pad))
            v_pad = F.pad(v_full, (0, 0, 0, pad))
        else:
            k_pad = k_full
            v_pad = v_full
        # Compressed (always available)
        k_comp, v_comp = self.compressed_pool(k_pad, v_pad)   # (B, H_kv, N_comp, D)
        # Super-pool reduces compressed → super
        n_comp = k_comp.shape[2]
        ratio = sb // cb
        pad_c = (-n_comp) % ratio
        if pad_c:
            k_comp_pad = F.pad(k_comp, (0, 0, 0, pad_c))
            v_comp_pad = F.pad(v_comp, (0, 0, 0, pad_c))
        else:
            k_comp_pad = k_comp
            v_comp_pad = v_comp
        k_super, v_super = self.super_pool(k_comp_pad, v_comp_pad)  # (B, H_kv, N_super, D)

        # ---- Branches --------------------------------------------------
        out_sliding, mass_s = self._sliding_branch(q, k_full, v_full)
        out_compressed, mass_c = self._compressed_branch(q, k_comp, v_comp, k_super, v_super)
        out_selected, mass_x = self._selected_branch(q, k_full, v_full, k_comp, k_super)

        # ---- Gate mix --------------------------------------------------
        gate = F.softmax(self.branch_gate, dim=-1)  # (H, 3)
        # Optional branch disabling (used by BranchAblation for A1 ablations):
        # zero out specified branches AFTER softmax so the surviving branches
        # are renormalised implicitly via the residual sum dropping out.
        disabled = getattr(self, "_disabled_branches", None)
        if disabled:
            mask = torch.ones(3, device=gate.device, dtype=gate.dtype)
            if "sliding" in disabled:
                mask[0] = 0.0
            if "selected" in disabled:
                mask[1] = 0.0
            if "compressed" in disabled:
                mask[2] = 0.0
            gate = gate * mask.view(1, 3)
            # Renormalise so remaining branches still sum to 1
            denom = gate.sum(dim=-1, keepdim=True).clamp(min=1e-6)
            gate = gate / denom
        # Permute to (B, T, H, D)
        outs = torch.stack([
            out_sliding.transpose(1, 2),
            out_selected.transpose(1, 2),
            out_compressed.transpose(1, 2),
        ], dim=-1)  # (B, T, H, D, 3)
        gate_b = gate.view(1, 1, self.num_q_heads, 1, 3)
        mixed = (outs * gate_b).sum(dim=-1)         # (B, T, H, D)
        merged = mixed.reshape(B, T, self.num_q_heads * self.head_dim)
        out = self.o_proj(merged)
        if self.output_scale is not None:
            out = out * self.output_scale

        bout = BranchOutputs(
            sliding=out_sliding, selected=out_selected, compressed=out_compressed,
            sliding_mass=mass_s, selected_mass=mass_x, compressed_mass=mass_c,
        )
        self._last_branch_outputs = bout
        return out, bout
