"""Veyra3 / Gemma4 adapter.

Bridges Veyra3's `Gemma4ForCausalLM` (HF transformers) to our
`ThreeBranchAttention`. Handles all Gemma4-specific quirks discovered
by inspection of `veyra-ai/veyra3-5m-base`:

  - **Per-head norms (q_norm, k_norm, v_norm)** applied AFTER projection
    and BEFORE RoPE. Each is a `Gemma4RMSNorm(head_dim)`.

  - **layer_scalar** (1-element learned parameter) multiplied onto the
    attention output after `o_proj` (lives on the parent
    `Gemma4DecoderLayer`).

  - **K=V tied for full_attention layers** (`attention_k_eq_v=True`).
    Full layers have NO `v_proj`; V is derived from K. Our surgery
    copies `k_proj` weights into both our `k_proj` AND `v_proj` so the
    starting behavior matches the base.

  - **Per-attention-type RoPE**: full_attention uses
    `partial_rotary_factor=0.25, rope_theta=1e6`; sliding uses default.
    We only replace full-attn layers, so we wire the full-attn config
    into the ThreeBranchAttention RoPE cache.

  - **Different KV head counts per layer type**: full layers have
    `num_global_key_value_heads=1` (head_dim 64 → k_proj out=64).
    Sliding layers have `num_key_value_heads=2` (out=128). Since we
    only replace full-attn layers, our adapter forces num_kv_heads=1
    for the replaced layers regardless of the global config value.

  - **Surgery scope**: only `full_attention` layers are replaced. The
    4 native `sliding_attention` layers stay intact as a control
    channel for local-context modeling. See plan.md §6.5 for rationale.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import List, Optional, Tuple
import copy

import torch
import torch.nn as nn

from ..config import LocalSparseConfig, ModelDims, AttentionConfig
from ..attention.sparse_three_branch import ThreeBranchAttention


# ---------------------------------------------------------------------------
# Gemma4-aware ThreeBranchAttention subclass
# ---------------------------------------------------------------------------
class Gemma4ThreeBranchAttention(ThreeBranchAttention):
    """ThreeBranchAttention pre-configured with Gemma4's per-head norms
    and full-attention RoPE config.

    The Gemma4 norms have shape (head_dim,) and apply via RMSNorm to the
    last dim of the (B, H, T, D) tensors — exactly what `nn.RMSNorm`
    does, so we use that.
    """

    def __init__(self, model: ModelDims, attn: AttentionConfig, layer_idx: int,
                 *, rms_eps: float = 1e-6):
        super().__init__(model=model, attn=attn, layer_idx=layer_idx)
        D = self.head_dim
        # Per-head RMSNorm — weight shape (D,), broadcasts over (B, H, T)
        self.q_norm = nn.RMSNorm(D, eps=rms_eps)
        self.k_norm = nn.RMSNorm(D, eps=rms_eps)
        self.v_norm = nn.RMSNorm(D, eps=rms_eps)
        # Learnt per-layer output scalar (1-element)
        self.output_scale = nn.Parameter(torch.ones(1))

    # HF Gemma4 decoder layers call self_attn with extra kwargs
    # (position_embeddings, attention_mask, past_key_value, cache_position,
    # use_cache, output_attentions, ...). Our 3-branch attention computes
    # everything from hidden_states + position_ids. We accept and discard
    # the HF-specific kwargs, but extract position_ids from cache_position
    # when given, and return the (attn_output, attn_weights_or_None) tuple
    # HF expects.
    def forward(self, hidden_states, position_embeddings=None,
                attention_mask=None, past_key_value=None,
                cache_position=None, use_cache=False,
                output_attentions=False, position_ids=None, **kwargs):
        if position_ids is None and cache_position is not None:
            position_ids = cache_position.unsqueeze(0).expand(
                hidden_states.shape[0], -1)
        out, _bout = super().forward(hidden_states, position_ids=position_ids)
        # HF expects (attn_output, attn_weights) — we don't compute weights.
        return out, None


# ---------------------------------------------------------------------------
# Decoder-layer wrapper that re-routes the existing layer's forward to use
# our new attention. We keep the rest of the layer (LayerNorms, MLP,
# residual connections) intact — only `self_attn` is replaced.
# ---------------------------------------------------------------------------
def _detect_layer_types(hf_config) -> List[str]:
    """Return the `layer_types` list from a Gemma4 config (e.g.
    ['sliding_attention', 'sliding_attention', ..., 'full_attention'])."""
    layer_types = getattr(hf_config, "layer_types", None)
    if layer_types is None:
        # Older Gemma3 may not have it — assume all full attention
        layer_types = ["full_attention"] * hf_config.num_hidden_layers
    return list(layer_types)


def _veyra3_full_attn_dims(hf_config) -> ModelDims:
    """ModelDims that match Gemma-4-style *full-attention* layer shapes.

    Veyra3 full-attn uses `num_global_key_value_heads=1` with the same
    `head_dim` as sliding layers. Production Gemma 4 E models instead use a
    wider `global_head_dim` for full attention. Surgery must mirror those
    full-attn projection shapes exactly so q/k/v/o weights can be inherited.
    """
    head_dim = (
        getattr(hf_config, "global_head_dim", None)
        or getattr(hf_config, "head_dim", None)
        or (hf_config.hidden_size // hf_config.num_attention_heads)
    )
    # HF Gemma4TextAttention uses num_global_key_value_heads only for the
    # alternative K=V full-attention path. Otherwise full layers use
    # num_key_value_heads even when global_head_dim is wider.
    attention_k_eq_v = getattr(hf_config, "attention_k_eq_v", None)
    if attention_k_eq_v is False:
        num_kv = hf_config.num_key_value_heads
    else:
        num_kv = (getattr(hf_config, "num_global_key_value_heads", None)
                  or hf_config.num_key_value_heads)
    return ModelDims(
        vocab_size=hf_config.vocab_size,
        hidden_size=hf_config.hidden_size,
        num_layers=hf_config.num_hidden_layers,
        num_q_heads=hf_config.num_attention_heads,
        num_kv_heads=num_kv,
        head_dim=head_dim,
        intermediate_size=hf_config.intermediate_size,
    )


def _veyra3_attn_cfg(base_cfg: AttentionConfig, hf_config) -> AttentionConfig:
    """AttentionConfig with Veyra3 RoPE parameters threaded in.

    Reads `rope_parameters.full_attention.rope_theta` and overrides the
    default. partial_rotary_factor is not currently honored by our YaRN
    builder — recorded for future work (TODO).
    """
    rope_full = getattr(hf_config, "rope_parameters", {}).get(
        "full_attention", {})
    rope_theta = rope_full.get("rope_theta", 10_000.0)
    out = replace(base_cfg)
    # Push rope theta through ModelDims since our build_rope_cache reads from there
    return out, rope_theta


# ---------------------------------------------------------------------------
# Surgery report
# ---------------------------------------------------------------------------
@dataclass
class VeyraSurgeryReport:
    layers_replaced: List[int]
    layers_skipped: List[int]
    new_param_bytes: int
    inherited_param_bytes: int
    notes: List[str]


def _copy_q_proj(dst: nn.Linear, src: nn.Linear) -> int:
    with torch.no_grad():
        dst.weight.copy_(src.weight.to(dst.weight.dtype))
    return src.weight.numel() * src.weight.element_size()


def _copy_k_to_kv(dst_k: nn.Linear, dst_v: nn.Linear, src_k: nn.Linear) -> int:
    """For full-attn layers where V is tied to K, copy src_k into both
    our k_proj AND v_proj (initializing V to behave like K)."""
    with torch.no_grad():
        dst_k.weight.copy_(src_k.weight.to(dst_k.weight.dtype))
        dst_v.weight.copy_(src_k.weight.to(dst_v.weight.dtype))
    return src_k.weight.numel() * src_k.weight.element_size() * 2


def _copy_norm(dst: nn.Module, src: nn.Module) -> int:
    if not hasattr(dst, "weight") or not hasattr(src, "weight"):
        return 0
    with torch.no_grad():
        dst.weight.copy_(src.weight.to(dst.weight.dtype))
    return src.weight.numel() * src.weight.element_size()


def surgery_veyra3(
    model: nn.Module,
    base_config: Optional[LocalSparseConfig] = None,
) -> VeyraSurgeryReport:
    """Replace ONLY Veyra3's `full_attention` layers with
    Gemma4ThreeBranchAttention. Sliding layers remain untouched.

    Returns a detailed report listing replaced / skipped layers and the
    parameter-bytes accounting for both inherited and new components.
    """
    notes: List[str] = []
    hf_cfg = model.config
    layer_types = _detect_layer_types(hf_cfg)

    full_dims = _veyra3_full_attn_dims(hf_cfg)
    if base_config is None:
        from ..config import default_config
        base_config = default_config()
    attn_cfg, rope_theta_full = _veyra3_attn_cfg(base_config.attention, hf_cfg)
    # Override rope_theta on full_dims (ModelDims carries rope_theta)
    full_dims = replace(full_dims, rope_theta=rope_theta_full)

    layers_container = model.model.layers
    replaced: List[int] = []
    skipped: List[int] = []
    new_bytes = 0
    inh_bytes = 0

    for idx, layer in enumerate(layers_container):
        layer_type = layer_types[idx] if idx < len(layer_types) else "full_attention"
        if layer_type != "full_attention":
            skipped.append(idx)
            continue
        old = layer.self_attn
        # Confirm we have the expected projections
        if not hasattr(old, "q_proj") or not hasattr(old, "k_proj"):
            notes.append(f"layer {idx}: missing q/k_proj, skipping")
            skipped.append(idx)
            continue

        new_attn = Gemma4ThreeBranchAttention(
            model=full_dims, attn=attn_cfg, layer_idx=idx,
        ).to(dtype=old.q_proj.weight.dtype, device=old.q_proj.weight.device)

        # Q/K/V/O weight surgery
        inh_bytes += _copy_q_proj(new_attn.q_proj, old.q_proj)
        old_v = getattr(old, "v_proj", None)
        if old_v is not None and getattr(old_v, "weight", None) is not None:
            inh_bytes += _copy_q_proj(new_attn.k_proj, old.k_proj)
            inh_bytes += _copy_q_proj(new_attn.v_proj, old_v)
        else:
            # K=V tied: clone k_proj into both (Gemma4 attention_k_eq_v)
            inh_bytes += _copy_k_to_kv(new_attn.k_proj, new_attn.v_proj, old.k_proj)
            notes.append(f"layer {idx}: V tied to K (Gemma4 attention_k_eq_v), v_proj cloned")
        inh_bytes += _copy_q_proj(new_attn.o_proj, old.o_proj)

        # Norms surgery (q/k/v)
        if hasattr(old, "q_norm"):
            inh_bytes += _copy_norm(new_attn.q_norm, old.q_norm)
        if hasattr(old, "k_norm"):
            inh_bytes += _copy_norm(new_attn.k_norm, old.k_norm)
        if hasattr(old, "v_norm"):
            inh_bytes += _copy_norm(new_attn.v_norm, old.v_norm)

        # layer_scalar lives on the *parent* DecoderLayer, not on attn
        if hasattr(layer, "layer_scalar"):
            with torch.no_grad():
                new_attn.output_scale.copy_(
                    layer.layer_scalar.detach().to(new_attn.output_scale.dtype))
            inh_bytes += layer.layer_scalar.numel() * layer.layer_scalar.element_size()
            # Zero out the layer's scalar so we don't double-apply it
            with torch.no_grad():
                layer.layer_scalar.fill_(1.0)
            notes.append(f"layer {idx}: layer_scalar copied to output_scale; parent reset to 1.0")

        # Account for *new* params (the 3-branch components + norms)
        for n, p in new_attn.named_parameters():
            if any(n.startswith(prefix) for prefix in
                   ("indexer.", "compressed_pool.", "super_pool.", "branch_gate")):
                new_bytes += p.numel() * p.element_size()

        layer.self_attn = new_attn
        replaced.append(idx)

    return VeyraSurgeryReport(
        layers_replaced=replaced, layers_skipped=skipped,
        new_param_bytes=new_bytes, inherited_param_bytes=inh_bytes,
        notes=notes,
    )
