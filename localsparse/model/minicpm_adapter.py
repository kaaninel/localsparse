"""MiniCPM5 / generic LlamaAttention surgery adapter.

Bridges HF LlamaAttention-style attention layers (used in MiniCPM5, Llama3,
Mistral, etc.) to our ThreeBranchAttention. This is the M1-target adapter.

Key difference from Veyra3/Gemma4:
  - LlamaAttention returns (attn_output, attn_weights, past_key_value) — 3 values
  - Gemma4Attention returns (attn_output, attn_weights) — 2 values
  - MiniCPM5 may have per-layer scale (scaling_factor) or other quirks

Usage:

    from localsparse.model.minicpm_adapter import surgery_minicpm
    report = surgery_minicpm(model, config)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List

import torch
import torch.nn as nn

from ..config import LocalSparseConfig, ModelDims, AttentionConfig
from ..attention.sparse_three_branch import ThreeBranchAttention


# ---------------------------------------------------------------------------
# LlamaAttention-compatible subclass (covers MiniCPM5, Llama3, Mistral, …)
# ---------------------------------------------------------------------------

class LlamaThreeBranchAttention(ThreeBranchAttention):
    """ThreeBranchAttention with HF LlamaAttention forward signature.

    LlamaAttention's forward returns (attn_output, attn_weights, past_key_value).
    We return (out, None, None).
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple] = None,
        **kwargs,
    ):
        if position_ids is None and cache_position is not None:
            position_ids = cache_position.unsqueeze(0).expand(
                hidden_states.shape[0], -1
            )
        out, _bout = ThreeBranchAttention.forward(
            self, hidden_states, position_ids=position_ids
        )
        # LlamaAttention returns 3 values: (output, weights, past_kv)
        return out, None, None


# ---------------------------------------------------------------------------
# Surgery helper
# ---------------------------------------------------------------------------

@dataclass
class MiniCPMSurgeryReport:
    layers_replaced: int
    layers_skipped: int
    new_param_bytes: int
    model_id: str
    notes: List[str]


def _has_standard_projections(mod: nn.Module) -> bool:
    return all(hasattr(mod, n) for n in ("q_proj", "k_proj", "v_proj", "o_proj"))


def _infer_model_dims(hf_config) -> ModelDims:
    """Infer ModelDims from a HF config (Llama/MiniCPM style)."""
    hidden_size = hf_config.hidden_size
    num_q = getattr(hf_config, "num_attention_heads", None) or \
            getattr(hf_config, "num_heads", None)
    num_kv = getattr(hf_config, "num_key_value_heads", None) or num_q
    head_dim = getattr(hf_config, "head_dim", None) or (hidden_size // num_q)
    vocab_size = hf_config.vocab_size
    num_layers = getattr(hf_config, "num_hidden_layers", None) or \
                 getattr(hf_config, "num_layers", None)
    rope_theta = getattr(hf_config, "rope_theta", 10000.0)

    return ModelDims(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_q_heads=num_q,
        num_kv_heads=num_kv,
        head_dim=head_dim,
        rope_theta=rope_theta,
    )


def surgery_minicpm(
    model: nn.Module,
    config: LocalSparseConfig,
    layer_indices: Optional[List[int]] = None,
) -> MiniCPMSurgeryReport:
    """Replace LlamaAttention layers in model with LlamaThreeBranchAttention.

    Args:
        model:          HF CausalLM (openbmb/MiniCPM5-1B or similar).
        config:         LocalSparseConfig controlling attention hyperparams.
        layer_indices:  Which layer indices to replace. None = replace all.

    Returns:
        MiniCPMSurgeryReport with statistics.
    """
    hf_config = model.config
    model_dims = _infer_model_dims(hf_config)
    attn_cfg = config.attention

    notes = []

    # Navigate to the decoder layers container
    # Typical HF layout: model.model.layers or model.layers
    layers_container = None
    for attr_path in [("model", "layers"), ("transformer", "layers"), ("layers",)]:
        obj = model
        try:
            for attr in attr_path:
                obj = getattr(obj, attr)
            layers_container = obj
            break
        except AttributeError:
            continue

    if layers_container is None:
        raise RuntimeError(
            "Cannot find decoder layers in model. Expected model.model.layers, "
            "model.transformer.layers, or model.layers."
        )

    replaced = 0
    skipped = 0
    new_bytes = 0

    for idx, layer in enumerate(layers_container):
        if layer_indices is not None and idx not in layer_indices:
            skipped += 1
            continue

        old_attn = getattr(layer, "self_attn", None)
        if old_attn is None or not _has_standard_projections(old_attn):
            skipped += 1
            notes.append(f"layer {idx}: skipped (no standard q/k/v/o_proj)")
            continue

        # Build replacement
        new_attn = LlamaThreeBranchAttention(
            model=model_dims, attn=attn_cfg, layer_idx=idx
        )
        new_attn = new_attn.to(
            dtype=old_attn.q_proj.weight.dtype,
            device=old_attn.q_proj.weight.device,
        )

        # Copy Q/K/V/O weights
        def _copy(dst: nn.Linear, src: nn.Linear):
            with torch.no_grad():
                dst.weight.copy_(src.weight.to(dst.weight.dtype))
                if src.bias is not None and dst.bias is not None:
                    dst.bias.copy_(src.bias.to(dst.bias.dtype))

        _copy(new_attn.q_proj, old_attn.q_proj)
        _copy(new_attn.k_proj, old_attn.k_proj)
        _copy(new_attn.v_proj, old_attn.v_proj)
        _copy(new_attn.o_proj, old_attn.o_proj)

        layer.self_attn = new_attn
        replaced += 1

        # Count new parameters
        for n, p in new_attn.named_parameters():
            if n.startswith(("indexer.", "compressed_pool.", "super_pool.", "branch_gate")):
                new_bytes += p.numel() * p.element_size()

    model_id = getattr(hf_config, "_name_or_path", "unknown")
    return MiniCPMSurgeryReport(
        layers_replaced=replaced,
        layers_skipped=skipped,
        new_param_bytes=new_bytes,
        model_id=model_id,
        notes=notes,
    )
