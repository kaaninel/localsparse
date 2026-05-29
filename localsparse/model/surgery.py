"""Surgery: swap MiniCPM5's LlamaAttention layers with ThreeBranchAttention.

We download a base model via `transformers.AutoModelForCausalLM`, iterate
its decoder layers, and replace each `self_attn` submodule with a
`ThreeBranchAttention` that:
  - inherits Q/K/V/O weights from the base attention (zero-loss copy)
  - initializes its new components (indexer, compressed pool, super pool,
    branch_gate) — branch_gate at zero so softmax = uniform 1/3 mix
    means the network starts with non-trivial mass on every branch.

This is intentionally permissive about what the base model class is
called (`LlamaAttention`, `MiniCPMAttention`, etc): we just probe for the
standard `q_proj/k_proj/v_proj/o_proj` linears.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence
import torch
import torch.nn as nn

from ..config import LocalSparseConfig, ModelDims, AttentionConfig
from ..attention.sparse_three_branch import ThreeBranchAttention


@dataclass
class SurgeryReport:
    layers_replaced: int
    layers_skipped: int
    bytes_in_new_params: int


def _has_standard_attn_projections(mod: nn.Module) -> bool:
    return all(hasattr(mod, n) for n in ("q_proj", "k_proj", "v_proj", "o_proj"))


def _copy_projection(dst: nn.Linear, src: nn.Linear) -> None:
    with torch.no_grad():
        dst.weight.copy_(src.weight.to(dst.weight.dtype))
        if src.bias is not None and dst.bias is not None:
            dst.bias.copy_(src.bias.to(dst.bias.dtype))


def _replace_attention(parent: nn.Module, attr: str,
                       old: nn.Module, model_dims: ModelDims,
                       attn_cfg: AttentionConfig, layer_idx: int) -> int:
    new = ThreeBranchAttention(model=model_dims, attn=attn_cfg, layer_idx=layer_idx)
    new = new.to(dtype=old.q_proj.weight.dtype, device=old.q_proj.weight.device)
    _copy_projection(new.q_proj, old.q_proj)
    _copy_projection(new.k_proj, old.k_proj)
    _copy_projection(new.v_proj, old.v_proj)
    _copy_projection(new.o_proj, old.o_proj)
    setattr(parent, attr, new)
    # Tally only the *new* parameters (indexer + pools + branch_gate).
    new_bytes = 0
    for n, p in new.named_parameters():
        if n.startswith(("indexer.", "compressed_pool.", "super_pool.", "branch_gate")):
            new_bytes += p.numel() * p.element_size()
    return new_bytes


def perform_surgery(
    model: nn.Module,
    config: LocalSparseConfig,
    *,
    layer_attr_path: Sequence[str] = ("model", "layers"),
    attn_attr: str = "self_attn",
    layer_indices: Optional[Sequence[int]] = None,
) -> SurgeryReport:
    """Mutates `model` in place. Returns a SurgeryReport."""
    layers_container = model
    for name in layer_attr_path:
        layers_container = getattr(layers_container, name)

    replaced = 0
    skipped = 0
    bytes_new = 0
    for idx, layer in enumerate(layers_container):
        if layer_indices is not None and idx not in layer_indices:
            skipped += 1
            continue
        old_attn = getattr(layer, attn_attr, None)
        if old_attn is None or not _has_standard_attn_projections(old_attn):
            skipped += 1
            continue
        bytes_new += _replace_attention(
            layer, attn_attr, old_attn, config.model, config.attention, idx)
        replaced += 1
    return SurgeryReport(layers_replaced=replaced, layers_skipped=skipped,
                         bytes_in_new_params=bytes_new)


def detect_model_dims(hf_config) -> ModelDims:
    """Read a HuggingFace transformers config into our ModelDims."""
    head_dim = getattr(hf_config, "head_dim", None) or (
        hf_config.hidden_size // hf_config.num_attention_heads
    )
    return ModelDims(
        vocab_size=hf_config.vocab_size,
        hidden_size=hf_config.hidden_size,
        num_layers=hf_config.num_hidden_layers,
        num_q_heads=hf_config.num_attention_heads,
        num_kv_heads=getattr(hf_config, "num_key_value_heads", hf_config.num_attention_heads),
        head_dim=head_dim,
        intermediate_size=hf_config.intermediate_size,
    )
