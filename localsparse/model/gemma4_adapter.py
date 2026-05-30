"""Gemma 4 E2B (and family) surgery adapter (plan §7.4).

Veyra3 is a Gemma-4-style 5M model and our existing `veyra_adapter.py` already
implements the surgery for that architecture (`Gemma4ThreeBranchAttention`,
RoPE threading, per-head norms, K=V tying, layer_scalar). The differences for
production Gemma 4 E2B / E4B / 31B variants are:

  1. **Layer location**: pure-text variants use `model.model.layers`;
     conditional-generation wrappers expose the text decoder at
     `model.model.language_model.layers`; older multimodal wrappers may use
     `model.language_model.model.layers`. We probe all known paths at runtime.

  2. **Per-Layer Embeddings (PLE)**: an auxiliary residual signal added INSIDE
     each decoder layer. Because surgery only replaces `self_attn`, the PLE
     add stays intact — no special handling needed.

  3. **Variant tokens for `layer_types`**: Gemma 4 may use values like
     `"full_attention"`, `"sliding_attention"`, or the legacy
     `Gemma3` strings. We treat any layer whose `layer_types[i]` is `None`
     or equals "full_attention" as a replacement target.

This file exposes:
  - `Gemma4ESurgeryReport`
  - `surgery_gemma4(model, base_config=None)` — replaces all full-attn layers
  - `find_decoder_layers(model)` — utility to probe the layer container
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn

from ..config import LocalSparseConfig
from .veyra_adapter import (
    Gemma4ThreeBranchAttention, VeyraSurgeryReport,
    _copy_norm, _copy_q_proj, _copy_k_to_kv,
    _detect_layer_types, _veyra3_attn_cfg, _veyra3_full_attn_dims,
)
from dataclasses import replace as _replace


@dataclass
class Gemma4ESurgeryReport(VeyraSurgeryReport):
    """Extends VeyraSurgeryReport with the discovered layer-container path."""
    layers_path: str = ""


def resolve_vocab_size(model: nn.Module) -> int:
    """Return vocab_size for any Gemma 4 variant (multimodal or pure-text).

    Gemma4ForConditionalGeneration nests the language config under
    ``config.text_config``; pure-text Gemma 4 exposes ``config.vocab_size``
    directly. Falls back to the embedding weight if neither is present.
    """
    cfg = model.config
    if hasattr(cfg, "vocab_size") and cfg.vocab_size is not None:
        return int(cfg.vocab_size)
    text_cfg = getattr(cfg, "text_config", None)
    if text_cfg is not None and getattr(text_cfg, "vocab_size", None) is not None:
        return int(text_cfg.vocab_size)
    # Fallback: read from the input embedding weight
    try:
        emb = model.get_input_embeddings()
        return int(emb.weight.shape[0])
    except Exception as e:
        raise AttributeError(
            f"Could not resolve vocab_size for {type(model).__name__}: {e}")


def find_decoder_layers(model: nn.Module) -> tuple[nn.ModuleList, str]:
    """Locate the decoder-layer container, returning (layers, path_string).

    Tries (in order):
      model.model.layers                       — pure text Gemma 4 causal LM
      model.model.language_model.layers        — Gemma4ForConditionalGeneration
      model.language_model.model.layers        — older multimodal wrapper
      model.language_model.layers              — decoder-only wrapper
      model.text_model.model.layers            — alt multimodal naming
      model.text_model.layers                  — alt decoder-only naming
    """
    candidates = [
        ("model.model.layers", lambda m: m.model.layers),
        ("model.model.language_model.layers",
         lambda m: m.model.language_model.layers),
        ("model.language_model.model.layers",
         lambda m: m.language_model.model.layers),
        ("model.language_model.layers",
         lambda m: m.language_model.layers),
        ("model.text_model.model.layers",
         lambda m: m.text_model.model.layers),
        ("model.text_model.layers",
         lambda m: m.text_model.layers),
    ]
    for path, getter in candidates:
        try:
            layers = getter(model)
            if isinstance(layers, (nn.ModuleList, list)) and len(layers) > 0:
                return layers, path
        except AttributeError:
            continue
    raise AttributeError(
        f"Could not locate decoder layers on model of type "
        f"{type(model).__name__}. Tried paths: "
        + ", ".join(p for p, _ in candidates))


def surgery_gemma4(
    model: nn.Module,
    base_config: Optional[LocalSparseConfig] = None,
) -> Gemma4ESurgeryReport:
    """Replace `full_attention` layers in Gemma 4 (any variant) with our
    `Gemma4ThreeBranchAttention`. Sliding layers are left untouched."""
    notes: List[str] = []
    hf_cfg = model.config
    # For multimodal wrappers, the language config nests one level down
    text_cfg = getattr(hf_cfg, "text_config", hf_cfg)
    layer_types = _detect_layer_types(text_cfg)
    full_dims = _veyra3_full_attn_dims(text_cfg)
    if base_config is None:
        from ..config import default_config
        base_config = default_config()
    attn_cfg, rope_theta_full = _veyra3_attn_cfg(base_config.attention, text_cfg)
    full_dims = _replace(full_dims, rope_theta=rope_theta_full)

    layers_container, layers_path = find_decoder_layers(model)
    replaced: List[int] = []
    skipped: List[int] = []
    new_bytes = 0
    inh_bytes = 0

    for idx, layer in enumerate(layers_container):
        layer_type = layer_types[idx] if idx < len(layer_types) else "full_attention"
        if layer_type and layer_type != "full_attention":
            skipped.append(idx)
            continue
        old = layer.self_attn
        if not hasattr(old, "q_proj") or not hasattr(old, "k_proj"):
            notes.append(f"layer {idx}: missing q/k_proj, skipping")
            skipped.append(idx)
            continue
        if getattr(old, "store_full_length_kv", False):
            notes.append(
                f"layer {idx}: stores shared KV for later HF layers, skipping")
            skipped.append(idx)
            continue

        new_attn = Gemma4ThreeBranchAttention(
            model=full_dims, attn=attn_cfg, layer_idx=idx,
        ).to(dtype=old.q_proj.weight.dtype, device=old.q_proj.weight.device)

        inh_bytes += _copy_q_proj(new_attn.q_proj, old.q_proj)
        old_v = getattr(old, "v_proj", None)
        if old_v is not None and getattr(old_v, "weight", None) is not None:
            inh_bytes += _copy_q_proj(new_attn.k_proj, old.k_proj)
            inh_bytes += _copy_q_proj(new_attn.v_proj, old_v)
        else:
            inh_bytes += _copy_k_to_kv(new_attn.k_proj, new_attn.v_proj, old.k_proj)
            notes.append(f"layer {idx}: V tied to K, v_proj cloned")
        inh_bytes += _copy_q_proj(new_attn.o_proj, old.o_proj)

        for n in ("q_norm", "k_norm", "v_norm"):
            if hasattr(old, n):
                inh_bytes += _copy_norm(getattr(new_attn, n), getattr(old, n))

        if hasattr(layer, "layer_scalar"):
            with torch.no_grad():
                new_attn.output_scale.copy_(
                    layer.layer_scalar.detach().to(new_attn.output_scale.dtype))
                layer.layer_scalar.fill_(1.0)
            notes.append(f"layer {idx}: layer_scalar transferred")

        for n, p in new_attn.named_parameters():
            if any(n.startswith(prefix) for prefix in
                   ("indexer.", "compressed_pool.", "super_pool.", "branch_gate")):
                new_bytes += p.numel() * p.element_size()

        layer.self_attn = new_attn
        replaced.append(idx)

    return Gemma4ESurgeryReport(
        layers_replaced=replaced, layers_skipped=skipped,
        new_param_bytes=new_bytes, inherited_param_bytes=inh_bytes,
        notes=notes, layers_path=layers_path,
    )
