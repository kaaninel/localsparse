"""Tests for the Gemma 4 E2B / family surgery adapter (plan §7.4).

Uses a tiny synthetic Gemma-4-style model so no HF download is needed.
Two wrappers are tested:
  1. Pure text:    `model.model.layers`
  2. Multimodal:   `model.language_model.model.layers`

Both should be detected by `find_decoder_layers` and surgery should
replace only `full_attention` layers in both cases.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from localsparse.model.gemma4_adapter import (
    Gemma4ESurgeryReport, find_decoder_layers, surgery_gemma4,
)
from localsparse.model.veyra_adapter import Gemma4ThreeBranchAttention
from tests.test_veyra_adapter import (
    _FakeFullAttn, _FakeGemma4Config, _FakeLayer, _FakeModel,
)


class _FakeMultimodalModel(nn.Module):
    """Mirrors HF Gemma 4 multimodal: `model.language_model.model.layers`."""
    def __init__(self):
        super().__init__()
        # text_config on the outer config — adapter should drill into it
        class _OuterCfg:
            text_config = _FakeGemma4Config()
        self.config = _OuterCfg()

        class _Inner(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([
                    _FakeLayer(is_full=(t == "full_attention"))
                    for t in _FakeGemma4Config.layer_types
                ])

        class _LM(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = _Inner()

        self.language_model = _LM()


# ---------------------------------------------------------------------------
# find_decoder_layers
# ---------------------------------------------------------------------------
def test_find_layers_pure_text():
    m = _FakeModel()
    layers, path = find_decoder_layers(m)
    assert path == "model.model.layers"
    assert len(layers) == 4


def test_find_layers_multimodal():
    m = _FakeMultimodalModel()
    layers, path = find_decoder_layers(m)
    assert path == "model.language_model.model.layers"
    assert len(layers) == 4


def test_find_layers_missing_raises():
    class _Bad(nn.Module):
        pass
    with pytest.raises(AttributeError):
        find_decoder_layers(_Bad())


# ---------------------------------------------------------------------------
# Surgery on pure-text
# ---------------------------------------------------------------------------
def test_surgery_pure_text_replaces_only_full_layers():
    m = _FakeModel()
    report = surgery_gemma4(m)
    assert isinstance(report, Gemma4ESurgeryReport)
    assert report.layers_replaced == [1, 3]
    assert report.layers_skipped == [0, 2]
    assert report.layers_path == "model.model.layers"
    assert isinstance(m.model.layers[1].self_attn, Gemma4ThreeBranchAttention)
    assert isinstance(m.model.layers[3].self_attn, Gemma4ThreeBranchAttention)


# ---------------------------------------------------------------------------
# Surgery on multimodal
# ---------------------------------------------------------------------------
def test_surgery_multimodal_replaces_only_full_layers():
    m = _FakeMultimodalModel()
    report = surgery_gemma4(m)
    assert report.layers_replaced == [1, 3]
    assert report.layers_skipped == [0, 2]
    assert report.layers_path == "model.language_model.model.layers"
    layers = m.language_model.model.layers
    assert isinstance(layers[1].self_attn, Gemma4ThreeBranchAttention)
    assert isinstance(layers[3].self_attn, Gemma4ThreeBranchAttention)


# ---------------------------------------------------------------------------
# Numerical: forward through a replaced layer is finite
# ---------------------------------------------------------------------------
def test_replaced_layer_forward_is_finite():
    torch.manual_seed(0)
    m = _FakeModel()
    surgery_gemma4(m)
    new_attn = m.model.layers[1].self_attn
    B, T, H = 2, 16, 32
    hidden = torch.randn(B, T, H)
    out, _ = new_attn(hidden)
    assert out.shape == hidden.shape
    assert torch.isfinite(out).all()


def test_gradient_flows_to_branch_params():
    """Gradients must reach the *new* branch components (indexer, pools, gate)."""
    torch.manual_seed(0)
    m = _FakeModel()
    surgery_gemma4(m)
    new_attn = m.model.layers[1].self_attn
    B, T, H = 1, 8, 32
    hidden = torch.randn(B, T, H, requires_grad=True)
    out, _ = new_attn(hidden)
    out.sum().backward()
    # Branch gate is one of the always-new params
    assert new_attn.branch_gate.grad is not None
    assert torch.isfinite(new_attn.branch_gate.grad).all()
