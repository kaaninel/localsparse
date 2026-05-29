"""Tests for the Veyra3 / Gemma4 adapter.

We synthesize a tiny Gemma4-shaped model with mixed sliding/full
attention layers (matching Veyra3's structure) so the test runs offline
without needing to download anything from HuggingFace.

Coverage matches the surgery checklist in plan.md §6.5:
  ✓ Replace only `full_attention` layers
  ✓ Inherit q_proj/k_proj/o_proj exactly
  ✓ Clone k_proj → v_proj when V is tied to K (attention_k_eq_v)
  ✓ Inherit q_norm/k_norm/v_norm weights
  ✓ Inherit layer_scalar onto output_scale; reset parent to 1.0
  ✓ Skip sliding layers (their self_attn must be untouched)
  ✓ Numerical: forward of a single layer produces non-NaN output
"""
from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from localsparse.config import default_config
from localsparse.model.veyra_adapter import (
    Gemma4ThreeBranchAttention, surgery_veyra3, VeyraSurgeryReport,
    _veyra3_full_attn_dims,
)
from localsparse.attention.sparse_three_branch import ThreeBranchAttention


# ---------------------------------------------------------------------------
# Tiny fake Gemma4 model
# ---------------------------------------------------------------------------
class _FakeGemma4Config:
    vocab_size = 256
    hidden_size = 32
    num_hidden_layers = 4
    num_attention_heads = 4
    num_key_value_heads = 2          # sliding layers
    num_global_key_value_heads = 1   # full layers
    intermediate_size = 64
    head_dim = 8
    layer_types = ["sliding_attention", "full_attention",
                   "sliding_attention", "full_attention"]
    rope_parameters = {
        "full_attention": {"rope_theta": 1_000_000.0, "partial_rotary_factor": 0.25},
        "sliding_attention": {"rope_theta": 10_000.0},
    }


class _FakeSlidingAttn(nn.Module):
    """Sliding-attn shape: q(32→32), k/v(32→16), o(32→32). Has q/k/v_norm."""
    def __init__(self, hidden=32, num_q=4, num_kv=2, head_dim=8):
        super().__init__()
        self.q_proj = nn.Linear(hidden, num_q * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, num_kv * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, num_kv * head_dim, bias=False)
        self.o_proj = nn.Linear(num_q * head_dim, hidden, bias=False)
        self.q_norm = nn.RMSNorm(head_dim)
        self.k_norm = nn.RMSNorm(head_dim)
        self.v_norm = nn.RMSNorm(head_dim)


class _FakeFullAttn(nn.Module):
    """Full-attn shape: q(32→32), k(32→8), NO v_proj (V=K tied), o(32→32). Has all 3 norms."""
    def __init__(self, hidden=32, num_q=4, num_kv=1, head_dim=8):
        super().__init__()
        self.q_proj = nn.Linear(hidden, num_q * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, num_kv * head_dim, bias=False)
        self.o_proj = nn.Linear(num_q * head_dim, hidden, bias=False)
        self.q_norm = nn.RMSNorm(head_dim)
        self.k_norm = nn.RMSNorm(head_dim)
        self.v_norm = nn.RMSNorm(head_dim)


class _FakeLayer(nn.Module):
    def __init__(self, is_full: bool):
        super().__init__()
        self.self_attn = _FakeFullAttn() if is_full else _FakeSlidingAttn()
        self.layer_scalar = nn.Parameter(torch.tensor([1.5]))


class _FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = _FakeGemma4Config()
        class Inner(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([
                    _FakeLayer(is_full=(t == "full_attention"))
                    for t in _FakeGemma4Config.layer_types
                ])
        self.model = Inner()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@pytest.fixture
def model():
    torch.manual_seed(0)
    return _FakeModel()


def test_only_full_attn_layers_replaced(model):
    report = surgery_veyra3(model)
    assert report.layers_replaced == [1, 3]
    assert report.layers_skipped == [0, 2]
    # Sliding layers untouched
    assert isinstance(model.model.layers[0].self_attn, _FakeSlidingAttn)
    assert isinstance(model.model.layers[2].self_attn, _FakeSlidingAttn)
    # Full layers replaced
    assert isinstance(model.model.layers[1].self_attn, Gemma4ThreeBranchAttention)
    assert isinstance(model.model.layers[3].self_attn, Gemma4ThreeBranchAttention)


def test_q_proj_weight_copied_exactly(model):
    pre = model.model.layers[1].self_attn.q_proj.weight.clone()
    surgery_veyra3(model)
    post = model.model.layers[1].self_attn.q_proj.weight
    torch.testing.assert_close(pre, post)


def test_v_tied_to_k_when_no_v_proj(model):
    # Pre-surgery: full layer has NO v_proj
    assert not hasattr(model.model.layers[1].self_attn, "v_proj")
    pre_k = model.model.layers[1].self_attn.k_proj.weight.clone()
    surgery_veyra3(model)
    new = model.model.layers[1].self_attn
    # K copied directly
    torch.testing.assert_close(new.k_proj.weight, pre_k)
    # V matches K (tied at init)
    torch.testing.assert_close(new.v_proj.weight, pre_k)


def test_norms_copied(model):
    pre_q = model.model.layers[1].self_attn.q_norm.weight.clone()
    pre_k = model.model.layers[1].self_attn.k_norm.weight.clone()
    pre_v = model.model.layers[1].self_attn.v_norm.weight.clone()
    surgery_veyra3(model)
    new = model.model.layers[1].self_attn
    torch.testing.assert_close(new.q_norm.weight, pre_q)
    torch.testing.assert_close(new.k_norm.weight, pre_k)
    torch.testing.assert_close(new.v_norm.weight, pre_v)


def test_layer_scalar_inherited_and_parent_reset(model):
    pre_scalar = float(model.model.layers[1].layer_scalar)
    assert pre_scalar == 1.5
    surgery_veyra3(model)
    new = model.model.layers[1].self_attn
    # output_scale picked up the value
    assert pytest.approx(float(new.output_scale)) == 1.5
    # Parent layer's layer_scalar reset to 1.0 to avoid double-application
    assert pytest.approx(float(model.model.layers[1].layer_scalar)) == 1.0


def test_full_attn_dims_uses_global_kv_heads(model):
    dims = _veyra3_full_attn_dims(model.config)
    assert dims.num_kv_heads == 1
    assert dims.num_q_heads == 4
    assert dims.head_dim == 8
    assert dims.hidden_size == 32


def test_full_attn_dims_uses_global_head_dim_when_present():
    class _Gemma4ETextConfig:
        vocab_size = 256
        hidden_size = 32
        num_hidden_layers = 4
        num_attention_heads = 4
        num_key_value_heads = 1
        num_global_key_value_heads = None
        intermediate_size = 64
        head_dim = 8
        global_head_dim = 16
        attention_k_eq_v = False

    dims = _veyra3_full_attn_dims(_Gemma4ETextConfig())
    assert dims.num_kv_heads == 1
    assert dims.num_q_heads == 4
    assert dims.head_dim == 16
    assert dims.hidden_size == 32


def test_forward_runs_no_nan(model):
    surgery_veyra3(model)
    x = torch.randn(1, 16, 32)
    out, _ = model.model.layers[1].self_attn(x)
    assert out.shape == (1, 16, 32)
    assert torch.isfinite(out).all()
    # Branch masses non-zero (read from the stashed BranchOutputs)
    branch = model.model.layers[1].self_attn._last_branch_outputs
    assert float(branch.sliding_mass) > 0
    assert float(branch.compressed_mass) > 0


def test_save_load_fidelity(model, tmp_path):
    """G0.5-save: save + load model and logits stay identical."""
    surgery_veyra3(model)
    x = torch.randn(1, 16, 32)
    model.eval()
    with torch.no_grad():
        out_pre, _ = model.model.layers[1].self_attn(x)
    sd = model.state_dict()
    torch.save(sd, tmp_path / "m.pt")

    fresh = _FakeModel()
    surgery_veyra3(fresh)
    fresh.load_state_dict(torch.load(tmp_path / "m.pt", weights_only=True))
    fresh.eval()
    with torch.no_grad():
        out_post, _ = fresh.model.layers[1].self_attn(x)
    torch.testing.assert_close(out_pre, out_post)


def test_inherited_vs_new_param_accounting(model):
    report = surgery_veyra3(model)
    # We replaced 2 full-attn layers
    assert report.inherited_param_bytes > 0
    assert report.new_param_bytes > 0
    # The new bytes should be considerably larger than just inherited
    # because of indexer + compressed_pool + super_pool
    assert "layer 1" in " ".join(report.notes) or "layer 3" in " ".join(report.notes)
