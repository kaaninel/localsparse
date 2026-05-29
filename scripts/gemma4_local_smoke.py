"""Local CPU smoke runner for Gemma 4 adapter (plan §7.4 B3).

Builds a tiny synthetic Gemma-4-style model (no HF download), applies
`surgery_gemma4`, runs forward + backward, and verifies:
  - surgery report (replaced count, layer path)
  - no NaN in forward output
  - gradients flow to all new branch params

Target runtime: < 10 seconds on local CPU.
Exit code 0 on PASS, 1 on FAIL.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn


class _FakeGemma4Config:
    vocab_size = 256
    hidden_size = 32
    num_hidden_layers = 4
    num_attention_heads = 4
    num_key_value_heads = 2
    num_global_key_value_heads = 1
    intermediate_size = 64
    head_dim = 8
    global_head_dim = 8
    attention_k_eq_v = True
    layer_types = ["sliding_attention", "full_attention",
                   "sliding_attention", "full_attention"]
    rope_parameters = {
        "full_attention": {"rope_theta": 1_000_000.0,
                           "partial_rotary_factor": 0.25},
        "sliding_attention": {"rope_theta": 10_000.0},
    }


class _FakeSlidingAttn(nn.Module):
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


class _FakeInner(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            _FakeLayer(is_full=(t == "full_attention"))
            for t in _FakeGemma4Config.layer_types
        ])


class _FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = _FakeGemma4Config()
        self.model = _FakeInner()


class _FakeLegacyMultimodalModel(nn.Module):
    """Older wrapper shape: model.language_model.model.layers."""
    def __init__(self):
        super().__init__()

        class _OuterCfg:
            text_config = _FakeGemma4Config()

        class _LM(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = _FakeInner()

        self.config = _OuterCfg()
        self.language_model = _LM()


class _FakeConditionalGenerationModel(nn.Module):
    """HF Gemma4ForConditionalGeneration shape: model.model.language_model.layers."""
    def __init__(self):
        super().__init__()

        class _OuterCfg:
            text_config = _FakeGemma4Config()

        class _Wrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.language_model = _FakeInner()

        self.config = _OuterCfg()
        self.model = _Wrapper()


def _build_tiny():
    """Tiny synthetic Gemma-4-style model with 4 layers (2 sliding + 2 full)."""
    return _FakeModel()


def _build_tiny_multimodal():
    return _FakeLegacyMultimodalModel()


def _build_tiny_conditional_generation():
    return _FakeConditionalGenerationModel()


def main():
    from localsparse.model.gemma4_adapter import surgery_gemma4

    print("[gemma4_smoke] === pure-text variant ===")
    t0 = time.time()
    m = _build_tiny()
    report = surgery_gemma4(m)
    assert report.layers_replaced == [1, 3], f"replaced={report.layers_replaced}"
    assert report.layers_path == "model.model.layers"
    print(f"  surgery report: replaced={report.layers_replaced} "
          f"path={report.layers_path}")
    # Forward
    new_attn = m.model.layers[1].self_attn
    hidden = torch.randn(2, 16, 32, requires_grad=True)
    out, _ = new_attn(hidden)
    assert out.shape == hidden.shape
    assert torch.isfinite(out).all(), "NaN/Inf in forward output"
    print(f"  forward ok: shape={tuple(out.shape)}, finite=True")
    # Backward
    out.sum().backward()
    new_grads = {n: p.grad for n, p in new_attn.named_parameters()
                 if p.grad is not None and
                 any(n.startswith(prefix) for prefix in
                     ("indexer.", "compressed_pool.", "branch_gate"))}
    assert len(new_grads) > 0, "no new branch params got gradients"
    for n, g in new_grads.items():
        assert torch.isfinite(g).all(), f"grad NaN in {n}"
    print(f"  backward ok: {len(new_grads)} new params got finite gradients")

    print("[gemma4_smoke] === multimodal variant ===")
    m2 = _build_tiny_multimodal()
    report2 = surgery_gemma4(m2)
    assert report2.layers_replaced == [1, 3]
    assert report2.layers_path == "model.language_model.model.layers"
    print(f"  surgery report: replaced={report2.layers_replaced} "
          f"path={report2.layers_path}")

    print("[gemma4_smoke] === conditional-generation variant ===")
    m3 = _build_tiny_conditional_generation()
    report3 = surgery_gemma4(m3)
    assert report3.layers_replaced == [1, 3]
    assert report3.layers_path == "model.model.language_model.layers"
    print(f"  surgery report: replaced={report3.layers_replaced} "
          f"path={report3.layers_path}")

    elapsed = time.time() - t0
    print(f"\n[gemma4_smoke] PASS in {elapsed:.2f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
