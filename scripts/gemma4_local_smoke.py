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


def _build_tiny():
    """Tiny synthetic Gemma-4-style model with 4 layers (2 sliding + 2 full)."""
    from tests.test_veyra_adapter import _FakeModel
    return _FakeModel()


def _build_tiny_multimodal():
    from tests.test_gemma4_adapter import _FakeMultimodalModel
    return _FakeMultimodalModel()


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

    elapsed = time.time() - t0
    print(f"\n[gemma4_smoke] PASS in {elapsed:.2f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
