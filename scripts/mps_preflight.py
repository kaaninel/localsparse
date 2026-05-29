#!/usr/bin/env python
"""MPS preflight: verify our 3-branch attention runs entirely on MPS
(no silent CPU fallback) on the user's M4 Air. Critical because
silent fallbacks cause 4–10× slowdowns.

Usage:
    .venv/bin/python scripts/mps_preflight.py [--device mps|cpu|cuda]

Reports per-op device placement and fires loudly if any of our
attention components land on CPU when we asked for MPS.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from localsparse.config import ModelDims, AttentionConfig, default_config
from localsparse.attention.sparse_three_branch import ThreeBranchAttention


def _resolve_device(requested: str) -> torch.device:
    if requested == "mps":
        if not torch.backends.mps.is_available():
            print("[warn] MPS not available on this machine; falling back to CPU")
            return torch.device("cpu")
        return torch.device("mps")
    if requested == "cuda":
        if not torch.cuda.is_available():
            print("[warn] CUDA not available; falling back to CPU")
            return torch.device("cpu")
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps", choices=["mps", "cpu", "cuda"])
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=2)
    args = ap.parse_args()

    device = _resolve_device(args.device)
    print(f"[preflight] device={device}, seq_len={args.seq_len}, batch={args.batch}")

    cfg = default_config()
    cfg.attention.sliding_window = 256
    cfg.attention.compressed_block = 32
    cfg.attention.super_block = 256
    cfg.attention.selected_top_k = 4
    cfg.attention.indexer_dim = 32

    model = ModelDims(num_layers=1, num_kv_heads=2, head_dim=64,
                      hidden_size=256, vocab_size=4096, num_q_heads=4,
                      intermediate_size=512)
    attn = ThreeBranchAttention(model=model, attn=cfg.attention, layer_idx=0)
    dtype = torch.float32 if device.type == "cpu" else torch.bfloat16
    attn = attn.to(device=device, dtype=dtype)

    x = torch.randn(args.batch, args.seq_len, model.hidden_size,
                    device=device, dtype=dtype)
    pos = torch.arange(args.seq_len, device=device).unsqueeze(0).expand(args.batch, -1)

    # Warm-up
    print("[preflight] warm-up forward...")
    out, _ = attn(x, position_ids=pos)
    if device.type == "mps":
        torch.mps.synchronize()
    print(f"[preflight] output shape: {tuple(out.shape)}, finite: {bool(torch.isfinite(out).all())}")

    # Time a few iterations
    print("[preflight] timing 5 fwd+bwd iterations...")
    optimizer = torch.optim.AdamW(attn.parameters(), lr=1e-4)
    t0 = time.time()
    for i in range(5):
        optimizer.zero_grad(set_to_none=True)
        out, _ = attn(x, position_ids=pos)
        loss = out.float().pow(2).mean()
        loss.backward()
        optimizer.step()
        if device.type == "mps":
            torch.mps.synchronize()
    elapsed = time.time() - t0
    tps = args.batch * args.seq_len * 5 / elapsed
    print(f"[preflight] elapsed={elapsed:.2f}s, tokens/sec ≈ {tps:.0f}, "
          f"steps/sec ≈ {5/elapsed:.2f}")

    # Run profiler if MPS to detect fallbacks
    if device.type == "mps":
        print("[preflight] running profiler to detect CPU fallbacks…")
        from torch.profiler import profile, ProfilerActivity
        with profile(activities=[ProfilerActivity.CPU]) as p:
            out, _ = attn(x, position_ids=pos)
            loss = out.float().pow(2).mean()
            loss.backward()
            torch.mps.synchronize()
        # Look for any op named with 'aten::' that ran on CPU during MPS run.
        # MPS-native ops are reported with 'mps::' prefix typically.
        events = p.key_averages()
        cpu_only = [e for e in events
                    if e.key.startswith("aten::")
                    and "mps" not in e.key.lower()
                    and e.cpu_time_total > 1000]  # >1ms
        if cpu_only:
            print("[preflight] ⚠️  Possible CPU fallback ops detected (top 10):")
            for e in sorted(cpu_only, key=lambda x: -x.cpu_time_total)[:10]:
                print(f"  {e.key:<40} cpu_time={e.cpu_time_total/1000:.1f}ms count={e.count}")
        else:
            print("[preflight] ✅  No obvious CPU fallback ops detected")

    print("[preflight] done.")


if __name__ == "__main__":
    main()
