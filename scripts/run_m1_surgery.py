#!/usr/bin/env python
"""Milestone 1 driver: download MiniCPM5-1B, run surgery, train ~200 steps,
emit a PPL regression report.

Usage on Colab:
    !cd /content/localsparse && python scripts/run_m1_surgery.py \
        --base openbmb/MiniCPM5-1B \
        --out  /content/localsparse_m1 \
        --steps 200 --batch-size 1 --seq-len 1024

The script intentionally:
  - does not depend on `accelerate` or `unsloth` (kept minimal for portability)
  - loads in bf16 if CUDA is available, fp32 otherwise (so the local CPU
    smoke test still works on tiny models supplied via `--base toy`)
  - saves a SurgeryReport JSON + the M1Stats JSON next to the checkpoint
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

# Add the repo to the path so this is runnable directly out of a checkout
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from localsparse.config import default_config
from localsparse.model.surgery import perform_surgery, detect_model_dims
from localsparse.training import M1Config, run_m1, synthetic_lm_batch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True,
                   help="HF model id or local path (or 'toy' for offline smoke test).")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--data", default="synthetic",
                   choices=["synthetic", "fineweb"])
    p.add_argument("--with-teacher", action="store_true",
                   help="Keep the unmodified base loaded for KL-regression loss.")
    return p.parse_args()


def _load_toy():
    """Tiny in-memory model so we can smoke-test the pipeline offline."""
    import torch.nn as nn
    from localsparse.attention.sparse_three_branch import ThreeBranchAttention
    from localsparse.config import ModelDims, AttentionConfig
    import torch.nn.functional as F

    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = type("C", (), dict(
                vocab_size=128, hidden_size=32, num_hidden_layers=2,
                num_attention_heads=4, num_key_value_heads=2,
                intermediate_size=64, head_dim=8))()
            self.tok = nn.Embedding(128, 32)
            self.norm = nn.LayerNorm(32)
            self.head = nn.Linear(32, 128, bias=False)
            class Inner(nn.Module):
                def __init__(self):
                    super().__init__()
                    class L(nn.Module):
                        def __init__(self):
                            super().__init__()
                            self.self_attn = type("FA", (nn.Module,), {})()
                            self.self_attn.q_proj = nn.Linear(32, 32, bias=False)
                            self.self_attn.k_proj = nn.Linear(32, 16, bias=False)
                            self.self_attn.v_proj = nn.Linear(32, 16, bias=False)
                            self.self_attn.o_proj = nn.Linear(32, 32, bias=False)
                    self.layers = nn.ModuleList([L() for _ in range(2)])
            self.model = Inner()

        def forward(self, input_ids, labels=None):
            x = self.tok(input_ids)
            position_ids = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
            for layer in self.model.layers:
                if hasattr(layer.self_attn, "forward") and isinstance(layer.self_attn, ThreeBranchAttention):
                    out, _ = layer.self_attn(x, position_ids=position_ids.expand(x.shape[0], -1))
                    x = self.norm(x + out)
            logits = self.head(x)
            loss = None
            if labels is not None:
                loss = F.cross_entropy(logits.view(-1, 128), labels.view(-1), ignore_index=-100)
            class O: pass
            o = O(); o.loss = loss; o.logits = logits
            return o
    return Toy(), None


def _load_hf(model_id: str, dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"[load] {model_id}")
    tok = AutoTokenizer.from_pretrained(model_id)
    base = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    return base, tok


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    if args.base == "toy":
        model, tok = _load_toy()
        cfg = default_config()
        # The toy model's dims need a config override
        from localsparse.config import LocalSparseConfig, ModelDims, AttentionConfig, WorkspaceConfig, Paths
        cfg = LocalSparseConfig(
            model=ModelDims(num_layers=2, num_kv_heads=2, head_dim=8, hidden_size=32,
                            vocab_size=128, num_q_heads=4, intermediate_size=64),
            attention=AttentionConfig(compressed_block=4, super_block=16, indexer_dim=4,
                                      sliding_window=16, selected_top_k=2,
                                      selection_layer_stride=1),
            workspace=WorkspaceConfig(),
            paths=Paths(root=args.out / "wks"),
        )
    else:
        model, tok = _load_hf(args.base, dtype)
        cfg = default_config()
        cfg.model = detect_model_dims(model.config)

    print(f"[surgery] beginning replacement")
    report = perform_surgery(model, cfg)
    print(f"[surgery] {report}")

    teacher = None
    if args.with_teacher and args.base != "toy":
        teacher, _ = _load_hf(args.base, dtype)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False

    if torch.cuda.is_available():
        model = model.cuda()
        if teacher is not None:
            teacher = teacher.cuda()

    if args.data == "synthetic":
        def batches():
            for s in range(args.steps):
                yield synthetic_lm_batch(args.batch_size, args.seq_len,
                                         vocab_size=cfg.model.vocab_size, seed=s)
        batch_iter = batches()
    else:
        from localsparse.training.data import FineWebStream
        if tok is None:
            raise SystemExit("--data fineweb requires a real tokenizer (not --base toy)")
        batch_iter = FineWebStream(tok, seq_len=args.seq_len, batch_size=args.batch_size)

    m1_cfg = M1Config(steps=args.steps, batch_size=args.batch_size,
                      seq_len=args.seq_len, lr=args.lr)
    stats = run_m1(model, teacher=teacher, batch_iter=batch_iter, cfg=m1_cfg)

    (args.out / "surgery_report.json").write_text(json.dumps({
        "layers_replaced": report.layers_replaced,
        "layers_skipped": report.layers_skipped,
        "bytes_in_new_params": report.bytes_in_new_params,
    }, indent=2))
    (args.out / "m1_stats.json").write_text(json.dumps({
        "step": stats.step,
        "final_loss": stats.losses[-1] if stats.losses else None,
        "first_loss": stats.losses[0] if stats.losses else None,
        "final_branch_mass": stats.branch_mass_history[-1] if stats.branch_mass_history else None,
    }, indent=2))
    print("[done] wrote", args.out / "m1_stats.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
