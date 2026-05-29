#!/usr/bin/env python
"""Capacity-calibration sweep (plan §6.6.1).

Trains the surgery'd Veyra3 model from scratch on factoid corpora of
varying size N, measures answer accuracy after fixed compute. The
"knee" sets the reference capacity for G4/G6 thresholds (which are
then derived, not invented).

Usage:
    .venv/bin/python scripts/factoid_capacity_sweep.py \\
        --model veyra-ai/veyra3-5m-base --out runs/m05 \\
        --n-list 64,128,256,512,1024,2048 --steps 800

Outputs:
    <out>/capacity_sweep/sweep.json
    <out>/capacity_sweep/run_N<n>/train.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from localsparse.logging import RunLogger, RunDirectory, FailureDetector
from localsparse.training.factoid_world import (
    build_world, render_corpus, make_lm_batches,
)
from localsparse.training.m05_runners import train_facts, eval_world


def _device(s):
    if s == "auto":
        if torch.backends.mps.is_available(): return torch.device("mps")
        if torch.cuda.is_available(): return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(s)


def _load_and_surgery(model_id, device, dtype):
    from transformers import AutoModelForCausalLM
    from localsparse.model.veyra_adapter import surgery_veyra3
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
    surgery_veyra3(model)
    return model.to(device)


def run_one(model_id, *, n_facts, out_dir, device, dtype, batch_size,
            seq_len, epochs, lr, repeats):
    model = _load_and_surgery(model_id, device, dtype)
    world = build_world(vocab_size=model.config.vocab_size, n_facts=n_facts,
                        seed=n_facts)
    # Auto-scale repeats so we always get at least 8 batches per epoch
    min_repeats = repeats
    tokens_per_fact_approx = 12
    needed_tokens = batch_size * seq_len * 8
    have = n_facts * min_repeats * tokens_per_fact_approx
    if have < needed_tokens:
        min_repeats = max(min_repeats, needed_tokens // (n_facts * tokens_per_fact_approx) + 1)
    stream = render_corpus(world, repeats_per_fact=min_repeats, seed=n_facts)
    batches = make_lm_batches(stream, batch_size=batch_size, seq_len=seq_len,
                              device=device)
    if not batches:
        return {"n_facts": n_facts, "error": "not enough tokens",
                "repeats_attempted": min_repeats}
    run = RunDirectory(root=out_dir)
    logger = RunLogger(run, print_every=50)
    detector = FailureDetector(branch_collapse_window=999_999)  # off
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    stats = train_facts(model, batches=batches, optimizer=opt, epochs=epochs,
                        logger=logger, detector=detector,
                        label_prefix=f"capacity_N{n_facts}")
    res = eval_world(model, world, device=device)
    return {
        "n_facts": n_facts,
        "accuracy": res["accuracy"],
        "final_loss": stats["final_loss"],
        "wall_seconds": stats["wall_seconds"],
        "n_train_batches": len(batches),
        "n_evaluated": res["n_evaluated"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="veyra-ai/veyra3-5m-base")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n-list", default="64,128,256,512,1024,2048",
                    help="comma list of fact counts to sweep")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--repeats", type=int, default=40,
                    help="render repeats per fact; auto-bumped if too few tokens")
    args = ap.parse_args()

    device = _device(args.device)
    dtype = torch.bfloat16 if device.type in {"mps", "cuda"} else torch.float32
    print(f"[sweep] device={device} dtype={dtype}")
    out_root = args.out / "capacity_sweep"
    out_root.mkdir(parents=True, exist_ok=True)

    results = []
    for n in [int(x) for x in args.n_list.split(",")]:
        print(f"\n[sweep] === N = {n} facts ===")
        d = out_root / f"run_N{n}"
        d.mkdir(parents=True, exist_ok=True)
        try:
            res = run_one(args.model, n_facts=n, out_dir=d, device=device,
                          dtype=dtype, batch_size=args.batch,
                          seq_len=args.seq_len, epochs=args.epochs, lr=args.lr,
                          repeats=args.repeats)
        except Exception as e:
            res = {"n_facts": n, "error": repr(e)}
        print(f"[sweep] N={n}: {res}")
        results.append(res)
        (out_root / "sweep.json").write_text(json.dumps(results, indent=2))

    # Derived gate thresholds (plan §6.6.1: G6 = 0.6 × weights-path acc at chosen N)
    valid = [r for r in results if "accuracy" in r]
    if valid:
        best = max(valid, key=lambda r: r["accuracy"])
        knee = next((r for r in valid if r["accuracy"] >= 0.5), best)
        derived = {
            "best_N": best["n_facts"], "best_accuracy": best["accuracy"],
            "knee_N": knee["n_facts"], "knee_accuracy": knee["accuracy"],
            "g4_threshold_mount_vs_nomount_ratio": 2.0,
            "g6_threshold_mount_vs_weights_ratio": 0.6,
            "g6_absolute_min_at_knee_N": 0.6 * knee["accuracy"],
        }
        (out_root / "derived_thresholds.json").write_text(json.dumps(derived, indent=2))
        print("\n[sweep] derived thresholds:")
        print(json.dumps(derived, indent=2))


if __name__ == "__main__":
    main()
