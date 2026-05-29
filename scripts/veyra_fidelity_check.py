#!/usr/bin/env python
"""Veyra3 HF-vs-ONNX fidelity check.

The model card warns the HF export is degraded vs the ONNX-int8 sibling
repo. Quantifies how degraded so we can decide whether to:
  - proceed with HF as-is (max abs logit diff < 0.1)
  - accept known fidelity loss in our deltas (0.1 ≤ diff ≤ 1.0)
  - switch to ONNX-wrapper baseline (diff > 1.0)

Plan reference: §6.2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-model", default="veyra-ai/veyra3-5m-base")
    ap.add_argument("--onnx-model", default="veyra-ai/veyra3-5m-base-onnx-int8")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n-tokens", type=int, default=64)
    ap.add_argument("--n-prompts", type=int, default=5)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[fidelity] loading HF model {args.hf_model}…")
    tok = AutoTokenizer.from_pretrained(args.hf_model)
    hf = AutoModelForCausalLM.from_pretrained(args.hf_model, dtype=torch.float32).eval()

    try:
        import onnxruntime as ort  # noqa: F401
    except ImportError:
        print("[fidelity] onnxruntime not installed; install with: pip install onnxruntime")
        print("[fidelity] skipping ONNX comparison. Writing HF-only stats.")
        _hf_only(hf, tok, args)
        return

    try:
        from optimum.onnxruntime import ORTModelForCausalLM
    except ImportError:
        print("[fidelity] optimum not installed; install with: pip install optimum[onnxruntime]")
        _hf_only(hf, tok, args)
        return

    print(f"[fidelity] loading ONNX model {args.onnx_model}…")
    onnx_model = ORTModelForCausalLM.from_pretrained(args.onnx_model)

    diffs = []
    samples = []
    for i, prompt in enumerate(_prompts(args.n_prompts)):
        ids = tok(prompt, return_tensors="pt", padding="max_length",
                  truncation=True, max_length=args.n_tokens).input_ids
        with torch.no_grad():
            hf_logits = hf(ids).logits.float().cpu().numpy()
        onnx_logits = onnx_model(input_ids=ids).logits
        if hasattr(onnx_logits, "cpu"):
            onnx_logits = onnx_logits.cpu().numpy()
        else:
            onnx_logits = np.asarray(onnx_logits)
        d = float(np.abs(hf_logits - onnx_logits).max())
        mean_d = float(np.abs(hf_logits - onnx_logits).mean())
        diffs.append({"prompt_idx": i, "max_abs_diff": d, "mean_abs_diff": mean_d})
        samples.append({"prompt": prompt, "max_abs_diff": d, "mean_abs_diff": mean_d})
        print(f"  [{i}] max_abs_diff={d:.4f} mean={mean_d:.4f}")

    overall_max = max(d["max_abs_diff"] for d in diffs)
    overall_mean = sum(d["mean_abs_diff"] for d in diffs) / len(diffs)

    if overall_max < 0.1:
        verdict = "use_hf_as_is"
    elif overall_max < 1.0:
        verdict = "use_hf_with_caveat"
    else:
        verdict = "switch_to_onnx_wrapper"

    out = {
        "hf_model": args.hf_model,
        "onnx_model": args.onnx_model,
        "n_prompts": args.n_prompts,
        "max_abs_diff_overall": overall_max,
        "mean_abs_diff_overall": overall_mean,
        "per_prompt": diffs,
        "verdict": verdict,
    }
    (args.out / "fidelity.json").write_text(json.dumps(out, indent=2))
    print(f"[fidelity] verdict: {verdict}")
    print(f"[fidelity] wrote {args.out / 'fidelity.json'}")


def _hf_only(hf, tok, args):
    """Fallback when ONNX dependencies absent."""
    samples = []
    for prompt in _prompts(args.n_prompts):
        ids = tok(prompt, return_tensors="pt").input_ids
        with torch.no_grad():
            logits = hf(ids).logits.float()
        samples.append({"prompt": prompt,
                        "logit_mean": float(logits.mean()),
                        "logit_std": float(logits.std())})
    out = {"hf_model": args.hf_model, "onnx_skipped": True, "samples": samples,
           "verdict": "onnx_comparison_skipped"}
    (args.out / "fidelity.json").write_text(json.dumps(out, indent=2))
    print(f"[fidelity] wrote (hf-only) {args.out / 'fidelity.json'}")


def _prompts(n):
    base = ["The capital of France is", "Once upon a time, there was",
            "To be or not to be,", "In a world where", "She walked into the room and"]
    return base[:n]


if __name__ == "__main__":
    main()
