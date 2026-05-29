#!/usr/bin/env python
"""Pre-benchmark Veyra3-5M-base before any surgery.

Records baseline metrics so post-surgery deltas in M0.5 are interpretable.

Outputs:
    <out>/pre_benchmark.json       — all metrics
    <out>/pre_samples.txt          — qualitative generation samples
    <out>/pre_logit_distribution.txt — entropy stats

Plan reference: §6.1
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch


def _device(s):
    if s == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="veyra-ai/veyra3-5m-base")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--n-ppl-tokens", type=int, default=20_000)
    ap.add_argument("--ctx-lens", default="1024,4096,8192", help="comma list")
    ap.add_argument("--n-samples", type=int, default=5)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    device = _device(args.device)
    if args.dtype == "auto":
        dtype = torch.bfloat16 if device.type in {"mps", "cuda"} else torch.float32
    else:
        dtype = getattr(torch, args.dtype)
    print(f"[bench] device={device} dtype={dtype}")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"[bench] loading {args.model}…")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[bench] model loaded: {n_params:,} params")

    out: dict = {
        "model_id": args.model,
        "device": str(device),
        "dtype": str(dtype),
        "n_params": n_params,
        "config_summary": {
            "vocab_size": getattr(model.config, "vocab_size", None),
            "hidden_size": getattr(model.config, "hidden_size", None),
            "num_layers": getattr(model.config, "num_hidden_layers", None),
            "num_q_heads": getattr(model.config, "num_attention_heads", None),
            "num_kv_heads": getattr(model.config, "num_key_value_heads", None),
            "head_dim": getattr(model.config, "head_dim", None),
            "layer_types": getattr(model.config, "layer_types", None),
            "max_position": getattr(model.config, "max_position_embeddings", None),
            "sliding_window": getattr(model.config, "sliding_window", None),
        },
    }

    # ---- 1. tokenizer round-trip ---------------------------------------
    print("[bench] tokenizer round-trip…")
    test_strings = [
        "hello world this is a test",
        "the quick brown fox jumps over the lazy dog",
        "she sells sea shells by the sea shore",
        "supercalifragilisticexpialidocious",
        "a b c d e f g 1 2 3 4 5 . , ? !",
    ]
    rt_pass = 0
    for s in test_strings:
        ids = tok(s, add_special_tokens=False).input_ids
        back = tok.decode(ids)
        if back.strip() == s.strip():
            rt_pass += 1
    out["tokenizer_roundtrip_pass"] = rt_pass
    out["tokenizer_roundtrip_total"] = len(test_strings)

    # ---- 2. logit entropy on random English ----------------------------
    print("[bench] logit entropy on random English…")
    prompt = ("The capital of France is Paris. Photosynthesis converts sunlight. "
              "Computers process binary code. Music consists of organized sound. " * 8)
    ids = tok(prompt, return_tensors="pt").input_ids[:, :512].to(device)
    with torch.no_grad():
        logits = model(ids).logits.float()
    probs = logits.softmax(dim=-1)
    entropy = -(probs * (probs + 1e-12).log()).sum(dim=-1)
    max_prob = probs.max(dim=-1).values
    out["logit_entropy_mean"] = float(entropy.mean())
    out["logit_entropy_std"] = float(entropy.std())
    out["logit_entropy_min"] = float(entropy.min())
    out["logit_max_prob_mean"] = float(max_prob.mean())

    # ---- 3. PPL on random English from tokenizer's training distribution
    # We can't download FineWeb cheaply here. Use a heuristic: tokenize the
    # tokenizer's own vocab joined together to produce ~varied English.
    print(f"[bench] approximate PPL on {args.n_ppl_tokens} synthetic English tokens…")
    # Build a token stream by re-tokenizing common English sentences
    corpus = (
        "The quick brown fox jumps over the lazy dog. " * 50 +
        "She sells sea shells by the sea shore. " * 50 +
        "How much wood would a woodchuck chuck. " * 50 +
        "To be or not to be that is the question. " * 50
    )
    ids = tok(corpus, return_tensors="pt").input_ids.to(device)
    if ids.shape[1] > args.n_ppl_tokens:
        ids = ids[:, :args.n_ppl_tokens]
    with torch.no_grad():
        nll = model(ids, labels=ids).loss.float()
    out["english_ppl"] = math.exp(float(nll))
    out["english_loss"] = float(nll)

    # ---- 4. inference speed sweep --------------------------------------
    print("[bench] inference speed sweep…")
    speed = {}
    for ctx_len in [int(x) for x in args.ctx_lens.split(",")]:
        if ctx_len > getattr(model.config, "max_position_embeddings", 4096):
            speed[ctx_len] = "skipped (>max_position)"
            continue
        x = torch.randint(0, out["config_summary"]["vocab_size"], (1, ctx_len), device=device)
        # Warm
        with torch.no_grad():
            _ = model(x).logits
        if device.type == "mps":
            torch.mps.synchronize()
        t0 = time.time()
        with torch.no_grad():
            _ = model(x).logits
        if device.type == "mps":
            torch.mps.synchronize()
        elapsed = time.time() - t0
        speed[ctx_len] = {
            "elapsed_sec": round(elapsed, 3),
            "tokens_per_sec": round(ctx_len / elapsed, 0),
        }
        print(f"  ctx={ctx_len:>5}: {elapsed:.3f}s → {ctx_len/elapsed:.0f} tok/s")
    out["inference_speed"] = speed

    # ---- 5. generation samples (qualitative) ---------------------------
    print(f"[bench] generating {args.n_samples} samples…")
    prompts = ["The", "Once upon a time", "Hello", "What is the",
               "In the year"][:args.n_samples]
    samples = []
    for p in prompts:
        inp = tok(p, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            gen = model.generate(inp, max_new_tokens=args.max_new_tokens,
                                 do_sample=False, pad_token_id=tok.eos_token_id)
        samples.append({
            "prompt": p,
            "generated": tok.decode(gen[0, inp.shape[1]:], skip_special_tokens=False),
        })

    # ---- write outputs --------------------------------------------------
    (args.out / "pre_benchmark.json").write_text(json.dumps(out, indent=2))
    (args.out / "pre_samples.txt").write_text(
        "\n\n".join(f"PROMPT: {s['prompt']}\nGEN:    {s['generated']}" for s in samples)
    )
    print(f"[bench] wrote {args.out / 'pre_benchmark.json'}")
    print(f"[bench] wrote {args.out / 'pre_samples.txt'}")


if __name__ == "__main__":
    main()
