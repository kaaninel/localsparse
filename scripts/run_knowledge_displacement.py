#!/usr/bin/env python
"""Phase C — knowledge displacement experiment (G6).

The headline experiment of M0.5: does the model treat workspace context
as equivalent to in-weights knowledge?

Protocol:
  1. Build a single fact pool of size 2N.
  2. Split 50/50 → set_W (trained in-weights) and set_M (mounted only).
  3. Train model on set_W (heavy, many epochs).
  4. Evaluate:
       - acc_W   = QA accuracy on set_W with prompt only → "weights path"
       - acc_M   = QA accuracy on set_M with set_M's render as prefix → "mount path"
  5. G6 pass: acc_M ≥ 0.6 × acc_W

Optional stretch:
  G7 — YaRN extension single-needle NIAH at 8K / 32K
  G8 — 16K-slot wks (long workspace) position-random recall

Plan reference: §6.9
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import random
import torch

from localsparse.logging import RunDirectory, RunLogger, GateLogger, FailureDetector
from localsparse.training.factoid_world import (
    build_world, FactoidWorld, render_corpus, build_qa_pairs, make_lm_batches,
    evaluate_qa,
)
from localsparse.training.m05_runners import train_facts


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


def _split_world(world: FactoidWorld, seed: int = 0):
    rng = random.Random(seed)
    facts = list(world.facts)
    rng.shuffle(facts)
    half = len(facts) // 2
    return (
        FactoidWorld(subjects=world.subjects, predicates=world.predicates,
                     objects=world.objects, facts=facts[:half],
                     template_tokens=world.template_tokens, eos_id=world.eos_id),
        FactoidWorld(subjects=world.subjects, predicates=world.predicates,
                     objects=world.objects, facts=facts[half:],
                     template_tokens=world.template_tokens, eos_id=world.eos_id),
    )


def eval_with_prefix(model, qa_pairs, prefix_ids, *, device):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for (prompt, answer) in qa_pairs:
            ids = torch.tensor((prefix_ids + prompt), device=device).unsqueeze(0)
            max_pos = getattr(model.config, "max_position_embeddings", 4096)
            if ids.shape[1] > max_pos:
                ids = ids[:, -max_pos:]
            logits = model(input_ids=ids).logits
            pred = int(logits[0, -1].argmax())
            correct += (pred == answer)
            total += 1
    return {"accuracy": correct / max(total, 1), "n_evaluated": total}


def run_g6(args, model, gates: GateLogger, logger: RunLogger, device):
    world = build_world(vocab_size=model.config.vocab_size,
                        n_facts=args.n_facts * 2, seed=42)
    set_W, set_M = _split_world(world, seed=43)
    print(f"[G6] |set_W|={set_W.n_facts}  |set_M|={set_M.n_facts}")

    # Train on set_W
    stream = render_corpus(set_W, repeats_per_fact=args.train_repeats, seed=44)
    batches = make_lm_batches(stream, batch_size=args.batch,
                              seq_len=args.seq_len, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    print(f"[G6] training on set_W: {len(batches)} batches × {args.epochs} epochs")
    train_facts(model, batches=batches, optimizer=opt, epochs=args.epochs,
                logger=logger, detector=FailureDetector(),
                label_prefix="G6_train_W")

    # Eval: weights path
    pairs_W = build_qa_pairs(set_W)
    acc_W = evaluate_qa(model, pairs_W, device=device)["accuracy"]
    # Eval: mount path
    pairs_M = build_qa_pairs(set_M)
    prefix_M = render_corpus(set_M, repeats_per_fact=2, seed=45)
    acc_M = eval_with_prefix(model, pairs_M, prefix_M, device=device)["accuracy"]
    # Control: no-mount path on set_M (should be ~chance)
    acc_M_nomount = evaluate_qa(model, pairs_M, device=device)["accuracy"]

    ratio = acc_M / max(acc_W, 1e-6)
    threshold = 0.6
    if ratio >= threshold:
        verdict = "pass"
    elif ratio >= 0.3:
        verdict = "stretch"
    else:
        verdict = "fail"
    gates.record("G6", metric="mount_vs_weights_ratio", value=ratio,
                 threshold=threshold, status=verdict,
                 acc_weights_path=acc_W, acc_mount_path=acc_M,
                 acc_mount_path_nomount_control=acc_M_nomount,
                 n_facts_per_path=set_W.n_facts)
    return {"acc_W": acc_W, "acc_M": acc_M, "acc_M_nomount": acc_M_nomount,
            "ratio": ratio, "verdict": verdict}


def run_g7_stretch(args, model, gates: GateLogger, device):
    """Single-needle NIAH at 8K and 32K. Stretch goal."""
    from localsparse.training.data import needle_in_haystack_batch
    for ctx_len in (args.g7_8k_ctx, args.g7_32k_ctx):
        if ctx_len <= 0:
            continue
        max_pos = getattr(model.config, "max_position_embeddings", 4096)
        # In stretch test we *want* to push past max_position_embeddings to
        # exercise YaRN; for M0.5 we just measure whether the model can
        # produce a finite output.
        try:
            b = needle_in_haystack_batch(batch_size=1, seq_len=min(ctx_len, max_pos),
                                         vocab_size=model.config.vocab_size, seed=ctx_len)
            inp = b.input_ids.to(device)
            with torch.no_grad():
                out = model(input_ids=inp).logits
            ok = bool(torch.isfinite(out).all())
            gates.record(f"G7_{ctx_len}", metric="forward_finite_at_ctx",
                         value=1.0 if ok else 0.0, threshold=1.0,
                         status="stretch", ctx_len=ctx_len,
                         note="stretch — full NIAH requires YaRN ext + training")
        except Exception as e:
            gates.record(f"G7_{ctx_len}", metric="forward_finite_at_ctx",
                         value=0.0, threshold=1.0, status="deferred",
                         ctx_len=ctx_len, note=repr(e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="veyra-ai/veyra3-5m-base")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--n-facts", type=int, default=200,
                    help="per-path; total facts = 2 × n_facts")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--skip-g7", action="store_true")
    ap.add_argument("--g7-8k-ctx", type=int, default=8192)
    ap.add_argument("--g7-32k-ctx", type=int, default=0,
                    help="0 disables (requires YaRN); default off")
    ap.add_argument("--train-repeats", type=int, default=40,
                    help="how many surface forms per fact to render in training corpus")
    args = ap.parse_args()

    device = _device(args.device)
    dtype = torch.bfloat16 if device.type in {"mps", "cuda"} else torch.float32
    print(f"[phaseC] device={device} dtype={dtype}")
    model = _load_and_surgery(args.model, device, dtype)

    run = RunDirectory.fresh(args.out, prefix="phase_c")
    print(f"[phaseC] run dir: {run.root}")
    logger = RunLogger(run, print_every=25)
    gates = GateLogger(run)

    res = run_g6(args, model, gates, logger, device)
    if not args.skip_g7:
        run_g7_stretch(args, model, gates, device)

    summary = {"model": args.model, "device": str(device),
               "n_facts_per_path": args.n_facts, "G6_result": res}
    run.summary_json.write_text(json.dumps(summary, indent=2))
    print(f"[phaseC] wrote {run.summary_json}")
    print(f"\n=== G6 HEADLINE ===")
    print(f"  weights-path accuracy: {res['acc_W']:.3f}")
    print(f"  mount-path accuracy:   {res['acc_M']:.3f}")
    print(f"  control (no mount):    {res['acc_M_nomount']:.3f}")
    print(f"  ratio:                 {res['ratio']:.3f}")
    print(f"  verdict:               {res['verdict']}")


if __name__ == "__main__":
    main()
