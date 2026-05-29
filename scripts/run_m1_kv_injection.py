#!/usr/bin/env python3
"""M1 — KV-injection knowledge-displacement pipeline.

Runs the G6 test with proper KV injection mounting instead of text-prepend.
Works on both:
  - veyra3-5m (fast local test / CPU validation)
  - MiniCPM5-1B (Colab A100 primary target)

Usage:
    python scripts/run_m1_kv_injection.py --model veyra3   [local test]
    python scripts/run_m1_kv_injection.py --model minicpm  [Colab A100]
    python scripts/run_m1_kv_injection.py --model minicpm --n_facts 200 --epochs 60
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime

import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from localsparse.config import LocalSparseConfig, ModelDims, AttentionConfig
from localsparse.training.factoid_world import build_world, render_corpus, make_lm_batches
from localsparse.workspace.kv_bank import WorkspaceKVBank
from localsparse.logging import RunLogger, RunDirectory


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

def load_veyra3(device, dtype):
    """Load Veyra3-5M with ThreeBranchAttention surgery."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from localsparse.model.veyra_adapter import surgery_veyra3

    print("[m1] Loading veyra3-5m-base...")
    model = AutoModelForCausalLM.from_pretrained(
        "veyra-ai/veyra3-5m-base",
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained("veyra-ai/veyra3-5m-base",
                                              trust_remote_code=True)

    cfg = LocalSparseConfig(
        model=ModelDims(
            vocab_size=model.config.vocab_size,
            hidden_size=model.config.hidden_size,
            num_layers=model.config.num_hidden_layers,
            num_q_heads=model.config.num_attention_heads,
            num_kv_heads=getattr(model.config, "num_key_value_heads",
                                 model.config.num_attention_heads),
            head_dim=getattr(model.config, "head_dim",
                             model.config.hidden_size // model.config.num_attention_heads),
        ),
        attention=AttentionConfig(
            sliding_window=1024,
            compressed_block=32,
            super_block=512,
            selected_top_k=8,
            indexer_dim=32,
        ),
    )

    report = surgery_veyra3(model, cfg)
    replaced_count = len(report.layers_replaced) if isinstance(report.layers_replaced, list) else report.layers_replaced
    skipped_count = len(report.layers_skipped) if isinstance(report.layers_skipped, list) else report.layers_skipped
    print(f"[m1] Surgery: replaced={replaced_count}, skipped={skipped_count}")
    return model, tokenizer, cfg


def load_minicpm5(device, dtype):
    """Load MiniCPM5-1B with LlamaThreeBranchAttention surgery."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from localsparse.model.minicpm_adapter import surgery_minicpm

    model_id = "openbmb/MiniCPM5-1B"
    print(f"[m1] Loading {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    cfg = LocalSparseConfig(
        model=ModelDims(
            vocab_size=model.config.vocab_size,
            hidden_size=model.config.hidden_size,
            num_layers=model.config.num_hidden_layers,
            num_q_heads=model.config.num_attention_heads,
            num_kv_heads=getattr(model.config, "num_key_value_heads",
                                 model.config.num_attention_heads),
            head_dim=getattr(model.config, "head_dim",
                             model.config.hidden_size // model.config.num_attention_heads),
        ),
        attention=AttentionConfig(
            sliding_window=4096,
            compressed_block=64,
            super_block=4096,
            selected_top_k=16,
            indexer_dim=64,
        ),
    )

    report = surgery_minicpm(model, cfg)
    print(f"[m1] Surgery: replaced={report.layers_replaced}, "
          f"skipped={report.layers_skipped}")
    if report.notes:
        for note in report.notes[:3]:
            print(f"  note: {note}")
    return model, tokenizer, cfg


# ---------------------------------------------------------------------------
# Gate: G6 KV-injection knowledge displacement
# ---------------------------------------------------------------------------

def run_g6(model, tokenizer, device, args, run_dir: Path) -> dict:
    """Core G6 gate with KV injection."""
    from localsparse.training.factoid_world import build_qa_pairs, evaluate_qa

    print(f"\n[G6] |set_W|={args.n_facts}  |set_M|={args.n_facts}")

    # Build vocab-aligned world
    vocab_size = model.config.vocab_size if hasattr(model, "config") else 256

    world_W = build_world(vocab_size=vocab_size, n_facts=args.n_facts, seed=0)
    world_M = build_world(vocab_size=vocab_size, n_facts=args.n_facts, seed=1)

    # ---- weights path: train on set_W ----
    run_dir.mkdir(parents=True, exist_ok=True)
    log_run = RunDirectory(root=run_dir / "g6_train")
    logger = RunLogger(log_run, print_every=50)
    corpus = render_corpus(world_W, repeats_per_fact=args.repeats)
    batches = make_lm_batches(corpus, batch_size=args.batch_size,
                               seq_len=args.seq_len, device=device)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01,
    )

    from localsparse.training.m05_runners import train_facts
    print(f"[G6] training on set_W: {len(batches)} batches × {args.epochs} epochs")
    t0 = time.time()
    stats = train_facts(
        model, batches=batches, optimizer=optimizer,
        epochs=args.epochs, logger=logger, label_prefix="g6_weights",
    )

    # Evaluate weights-path
    pairs_W = build_qa_pairs(world_W)
    weights_result = evaluate_qa(model, pairs_W, device=device)
    weights_acc = weights_result["accuracy"]
    print(f"[G6] weights-path accuracy: {weights_acc:.3f}")

    # ---- mount path: encode set_M into KV bank ----
    corpus_M = render_corpus(world_M, repeats_per_fact=1)
    # Decode token ids to a string the tokenizer can re-encode
    # (workspace "text" = the factoid corpus decoded through the real tokenizer)
    ws_text = tokenizer.decode(corpus_M, skip_special_tokens=True)
    bank = WorkspaceKVBank()
    print(f"[G6] encoding workspace (set_M, {len(corpus_M)} tokens) into KV bank...")
    bank.encode(model, ws_text, tokenizer, device, max_length=args.bank_max_length)
    print(f"[G6] bank encoded: {bank.workspace_seq_len()} tokens across "
          f"{len(bank._bank)} layers")

    # Mount-path accuracy
    pairs_M = build_qa_pairs(world_M)
    with bank.inject(model):
        mount_result = evaluate_qa(model, pairs_M, device=device)
    mount_acc = mount_result["accuracy"]

    # Control: no mount (should be ~random for held-out facts)
    control_result = evaluate_qa(model, pairs_M, device=device)
    control_acc = control_result["accuracy"]

    ratio = mount_acc / max(weights_acc, 1e-6)
    threshold = args.g6_threshold
    status = "pass" if ratio >= threshold else "fail"

    print(f"\n  ❓ G6: mount_vs_weights_ratio={ratio:.4f} (≥ {threshold:.4f})")
    icon = "✅" if status == "pass" else "❌"
    print(f"  {icon} G6: {status}")

    result = {
        "gate": "G6",
        "metric": "mount_vs_weights_ratio",
        "value": ratio,
        "threshold": threshold,
        "status": status,
        "weights_accuracy": weights_acc,
        "mount_accuracy": mount_acc,
        "control_accuracy": control_acc,
        "wall_seconds": time.time() - t0,
    }

    print(f"\n=== G6 HEADLINE ===")
    print(f"  weights-path accuracy: {weights_acc:.3f}")
    print(f"  mount-path accuracy:   {mount_acc:.3f}")
    print(f"  control (no mount):    {control_acc:.3f}")
    print(f"  ratio:                 {ratio:.3f}")
    print(f"  verdict:               {status}  ")

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="M1 KV-injection pipeline")
    parser.add_argument("--model", choices=["veyra3", "minicpm"], default="veyra3")
    parser.add_argument("--device", default=None,
                        help="cuda / mps / cpu (auto-detect if not set)")
    parser.add_argument("--dtype", default="bfloat16",
                        choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--n_facts", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--bank_max_length", type=int, default=512,
                        help="Max tokens for workspace KV encoding")
    parser.add_argument("--g6_threshold", type=float, default=0.6)
    parser.add_argument("--run_dir", type=str, default=None)
    args = parser.parse_args()

    # Device
    if args.device is None:
        if torch.cuda.is_available():
            args.device = "cuda"
        elif torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    print(f"[m1] device={args.device} dtype={args.dtype}")

    # Run dir
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.run_dir) if args.run_dir else \
              ROOT / "runs" / "m1" / f"kv_injection_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[m1] run dir: {run_dir}")

    # Load model
    if args.model == "veyra3":
        model, tokenizer, cfg = load_veyra3(device, dtype)
    else:
        model, tokenizer, cfg = load_minicpm5(device, dtype)

    # G6
    result = run_g6(model, tokenizer, device, args, run_dir)

    # Save summary
    summary_path = run_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[m1] wrote {summary_path}")

    # Final verdict
    if result["status"] == "pass":
        print("\n✅ G6 PASS — KV injection works. Ready for Colab full run.")
    else:
        ratio = result["value"]
        print(f"\n❌ G6 FAIL (ratio={ratio:.3f}). "
              f"Increase epochs/bank_max_length or check surgery.")

    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
