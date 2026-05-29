"""Phase C — Gemma 4 E2B G-gate validation (plan §7.5).

Sections:
  c3   smoke: N=32, 50 steps — must be first cell on Colab
  c1   prebench: PPL + tok/s before & after surgery
  c2   real G6: N=512, train-to-convergence (cap 4000 steps), bank-mounted set_M
  all  c3 → c1 → c2

Usage:
  python scripts/bench_gemma4.py --section c3 --device cuda --dtype bfloat16
  python scripts/bench_gemma4.py --section all --device cuda --dtype bfloat16
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn

from localsparse.config import LocalSparseConfig
from localsparse.logging import RunDirectory, RunLogger
from localsparse.training.convergence import PlateauDetector
from localsparse.training.factoid_world import (
    build_qa_pairs, build_world, evaluate_qa,
    make_lm_batches, render_corpus,
)
from localsparse.training.m15_runners import train_to_convergence
from localsparse.workspace.kv_bank import WorkspaceKVBank


DEFAULT_MODEL_ID = "google/gemma-4-E2B"


def parse_device(s: str) -> torch.device:
    if s == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(s)


def parse_dtype(s: str) -> torch.dtype:
    return {"float32": torch.float32, "float16": torch.float16,
            "bfloat16": torch.bfloat16}[s]


def load_gemma4(model_id: str, device: torch.device, dtype: torch.dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from localsparse.model.gemma4_adapter import surgery_gemma4
    print(f"[bench_gemma4] Loading {model_id} ...")
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
    report = surgery_gemma4(model)
    model = model.to(device)
    print(f"[bench_gemma4] Surgery: replaced {len(report.layers_replaced)} "
          f"layers via {report.layers_path}")
    print(f"[bench_gemma4] Skipped: {len(report.layers_skipped)} sliding layers")
    return model, tok, report


def make_logger(run_dir: Path, label: str) -> RunLogger:
    sub = run_dir / label
    sub.mkdir(parents=True, exist_ok=True)
    return RunLogger(RunDirectory(root=sub), print_every=50)


# ---------------------------------------------------------------------------
# C3: smoke
# ---------------------------------------------------------------------------

def section_c3(args, run_dir, model, tok):
    print("[c3] smoke: N=32, 50 steps, ctx 256, batch 1")
    world = build_world(vocab_size=model.config.vocab_size, n_facts=32, seed=99)
    token_stream = render_corpus(world, repeats_per_fact=4)
    batches = make_lm_batches(token_stream, batch_size=1, seq_len=256,
                              device=args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
    stats = train_to_convergence(
        model, batches, opt, max_steps=50,
        logger=make_logger(run_dir, "c3"),
    )
    pairs = build_qa_pairs(world)
    eval_res = evaluate_qa(model, pairs, device=args.device)
    acc = eval_res["accuracy"]
    return {
        "section": "C3",
        "n_facts": 32,
        "accuracy": acc,
        "pass_criterion": "accuracy > 0 after 50 steps",
        "status": "pass" if acc > 0 else "fail",
        **stats,
    }


# ---------------------------------------------------------------------------
# C1: prebench (PPL + tok/s)
# ---------------------------------------------------------------------------

def section_c1(args, run_dir, model, tok):
    print("[c1] PPL + tok/s on synthetic random tokens (FineWeb sample optional)")
    out: Dict[str, Any] = {"section": "C1", "ctx_results": []}
    for T in [1024, 4096]:
        try:
            x = torch.randint(0, model.config.vocab_size, (1, T),
                              device=args.device)
            torch.cuda.synchronize() if args.device.type == "cuda" else None
            t0 = time.time()
            with torch.no_grad():
                o = model(input_ids=x, labels=x)
            torch.cuda.synchronize() if args.device.type == "cuda" else None
            dt = time.time() - t0
            ppl = float(torch.exp(o.loss.detach()))
            tps = T / dt
            out["ctx_results"].append({
                "ctx": T, "ppl": ppl,
                "fwd_seconds": dt, "tok_per_s": tps,
            })
            print(f"  ctx={T}: ppl={ppl:.2f}, tok/s={tps:.0f}")
        except RuntimeError as e:
            out["ctx_results"].append({
                "ctx": T, "status": "error", "error": str(e)[:200],
            })
            if args.device.type == "cuda":
                torch.cuda.empty_cache()
    out["status"] = "info"
    return out


# ---------------------------------------------------------------------------
# C2: real G6
# ---------------------------------------------------------------------------

def section_c2(args, run_dir, model, tok):
    print(f"[c2] G6: N={args.c2_n_facts}, cap={args.c2_max_steps} steps")
    vocab = model.config.vocab_size
    # set_W (trained in weights) and set_M (mounted via bank)
    world_W = build_world(vocab_size=vocab, n_facts=args.c2_n_facts, seed=100)
    world_M = build_world(vocab_size=vocab, n_facts=args.c2_n_facts, seed=200)

    # 1. weights path
    print("[c2] training weights-path on set_W ...")
    token_stream = render_corpus(world_W, repeats_per_fact=10)
    batches = make_lm_batches(token_stream, batch_size=args.c2_batch_size,
                              seq_len=args.c2_seq_len, device=args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
    stats_W = train_to_convergence(
        model, batches, opt,
        max_steps=args.c2_max_steps,
        logger=make_logger(run_dir, "c2_train"),
        label_prefix="c2",
    )
    pairs_W = build_qa_pairs(world_W)
    w_acc = evaluate_qa(model, pairs_W, device=args.device)["accuracy"]
    print(f"[c2] weights-path accuracy on set_W: {w_acc:.3f} "
          f"(converged={stats_W['converged']} @ step {stats_W['converged_at']})")

    # 2. mount path: encode set_M into bank, eval set_M with bank
    print("[c2] encoding set_M into workspace KV bank ...")
    ws_tokens = render_corpus(world_M, repeats_per_fact=1)
    ws_text = tok.decode(ws_tokens, skip_special_tokens=True)
    bank = WorkspaceKVBank()
    bank.encode(model, ws_text, tok, args.device,
                max_length=args.c2_bank_max_length)
    pairs_M = build_qa_pairs(world_M)
    with bank.inject(model):
        m_acc = evaluate_qa(model, pairs_M, device=args.device)["accuracy"]
    # control: same set_M without mount
    control_acc = evaluate_qa(model, pairs_M, device=args.device)["accuracy"]

    ratio = m_acc / max(w_acc, 1e-6)
    passed = ratio >= args.c2_threshold

    print(f"\n=== C2 HEADLINE ===")
    print(f"  weights-path accuracy: {w_acc:.3f}")
    print(f"  mount-path accuracy:   {m_acc:.3f}")
    print(f"  control (no mount):    {control_acc:.3f}")
    print(f"  ratio (mount/weights): {ratio:.3f}")
    print(f"  threshold:             {args.c2_threshold:.3f}")
    print(f"  verdict:               {'PASS' if passed else 'FAIL'}")

    return {
        "section": "C2",
        "weights_accuracy": w_acc,
        "mount_accuracy": m_acc,
        "control_accuracy": control_acc,
        "ratio": ratio,
        "threshold": args.c2_threshold,
        "n_facts": args.c2_n_facts,
        "status": "pass" if passed else "fail",
        **stats_W,
    }


SECTIONS = {"c3": section_c3, "c1": section_c1, "c2": section_c2}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--section", default="all", help="c1|c2|c3|all")
    p.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--run_dir", default=None)
    p.add_argument("--c2_n_facts", type=int, default=512)
    p.add_argument("--c2_max_steps", type=int, default=4000)
    p.add_argument("--c2_batch_size", type=int, default=4)
    p.add_argument("--c2_seq_len", type=int, default=512)
    p.add_argument("--c2_bank_max_length", type=int, default=1024)
    p.add_argument("--c2_threshold", type=float, default=0.6)
    args = p.parse_args()
    args.device = parse_device(args.device)
    args.dtype = parse_dtype(args.dtype)
    run_dir = Path(args.run_dir or
                   f"runs/bench_gemma4/{time.strftime('%Y%m%d_%H%M%S')}")
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[bench_gemma4] device={args.device} dtype={args.dtype}")
    print(f"[bench_gemma4] run dir: {run_dir}")

    model, tok, report = load_gemma4(args.model_id, args.device, args.dtype)
    (run_dir / "surgery_report.json").write_text(json.dumps({
        "layers_replaced": report.layers_replaced,
        "layers_skipped": report.layers_skipped,
        "layers_path": report.layers_path,
        "new_param_bytes": report.new_param_bytes,
        "inherited_param_bytes": report.inherited_param_bytes,
        "notes": report.notes,
    }, indent=2))

    sections = ["c3", "c1", "c2"] if args.section == "all" else [args.section]
    summary: Dict[str, Any] = {"sections": {}}
    for s in sections:
        print(f"\n=== [STAGE {s.upper()}] ===")
        try:
            result = SECTIONS[s](args, run_dir, model, tok)
        except Exception as e:
            result = {"section": s.upper(), "status": "error",
                      "error": str(e)[:500]}
            import traceback; traceback.print_exc()
        (run_dir / f"{s}.json").write_text(json.dumps(result, indent=2,
                                                       default=str))
        summary["sections"][s] = {"status": result.get("status")}
        # Gate: stop early if c3 fails
        if s == "c3" and result.get("status") == "fail":
            print("\n[bench_gemma4] STOP — c3 smoke failed; skipping c1/c2")
            break

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[bench_gemma4] wrote {run_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
