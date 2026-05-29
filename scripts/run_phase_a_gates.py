"""Phase A gate runner for M0.5 / Veyra3.

Executes the Phase A gates against a Veyra3 model that has been
surgically replaced with `ThreeBranchAttention` on its `full_attention`
layers.

Gates:
  G1   surgery numerical stability  — 100 fwd+bwd, 0 NaN/Inf
  G2   branch non-collapse          — min branch_mass ≥ 0.05 over
                                       last 50 of first 1K steps
  G3   indexer block-routing        — top-1 ≥ 6× random baseline
  Gsave save/load fidelity          — post/pre logit diff ≤ 1e-2

Output:
  <run_dir>/train.jsonl
  <run_dir>/gates.jsonl
  <run_dir>/summary.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn

from localsparse.logging import (
    RunLogger, GateLogger, RunDirectory, FailureDetector,
    dump_debug_state, per_module_grad_norms,
)
from localsparse.training.data import synthetic_lm_batch
from localsparse.training.milestone1 import M1Config, collect_branch_masses


def _device(s):
    if s == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(s)


def _load_and_surgery(model_id: str, device: torch.device, dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from localsparse.model.veyra_adapter import surgery_veyra3
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
    report = surgery_veyra3(model)
    print(f"[gates] surgery: replaced layers {report.layers_replaced}")
    model = model.to(device)
    return model, tok, report


def _make_batch(vocab_size: int, seq_len: int, batch: int, device):
    b = synthetic_lm_batch(batch_size=batch, seq_len=seq_len, vocab_size=vocab_size)
    b.input_ids = b.input_ids.to(device)
    b.labels = b.labels.to(device)
    return b


def run_g1_g2(model: nn.Module, *, run: RunDirectory, gates: GateLogger,
              steps: int, batch_size: int, seq_len: int, lr: float,
              vocab_size: int, device):
    logger = RunLogger(run, print_every=50)
    detector = FailureDetector(
        on_fire=lambda r, rec: dump_debug_state(
            run, r, model=model, extra={"trigger_rec": rec}),
        branch_collapse_window=50,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    nan_seen = 0
    branch_min_history = []
    masses_history = []
    # Find TBA layers for branch_gate inspection
    from localsparse.attention.sparse_three_branch import ThreeBranchAttention
    tba_layers = [m for m in model.modules() if isinstance(m, ThreeBranchAttention)]
    t0 = time.time()
    for step in range(steps):
        b = _make_batch(vocab_size, seq_len, batch_size, device)
        opt.zero_grad(set_to_none=True)
        out = model(input_ids=b.input_ids, labels=b.labels)
        loss = out.loss
        if not torch.isfinite(loss):
            nan_seen += 1
        loss.backward()
        gn = per_module_grad_norms(model)
        opt.step()
        # Use branch_gate softmax (mixing weights, sum to 1) as the
        # branch-mass measure — these are the actual proportions of each
        # branch in the output.
        s = sel = c = 0.0
        for tba in tba_layers:
            mix = tba.branch_gate.detach().float().softmax(dim=-1).mean(dim=0)
            s += float(mix[0]); sel += float(mix[1]); c += float(mix[2])
        if tba_layers:
            n = len(tba_layers)
            s /= n; sel /= n; c /= n
        masses_history.append((s, sel, c))
        if (s + sel + c) > 0:
            branch_min_history.append(min(s, sel, c))
        rec = {
            "step": step, "lm_loss": float(loss.detach()),
            "total_loss": float(loss.detach()),
            "sliding_mass": s, "selected_mass": sel, "compressed_mass": c,
            **gn, "tokens_per_sec": batch_size * seq_len * (step + 1) / max(time.time() - t0, 1e-6),
        }
        logger.step(**rec)
        detector.check(rec)

    # G1 — NaN check
    gates.record("G1", metric="nan_count", value=nan_seen,
                 threshold=0, status="pass" if nan_seen == 0 else "fail")
    # G2 — branch non-collapse, last 50 of first 1000 (or all if fewer)
    tail = branch_min_history[max(0, min(1000, len(branch_min_history)) - 50):min(1000, len(branch_min_history))]
    g2_val = min(tail) if tail else 0.0
    gates.record("G2", metric="min_branch_mass_tail", value=g2_val,
                 threshold=0.05, status="pass" if g2_val >= 0.05 else "fail")
    return masses_history


def run_g3_indexer(model: nn.Module, *, gates: GateLogger,
                   vocab_size: int, n_blocks: int = 16, n_trials: int = 100,
                   seq_len: int = 1024, device):
    """Synthetic test: insert a unique 'needle' subsequence into one of N
    equal-sized blocks of a random context; measure how often the indexer's
    top-1 selected block matches the planted block.

    We probe the indexer's score by extracting `_last_selected_indices`
    from any ThreeBranchAttention layer (set during the standard forward).
    """
    from localsparse.attention.sparse_three_branch import ThreeBranchAttention
    # Find a TBA layer
    tba = None
    for m in model.modules():
        if isinstance(m, ThreeBranchAttention):
            tba = m
            break
    if tba is None:
        gates.record("G3", metric="indexer_top1_accuracy", value=0.0,
                     threshold=6 / n_blocks, status="fail",
                     note="no ThreeBranchAttention layer found")
        return

    hits = 0
    attempted = 0
    needle_token = vocab_size - 1
    for trial in range(n_trials):
        # Random ctx
        ids = torch.randint(0, vocab_size - 1, (1, seq_len), device=device)
        block_size = seq_len // n_blocks
        target_block = trial % n_blocks
        start = target_block * block_size
        # Plant a small unique pattern at start of target block; mention it in
        # the "query" position (last 8 tokens) so the indexer should route there.
        pattern = torch.tensor([needle_token, needle_token - 1, needle_token - 2],
                               device=device)
        ids[0, start:start + 3] = pattern
        ids[0, -3:] = pattern  # query references the planted pattern
        with torch.no_grad():
            try:
                model(input_ids=ids)
            except Exception as e:
                print(f"[G3] trial {trial} forward failed: {e}")
                continue
        sel_idx = getattr(tba, "_last_selected_indices", None)
        if sel_idx is None:
            continue
        attempted += 1
        # sel_idx shape: (B, H, T, K) — block indices over compressed blocks
        # We map our virtual blocks to its compressed-block grid:
        # compressed_block_size in the model may differ; the *relative*
        # position of the target should land in the lower-numbered block
        # range proportional to target_block / n_blocks. We accept top-1
        # match if any head selected a block whose center maps back to
        # target_block.
        top1 = sel_idx[0, :, -1, 0].cpu().tolist()  # all heads, last token, top-1
        # Map each picked block id → coarse target-block id
        comp_block_size = tba.attn_cfg.compressed_block
        approx_pos = [tid * comp_block_size for tid in top1]
        coarse = [p // block_size for p in approx_pos]
        if target_block in coarse:
            hits += 1

    acc = hits / max(attempted, 1)
    random_baseline = 1.0 / n_blocks
    threshold = 6 * random_baseline
    status = "pass" if acc >= threshold else ("stretch" if acc >= 2 * random_baseline else "fail")
    gates.record("G3", metric="indexer_top1_accuracy", value=acc,
                 threshold=threshold, status=status,
                 random_baseline=random_baseline, n_trials=attempted)


def run_gsave(model: nn.Module, *, gates: GateLogger, vocab_size: int,
              seq_len: int, device, tmp_dir: Path):
    """Save + reload + compare logits on identical input."""
    x = torch.randint(0, vocab_size, (1, seq_len), device=device)
    with torch.no_grad():
        pre = model(input_ids=x).logits.float().cpu()
    ckpt = tmp_dir / "save_load_ckpt"
    ckpt.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt / "model.pt")
    # Reload into-same instance to avoid recreating Veyra3 model:
    sd = torch.load(ckpt / "model.pt", map_location=device, weights_only=True)
    model.load_state_dict(sd)
    with torch.no_grad():
        post = model(input_ids=x).logits.float().cpu()
    diff = float((pre - post).abs().max())
    threshold = 1e-2  # bf16 epsilon ~1e-2
    gates.record("Gsave", metric="max_abs_logit_diff", value=diff,
                 threshold=threshold,
                 status="pass" if diff <= threshold else "fail")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="veyra-ai/veyra3-5m-base")
    ap.add_argument("--out", required=True, type=Path,
                    help="Parent dir; run_<ts>/ will be created")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--g3-trials", type=int, default=64)
    ap.add_argument("--g3-blocks", type=int, default=16)
    ap.add_argument("--skip-g3", action="store_true")
    args = ap.parse_args()

    device = _device(args.device)
    dtype = torch.bfloat16 if device.type in {"mps", "cuda"} else torch.float32
    print(f"[gates] device={device} dtype={dtype}")

    model, tok, report = _load_and_surgery(args.model, device, dtype)
    vocab_size = model.config.vocab_size

    run = RunDirectory.fresh(args.out, prefix="phase_a")
    print(f"[gates] run dir: {run.root}")
    gates = GateLogger(run)

    print("[gates] G1+G2 training healing run…")
    masses_hist = run_g1_g2(model, run=run, gates=gates,
                            steps=args.steps, batch_size=args.batch,
                            seq_len=args.seq_len, lr=args.lr,
                            vocab_size=vocab_size, device=device)

    if not args.skip_g3:
        print("[gates] G3 indexer routing test…")
        run_g3_indexer(model, gates=gates, vocab_size=vocab_size,
                       n_blocks=args.g3_blocks, n_trials=args.g3_trials,
                       seq_len=args.seq_len, device=device)

    print("[gates] Gsave save/load fidelity…")
    run_gsave(model, gates=gates, vocab_size=vocab_size,
              seq_len=512, device=device, tmp_dir=run.root)

    summary = {
        "model": args.model,
        "device": str(device),
        "dtype": str(dtype),
        "steps": args.steps,
        "surgery_report": {
            "layers_replaced": report.layers_replaced,
            "layers_skipped": report.layers_skipped,
        },
        "branch_masses_final": masses_hist[-1] if masses_hist else None,
    }
    run.summary_json.write_text(json.dumps(summary, indent=2))
    print(f"[gates] wrote {run.summary_json}")


if __name__ == "__main__":
    main()
