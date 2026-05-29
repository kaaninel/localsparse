"""Gemma 4 E2B evaluation bundle: PPL, G6 synthetic, RAG-acc real.

Used by per-stage gates in `notebooks/gemma4_e2b_training.ipynb` and by S4
final eval to produce the model card summary.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

from ..training.factoid_world import (
    build_qa_pairs, build_world, evaluate_qa, make_lm_batches, render_corpus,
)
from ..workspace.kv_bank import WorkspaceKVBank


# ---------------------------------------------------------------------------
# Held-out PPL on streaming batches (used after S2 distill)
# ---------------------------------------------------------------------------
def eval_perplexity(
    model: torch.nn.Module,
    batches: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    *,
    max_batches: int = 20,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """Compute average token-level CE and PPL across up to `max_batches`."""
    model.eval()
    tot_loss = 0.0
    tot_tokens = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(batches):
            if device is not None:
                x = x.to(device); y = y.to(device)
            out = model(input_ids=x, labels=y)
            n = (y != -100).sum().item() if (y == -100).any() else y.numel()
            tot_loss += float(out.loss.detach()) * n
            tot_tokens += n
            if i + 1 >= max_batches:
                break
    avg = tot_loss / max(tot_tokens, 1)
    return {"loss": avg, "ppl": math.exp(min(avg, 20.0)),
            "n_tokens": tot_tokens}


# ---------------------------------------------------------------------------
# G6 synthetic — weights vs. bank-mount on factoid worlds
# ---------------------------------------------------------------------------
def eval_g6_synthetic(
    model: torch.nn.Module,
    tokenizer: Any,
    *,
    device: torch.device,
    n_facts: int = 64,
    seed_weights: int = 100,
    seed_mount: int = 200,
    bank_max_length: int = 1024,
) -> Dict[str, float]:
    """Train-once weights set vs. mount-only set both evaluated.

    Note: This is *eval-only* — assumes the model already has whatever
    training the caller wants. For the actual G6 *training* see
    `bench_gemma4.section_c2`. Use this as a gate metric in the notebook
    after C2 finishes.
    """
    vocab = model.config.vocab_size if hasattr(model, "config") else 32_000
    world_W = build_world(vocab_size=vocab, n_facts=n_facts, seed=seed_weights)
    world_M = build_world(vocab_size=vocab, n_facts=n_facts, seed=seed_mount)

    pairs_W = build_qa_pairs(world_W)
    w_acc = evaluate_qa(model, pairs_W, device=device)["accuracy"]

    ws_tokens = render_corpus(world_M, repeats_per_fact=1)
    ws_text = tokenizer.decode(ws_tokens, skip_special_tokens=True)
    bank = WorkspaceKVBank()
    bank.encode(model, ws_text, tokenizer, device, max_length=bank_max_length)
    pairs_M = build_qa_pairs(world_M)
    with bank.inject(model):
        m_acc = evaluate_qa(model, pairs_M, device=device)["accuracy"]
    control_acc = evaluate_qa(model, pairs_M, device=device)["accuracy"]

    ratio = m_acc / max(w_acc, 1e-6)
    return {
        "g6_weights_acc": w_acc,
        "g6_mount_acc": m_acc,
        "g6_control_acc": control_acc,
        "g6_ratio_mount_over_weights": ratio,
        "n_facts": n_facts,
    }


# ---------------------------------------------------------------------------
# RAG accuracy on SQuAD-style batches (used after S3)
# ---------------------------------------------------------------------------
def eval_rag_accuracy(
    model: torch.nn.Module,
    tokenizer: Any,
    rag_batches: Iterable[Any],
    *,
    device: torch.device,
    max_batches: int = 8,
    bank_max_length: int = 1024,
) -> Dict[str, float]:
    """Compare answer-first-token accuracy with vs. without bank mount.

    For each batch we encode the passage as a bank and check whether the
    model's argmax token at the answer position matches the gold first
    answer token. This is a crude but cheap metric.
    """
    model.eval()
    n_total = 0
    n_mount_correct = 0
    n_nomount_correct = 0
    with torch.no_grad():
        for bi, rb in enumerate(rag_batches):
            if bi >= max_batches:
                break
            x = rb.input_ids.to(device)
            y = rb.labels.to(device)
            target_pos = (y != -100).int().argmax(dim=-1)  # first answer pos
            # Build a single bank that holds all passages concatenated.
            bank_text = "\n\n".join(rb.bank_texts)
            bank = WorkspaceKVBank()
            try:
                bank.encode(model, bank_text, tokenizer, device,
                            max_length=bank_max_length)
            except Exception as e:
                print(f"[rag-eval] bank encode failed: {e}")
                continue
            with bank.inject(model):
                logits_m = model(input_ids=x).logits
            logits_n = model(input_ids=x).logits
            for i in range(x.shape[0]):
                tp = int(target_pos[i].item())
                gold = int(y[i, tp].item()) if y[i, tp].item() != -100 else None
                if gold is None:
                    continue
                pred_m = int(logits_m[i, tp - 1].argmax().item())
                pred_n = int(logits_n[i, tp - 1].argmax().item())
                n_total += 1
                if pred_m == gold:
                    n_mount_correct += 1
                if pred_n == gold:
                    n_nomount_correct += 1
    if n_total == 0:
        return {"rag_mount_acc": 0.0, "rag_nomount_acc": 0.0,
                "rag_ratio_mount_over_nomount": 0.0, "rag_n": 0}
    m_acc = n_mount_correct / n_total
    n_acc = n_nomount_correct / n_total
    ratio = m_acc / max(n_acc, 1e-6)
    return {
        "rag_mount_acc": m_acc,
        "rag_nomount_acc": n_acc,
        "rag_ratio_mount_over_nomount": ratio,
        "rag_n": n_total,
    }
