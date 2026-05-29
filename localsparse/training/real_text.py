"""Streaming real-text dataset wrappers for Gemma 4 E2B training (§8 S2/S3).

Two corpus sources:
  - `streaming_fineweb_batches`: HuggingFaceFW/fineweb-edu streaming →
    `(input_ids, labels)` batches used by S2 (distill on real text).
  - `rag_batches_from_squad`: SQuAD v1.1 → `(bank_text, question, answer_ids)`
    used by S3 (RAG workspace training).

Both functions return generators that yield batches **forever** (cycling the
underlying stream) so the training loop can hit any step budget without
worrying about epoch boundaries. Each function is offline-friendly: if HF
datasets streaming is unavailable, a synthetic fallback yields random tokens
so smoke tests still run on the local machine.
"""
from __future__ import annotations

import itertools
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# FineWeb streaming
# ---------------------------------------------------------------------------
def streaming_fineweb_batches(
    tokenizer: Any,
    *,
    batch_size: int = 4,
    seq_len: int = 512,
    device: torch.device = torch.device("cpu"),
    dataset_name: str = "HuggingFaceFW/fineweb-edu",
    split: str = "train",
    text_column: str = "text",
    n_total_tokens: Optional[int] = None,
    seed: int = 0,
    fallback_vocab_size: Optional[int] = None,
) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
    """Yield `(input_ids, labels)` packed sequences from FineWeb-Edu.

    Texts are tokenized on the fly and packed into contiguous `seq_len`
    chunks. Labels = input_ids (next-token loss handled externally).

    If `n_total_tokens` is given, the generator stops after that many tokens
    have been served; otherwise it cycles forever.

    Falls back to random tokens if HF datasets can't stream (offline test).
    """
    served = 0

    try:
        from datasets import load_dataset
        ds = load_dataset(dataset_name, split=split, streaming=True)
        # Determinism: per-session shuffle is best-effort with streaming
        try:
            ds = ds.shuffle(seed=seed, buffer_size=10_000)
        except Exception:
            pass
        iterator = iter(ds)
    except Exception as e:  # pragma: no cover - offline path
        print(f"[fineweb] streaming unavailable ({e}); using random fallback")
        iterator = None

    vocab_size = (
        fallback_vocab_size
        or getattr(tokenizer, "vocab_size", None)
        or 32_000
    )
    rng = random.Random(seed)

    buf: List[int] = []
    while True:
        # Refill buffer
        while len(buf) < batch_size * seq_len:
            if iterator is not None:
                try:
                    row = next(iterator)
                except StopIteration:
                    iterator = iter(load_dataset(
                        dataset_name, split=split, streaming=True))
                    continue
                txt = row.get(text_column, "")
                if not isinstance(txt, str) or len(txt) < 4:
                    continue
                ids = tokenizer.encode(txt, add_special_tokens=False)
                buf.extend(ids)
            else:
                buf.extend(rng.randint(0, vocab_size - 1)
                           for _ in range(seq_len))

        chunk = buf[: batch_size * seq_len]
        buf = buf[batch_size * seq_len:]
        x = torch.tensor(chunk, dtype=torch.long, device=device)
        x = x.view(batch_size, seq_len)
        served += x.numel()
        yield x, x.clone()
        if n_total_tokens is not None and served >= n_total_tokens:
            return


# ---------------------------------------------------------------------------
# SQuAD RAG batches
# ---------------------------------------------------------------------------
@dataclass
class RagBatch:
    """A single RAG-style training batch.

    `bank_texts` are encoded by `WorkspaceKVBank.encode` per-batch (small N),
    then `inject` is held while the LM is forwarded on the question/answer
    sequence. The answer-ids are used as labels (with question tokens masked
    out by setting label=-100 for the prompt span).
    """

    bank_texts: List[str]
    input_ids: torch.Tensor      # (B, T)
    labels: torch.Tensor         # (B, T)  -100 where not predicting
    questions: List[str]
    answers: List[str]


def rag_batches_from_squad(
    tokenizer: Any,
    *,
    batch_size: int = 2,
    qa_max_length: int = 256,
    device: torch.device = torch.device("cpu"),
    split: str = "train",
    seed: int = 0,
    n_total: Optional[int] = None,
) -> Generator[RagBatch, None, None]:
    """Yield RAG-style batches sourced from SQuAD v1.1.

    Each batch contains `batch_size` items, each item has:
      bank_text = the passage (context)
      input_ids = "Q: {question}\\nA: {answer}" tokenized
      labels    = same, with the prompt span masked to -100

    Falls back to a tiny synthetic set if `datasets` unavailable.
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("squad", split=split)
        ds = ds.shuffle(seed=seed)
        rows: Iterable[Dict[str, Any]] = ds
    except Exception as e:  # pragma: no cover - offline path
        print(f"[squad] dataset unavailable ({e}); using synthetic fallback")
        rows = _synthetic_squad_rows(seed=seed, n=max(8, batch_size * 4))

    served = 0
    rows_iter = iter(rows)
    while True:
        batch_rows: List[Dict[str, Any]] = []
        for _ in range(batch_size):
            try:
                batch_rows.append(next(rows_iter))
            except StopIteration:
                rows_iter = iter(rows)
                batch_rows.append(next(rows_iter))

        bank_texts: List[str] = []
        ids_list: List[List[int]] = []
        labels_list: List[List[int]] = []
        qs: List[str] = []
        ans: List[str] = []
        for row in batch_rows:
            ctx = row.get("context") or row.get("passage") or ""
            q = row.get("question") or ""
            a = _first_answer(row)
            bank_texts.append(ctx)
            qs.append(q)
            ans.append(a)
            prompt = f"Q: {q}\nA:"
            full = prompt + " " + a
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            full_ids = tokenizer.encode(full, add_special_tokens=False)
            full_ids = full_ids[:qa_max_length]
            n_pad = qa_max_length - len(full_ids)
            input_ids = full_ids + [0] * n_pad
            labels = list(full_ids)
            for i in range(min(len(prompt_ids), len(labels))):
                labels[i] = -100
            for i in range(len(full_ids), qa_max_length):
                labels.append(-100)
            ids_list.append(input_ids)
            labels_list.append(labels)

        x = torch.tensor(ids_list, dtype=torch.long, device=device)
        y = torch.tensor(labels_list, dtype=torch.long, device=device)
        served += batch_size
        yield RagBatch(
            bank_texts=bank_texts, input_ids=x, labels=y,
            questions=qs, answers=ans,
        )
        if n_total is not None and served >= n_total:
            return


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _first_answer(row: Dict[str, Any]) -> str:
    a = row.get("answers", {})
    if isinstance(a, dict):
        texts = a.get("text", [])
        if texts:
            return texts[0]
    if isinstance(row.get("answer"), str):
        return row["answer"]
    return ""


def _synthetic_squad_rows(*, seed: int, n: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        ctx = f"Synthetic context {i}: alpha_{rng.randint(0, 999)} bravo charlie."
        q = f"What is the value of token {i}?"
        a = f"value_{i}"
        rows.append({"context": ctx, "question": q,
                     "answers": {"text": [a]}})
    return rows
