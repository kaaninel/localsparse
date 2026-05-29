"""Synthetic / streamed training data builders.

For local validation we use:
  - `synthetic_lm_batch`: random token ids (PPL won't drop but the
    training loop will run end-to-end).
  - `needle_in_haystack_batch`: classic NIAH for long-ctx eval; the
    haystack is built from random tokens with a planted "magic number"
    near a chosen position, and the question asks to retrieve it.

For real training (Colab) `FineWebStream` and `RoutingSyntheticStream`
lazy-import HuggingFace `datasets`. Both produce dict batches with
`input_ids`, `attention_mask`, and `labels`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional, List
import random
import torch


@dataclass
class Batch:
    input_ids: torch.Tensor      # (B, T)
    labels: torch.Tensor         # (B, T) — -100 for ignored positions
    attention_mask: Optional[torch.Tensor] = None
    needle_position: Optional[List[int]] = None
    needle_value: Optional[List[int]] = None


def synthetic_lm_batch(batch_size: int, seq_len: int, vocab_size: int,
                       *, seed: int = 0) -> Batch:
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(1, vocab_size, (batch_size, seq_len), generator=g)
    return Batch(input_ids=ids, labels=ids.clone())


def needle_in_haystack_batch(batch_size: int, seq_len: int, vocab_size: int,
                             *, seed: int = 0) -> Batch:
    """Plant a unique 'magic number' token at a random position, then ask
    the model to repeat it. We just verify that LM loss on the needle
    position behaves; full eval logic lives in eval/niah.py."""
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(1, vocab_size - 16, (batch_size, seq_len), generator=g)
    needle_positions: list[int] = []
    needle_values: list[int] = []
    for b in range(batch_size):
        pos = int(torch.randint(seq_len // 4, 3 * seq_len // 4, (1,), generator=g).item())
        # Make the needle a token in the high range so it's clearly distinct
        val = int(torch.randint(vocab_size - 16, vocab_size, (1,), generator=g).item())
        ids[b, pos] = val
        needle_positions.append(pos)
        needle_values.append(val)
    labels = torch.full_like(ids, -100)
    for b, pos in enumerate(needle_positions):
        labels[b, pos] = ids[b, pos]
    return Batch(input_ids=ids, labels=labels,
                 needle_position=needle_positions,
                 needle_value=needle_values)


# ---------------------------------------------------------------------------
# HF datasets-backed streams (lazy import; used during real training only)
# ---------------------------------------------------------------------------
class FineWebStream:
    """Streaming loader from HuggingFace `HuggingFaceFW/fineweb` or `Ultra-FineWeb`.

    Yields Batch objects pre-tokenized with the supplied tokenizer.
    """

    def __init__(self, tokenizer, *, dataset: str = "HuggingFaceFW/fineweb",
                 split: str = "train", seq_len: int = 4096, batch_size: int = 1,
                 streaming: bool = True):
        from datasets import load_dataset
        self.ds = load_dataset(dataset, split=split, streaming=streaming)
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[Batch]:
        buf: list[int] = []
        out_rows: list[list[int]] = []
        for ex in self.ds:
            text = ex.get("text") or ex.get("content") or ""
            if not text:
                continue
            ids = self.tokenizer(text, add_special_tokens=False).input_ids
            buf.extend(ids)
            while len(buf) >= self.seq_len:
                out_rows.append(buf[: self.seq_len])
                buf = buf[self.seq_len :]
                if len(out_rows) == self.batch_size:
                    t = torch.tensor(out_rows, dtype=torch.long)
                    yield Batch(input_ids=t, labels=t.clone())
                    out_rows = []
