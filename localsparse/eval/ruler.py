"""RULER-style needle-in-haystack eval at multiple context lengths.

Generates a haystack of `ctx_len` random tokens, embeds K needle facts
("the magic number 42 is hidden here"), then queries the model for each
needle. We don't have a tokenizer here so the harness is generic over a
callable `predict(prompt_ids) -> answer_ids` interface — Colab plugs in
the tokenizer + HFBackend.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List
import random


@dataclass
class RulerResult:
    ctx_len: int
    n_needles: int
    n_recovered: int

    @property
    def accuracy(self) -> float:
        return self.n_recovered / max(1, self.n_needles)


def make_haystack(ctx_len: int, needles: List[str], filler: str,
                  *, seed: int = 0) -> tuple[str, list[int]]:
    """Insert each `needle` at a random position inside `filler` until total
    length ≈ ctx_len chars. Returns (haystack, positions)."""
    random.seed(seed)
    haystack = (filler + " ") * max(1, ctx_len // max(1, len(filler) + 1))
    haystack = haystack[:ctx_len]
    positions = sorted(random.sample(range(len(haystack)), len(needles)))
    chars = list(haystack)
    for pos, n in zip(positions, needles):
        chars[pos : pos + len(n)] = list(n)
    return "".join(chars), positions


def ruler_eval(
    predict_fn: Callable[[str], str],
    *,
    ctx_len: int = 32_768,
    n_needles: int = 5,
    filler: str = "the quick brown fox jumps over the lazy dog ",
    seed: int = 0,
) -> RulerResult:
    """Run a single ctx-len, n-needle eval. Returns RulerResult."""
    needles = [f"the magic number {i+1} is {1000 + i}." for i in range(n_needles)]
    haystack, _ = make_haystack(ctx_len, needles, filler, seed=seed)
    recovered = 0
    for i in range(n_needles):
        prompt = (
            f"{haystack}\n\n"
            f"Question: what is the magic number {i+1}? Answer with just the number."
        )
        answer = predict_fn(prompt)
        if str(1000 + i) in answer:
            recovered += 1
    return RulerResult(ctx_len=ctx_len, n_needles=n_needles, n_recovered=recovered)
