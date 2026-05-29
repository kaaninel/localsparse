"""Mount-vs-flat custom eval (M6).

For each Q&A pair we evaluate accuracy two ways:
  - flat: the supporting passage is concatenated into the prompt.
  - mounted: the passage is loaded into a workspace and the model
    invokes `workspace.mount` to access it.

Target: mounted ≥ 70% of flat accuracy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Dict


@dataclass
class MountVsFlatResult:
    flat_correct: int
    mounted_correct: int
    n: int

    @property
    def flat_accuracy(self) -> float:
        return self.flat_correct / max(1, self.n)

    @property
    def mounted_accuracy(self) -> float:
        return self.mounted_correct / max(1, self.n)

    @property
    def parity(self) -> float:
        return self.mounted_accuracy / max(1e-9, self.flat_accuracy)


@dataclass
class QAItem:
    passage: str
    question: str
    answer: str


def mount_vs_flat_eval(
    items: List[QAItem],
    flat_predict: Callable[[str, str], str],
    mounted_predict: Callable[[str, str], str],
    *,
    judge: Callable[[str, str], bool] = lambda gold, got: gold.lower().strip() in got.lower(),
) -> MountVsFlatResult:
    flat_c = 0
    mnt_c = 0
    for it in items:
        flat_out = flat_predict(it.passage, it.question)
        mnt_out = mounted_predict(it.passage, it.question)
        if judge(it.answer, flat_out):
            flat_c += 1
        if judge(it.answer, mnt_out):
            mnt_c += 1
    return MountVsFlatResult(flat_correct=flat_c, mounted_correct=mnt_c, n=len(items))
