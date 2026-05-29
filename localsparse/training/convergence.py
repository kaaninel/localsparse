"""Plateau-detection helpers for "train to convergence" discipline (plan §7.3).

M0.5 cut training short. M1.5 requires every benchmark to train until loss
plateaus, with a generous safety cap. These helpers provide the detector and
a `train_to_convergence` loop usable from any bench script.

Plateau definition (default):
  - Maintain a window of N most recent steps (default 100)
  - Plateau = `K` consecutive windows where relative improvement
    (mean(prev_window) - mean(curr_window)) / mean(prev_window) < `tol`
  - Default: K=3, tol=0.01 → 3 consecutive 100-step windows with <1% improvement
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional


@dataclass
class PlateauDetector:
    """Detects training-loss plateau via rolling-window relative improvement.

    Call `update(loss)` every step. After enough history, `should_stop()`
    returns True when convergence is detected.
    """
    window: int = 100
    consecutive: int = 3
    tol: float = 0.01
    min_steps: int = 200  # never declare convergence before this many steps

    _losses: Deque[float] = field(default_factory=deque, repr=False)
    _means: Deque[float] = field(default_factory=deque, repr=False)
    _ok_count: int = 0
    _step: int = 0
    converged_at: Optional[int] = None

    def update(self, loss: float) -> None:
        self._step += 1
        self._losses.append(float(loss))
        if len(self._losses) > self.window:
            self._losses.popleft()
        if len(self._losses) == self.window:
            mean = sum(self._losses) / self.window
            self._means.append(mean)
            if len(self._means) > self.consecutive + 1:
                self._means.popleft()
            self._check()

    def _check(self) -> None:
        if self._step < self.min_steps:
            return
        if len(self._means) < 2:
            return
        prev = self._means[-2]
        curr = self._means[-1]
        if prev <= 0:
            return
        rel = (prev - curr) / abs(prev)
        if rel < self.tol:
            self._ok_count += 1
            if self._ok_count >= self.consecutive and self.converged_at is None:
                self.converged_at = self._step
        else:
            self._ok_count = 0

    @property
    def converged(self) -> bool:
        return self.converged_at is not None

    def should_stop(self) -> bool:
        return self.converged
