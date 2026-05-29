"""Run-level logging + analytics infrastructure for M0.5 iteration.

Why this exists: when an experiment at 5M scale fails, we need to know
*why* and *where* fast. Sparse-attention training has many silent
failure modes (branch collapse, indexer never trains, NaN in
compressed pool, etc.) — each of which we want to detect automatically
and freeze the failing state for inspection in a notebook.

Components:
  - `RunLogger`: writes one JSONL per training step
  - `GateLogger`: writes one JSONL per gate evaluation
  - `FailureDetector`: monitors loss/grad/branch stats, fires alerts,
    triggers debug-on-fail dumps
  - `RunDirectory`: standard layout for one run

Default backend is plain JSONL files — zero deps, works on the user's
M4 Air without W&B/TensorBoard.
"""
from __future__ import annotations

import json
import time
import datetime as dt
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable

import torch


@dataclass
class RunDirectory:
    root: Path
    train_jsonl: Path = field(init=False)
    gates_jsonl: Path = field(init=False)
    summary_json: Path = field(init=False)

    def __post_init__(self):
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.train_jsonl = self.root / "train.jsonl"
        self.gates_jsonl = self.root / "gates.jsonl"
        self.summary_json = self.root / "summary.json"

    @classmethod
    def fresh(cls, parent: Path, prefix: str = "run") -> "RunDirectory":
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        return cls(root=Path(parent) / f"{prefix}_{ts}")

    def debug_dir(self, reason: str) -> Path:
        d = self.root / f"debug_{reason}_{int(time.time())}"
        d.mkdir(parents=True, exist_ok=True)
        return d


def _json_safe(o: Any) -> Any:
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist() if o.numel() < 64 else f"<tensor shape={tuple(o.shape)}>"
    if isinstance(o, Path):
        return str(o)
    return repr(o)


def _append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record, default=_json_safe) + "\n")


class RunLogger:
    """Writes one JSON record per step to train.jsonl."""

    def __init__(self, run: RunDirectory, *, print_every: int = 50):
        self.run = run
        self.print_every = print_every
        self._t0 = time.time()
        self._records_written = 0

    def step(self, **fields) -> None:
        rec = {
            "step": fields.pop("step", self._records_written),
            "wall": round(time.time() - self._t0, 3),
            **fields,
        }
        _append_jsonl(self.run.train_jsonl, rec)
        self._records_written += 1
        if self.print_every and (self._records_written % self.print_every == 0):
            print(self._format(rec))

    @staticmethod
    def _format(rec: Dict[str, Any]) -> str:
        parts = [f"[step {rec['step']:>5}]"]
        for k in ("lm_loss", "total_loss", "branch_balance",
                  "sliding_mass", "selected_mass", "compressed_mass",
                  "grad_norm_total", "tokens_per_sec"):
            if k in rec and rec[k] is not None:
                parts.append(f"{k}={rec[k]:.4f}")
        return " ".join(parts)


class GateLogger:
    """Writes one JSON record per gate evaluation."""

    def __init__(self, run: RunDirectory):
        self.run = run

    def record(self, gate_id: str, *, metric: str, value: float,
               threshold: float, status: str, **extra) -> None:
        assert status in {"pass", "fail", "stretch", "deferred"}
        rec = {
            "ts": dt.datetime.now().isoformat(timespec="seconds"),
            "gate_id": gate_id,
            "metric": metric,
            "value": float(value),
            "threshold": float(threshold),
            "status": status,
            **extra,
        }
        _append_jsonl(self.run.gates_jsonl, rec)
        marker = {"pass": "✅", "fail": "❌", "stretch": "🟡", "deferred": "⏸"}[status]
        print(f"  {marker} {gate_id}: {metric}={value:.4f} (≥ {threshold:.4f})")


@dataclass
class _DetectorState:
    consec_branch_collapse: int = 0
    grad_norm_window: deque = field(default_factory=lambda: deque(maxlen=100))
    loss_window: deque = field(default_factory=lambda: deque(maxlen=200))


class FailureDetector:
    """Watches training records and fires callbacks on detected failures.

    Detected modes:
      - NaN/Inf in any loss field      → 'nan'
      - min branch_mass < threshold for N → 'branch_collapse'
      - grad_norm > Mx rolling median  → 'explosion'
      - lm_loss not improving in window → 'stuck'
    """

    def __init__(self, on_fire: Optional[Callable[[str, Dict[str, Any]], None]] = None,
                 *, branch_collapse_threshold: float = 0.01,
                 branch_collapse_window: int = 50,
                 grad_explosion_multiplier: float = 100.0,
                 stuck_window: int = 200, stuck_rel_threshold: float = 0.01):
        self.on_fire = on_fire or (lambda reason, rec: None)
        self.s = _DetectorState()
        self.branch_collapse_threshold = branch_collapse_threshold
        self.branch_collapse_window = branch_collapse_window
        self.grad_explosion_multiplier = grad_explosion_multiplier
        self.stuck_window = stuck_window
        self.stuck_rel_threshold = stuck_rel_threshold
        self.fired: List[str] = []

    def check(self, rec: Dict[str, Any]) -> List[str]:
        fired_now: List[str] = []
        for k in ("lm_loss", "total_loss", "branch_balance",
                  "routing_loss", "kl_loss"):
            if k in rec and rec[k] is not None:
                v = rec[k]
                if not (v == v) or v in (float("inf"), float("-inf")):
                    fired_now.append("nan")
                    break

        masses = [rec.get(k) for k in ("sliding_mass", "selected_mass", "compressed_mass")]
        masses = [m for m in masses if m is not None]
        if masses and min(masses) < self.branch_collapse_threshold:
            self.s.consec_branch_collapse += 1
            if self.s.consec_branch_collapse >= self.branch_collapse_window:
                fired_now.append("branch_collapse")
                self.s.consec_branch_collapse = 0
        else:
            self.s.consec_branch_collapse = 0

        gn = rec.get("grad_norm_total")
        if gn is not None and gn > 0:
            self.s.grad_norm_window.append(gn)
            if len(self.s.grad_norm_window) >= 20:
                sorted_w = sorted(self.s.grad_norm_window)
                median = sorted_w[len(sorted_w) // 2]
                if median > 0 and gn > self.grad_explosion_multiplier * median:
                    fired_now.append("explosion")

        lm = rec.get("lm_loss")
        if lm is not None:
            self.s.loss_window.append(lm)
            if len(self.s.loss_window) >= self.stuck_window:
                half = self.stuck_window // 2
                early = sum(list(self.s.loss_window)[:half]) / half
                late = sum(list(self.s.loss_window)[-half:]) / half
                if early > 0 and (early - late) / early < self.stuck_rel_threshold:
                    fired_now.append("stuck")
                    self.s.loss_window.clear()

        for reason in fired_now:
            if reason not in self.fired:
                self.fired.append(reason)
            self.on_fire(reason, rec)
        return fired_now


def dump_debug_state(
    run: RunDirectory, reason: str,
    *, model: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    batch: Optional[Dict[str, torch.Tensor]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    d = run.debug_dir(reason)
    if model is not None:
        torch.save(model.state_dict(), d / "model.pt")
    if optimizer is not None:
        torch.save(optimizer.state_dict(), d / "optimizer.pt")
    if batch is not None:
        torch.save({k: v for k, v in batch.items()}, d / "batch.pt")
    if extra is not None:
        (d / "extra.json").write_text(json.dumps(extra, indent=2, default=_json_safe))
    print(f"[debug-on-fail:{reason}] state saved to {d}")
    return d


def per_module_grad_norms(model: torch.nn.Module,
                          *, prefixes: List[str] = None) -> Dict[str, float]:
    """Return total grad-norm + per-prefix grad-norm."""
    prefixes = prefixes or ["indexer", "compressed_pool", "super_pool",
                            "q_proj", "k_proj", "v_proj", "o_proj",
                            "branch_gate"]
    sums: Dict[str, float] = {f"grad_norm_{p}": 0.0 for p in prefixes}
    total_sq = 0.0
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        sq = float(p.grad.detach().pow(2).sum())
        total_sq += sq
        for pref in prefixes:
            if pref in name:
                sums[f"grad_norm_{pref}"] += sq
                break
    out = {"grad_norm_total": total_sq ** 0.5}
    for k, v in sums.items():
        out[k] = v ** 0.5
    return out
