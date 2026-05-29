"""Stage runner with explicit gates and push-on-pass semantics.

A *stage* is a unit of work in the Gemma 4 E2B training notebook that:
  1. Runs a training/eval function
  2. Computes a gate verdict (pass/fail) from the returned metrics
  3. If pass: pushes the current state to HF via a HubCheckpointer
  4. If fail: pushes a `-failed` snapshot for debugging and stops the chain

Stages are sequential; a failed stage prevents later stages from running so
no resources are wasted chasing a path that already broke.
"""
from __future__ import annotations

import time
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from ..hub.checkpointing import HubCheckpointer, StageRecord


GateFn = Callable[[Dict[str, Any]], Tuple[bool, str]]
StageFn = Callable[[], Dict[str, Any]]


@dataclass
class StageResult:
    stage_id: str
    status: str
    metrics: Dict[str, Any]
    gate_message: str
    wall_seconds: float
    record: StageRecord


def run_stage(
    stage_id: str,
    fn: StageFn,
    gate: GateFn,
    checkpointer: Optional[HubCheckpointer] = None,
    model: Optional[torch.nn.Module] = None,
    tokenizer: Optional[Any] = None,
    extra_files: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> StageResult:
    """Execute one stage, apply the gate, and push the result.

    Returns a StageResult. On failure or exception, also pushes a
    `-failed` snapshot if checkpointer is provided.
    """
    if verbose:
        _banner(f"STAGE {stage_id} — start")

    started = time.time()
    metrics: Dict[str, Any] = {}
    status = "fail"
    gate_msg = ""
    try:
        metrics = fn() or {}
        passed, gate_msg = gate(metrics)
        status = "pass" if passed else "fail"
    except Exception as e:
        gate_msg = f"exception: {type(e).__name__}: {e}"
        metrics = {"error": str(e)[:500], "traceback": traceback.format_exc()[:2000]}
        status = "fail"
        if verbose:
            traceback.print_exc()

    finished = time.time()
    wall = finished - started

    record = StageRecord(
        stage_id=stage_id,
        status=status,
        metrics=metrics,
        gate_message=gate_msg,
        started_at=started,
        finished_at=finished,
        wall_seconds=wall,
    )

    if verbose:
        _print_metrics(metrics)
        verdict = "✅ PASS" if status == "pass" else "❌ FAIL"
        print(f"\n{verdict} — {stage_id} ({wall:.1f}s)\n  gate: {gate_msg}")

    # Push to hub
    if checkpointer is not None:
        push_id = stage_id if status == "pass" else f"{stage_id}-failed"
        try:
            checkpointer.push_stage(
                stage_id=push_id,
                model=model, tokenizer=tokenizer,
                record=record, extra_files=extra_files,
            )
            if verbose:
                print(f"  ☁ pushed to hf: {checkpointer.repo_id}/{push_id}")
        except Exception as e:
            print(f"  ⚠ push failed: {e}")

    if verbose:
        _banner(f"STAGE {stage_id} — end ({status})")

    return StageResult(
        stage_id=stage_id, status=status, metrics=metrics,
        gate_message=gate_msg, wall_seconds=wall, record=record,
    )


def _banner(text: str) -> None:
    line = "=" * max(40, len(text) + 8)
    print(f"\n{line}\n=== {text} ===\n{line}")


def _print_metrics(metrics: Dict[str, Any]) -> None:
    if not metrics:
        return
    print("  metrics:")
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            print(f"    {k:40s} = {v}")
        elif isinstance(v, dict):
            print(f"    {k}:")
            for kk, vv in v.items():
                print(f"      {kk:38s} = {vv}")
        else:
            s = str(v)
            if len(s) > 200:
                s = s[:200] + "..."
            print(f"    {k:40s} = {s}")


# ---------------------------------------------------------------------------
# Common gate factories
# ---------------------------------------------------------------------------
def gate_threshold(key: str, op: str, value: float) -> GateFn:
    """Build a gate that compares `metrics[key]` to `value`."""
    ops = {">=": (lambda a, b: a >= b), ">": (lambda a, b: a > b),
           "<=": (lambda a, b: a <= b), "<": (lambda a, b: a < b),
           "==": (lambda a, b: a == b)}
    cmp = ops[op]

    def _g(metrics: Dict[str, Any]) -> Tuple[bool, str]:
        got = metrics.get(key)
        if got is None:
            return False, f"missing metric {key!r}"
        ok = cmp(got, value)
        return ok, f"{key}={got:.4f} {op} {value:.4f} -> {'pass' if ok else 'fail'}"

    return _g


def gate_all(*gates: GateFn) -> GateFn:
    def _g(metrics: Dict[str, Any]) -> Tuple[bool, str]:
        msgs = []
        ok = True
        for g in gates:
            o, m = g(metrics)
            ok = ok and o
            msgs.append(m)
        return ok, " ; ".join(msgs)
    return _g


def gate_always_pass(msg: str = "info-only") -> GateFn:
    def _g(metrics: Dict[str, Any]) -> Tuple[bool, str]:
        return True, msg
    return _g
