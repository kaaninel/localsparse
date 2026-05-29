"""Tests for the run logger + failure detectors."""
from __future__ import annotations

import json
import math
import torch
import torch.nn as nn
import pytest

from localsparse.logging import (
    RunLogger, GateLogger, RunDirectory, FailureDetector,
    dump_debug_state, per_module_grad_norms,
)


def test_run_directory_creates_files(tmp_path):
    rd = RunDirectory.fresh(tmp_path)
    assert rd.root.exists()
    assert rd.train_jsonl.parent.exists()


def test_run_logger_appends_records(tmp_path):
    rd = RunDirectory(root=tmp_path / "r1")
    logger = RunLogger(rd, print_every=0)
    for s in range(3):
        logger.step(step=s, lm_loss=2.0 - 0.1 * s, sliding_mass=0.3,
                    selected_mass=0.3, compressed_mass=0.4)
    lines = rd.train_jsonl.read_text().strip().splitlines()
    assert len(lines) == 3
    recs = [json.loads(l) for l in lines]
    assert recs[0]["step"] == 0
    assert recs[-1]["lm_loss"] == pytest.approx(1.8)


def test_gate_logger_writes_status(tmp_path):
    rd = RunDirectory(root=tmp_path / "r2")
    g = GateLogger(rd)
    g.record("G1", metric="nan_count", value=0.0, threshold=0.0, status="pass")
    g.record("G2", metric="min_branch_mass", value=0.02, threshold=0.05, status="fail")
    lines = rd.gates_jsonl.read_text().strip().splitlines()
    assert len(lines) == 2
    recs = [json.loads(l) for l in lines]
    assert recs[0]["status"] == "pass"
    assert recs[1]["status"] == "fail"


def test_failure_detector_catches_nan():
    fired = []
    d = FailureDetector(on_fire=lambda r, _: fired.append(r))
    out = d.check({"lm_loss": float("nan"), "sliding_mass": 0.3,
                   "selected_mass": 0.3, "compressed_mass": 0.4})
    assert "nan" in out
    assert "nan" in fired


def test_failure_detector_catches_branch_collapse():
    fired = []
    d = FailureDetector(on_fire=lambda r, _: fired.append(r),
                        branch_collapse_window=3, branch_collapse_threshold=0.01)
    rec = {"sliding_mass": 0.001, "selected_mass": 0.5, "compressed_mass": 0.5}
    out = []
    for _ in range(3):
        out.append(d.check(rec))
    assert "branch_collapse" in out[-1]
    assert "branch_collapse" in fired


def test_failure_detector_recovers_branch_collapse():
    d = FailureDetector(branch_collapse_window=3, branch_collapse_threshold=0.01)
    d.check({"sliding_mass": 0.001, "selected_mass": 0.5, "compressed_mass": 0.5})
    d.check({"sliding_mass": 0.001, "selected_mass": 0.5, "compressed_mass": 0.5})
    # Recovery: high mass again
    out = d.check({"sliding_mass": 0.4, "selected_mass": 0.3, "compressed_mass": 0.3})
    assert "branch_collapse" not in out
    assert d.s.consec_branch_collapse == 0


def test_failure_detector_explosion():
    fired = []
    d = FailureDetector(on_fire=lambda r, _: fired.append(r),
                        grad_explosion_multiplier=10.0)
    # Establish baseline
    for _ in range(25):
        d.check({"grad_norm_total": 1.0})
    out = d.check({"grad_norm_total": 100.0})
    assert "explosion" in out


def test_failure_detector_stuck():
    d = FailureDetector(stuck_window=20, stuck_rel_threshold=0.01)
    fired_any = False
    for _ in range(20):
        out = d.check({"lm_loss": 2.0})  # exactly flat
        if "stuck" in out:
            fired_any = True
    assert fired_any


def test_dump_debug_state(tmp_path):
    rd = RunDirectory(root=tmp_path / "r3")
    model = nn.Linear(4, 4)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batch = {"input_ids": torch.zeros(2, 4, dtype=torch.long)}
    d = dump_debug_state(rd, "test_reason", model=model, optimizer=opt,
                         batch=batch, extra={"why": "manual"})
    assert (d / "model.pt").exists()
    assert (d / "optimizer.pt").exists()
    assert (d / "batch.pt").exists()
    assert (d / "extra.json").exists()


def test_per_module_grad_norms():
    model = nn.Sequential()
    model.add_module("indexer_q_proj", nn.Linear(4, 4))
    model.add_module("o_proj", nn.Linear(4, 4))
    x = torch.randn(2, 4)
    y = model(x).sum()
    y.backward()
    norms = per_module_grad_norms(model, prefixes=["indexer", "o_proj"])
    assert norms["grad_norm_total"] > 0
    assert norms["grad_norm_indexer"] > 0
    assert norms["grad_norm_o_proj"] > 0
