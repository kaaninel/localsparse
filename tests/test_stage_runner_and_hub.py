"""Smoke tests for stage_runner and HubCheckpointer (offline / mocked)."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from localsparse.training.stage_runner import (
    gate_threshold,
    gate_all,
    gate_always_pass,
    run_stage,
)


def test_gate_threshold_passes_when_value_meets():
    g = gate_threshold("acc", ">=", 0.5)
    ok, msg = g({"acc": 0.6})
    assert ok
    assert "acc" in msg


def test_gate_threshold_fails_when_value_below():
    g = gate_threshold("acc", ">=", 0.5)
    ok, msg = g({"acc": 0.4})
    assert not ok


def test_gate_threshold_missing_key_fails():
    g = gate_threshold("acc", ">=", 0.5)
    ok, _ = g({"other": 1.0})
    assert not ok


def test_gate_all_combines_pass():
    g = gate_all(gate_threshold("a", ">=", 0), gate_threshold("b", "<=", 10))
    ok, _ = g({"a": 1, "b": 5})
    assert ok


def test_gate_all_combines_fail():
    g = gate_all(gate_threshold("a", ">=", 0), gate_threshold("b", "<=", 10))
    ok, _ = g({"a": 1, "b": 11})
    assert not ok


def test_gate_always_pass():
    g = gate_always_pass("yo")
    ok, msg = g({})
    assert ok and msg == "yo"


def test_run_stage_no_checkpointer_pass():
    def fn():
        return {"acc": 0.9}
    res = run_stage(
        "test-stage", fn, gate_threshold("acc", ">=", 0.5),
        checkpointer=None, model=None, tokenizer=None, verbose=False,
    )
    assert res.status == "pass"
    assert res.metrics["acc"] == 0.9


def test_run_stage_no_checkpointer_fail():
    def fn():
        return {"acc": 0.1}
    res = run_stage(
        "test-stage", fn, gate_threshold("acc", ">=", 0.5),
        checkpointer=None, model=None, tokenizer=None, verbose=False,
    )
    assert res.status == "fail"


def test_run_stage_exception_returns_fail():
    def fn():
        raise RuntimeError("boom")
    res = run_stage(
        "explode", fn, gate_always_pass(),
        checkpointer=None, verbose=False,
    )
    assert res.status == "fail"
    assert "boom" in res.gate_message or "error" in res.gate_message.lower()


def test_hub_checkpointer_init_does_not_call_hf(tmp_path):
    from localsparse.hub.checkpointing import HubCheckpointer
    fake_api = MagicMock()
    fake_api.repo_exists.return_value = False
    cp = HubCheckpointer(
        repo_id="fake/repo", local_root=tmp_path,
        token="fake_token", private=False, api=fake_api,
    )
    assert cp.repo_id == "fake/repo"
    fake_api.create_repo.assert_called_once()


def test_install_shutdown_hooks_registers_atexit():
    from localsparse.hub.checkpointing import install_shutdown_hooks
    calls = []

    def cb():
        calls.append(1)

    # Just confirm the call doesn't error; we don't trigger atexit here.
    install_shutdown_hooks(cb)
    # If it returned without raising, success.
