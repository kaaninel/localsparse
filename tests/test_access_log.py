"""Tests for AccessLog."""
from __future__ import annotations

from localsparse.storage.access_log import AccessLog


def test_record_and_score(tmp_path):
    log = AccessLog(tmp_path / "al.lmdb")
    log.record_hit("physics", "r#1")
    log.record_hit("physics", "r#1")
    log.record_hit("physics", "r#1")
    s = log.access_score("physics", "r#1")
    # 3 hits, recency≈1 → score ≈ 3
    assert s > 2.5
    log.close()


def test_cross_hits_boost_score(tmp_path):
    log = AccessLog(tmp_path / "al.lmdb")
    log.record_hit("physics", "r#1")
    log.record_cross_hit("physics", "r#1", "ml")
    log.record_cross_hit("physics", "r#1", "ml")
    s_with_cross = log.access_score("physics", "r#1", cross_weight=2.0)
    # base ~1 + cross 2*2 = 5
    assert s_with_cross > 4
    log.close()


def test_decrement_cross(tmp_path):
    log = AccessLog(tmp_path / "al.lmdb")
    log.record_cross_hit("physics", "r#1", "ml")
    log.record_cross_hit("physics", "r#1", "ml")
    log.decrement_cross("physics", "r#1", "ml", delta=1)
    s = log.access_score("physics", "r#1", cross_weight=1.0)
    # 1 remaining cross hit
    assert s >= 1.0 and s < 2.0
    log.close()
