"""Tests for the LMDB registry."""
from __future__ import annotations

import numpy as np
import time
from pathlib import Path

from localsparse.storage.registry import (
    Registry, WorkspaceMeta, ConsolidationCandidate, ProvenanceRecord,
)


def test_workspace_crud(tmp_path):
    r = Registry(tmp_path / "reg.lmdb")
    meta = WorkspaceMeta(
        name="physics", slab_path=str(tmp_path / "physics.slab"),
        created_at=time.time(), last_used_at=time.time(),
    )
    r.put_workspace(meta)
    got = r.get_workspace("physics")
    assert got is not None and got.name == "physics"
    lst = r.list_workspaces()
    assert len(lst) == 1
    r.delete_workspace("physics")
    assert r.get_workspace("physics") is None
    r.close()


def test_embeddings_roundtrip(tmp_path):
    r = Registry(tmp_path / "reg.lmdb")
    vec = np.random.randn(256).astype(np.float32)
    r.put_embedding("math", vec)
    got = r.get_embedding("math", 256)
    np.testing.assert_array_equal(got, vec)
    r.close()


def test_candidate_threshold(tmp_path):
    r = Registry(tmp_path / "reg.lmdb")
    # 1st hit
    c1 = r.bump_candidate("physics", "r#1", "ml", "q-hash-1", threshold=3)
    assert c1 is None
    # 2nd hit, different query
    c2 = r.bump_candidate("physics", "r#1", "ml", "q-hash-2", threshold=3)
    assert c2 is None
    # 3rd distinct query → cross threshold
    c3 = r.bump_candidate("physics", "r#1", "ml", "q-hash-3", threshold=3)
    assert c3 is not None
    assert c3.hits == 3
    # duplicate query does NOT bump
    c4 = r.bump_candidate("physics", "r#1", "ml", "q-hash-1", threshold=3)
    # already at threshold so still returned
    assert c4 is not None
    assert c4.hits == 3
    lst = r.list_candidates()
    assert any(c.src_wks == "physics" for c in lst)
    r.delete_candidate("physics", "r#1", "ml")
    assert r.list_candidates() == []
    r.close()


def test_provenance(tmp_path):
    r = Registry(tmp_path / "reg.lmdb")
    p = ProvenanceRecord(
        consolidation_id="cid-1",
        src_wks="physics", src_region="r#1",
        dst_wks="ml", dst_region="r#42",
        mode="research", created_at=time.time(),
        content_hash="abc123",
    )
    r.put_provenance(p)
    got = r.get_provenance("cid-1")
    assert got is not None and got.mode == "research"
    lst = r.list_provenance(dst_wks="ml")
    assert len(lst) == 1
    r.delete_provenance("cid-1")
    assert r.get_provenance("cid-1") is None
    r.close()


def test_global_counters(tmp_path):
    r = Registry(tmp_path / "reg.lmdb")
    assert r.get_global_int("total_disk_bytes") == 0
    r.set_global_int("total_disk_bytes", 1024)
    assert r.get_global_int("total_disk_bytes") == 1024
    r.close()
