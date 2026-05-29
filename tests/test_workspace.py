"""Workspace manager / eviction / consolidation tests."""
from __future__ import annotations

import os
import pytest
from pathlib import Path

from localsparse.config import LocalSparseConfig, ModelDims, AttentionConfig, WorkspaceConfig, Paths
from localsparse.workspace import (
    WorkspaceManager, DummyEncoder, ConsolidationOrchestrator, MockSearcher,
)


@pytest.fixture
def tiny_config(tmp_path):
    cfg = LocalSparseConfig(
        model=ModelDims(
            num_layers=4, num_kv_heads=2, head_dim=16, hidden_size=64,
            vocab_size=512, num_q_heads=4, intermediate_size=128,
        ),
        attention=AttentionConfig(
            sliding_window=64, compressed_block=8, super_block=64,
            selected_top_k=2, indexer_dim=8, selection_layer_stride=2,
        ),
        workspace=WorkspaceConfig(
            per_workspace_slot_cap=2048,
            working_context_tokens=512,
            consolidation_max_bytes=4096,
            research_calls_per_session=10,
        ),
        paths=Paths(root=tmp_path / "ls_home"),
    )
    return cfg


@pytest.fixture
def mgr(tiny_config):
    m = WorkspaceManager(config=tiny_config, encoder=DummyEncoder())
    yield m
    m.close()


def test_create_and_list(mgr):
    mgr.create("physics", source="newton laws of motion")
    mgr.create("math", source="ring theory category basics")
    names = sorted(w.name for w in mgr.list())
    assert names == ["math", "physics"]
    assert mgr.list()[0].slot_count > 0


def test_mount_unmount(mgr):
    mgr.create("physics", source="energy and matter")
    mid1 = mgr.mount("physics")
    mid2 = mgr.mount("physics")
    assert sorted(mgr.mounted_workspaces()) == ["physics"]
    mgr.unmount(mid1)
    assert "physics" in mgr.mounted_workspaces()
    mgr.unmount(mid2)
    assert mgr.mounted_workspaces() == []


def test_delete(mgr):
    mgr.create("physics", source="x")
    path = Path(mgr.registry.get_workspace("physics").slab_path)
    assert path.exists()
    mgr.delete("physics")
    assert not path.exists()
    assert mgr.registry.get_workspace("physics") is None


def test_fork(mgr):
    mgr.create("physics", source="dark matter universe expansion")
    mgr.fork("physics", "cosmology")
    assert mgr.registry.get_workspace("cosmology").slot_count > 0


def test_eviction_triggers(mgr, tiny_config):
    """Append enough times that we cross the 95% threshold, and verify
    eviction is actually invoked (file shrinks at least once)."""
    from pathlib import Path
    mgr.create("big", source="seed")
    path = Path(mgr.registry.get_workspace("big").slab_path)
    sizes = []
    for i in range(40):
        mgr.append("big", f"chunk {i} " + " ".join(["word"] * 100))
        sizes.append(path.stat().st_size)
    # Eviction must have caused at least one shrink event.
    shrinks = sum(1 for a, b in zip(sizes, sizes[1:]) if b < a)
    assert shrinks >= 1, f"no eviction shrink detected; sizes={sizes[-10:]}"


def test_eviction_function_direct(mgr):
    """Call eviction directly and verify fine bit gets cleared."""
    mgr.create("big", source="seed " * 200)  # populate with multiple super blocks
    meta = mgr.registry.get_workspace("big")
    # Run eviction directly
    mgr._evict("big", meta)
    new_meta = mgr.registry.get_workspace("big")
    assert new_meta.tier_flags & 0b100 == 0, "fine bit must be cleared post-evict"
    assert new_meta.tier_flags & 0b011 == 0b011, "super and compressed bits must remain"


def test_pin_unpin(mgr):
    mgr.create("physics", source="x")
    mgr.pin("physics", weight=2.0)
    meta = mgr.registry.get_workspace("physics")
    assert meta.pinned and meta.pin_weight == 2.0
    mgr.unpin("physics")
    assert not mgr.registry.get_workspace("physics").pinned


def test_cross_access_threshold_creates_candidate(mgr):
    mgr.create("physics", source="x")
    mgr.create("ml", source="y")
    c1 = mgr.log_cross_access("physics", "r#1", "ml", "q1")
    c2 = mgr.log_cross_access("physics", "r#1", "ml", "q2")
    c3 = mgr.log_cross_access("physics", "r#1", "ml", "q3")
    assert c3 is not None
    cands = mgr.registry.list_candidates()
    assert any(c.src_wks == "physics" and c.dst_wks == "ml" for c in cands)


def test_consolidation_research(mgr):
    mgr.create("physics", source="x")
    mgr.create("ml", source="y")
    corpus = {
        "https://example.com/manifold": "manifold learning local linear embedding lle",
        "https://example.com/tensors": "tensor product symmetric basis riemannian",
    }
    orch = ConsolidationOrchestrator(mgr, searcher=MockSearcher(corpus))
    res = orch.consolidate(src="physics", region="r#1", dst="ml", mode="research")
    assert res.appended_tokens > 0
    assert res.mode == "research"
    # Provenance recorded
    provs = mgr.registry.list_provenance(dst_wks="ml")
    assert len(provs) == 1
    assert provs[0].mode == "research"
    # Candidate removed
    assert not mgr.registry.list_candidates()


def test_consolidation_rewrite_offline(mgr):
    mgr.create("physics", source="x")
    mgr.create("ml", source="y")
    orch = ConsolidationOrchestrator(mgr, searcher=MockSearcher({}))
    res = orch.consolidate(src="physics", region="r#1", dst="ml", mode="rewrite")
    assert res.appended_tokens > 0
    assert res.mode == "rewrite"


def test_consolidation_session_budget(mgr):
    mgr.create("a", source="x")
    mgr.create("b", source="y")
    orch = ConsolidationOrchestrator(mgr, searcher=MockSearcher({}))
    orch.cfg.research_calls_per_session = 2  # type: ignore
    orch.consolidate(src="a", region="r#1", dst="b", mode="rewrite")
    orch.consolidate(src="a", region="r#2", dst="b", mode="rewrite")
    with pytest.raises(RuntimeError, match="Research budget"):
        orch.consolidate(src="a", region="r#3", dst="b", mode="rewrite")


def test_deconsolidate(mgr):
    mgr.create("a", source="x")
    mgr.create("b", source="y")
    orch = ConsolidationOrchestrator(mgr, searcher=MockSearcher({}))
    res = orch.consolidate(src="a", region="r#1", dst="b", mode="rewrite")
    orch.deconsolidate(res.consolidation_id)
    assert mgr.registry.get_provenance(res.consolidation_id) is None


def test_promote_region_reuses_research(mgr):
    mgr.create("a", source="x")
    corpus = {"https://example.com/a": "alpha beta gamma topic content"}
    orch = ConsolidationOrchestrator(mgr, searcher=MockSearcher(corpus))
    res = orch.promote_region(wks="a", region="r#1")
    assert res.appended_tokens > 0
    assert res.dst_wks == "a"
