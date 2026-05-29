"""End-to-end agent loop test with MockBackend."""
from __future__ import annotations

import json
import pytest
from localsparse.config import LocalSparseConfig, ModelDims, AttentionConfig, WorkspaceConfig, Paths
from localsparse.agent import LocalSparseAgent, MockBackend
from localsparse.workspace.consolidation import MockSearcher


@pytest.fixture
def cfg(tmp_path):
    return LocalSparseConfig(
        model=ModelDims(num_layers=2, num_kv_heads=2, head_dim=8, hidden_size=16,
                        vocab_size=64, num_q_heads=2, intermediate_size=32),
        attention=AttentionConfig(compressed_block=4, super_block=16, indexer_dim=4,
                                  sliding_window=16, selected_top_k=2,
                                  selection_layer_stride=1),
        workspace=WorkspaceConfig(per_workspace_slot_cap=4096,
                                  consolidation_max_bytes=2048,
                                  research_calls_per_session=10),
        paths=Paths(root=tmp_path),
    )


def test_agent_no_tool_response(cfg):
    backend = MockBackend(["Hello there, I have no tools to call."])
    with LocalSparseAgent(config=cfg, backend=backend) as a:
        resp = a.chat("hi")
        assert "Hello there" in resp


def test_agent_with_tool_call(cfg):
    """Agent should dispatch the tool and then receive a follow-up to finalize."""
    scripted = [
        # First turn: assistant emits a tool call
        '<tool_call>{"name":"workspace.create","arguments":{"name":"physics","source":"newton"}}</tool_call>',
        # Second turn: assistant finalizes
        "Created physics workspace.",
    ]
    backend = MockBackend(scripted)
    searcher = MockSearcher({})
    with LocalSparseAgent(config=cfg, backend=backend, searcher=searcher) as a:
        resp = a.chat("create a physics workspace")
        assert "Created physics" in resp
        # Verify the workspace actually exists
        meta = a.manager.registry.get_workspace("physics")
        assert meta is not None
        # Verify the backend saw two prompts (one before, one after tool)
        assert len(backend.calls) == 2


def test_agent_run_tool_direct(cfg):
    with LocalSparseAgent(config=cfg, backend=None) as a:
        out = a.run_tool("workspace.create", name="math", source="ring theory")
        assert out["name"] == "math"
        out = a.run_tool("workspace.list")
        names = [w["name"] for w in out["workspaces"]]
        assert "math" in names


def test_agent_workspace_context_includes_pending_candidates(cfg):
    with LocalSparseAgent(config=cfg, backend=None) as a:
        a.run_tool("workspace.create", name="physics", source="x")
        a.run_tool("workspace.create", name="ml", source="y")
        # Force a candidate
        a.manager.log_cross_access("physics", "r#1", "ml", "q1")
        a.manager.log_cross_access("physics", "r#1", "ml", "q2")
        a.manager.log_cross_access("physics", "r#1", "ml", "q3")
        wc = a._build_workspace_context()
        # one candidate should be present
        assert any(c.src == "physics" and c.dst == "ml" for c in wc.candidates)
