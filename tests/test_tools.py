"""Tests for tool parsing, registry, workspace tools, web tools."""
from __future__ import annotations

import pytest
from localsparse.tools import (
    parse_tool_calls, format_tool_response, ToolRegistry,
    register_workspace_tools, register_web_tools, HtmlCleaner,
)
from localsparse.tools.parser import ToolCall


def test_parse_single_json_call():
    text = 'Sure! <tool_call>{"name":"workspace.mount","arguments":{"name":"physics"}}</tool_call>'
    calls, rem = parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].name == "workspace.mount"
    assert calls[0].arguments == {"name": "physics"}


def test_parse_multiple_calls():
    text = (
        '<tool_call>{"name":"workspace.list","arguments":{}}</tool_call>'
        ' some text '
        '<tool_call>{"name":"web.search","arguments":{"query":"foo","k":3}}</tool_call>'
    )
    calls, _ = parse_tool_calls(text)
    assert [c.name for c in calls] == ["workspace.list", "web.search"]
    assert calls[1].arguments == {"query": "foo", "k": 3}


def test_parse_fallback_kvp():
    text = "<tool_call>workspace.delete(name=physics)</tool_call>"
    calls, _ = parse_tool_calls(text)
    assert calls[0].name == "workspace.delete"
    assert calls[0].arguments == {"name": "physics"}


def test_parse_malformed_ignored():
    text = "<tool_call>not even close</tool_call>"
    calls, _ = parse_tool_calls(text)
    assert calls == []


def test_registry_dispatch_unknown():
    reg = ToolRegistry()
    call = ToolCall(name="missing.tool", arguments={}, raw="")
    resp = reg.dispatch(call)
    assert "unknown tool" in resp


def test_registry_dispatch_success():
    reg = ToolRegistry()
    reg.register("add", lambda a, b: {"sum": a + b}, description="add two ints")
    call = ToolCall(name="add", arguments={"a": 1, "b": 2}, raw="")
    resp = reg.dispatch(call)
    assert "\"sum\": 3" in resp


def test_registry_dispatch_bad_args():
    reg = ToolRegistry()
    reg.register("add", lambda a, b: a + b)
    call = ToolCall(name="add", arguments={"a": 1}, raw="")
    resp = reg.dispatch(call)
    assert "bad arguments" in resp


def test_html_cleaner():
    raw = "<html><head><style>x{color:red}</style></head><body><p>Hello&nbsp;<b>world</b></p></body></html>"
    out = HtmlCleaner().clean(raw)
    assert out == "Hello world"


def test_workspace_tools_integration(tmp_path):
    from localsparse.config import LocalSparseConfig, ModelDims, AttentionConfig, WorkspaceConfig, Paths
    from localsparse.workspace import WorkspaceManager, ConsolidationOrchestrator, DummyEncoder, MockSearcher

    cfg = LocalSparseConfig(
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
    with WorkspaceManager(config=cfg, encoder=DummyEncoder()) as mgr:
        orch = ConsolidationOrchestrator(mgr, searcher=MockSearcher({"https://e/a": "alpha beta gamma"}))
        reg = ToolRegistry()
        register_workspace_tools(reg, mgr, orch)
        # create
        out = reg.dispatch(ToolCall(name="workspace.create",
                                    arguments={"name": "physics", "source": "newton energy"}, raw=""))
        assert "\"name\": \"physics\"" in out
        # list
        out = reg.dispatch(ToolCall(name="workspace.list", arguments={}, raw=""))
        assert "physics" in out
        # mount
        out = reg.dispatch(ToolCall(name="workspace.mount", arguments={"name": "physics"}, raw=""))
        assert "mount_id" in out
        # consolidate
        reg.dispatch(ToolCall(name="workspace.create",
                              arguments={"name": "ml", "source": "x"}, raw=""))
        out = reg.dispatch(ToolCall(name="workspace.consolidate",
                                    arguments={"src": "physics", "region": "r#1",
                                               "dst": "ml", "mode": "research"}, raw=""))
        assert "consolidation_id" in out
