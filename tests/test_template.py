"""Tests for chat template."""
from __future__ import annotations

from localsparse.chat import (
    ChatMessage, WorkspaceContextBlock, MountInfo, PinInfo, CandidateInfo,
    render_prompt, split_assistant_response,
)


def test_render_prompt_minimal():
    out = render_prompt(
        system_prompt="be helpful",
        messages=[ChatMessage(role="user", content="hi")],
    )
    assert "<|im_start|>system" in out
    assert "be helpful" in out
    assert "<|im_start|>user\nhi<|im_end|>" in out
    assert out.endswith("<|im_start|>assistant\n")


def test_render_with_workspace_context():
    wc = WorkspaceContextBlock(
        mounted=[MountInfo(name="physics", tier="L0", slot_count=12345)],
        pinned=[PinInfo(name="math", weight=2.0)],
        candidates=[CandidateInfo(src="physics", region="r#1024", dst="ml", hits=3)],
    )
    out = render_prompt(
        system_prompt="sys",
        messages=[ChatMessage(role="user", content="q")],
        workspace_context=wc,
    )
    assert "<workspace_context>" in out
    assert 'name="physics"' in out
    assert 'weight="2.0"' in out
    assert 'hits="3"' in out


def test_render_with_tools():
    tools = [{"name": "workspace.list", "description": "list", "parameters": []}]
    out = render_prompt(system_prompt="s", messages=[], tools=tools)
    assert "<tools>" in out
    assert "workspace.list" in out


def test_split_think_block():
    text = "<think>internal monologue</think>\n\nVisible answer."
    think, visible = split_assistant_response(text)
    assert think == "internal monologue"
    assert visible == "Visible answer."


def test_split_no_think():
    text = "Just an answer."
    think, visible = split_assistant_response(text)
    assert think is None
    assert visible == "Just an answer."
