"""LocalSparseAgent: glues model + tools + workspaces + chat template.

For v1 we expose two run modes:

  - `chat(message)`: interactive single-shot generation. The model emits
    tool calls; the agent dispatches them, appends the responses, and
    re-prompts until generation finishes.

  - `run_loop(initial)`: long-form multi-turn workflow driver. Surfaces
    pending consolidation candidates each turn via the
    `<workspace_context>` block.

The actual generation is delegated to a `GenerateBackend` interface so
we can plug in (1) a HuggingFace transformers pipeline once the model
weights are loaded and (2) a `MockBackend` for tests that returns
scripted tool-call sequences.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Protocol, Dict, Any
import json

from ..config import LocalSparseConfig, default_config
from ..workspace import WorkspaceManager, ConsolidationOrchestrator
from ..workspace.consolidation import Searcher, MockSearcher
from ..tools import (
    ToolRegistry, parse_tool_calls, format_tool_response,
    register_workspace_tools, register_web_tools,
)
from ..chat import (
    ChatMessage, WorkspaceContextBlock, MountInfo, PinInfo, CandidateInfo,
    render_prompt, split_assistant_response,
)


# ---------------------------------------------------------------------------
# Generation backend interface
# ---------------------------------------------------------------------------
class GenerateBackend(Protocol):
    def generate(self, prompt: str, *, max_new_tokens: int = 512,
                 stop: Optional[List[str]] = None) -> str: ...


class MockBackend:
    """Replays a scripted sequence of assistant responses (for tests)."""

    def __init__(self, responses: List[str]):
        self._responses = list(responses)
        self.calls: list[str] = []

    def generate(self, prompt: str, *, max_new_tokens: int = 512,
                 stop: Optional[List[str]] = None) -> str:
        self.calls.append(prompt)
        if not self._responses:
            return ""
        return self._responses.pop(0)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------
@dataclass
class AgentState:
    messages: List[ChatMessage] = field(default_factory=list)
    system_prompt: str = (
        "You are LocalSparse, a small dense agent that uses workspaces to "
        "extend its knowledge. Prefer mounting an existing workspace over "
        "rereading the web. Use <think>…</think> for reasoning. Emit tool "
        "calls as <tool_call>JSON</tool_call>. Be concise."
    )


class LocalSparseAgent:
    def __init__(
        self,
        config: Optional[LocalSparseConfig] = None,
        backend: Optional[GenerateBackend] = None,
        searcher: Optional[Searcher] = None,
    ):
        self.config = config or default_config()
        self.manager = WorkspaceManager(config=self.config)
        self.searcher = searcher or MockSearcher({})
        self.orchestrator = ConsolidationOrchestrator(
            self.manager, searcher=self.searcher)
        self.tools = ToolRegistry()
        register_workspace_tools(self.tools, self.manager, self.orchestrator)
        register_web_tools(self.tools, self.searcher)
        self.backend = backend                         # may be None for offline tests
        self.state = AgentState()
        self._max_tool_iterations = 8

    def close(self) -> None:
        self.manager.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ---- workspace_context snapshot --------------------------------------
    def _build_workspace_context(self) -> WorkspaceContextBlock:
        all_wks = self.manager.list()
        mounted_names = set(self.manager.mounted_workspaces())
        mounted = [
            MountInfo(name=m.name, slot_count=m.slot_count,
                      tier=("L2" if m.tier_flags & 0b001 and not (m.tier_flags & 0b100)
                            else "L1" if not (m.tier_flags & 0b100)
                            else "L0"))
            for m in all_wks if m.name in mounted_names
        ]
        pinned = [PinInfo(name=m.name, weight=m.pin_weight)
                  for m in all_wks if m.pinned]
        cands = [
            CandidateInfo(src=c.src_wks, region=c.src_region, dst=c.dst_wks,
                          hits=c.hits)
            for c in self.orchestrator.pending_candidates()
        ]
        return WorkspaceContextBlock(mounted=mounted, pinned=pinned, candidates=cands)

    # ---- single chat turn -------------------------------------------------
    def chat(self, user_message: str) -> str:
        if self.backend is None:
            raise RuntimeError("No generation backend attached.")
        self.state.messages.append(ChatMessage(role="user", content=user_message))

        for _ in range(self._max_tool_iterations):
            prompt = render_prompt(
                system_prompt=self.state.system_prompt,
                messages=self.state.messages,
                workspace_context=self._build_workspace_context(),
                tools=self.tools.describe(),
            )
            resp = self.backend.generate(prompt, stop=["<|im_end|>"])
            calls, remainder = parse_tool_calls(resp)
            if not calls:
                self.state.messages.append(ChatMessage(role="assistant", content=resp))
                return resp
            # Dispatch all tool calls and feed results back as tool messages.
            self.state.messages.append(ChatMessage(role="assistant", content=resp))
            tool_payload_parts = []
            for c in calls:
                tool_payload_parts.append(self.tools.dispatch(c))
            self.state.messages.append(ChatMessage(
                role="tool", content="\n".join(tool_payload_parts)))
        # Reached iteration cap
        return "[tool-iteration cap reached]"

    # ---- programmatic API (no generation, direct tool calls) ------------
    def run_tool(self, tool_name: str, /, **kwargs) -> Any:
        """Direct in-process tool invocation. Used by CLI and tests."""
        tool = self.tools.get(tool_name)
        if tool is None:
            raise KeyError(f"unknown tool {tool_name!r}")
        return tool(**kwargs)
