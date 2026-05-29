"""LocalSparse chat template.

Extends MiniCPM5's native template (append-only) with a
`<workspace_context>` block injected at the start of every system message.
The base model's `<think>` and `<tool_call>` syntax are preserved
unchanged so we don't invalidate any pretrained tool-use behavior.

Layout of a full prompt:

    <|im_start|>system
    <workspace_context>
      <mounted name="physics" tier="L0"/>
      <pinned name="math" weight="1.5"/>
      <consolidation_candidate src="physics" region="r#1024" dst="ml" hits="3"/>
    </workspace_context>
    {tool descriptions JSON}
    {original system prompt content}
    <|im_end|>
    <|im_start|>user
    {user_message}<|im_end|>
    <|im_start|>assistant
    {assistant_message, optionally with <think>…</think> and <tool_call>…</tool_call> blocks}<|im_end|>

The exact special tokens (<|im_start|>, <|im_end|>) come from
MiniCPM5's tokenizer; we string-format them here for readability and the
real tokenizer's BPE handles encoding.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


IM_START = "<|im_start|>"
IM_END = "<|im_end|>"


@dataclass
class MountInfo:
    name: str
    tier: str = "L0"          # L0=fine, L1=compressed, L2=super-only
    slot_count: int = 0


@dataclass
class PinInfo:
    name: str
    weight: float = 1.0


@dataclass
class CandidateInfo:
    src: str
    region: str
    dst: str
    hits: int
    age_days: int = 0


@dataclass
class ChatMessage:
    role: str           # "system" | "user" | "assistant" | "tool"
    content: str


@dataclass
class WorkspaceContextBlock:
    mounted: List[MountInfo] = field(default_factory=list)
    pinned: List[PinInfo] = field(default_factory=list)
    candidates: List[CandidateInfo] = field(default_factory=list)

    def render(self) -> str:
        lines = ["<workspace_context>"]
        for m in self.mounted:
            lines.append(f'  <mounted name="{m.name}" tier="{m.tier}" slots="{m.slot_count}"/>')
        for p in self.pinned:
            lines.append(f'  <pinned name="{p.name}" weight="{p.weight}"/>')
        for c in self.candidates:
            lines.append(
                f'  <consolidation_candidate src="{c.src}" region="{c.region}" '
                f'dst="{c.dst}" hits="{c.hits}" age_days="{c.age_days}"/>'
            )
        lines.append("</workspace_context>")
        return "\n".join(lines)


def render_tool_descriptions(tools: List[Dict[str, Any]]) -> str:
    if not tools:
        return ""
    return "<tools>\n" + json.dumps(tools, indent=2) + "\n</tools>"


def render_prompt(
    *,
    system_prompt: str,
    messages: List[ChatMessage],
    workspace_context: Optional[WorkspaceContextBlock] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    add_generation_prefix: bool = True,
) -> str:
    """Assemble the full prompt string ready to feed to the tokenizer."""
    parts: list[str] = []

    sys_payload = []
    if workspace_context is not None:
        sys_payload.append(workspace_context.render())
    if tools:
        sys_payload.append(render_tool_descriptions(tools))
    if system_prompt:
        sys_payload.append(system_prompt)
    if sys_payload:
        parts.append(f"{IM_START}system\n" + "\n".join(sys_payload) + f"{IM_END}")

    for msg in messages:
        parts.append(f"{IM_START}{msg.role}\n{msg.content}{IM_END}")

    if add_generation_prefix:
        parts.append(f"{IM_START}assistant\n")

    return "\n".join(parts)


def split_assistant_response(text: str) -> tuple[Optional[str], str]:
    """Separate <think>…</think> chain-of-thought from the user-facing reply."""
    import re
    m = re.match(r"\s*<think>(.*?)</think>\s*(.*)", text, re.DOTALL)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return None, text.strip()
