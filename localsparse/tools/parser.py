"""MiniCPM5-native XML tool-call parser.

Ports the lightweight `minicpm5` parser semantics from SGLang.  Format
(produced by the base model and recognized in inbound assistant text):

    <tool_call>
    {"name": "workspace.mount", "arguments": {"name": "physics"}}
    </tool_call>

We also accept the slightly looser form used by some MiniCPM training
data where the body is bare key=value pairs.  The JSON form is canonical.

The parser is *stream-aware*: it can be fed partial assistant text and
will return the list of completed tool calls plus any trailing buffer
that doesn't yet form a complete call.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional, List, Tuple


TOOL_OPEN = "<tool_call>"
TOOL_CLOSE = "</tool_call>"
_PATTERN = re.compile(
    re.escape(TOOL_OPEN) + r"(.*?)" + re.escape(TOOL_CLOSE),
    re.DOTALL,
)


@dataclass
class ToolCall:
    name: str
    arguments: dict
    raw: str            # the original block (without tags)
    call_id: Optional[str] = None


def parse_tool_calls(text: str) -> tuple[List[ToolCall], str]:
    """Extract all complete <tool_call>…</tool_call> blocks from `text`.

    Returns: (calls, remainder)
      calls       — list of parsed ToolCalls (skips invalid bodies silently).
      remainder   — `text` with the matched blocks removed and any
                    incomplete trailing `<tool_call>` prefix kept for
                    streaming continuation.
    """
    calls: List[ToolCall] = []
    matches = list(_PATTERN.finditer(text))
    last_end = 0
    parts: list[str] = []
    for m in matches:
        parts.append(text[last_end:m.start()])
        body = m.group(1).strip()
        last_end = m.end()
        try:
            obj = json.loads(body)
            name = obj.get("name")
            args = obj.get("arguments") or obj.get("args") or {}
            if isinstance(name, str):
                calls.append(ToolCall(
                    name=name, arguments=args if isinstance(args, dict) else {},
                    raw=body, call_id=obj.get("id"),
                ))
        except json.JSONDecodeError:
            # MiniCPM occasionally emits compact non-JSON; tolerate by
            # trying a name(arg=val,arg=val) parse.
            mm = re.match(r"^([\w.\-]+)\((.*)\)$", body)
            if mm:
                name = mm.group(1)
                kvs = mm.group(2)
                args = {}
                for kv in re.findall(r"(\w+)=([^,]+)", kvs):
                    args[kv[0]] = kv[1].strip().strip('"').strip("'")
                calls.append(ToolCall(name=name, arguments=args, raw=body))
        # else: silently drop malformed
    parts.append(text[last_end:])
    remainder = "".join(parts)

    # Preserve any incomplete trailing TOOL_OPEN for streaming.
    if TOOL_OPEN in remainder and TOOL_CLOSE not in remainder.split(TOOL_OPEN, 1)[1]:
        # The remainder still has an unclosed open tag → caller may resume.
        pass
    return calls, remainder


def format_tool_response(name: str, result: dict | str, call_id: Optional[str] = None) -> str:
    """Render a tool result in the format the base model expects in the
    next turn's `<tool_response>` block."""
    payload = {"name": name, "content": result}
    if call_id is not None:
        payload["id"] = call_id
    return f"<tool_response>\n{json.dumps(payload)}\n</tool_response>"
