"""Tool registry + dispatcher.

The registry holds all callable tools exposed to the model.  A `Tool` is
a Python callable with a JSON-schema-style description; the dispatcher
takes a parsed `ToolCall` and runs the corresponding callable, returning
the JSON-serializable result (or an error payload).
"""
from __future__ import annotations

import inspect
import json
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .parser import ToolCall, format_tool_response


@dataclass
class Tool:
    name: str
    fn: Callable[..., Any]
    description: str = ""
    schema: Dict[str, Any] = field(default_factory=dict)   # JSON-schema-ish

    def __call__(self, **kwargs):
        return self.fn(**kwargs)


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, name: str, fn: Callable[..., Any],
                 description: str = "", schema: Optional[Dict] = None) -> None:
        if name in self._tools:
            raise ValueError(f"Tool {name!r} already registered")
        self._tools[name] = Tool(name=name, fn=fn, description=description,
                                 schema=schema or {})

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def list_tools(self) -> List[Tool]:
        return list(self._tools.values())

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def dispatch(self, call: ToolCall) -> str:
        """Execute `call` and return a serialized `<tool_response>` string."""
        tool = self.get(call.name)
        if tool is None:
            return format_tool_response(
                call.name, {"error": f"unknown tool {call.name!r}"},
                call_id=call.call_id,
            )
        try:
            result = tool(**call.arguments)
            if not isinstance(result, (dict, str, list, int, float, bool, type(None))):
                # best-effort serialize for custom objects
                result = repr(result)
            return format_tool_response(call.name, result, call_id=call.call_id)
        except TypeError as e:
            # argument mismatch
            return format_tool_response(
                call.name,
                {"error": f"bad arguments: {e}"},
                call_id=call.call_id,
            )
        except Exception as e:
            return format_tool_response(
                call.name,
                {"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc(limit=2)},
                call_id=call.call_id,
            )

    # ---- introspection: emit a JSON-schema list for the chat template
    def describe(self) -> List[Dict[str, Any]]:
        out = []
        for t in self._tools.values():
            sig = inspect.signature(t.fn)
            params = []
            for name, p in sig.parameters.items():
                if name == "self":
                    continue
                params.append({
                    "name": name,
                    "required": p.default is inspect.Parameter.empty,
                })
            out.append({
                "name": t.name,
                "description": t.description,
                "parameters": params,
                "schema": t.schema,
            })
        return out
