"""Chat template + workspace_context block."""
from .template import (  # noqa: F401
    ChatMessage, WorkspaceContextBlock, MountInfo, PinInfo, CandidateInfo,
    render_prompt, render_tool_descriptions, split_assistant_response,
    IM_START, IM_END,
)
