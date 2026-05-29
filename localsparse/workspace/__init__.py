"""Workspace layer: high-level manager, eviction, consolidation."""
from .manager import WorkspaceManager, Encoder, DummyEncoder  # noqa: F401
from .eviction import run_eviction  # noqa: F401
from .consolidation import (
    ConsolidationOrchestrator, ConsolidationResult, MockSearcher,  # noqa: F401
    MockSynthesizer, SearchResult,
)
