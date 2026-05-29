"""Cross-workspace consolidation orchestrator.

Implements the `research` and `rewrite` modes that back
`workspace.consolidate` and `workspace.promote_region`.

The orchestrator is intentionally *small*: it sequences calls to
`web.search`, `web.fetch`, the model's `synthesize_consolidated_text`
method, and the workspace manager's `append`.  The agent only ever sees
the composite op (see plan.md §2.7).
"""
from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Protocol

from ..storage.registry import ProvenanceRecord
from ..config import default_config


@dataclass
class SearchResult:
    url: str
    title: str
    snippet: str


class Searcher(Protocol):
    def search(self, query: str, k: int = 5) -> List[SearchResult]: ...
    def fetch(self, url: str, max_bytes: int = 256 * 1024) -> str: ...


class MockSearcher:
    """In-memory deterministic searcher for unit/integration tests.

    Seeded with a `corpus: dict[str, str]` where keys are URLs and values
    are the page text.  `search` returns the top-k URLs whose page text
    has the highest naive bag-of-words overlap with the query.
    """

    def __init__(self, corpus: dict[str, str]):
        self.corpus = corpus

    def search(self, query: str, k: int = 5) -> List[SearchResult]:
        q = set(query.lower().split())
        scored = []
        for url, text in self.corpus.items():
            score = len(q & set(text.lower().split()))
            scored.append((score, url, text))
        scored.sort(reverse=True)
        return [
            SearchResult(url=u, title=u.rsplit("/", 1)[-1], snippet=t[:200])
            for _, u, t in scored[:k]
        ]

    def fetch(self, url: str, max_bytes: int = 256 * 1024) -> str:
        return self.corpus.get(url, "")[:max_bytes]


class Synthesizer(Protocol):
    """Either the model itself or, in tests, a deterministic stub."""

    def topic_from_region(self, src_wks: str, src_region: str) -> str: ...
    def synthesize(self, *, topic: str, fetched_docs: list[str],
                   cross_refs: list[str]) -> str: ...


class MockSynthesizer:
    """Topic = "<wks> <region>", synthesis = concatenation of docs+refs."""

    def topic_from_region(self, src_wks: str, src_region: str) -> str:
        return f"{src_wks} {src_region}"

    def synthesize(self, *, topic: str, fetched_docs: list[str],
                   cross_refs: list[str]) -> str:
        parts = [f"# {topic}"]
        for i, d in enumerate(fetched_docs):
            parts.append(f"\n## source-{i}\n{d}")
        if cross_refs:
            parts.append("\n## cross-references\n" + "\n".join(cross_refs))
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
@dataclass
class ConsolidationResult:
    consolidation_id: str
    dst_wks: str
    dst_region: str
    appended_tokens: int
    mode: str
    provenance: ProvenanceRecord


class ConsolidationOrchestrator:
    def __init__(
        self,
        manager,                       # WorkspaceManager (avoid circular import)
        searcher: Optional[Searcher] = None,
        synthesizer: Optional[Synthesizer] = None,
    ):
        self.mgr = manager
        self.cfg = manager.config.workspace
        self.searcher = searcher or MockSearcher({})
        self.synthesizer = synthesizer or MockSynthesizer()
        self._calls_this_session = 0

    # ---- public entry points --------------------------------------------
    def consolidate(
        self, *, src: str, region: str, dst: str, mode: str = "research",
    ) -> ConsolidationResult:
        """Execute one consolidation. Enforces session and per-wks budgets."""
        self._enforce_session_budget()
        self._enforce_dst_budget(dst)

        if mode == "research":
            text = self._do_research(src=src, region=region)
        elif mode == "rewrite":
            text = self._do_rewrite(src=src, region=region)
        else:
            raise ValueError(f"unknown consolidation mode {mode!r}")

        # Append synthesized text into destination workspace.
        start, end = self.mgr.append(dst, text)
        # Track consolidated slot consumption on the destination.
        dst_meta = self.mgr.registry.get_workspace(dst)
        if dst_meta is not None:
            dst_meta.consolidated_slots += (end - start)
            self.mgr.registry.put_workspace(dst_meta)

        cid = uuid.uuid4().hex[:16]
        h = hashlib.sha256(text.encode()).hexdigest()[:32]
        prov = ProvenanceRecord(
            consolidation_id=cid,
            src_wks=src, src_region=region,
            dst_wks=dst, dst_region=f"r#{start}-{end}",
            mode=mode, created_at=time.time(),
            content_hash=h,
        )
        self.mgr.registry.put_provenance(prov)
        # Candidate is satisfied — drop it from pending so it doesn't
        # immediately re-surface.
        self.mgr.registry.delete_candidate(src, region, dst)
        return ConsolidationResult(
            consolidation_id=cid, dst_wks=dst, dst_region=prov.dst_region,
            appended_tokens=end - start, mode=mode, provenance=prov,
        )

    def promote_region(self, *, wks: str, region: str) -> ConsolidationResult:
        """Re-encode a compressed region back to fine fidelity (research-mode)."""
        return self.consolidate(src=wks, region=region, dst=wks, mode="research")

    def deconsolidate(self, consolidation_id: str) -> None:
        """Delete a consolidation's provenance and decrement cross-counters.

        We do NOT physically remove the appended bytes (that would require
        slab compaction). Future eviction passes will naturally remove
        the content once it ages out.
        """
        prov = self.mgr.registry.get_provenance(consolidation_id)
        if prov is None:
            return
        self.mgr.access_log.decrement_cross(
            prov.src_wks, prov.src_region, prov.dst_wks, delta=1)
        self.mgr.registry.delete_provenance(consolidation_id)

    def pending_candidates(self):
        return self.mgr.registry.list_candidates(
            ttl_days=self.cfg.consolidation_window_days)

    # ---- modes -----------------------------------------------------------
    def _do_research(self, *, src: str, region: str) -> str:
        topic = self.synthesizer.topic_from_region(src, region)
        results = self.searcher.search(topic, k=self.cfg.consolidation_max_results)
        fetched: list[str] = []
        budget = self.cfg.consolidation_max_bytes
        for r in results:
            doc = self.searcher.fetch(r.url, max_bytes=budget)
            fetched.append(doc)
            budget -= len(doc)
            if budget <= 0:
                break
        cross_refs = self._collect_cross_refs(exclude={src})
        return self.synthesizer.synthesize(
            topic=topic, fetched_docs=fetched, cross_refs=cross_refs)

    def _do_rewrite(self, *, src: str, region: str) -> str:
        topic = self.synthesizer.topic_from_region(src, region)
        cross_refs = self._collect_cross_refs(exclude={src})
        return self.synthesizer.synthesize(
            topic=topic, fetched_docs=[], cross_refs=cross_refs)

    def _collect_cross_refs(self, *, exclude: set[str]) -> list[str]:
        """Return short descriptors of currently-mounted workspaces (for the
        synthesizer to weave in)."""
        return [w for w in self.mgr.mounted_workspaces() if w not in exclude]

    # ---- budgets ---------------------------------------------------------
    def _enforce_session_budget(self) -> None:
        self._calls_this_session += 1
        if self._calls_this_session > self.cfg.research_calls_per_session:
            raise RuntimeError(
                f"Research budget exceeded "
                f"({self.cfg.research_calls_per_session}/session)")

    def _enforce_dst_budget(self, dst: str) -> None:
        meta = self.mgr.registry.get_workspace(dst)
        if meta is None:
            raise FileNotFoundError(dst)
        budget_slots = int(meta.slot_cap * self.cfg.consolidation_budget_fraction)
        if meta.consolidated_slots >= budget_slots:
            raise RuntimeError(
                f"Per-workspace consolidation budget exceeded for {dst!r}: "
                f"{meta.consolidated_slots}/{budget_slots} slots")
