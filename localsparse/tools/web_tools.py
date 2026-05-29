"""Web tools: `web.search` and `web.fetch`.

For local tests we use an injectable `Searcher` (the same one the
consolidation orchestrator uses).  In production this is wired to a real
HTTP search backend (e.g. DuckDuckGo HTML scrape, or a paid API) plus
`httpx` for fetching.

The fetch tool returns plaintext only — HTML stripped with a tiny
built-in cleaner so we don't pull in BeautifulSoup as a hard dep.
"""
from __future__ import annotations

import html
import re
from typing import Optional, Protocol, List

from ..workspace.consolidation import Searcher, SearchResult, MockSearcher
from .registry import ToolRegistry


class HtmlCleaner:
    """Minimal HTML → plaintext.  Good enough for unit tests and a
    reasonable fallback when no Readability is installed."""

    _SCRIPT = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
    _TAG = re.compile(r"<[^>]+>")
    _WS = re.compile(r"\s+")

    def clean(self, raw: str) -> str:
        raw = self._SCRIPT.sub(" ", raw)
        raw = self._TAG.sub(" ", raw)
        raw = html.unescape(raw)
        return self._WS.sub(" ", raw).strip()


def register_web_tools(
    reg: ToolRegistry,
    searcher: Searcher,
    cleaner: Optional[HtmlCleaner] = None,
    *,
    max_fetch_bytes: int = 256 * 1024,
) -> None:
    cleaner = cleaner or HtmlCleaner()

    def web_search(query: str, k: int = 5) -> dict:
        results = searcher.search(query, k=k)
        return {"query": query, "results": [
            {"url": r.url, "title": r.title, "snippet": r.snippet}
            for r in results
        ]}

    def web_fetch(url: str, max_bytes: Optional[int] = None) -> dict:
        mb = max_bytes if max_bytes is not None else max_fetch_bytes
        raw = searcher.fetch(url, max_bytes=mb)
        text = cleaner.clean(raw)
        return {"url": url, "text": text[:mb], "bytes": len(text)}

    reg.register("web.search", web_search,
                 description="Search the web; returns up to k URL/title/snippet entries.")
    reg.register("web.fetch", web_fetch,
                 description="Fetch a URL and return the cleaned plaintext content.")
