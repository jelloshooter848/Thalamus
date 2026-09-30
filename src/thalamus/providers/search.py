"""Web search as a sense organ (Tavily). THALAMUS runs the search itself, so every query and
result is visible and gated by JEV before any of it reaches the cortex."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Literal, Protocol

SEARCH_KEY_ENV = "TAVILY_API_KEY"
Topic = Literal["general", "news", "finance"]


@dataclass(frozen=True)
class WebResult:
    title: str
    url: str
    content: str
    score: float = 0.0


@dataclass
class SearchOutcome:
    query: str
    results: list[WebResult] = field(default_factory=list)
    latency_ms: float = 0.0
    calls: int = 1  # billable requests (search or extract)


class SearchProvider(Protocol):
    async def search(self, query: str, *, topic: Topic, max_results: int) -> SearchOutcome: ...

    async def read(self, url: str) -> SearchOutcome: ...


def search_key() -> str | None:
    return os.environ.get(SEARCH_KEY_ENV, "").strip() or None


class TavilySearch:
    def __init__(self, api_key: str | None = None, depth: str = "basic", client=None) -> None:
        from tavily import AsyncTavilyClient

        self._client = client or AsyncTavilyClient(api_key=api_key or search_key())
        self._depth = depth

    async def search(self, query: str, *, topic: Topic = "general", max_results: int = 6) -> SearchOutcome:
        started = time.perf_counter()
        data = await self._client.search(
            query, search_depth=self._depth, topic=topic, max_results=max_results, timeout=20
        )
        return SearchOutcome(
            query=query,
            results=[
                WebResult(r.get("title", ""), r.get("url", ""), r.get("content", ""), float(r.get("score", 0.0)))
                for r in data.get("results", [])
            ],
            latency_ms=(time.perf_counter() - started) * 1000,
        )

    async def read(self, url: str) -> SearchOutcome:
        """Read one page the user pointed at (Tavily extract)."""
        started = time.perf_counter()
        data = await self._client.extract(url, format="text", timeout=20)
        return SearchOutcome(
            query=url,
            results=[
                WebResult(r.get("title") or r.get("url", url), r.get("url", url), r.get("raw_content") or "", 1.0)
                for r in data.get("results", [])
            ],
            latency_ms=(time.perf_counter() - started) * 1000,
        )
