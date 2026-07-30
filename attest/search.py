"""Independent web search client (A3).

Used by `attest/verifiers/independent.py` to re-search the open web and
cross-check the *source*, independent of whatever the pipeline retrieved
(PLAN.md §2, §6). Tavily is the primary provider; DuckDuckGo is the
fallback. `SearchResult` is this module's own type — it is not part of the
frozen `attest.models` contract.

Structured the same way `attest/llm.py` structures its provider chain: a
small module-level list of `(name, async_call_fn)` tuples tried in order.
Tests monkeypatch `_PROVIDER_CHAIN` directly rather than patching private
function names, since the chain captures function references at import
time (see `tests/test_contracts.py` for the equivalent pattern against
`attest.llm._PROVIDER_CHAIN`).

Every call is cached to disk under `./data/search_cache`, keyed by a hash of
the query and `k`, so a demo run never depends on a live search API — a
cached run is a valid fallback path on its own.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path

import structlog
from pydantic import BaseModel

from attest.config import settings

logger = structlog.get_logger(__name__)

_CACHE_DIR = Path("./data/search_cache")


class SearchResult(BaseModel):
    """One organic web search result.

    Deliberately not in `attest.models` — only `IndependentVerifier`
    consumes this shape, so it doesn't need to be part of the frozen
    cross-component contract.
    """

    url: str
    title: str
    snippet: str
    published_date: str | None = None


SearchCall = Callable[[str, int], Awaitable[list[SearchResult]]]


async def _search_tavily(query: str, k: int) -> list[SearchResult]:
    from tavily import AsyncTavilyClient

    client = AsyncTavilyClient(api_key=settings.tavily_api_key)
    response = await client.search(query, max_results=k, search_depth="basic")
    results: list[SearchResult] = []
    for item in response.get("results", []):
        results.append(
            SearchResult(
                url=item.get("url", ""),
                title=item.get("title", ""),
                snippet=item.get("content", ""),
                published_date=item.get("published_date"),
            )
        )
    return results


async def _search_duckduckgo(query: str, k: int) -> list[SearchResult]:
    # Deferred import: `duckduckgo-search` is NOT currently a declared
    # project dependency (see this agent's final report — it needs adding
    # to pyproject.toml centrally by A0). Importing it lazily, inside the
    # function body, mirrors the pattern in attest/llm.py's
    # _call_anthropic/_call_groq/_call_gemini so this module doesn't
    # hard-fail at import time before the package is actually installed.
    from duckduckgo_search import DDGS

    def _run_sync() -> list[dict]:
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=k))

    rows = await asyncio.to_thread(_run_sync)
    return [
        SearchResult(
            url=row.get("href", ""),
            title=row.get("title", ""),
            snippet=row.get("body", ""),
            published_date=None,
        )
        for row in rows
    ]


# Tried in order; Tavily first, DuckDuckGo fallback (PLAN.md §6). Tests
# monkeypatch this list directly, e.g.:
#   monkeypatch.setattr(search, "_PROVIDER_CHAIN", [("tavily", fake_call)])
_PROVIDER_CHAIN: list[tuple[str, SearchCall]] = [
    ("tavily", _search_tavily),
    ("duckduckgo", _search_duckduckgo),
]


def _cache_path(query: str, k: int) -> Path:
    digest = hashlib.sha256(f"{query}::{k}".encode()).hexdigest()
    return _CACHE_DIR / f"{digest}.json"


def _read_cache(query: str, k: int) -> list[SearchResult] | None:
    """Return cached results, or None on a miss / corrupt entry / expired TTL.

    An entry older than `settings.search_cache_ttl_seconds` is treated as a
    miss so the independent verifier can actually observe that the world
    changed — an immortal cache would structurally prevent STALE detection.
    """
    path = _cache_path(query, k)
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        # Legacy entries were a bare list with no timestamp; treat them as
        # expired rather than trusting them forever.
        if not isinstance(raw, dict):
            logger.info("search_cache_legacy_entry_ignored", path=str(path))
            return None
        cached_at = datetime.fromisoformat(raw["cached_at"])
        age_seconds = (datetime.now(UTC) - cached_at).total_seconds()
        if age_seconds > settings.search_cache_ttl_seconds:
            logger.info(
                "search_cache_expired",
                query=query,
                age_seconds=round(age_seconds, 1),
                ttl_seconds=settings.search_cache_ttl_seconds,
            )
            return None
        return [SearchResult.model_validate(item) for item in raw["results"]]
    except Exception as exc:  # noqa: BLE001 - corrupt/partial cache entry is just a miss
        logger.warning("search_cache_read_failed", path=str(path), error=str(exc))
        return None


def _write_cache(query: str, k: int, results: list[SearchResult]) -> None:
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            {
                "cached_at": datetime.now(UTC).isoformat(),
                "results": [r.model_dump() for r in results],
            }
        )
        _cache_path(query, k).write_text(payload, encoding="utf-8")
    except OSError as exc:  # noqa: BLE001 - caching is best-effort, never fatal
        logger.warning("search_cache_write_failed", query_digest=_query_digest(query), error=str(exc))


def _query_digest(query: str) -> str:
    """Short, stable, non-reversible id for a search query, for use in logs.

    The independent verifier builds its query by rewriting claim text, and claim
    text comes from the answer of whatever pipeline installed the SDK. Logging
    the query verbatim therefore copies the host application's — in a real
    deployment, its users' — content into our log stream, where it outlives the
    run and lands in the platform's log retention. A digest still lets us
    correlate cache misses and provider failures for one query across log lines,
    which is the only reason the field existed.
    """
    return hashlib.sha256(query.encode("utf-8")).hexdigest()[:12]


async def search(query: str, k: int = 5) -> list[SearchResult]:
    """Search the open web: Tavily primary, DuckDuckGo fallback.

    Results are cached to disk keyed by a hash of `(query, k)`. A cache hit
    within `settings.search_cache_ttl_seconds` short-circuits every provider
    call, so demo day never depends on a live search API being reachable;
    past the TTL the entry is refetched so STALE detection stays possible.
    """
    cached = _read_cache(query, k)
    if cached is not None:
        return cached

    failures: list[str] = []
    for name, call in _PROVIDER_CHAIN:
        if name == "tavily" and not settings.tavily_api_key:
            continue
        try:
            results = await call(query, k)
        except Exception as exc:  # noqa: BLE001 - any provider failure triggers fallback
            logger.warning("search_provider_failed", provider=name, error=str(exc))
            failures.append(f"{name}: {exc}")
            continue

        logger.info(
            "search_call",
            provider=name,
            query_digest=_query_digest(query),
            k=k,
            n_results=len(results),
        )
        _write_cache(query, k, results)
        return results

    logger.warning(
        "search_all_providers_failed", query_digest=_query_digest(query), failures=failures
    )
    return []


__all__ = ["search", "SearchResult"]
