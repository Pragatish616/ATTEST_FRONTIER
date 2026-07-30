"""Tests for attest.search.

No real Tavily/DuckDuckGo calls: `search._PROVIDER_CHAIN` is monkeypatched
directly, mirroring how tests/test_contracts.py patches
`attest.llm._PROVIDER_CHAIN` (patching private function names wouldn't work
since the chain list captures function references at import time).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from attest import search
from attest.search import SearchResult


@pytest.fixture(autouse=True)
def _isolated_cache_dir(tmp_path, monkeypatch):
    """Never touch the real ./data/search_cache from tests."""
    monkeypatch.setattr(search, "_CACHE_DIR", tmp_path / "search_cache")


def _fake_result(url: str = "https://example.com/a") -> SearchResult:
    return SearchResult(url=url, title="Example", snippet="some snippet text", published_date=None)


async def test_search_uses_first_provider_that_succeeds(monkeypatch):
    async def fake_tavily(query: str, k: int) -> list[SearchResult]:
        return [_fake_result("https://tavily.example/1")]

    async def fake_ddg(query: str, k: int) -> list[SearchResult]:
        raise AssertionError("duckduckgo should not be called when tavily succeeds")

    monkeypatch.setattr(search, "_PROVIDER_CHAIN", [("tavily", fake_tavily), ("duckduckgo", fake_ddg)])
    monkeypatch.setattr(search.settings, "tavily_api_key", "fake-key")

    results = await search.search("who won the election", k=3)

    assert len(results) == 1
    assert results[0].url == "https://tavily.example/1"


async def test_search_falls_back_to_duckduckgo_on_tavily_failure(monkeypatch):
    async def fake_tavily(query: str, k: int) -> list[SearchResult]:
        raise RuntimeError("tavily rate limited")

    async def fake_ddg(query: str, k: int) -> list[SearchResult]:
        return [_fake_result("https://ddg.example/1")]

    monkeypatch.setattr(search, "_PROVIDER_CHAIN", [("tavily", fake_tavily), ("duckduckgo", fake_ddg)])
    monkeypatch.setattr(search.settings, "tavily_api_key", "fake-key")

    results = await search.search("some query", k=3)

    assert len(results) == 1
    assert results[0].url == "https://ddg.example/1"


async def test_search_skips_tavily_when_no_api_key_configured(monkeypatch):
    async def fake_tavily(query: str, k: int) -> list[SearchResult]:
        raise AssertionError("tavily should be skipped when no api key is configured")

    async def fake_ddg(query: str, k: int) -> list[SearchResult]:
        return [_fake_result("https://ddg.example/1")]

    monkeypatch.setattr(search, "_PROVIDER_CHAIN", [("tavily", fake_tavily), ("duckduckgo", fake_ddg)])
    monkeypatch.setattr(search.settings, "tavily_api_key", None)

    results = await search.search("some query", k=3)

    assert len(results) == 1
    assert results[0].url == "https://ddg.example/1"


async def test_search_returns_empty_list_when_all_providers_fail(monkeypatch):
    async def fake_tavily(query: str, k: int) -> list[SearchResult]:
        raise RuntimeError("down")

    async def fake_ddg(query: str, k: int) -> list[SearchResult]:
        raise RuntimeError("also down")

    monkeypatch.setattr(search, "_PROVIDER_CHAIN", [("tavily", fake_tavily), ("duckduckgo", fake_ddg)])
    monkeypatch.setattr(search.settings, "tavily_api_key", "fake-key")

    results = await search.search("some query", k=3)

    assert results == []


async def test_search_caches_results_to_disk(monkeypatch, tmp_path):
    call_count = 0

    async def fake_tavily(query: str, k: int) -> list[SearchResult]:
        nonlocal call_count
        call_count += 1
        return [_fake_result("https://tavily.example/1")]

    monkeypatch.setattr(search, "_PROVIDER_CHAIN", [("tavily", fake_tavily)])
    monkeypatch.setattr(search.settings, "tavily_api_key", "fake-key")

    first = await search.search("cached query", k=2)
    second = await search.search("cached query", k=2)

    assert call_count == 1  # second call served from disk cache, not the provider
    assert first == second

    cache_dir = search._CACHE_DIR
    assert cache_dir.exists()
    cached_files = list(cache_dir.glob("*.json"))
    assert len(cached_files) == 1
    on_disk = json.loads(cached_files[0].read_text(encoding="utf-8"))
    assert on_disk["results"][0]["url"] == "https://tavily.example/1"
    assert "cached_at" in on_disk  # TTL needs a timestamp to compare against


async def test_search_cache_is_keyed_by_query_and_k(monkeypatch):
    calls: list[tuple[str, int]] = []

    async def fake_tavily(query: str, k: int) -> list[SearchResult]:
        calls.append((query, k))
        return [_fake_result(f"https://tavily.example/{len(calls)}")]

    monkeypatch.setattr(search, "_PROVIDER_CHAIN", [("tavily", fake_tavily)])
    monkeypatch.setattr(search.settings, "tavily_api_key", "fake-key")

    await search.search("same query", k=3)
    await search.search("same query", k=5)  # different k -> different cache key

    assert len(calls) == 2


async def test_search_cache_survives_corrupt_file(monkeypatch, tmp_path):
    async def fake_tavily(query: str, k: int) -> list[SearchResult]:
        return [_fake_result("https://tavily.example/1")]

    monkeypatch.setattr(search, "_PROVIDER_CHAIN", [("tavily", fake_tavily)])
    monkeypatch.setattr(search.settings, "tavily_api_key", "fake-key")

    cache_dir = search._CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = search._cache_path("broken query", 5)
    path.write_text("not valid json{{{", encoding="utf-8")

    results = await search.search("broken query", k=5)

    assert len(results) == 1  # falls through to the provider instead of crashing


# ---------------------------------------------------------------------------
# Cache TTL (regression: an immortal cache structurally prevented the
# independent verifier from ever noticing its evidence had gone stale)
# ---------------------------------------------------------------------------


async def test_expired_cache_entry_triggers_a_live_provider_call(monkeypatch):
    call_count = 0

    async def fake_tavily(query: str, k: int) -> list[SearchResult]:
        nonlocal call_count
        call_count += 1
        return [
            SearchResult(
                title=f"result {call_count}",
                url=f"https://tavily.example/{call_count}",
                snippet="s",
            )
        ]

    monkeypatch.setattr(search, "_PROVIDER_CHAIN", [("tavily", fake_tavily)])
    monkeypatch.setattr(search.settings, "tavily_api_key", "fake-key")

    first = await search.search("ttl query", k=2)
    assert call_count == 1

    # Age the entry past the TTL by rewriting cached_at into the past.
    path = search._cache_path("ttl query", k=2)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["cached_at"] = (
        datetime.now(UTC) - timedelta(seconds=search.settings.search_cache_ttl_seconds + 60)
    ).isoformat()
    path.write_text(json.dumps(payload), encoding="utf-8")

    second = await search.search("ttl query", k=2)

    assert call_count == 2, "expired entry must not be served from cache"
    assert second != first  # genuinely refetched, so new information can surface


async def test_fresh_cache_entry_is_still_served_without_a_provider_call(monkeypatch):
    call_count = 0

    async def fake_tavily(query: str, k: int) -> list[SearchResult]:
        nonlocal call_count
        call_count += 1
        return [SearchResult(title="t", url="https://tavily.example/fresh", snippet="s")]

    monkeypatch.setattr(search, "_PROVIDER_CHAIN", [("tavily", fake_tavily)])
    monkeypatch.setattr(search.settings, "tavily_api_key", "fake-key")

    await search.search("fresh query", k=1)
    await search.search("fresh query", k=1)

    assert call_count == 1  # demo-day fast path intact


async def test_legacy_cache_entry_without_timestamp_is_a_miss(monkeypatch):
    call_count = 0

    async def fake_tavily(query: str, k: int) -> list[SearchResult]:
        nonlocal call_count
        call_count += 1
        return [SearchResult(title="t", url="https://tavily.example/new", snippet="s")]

    monkeypatch.setattr(search, "_PROVIDER_CHAIN", [("tavily", fake_tavily)])
    monkeypatch.setattr(search.settings, "tavily_api_key", "fake-key")

    # Pre-TTL format: a bare list with no cached_at.
    search._CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = search._cache_path("legacy query", k=1)
    path.write_text(
        json.dumps([{"title": "old", "url": "https://old.example", "snippet": "s"}]),
        encoding="utf-8",
    )

    results = await search.search("legacy query", k=1)

    assert call_count == 1
    assert results[0].url == "https://tavily.example/new"
