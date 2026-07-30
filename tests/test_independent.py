"""Tests for attest.verifiers.independent.IndependentVerifier.

No real network, no real LLM calls: `attest.llm._PROVIDER_CHAIN` is
monkeypatched the same way tests/test_contracts.py does it, and
`IndependentVerifier`'s own `search()` call is monkeypatched via the module
attribute it was imported into (`independent.search`), never a real
Tavily/DuckDuckGo call.

Covers the STALE decision logic pinned down in the A3 brief:
- STALE fires only when prior_entailment=GROUNDED, fresh evidence
  contradicts the claim, AND >=2 independent sources agree on that
  contradiction.
- A single contradicting source must NOT be enough to trigger STALE.
- Thin evidence (no search results at all) must fall back to the
  verifier's own judgment, never guessing at staleness.
- prior_entailment=None (standalone / no orchestrator) must never produce
  STALE, no matter how strong the contradiction signal is.
"""

from __future__ import annotations

import json
from uuid import uuid4

from attest import llm
from attest.models import Claim, Verdict, VerifyContext
from attest.search import SearchResult
from attest.verifiers import independent as independent_module
from attest.verifiers.independent import IndependentVerifier


def _make_claim(text: str = "The Eiffel Tower is the tallest structure in Paris.") -> Claim:
    return Claim(id=uuid4(), run_id=uuid4(), claim_index=0, text=text)


def _make_ctx(prior_entailment: Verdict | None = None) -> VerifyContext:
    return VerifyContext(
        run_id=uuid4(),
        query="how tall is the eiffel tower",
        answer="The Eiffel Tower is the tallest structure in Paris.",
        prior_entailment=prior_entailment,
    )


def _fake_search_results(n: int) -> list[SearchResult]:
    return [
        SearchResult(
            url=f"https://source-{i}.example/article",
            title=f"Source {i}",
            snippet="The Eiffel Tower is no longer the tallest structure in Paris as of 2024.",
            published_date="2024-06-01",
        )
        for i in range(n)
    ]


def _sequential_fake_call(responses: list[str]):
    """Build a fake provider-chain call that returns each response in order.

    Mirrors the pattern in tests/test_contracts.py's retry test: a plain
    iterator popped on each call, since attest.llm.complete calls the same
    (name, fn) tuple for both the "fast" rewrite call and the "judge" call.
    """
    remaining = iter(responses)

    async def fake_call(prompt: str, model: str) -> tuple[str, int, int]:
        text = next(remaining)
        return text, 10, 5

    return fake_call


def _rewrite_response(query: str = "eiffel tower tallest structure paris") -> str:
    return json.dumps({"query": query})


def _judge_response(verdict: str, rationale: str, evidence: list[dict]) -> str:
    return json.dumps({"verdict": verdict, "rationale": rationale, "evidence": evidence})


async def test_stale_fires_with_grounded_prior_and_two_agreeing_sources(monkeypatch):
    monkeypatch.setattr(independent_module, "search", _fake_search(2))
    responses = [
        _rewrite_response(),
        _judge_response(
            "CONTRADICTED",
            "Two recent sources say the claim is no longer true.",
            [
                {"source_index": 0, "quote_span": [0, 20], "stance": "refute"},
                {"source_index": 1, "quote_span": [0, 20], "stance": "refute"},
            ],
        ),
    ]
    monkeypatch.setattr(llm, "_PROVIDER_CHAIN", [("anthropic", _sequential_fake_call(responses))])

    verifier = IndependentVerifier()
    claim = _make_claim()
    ctx = _make_ctx(prior_entailment=Verdict.GROUNDED)

    result = await verifier.verify(claim, ctx)

    assert result.verdict == Verdict.STALE
    assert result.verifier == "independent"
    assert result.claim_id == claim.id
    assert len(result.evidence) == 2
    assert {e.url for e in result.evidence} == {
        "https://source-0.example/article",
        "https://source-1.example/article",
    }


async def test_stale_does_not_fire_with_only_one_contradicting_source(monkeypatch):
    monkeypatch.setattr(independent_module, "search", _fake_search(2))
    responses = [
        _rewrite_response(),
        _judge_response(
            "CONTRADICTED",
            "One source disagrees with the claim.",
            [
                {"source_index": 0, "quote_span": [0, 20], "stance": "refute"},
            ],
        ),
    ]
    monkeypatch.setattr(llm, "_PROVIDER_CHAIN", [("anthropic", _sequential_fake_call(responses))])

    verifier = IndependentVerifier()
    claim = _make_claim()
    ctx = _make_ctx(prior_entailment=Verdict.GROUNDED)

    result = await verifier.verify(claim, ctx)

    assert result.verdict != Verdict.STALE
    assert result.verdict == Verdict.CONTRADICTED
    assert "deferring" in result.rationale.lower()


async def test_thin_evidence_falls_back_to_own_judgment_without_asserting_stale(monkeypatch):
    monkeypatch.setattr(independent_module, "search", _fake_search(0))
    # Only the rewrite call should happen; if the judge LLM were called too,
    # this iterator would raise StopIteration and fail the test loudly.
    responses = [_rewrite_response()]
    monkeypatch.setattr(llm, "_PROVIDER_CHAIN", [("anthropic", _sequential_fake_call(responses))])

    verifier = IndependentVerifier()
    claim = _make_claim()
    ctx = _make_ctx(prior_entailment=Verdict.GROUNDED)

    result = await verifier.verify(claim, ctx)

    assert result.verdict != Verdict.STALE
    assert result.verdict == Verdict.UNVERIFIABLE
    assert "thin" in result.rationale.lower() or "no results" in result.rationale.lower()


async def test_standalone_without_prior_entailment_never_emits_stale(monkeypatch):
    monkeypatch.setattr(independent_module, "search", _fake_search(2))
    # Even with a strong two-source contradiction, STALE structurally cannot
    # fire without a prior GROUNDED entailment verdict to disagree with.
    responses = [
        _rewrite_response(),
        _judge_response(
            "CONTRADICTED",
            "Two sources disagree with the claim.",
            [
                {"source_index": 0, "quote_span": [0, 20], "stance": "refute"},
                {"source_index": 1, "quote_span": [0, 20], "stance": "refute"},
            ],
        ),
    ]
    monkeypatch.setattr(llm, "_PROVIDER_CHAIN", [("anthropic", _sequential_fake_call(responses))])

    verifier = IndependentVerifier()
    claim = _make_claim()
    ctx = _make_ctx(prior_entailment=None)

    result = await verifier.verify(claim, ctx)

    assert result.verdict != Verdict.STALE
    assert result.verdict == Verdict.CONTRADICTED


async def test_standalone_grounded_judgment_returned_as_is(monkeypatch):
    monkeypatch.setattr(independent_module, "search", _fake_search(2))
    responses = [
        _rewrite_response(),
        _judge_response(
            "GROUNDED",
            "Sources confirm the claim.",
            [{"source_index": 0, "quote_span": [0, 20], "stance": "support"}],
        ),
    ]
    monkeypatch.setattr(llm, "_PROVIDER_CHAIN", [("anthropic", _sequential_fake_call(responses))])

    verifier = IndependentVerifier()
    claim = _make_claim()
    ctx = _make_ctx(prior_entailment=None)

    result = await verifier.verify(claim, ctx)

    assert result.verdict == Verdict.GROUNDED
    assert result.evidence[0].stance == "support"


async def test_verify_never_raises_when_all_llm_providers_fail(monkeypatch):
    monkeypatch.setattr(independent_module, "search", _fake_search(2))
    monkeypatch.setattr(llm, "_PROVIDER_CHAIN", [])  # no provider configured at all

    verifier = IndependentVerifier()
    claim = _make_claim()
    ctx = _make_ctx(prior_entailment=Verdict.GROUNDED)

    result = await verifier.verify(claim, ctx)

    assert result.verdict == Verdict.UNVERIFIABLE
    assert result.verdict != Verdict.STALE


async def test_judge_ignores_out_of_range_source_index(monkeypatch):
    monkeypatch.setattr(independent_module, "search", _fake_search(1))
    responses = [
        _rewrite_response(),
        _judge_response(
            "CONTRADICTED",
            "hallucinated a second source that does not exist",
            [
                {"source_index": 0, "quote_span": [0, 10], "stance": "refute"},
                {"source_index": 5, "quote_span": [0, 10], "stance": "refute"},
            ],
        ),
    ]
    monkeypatch.setattr(llm, "_PROVIDER_CHAIN", [("anthropic", _sequential_fake_call(responses))])

    verifier = IndependentVerifier()
    claim = _make_claim()
    ctx = _make_ctx(prior_entailment=Verdict.GROUNDED)

    result = await verifier.verify(claim, ctx)

    # Only one real (in-range) source refutes -> STALE must not fire.
    assert result.verdict != Verdict.STALE
    assert len(result.evidence) == 1


def _fake_search(n: int):
    async def fake(query: str, k: int = 5) -> list[SearchResult]:
        return _fake_search_results(n)

    return fake
