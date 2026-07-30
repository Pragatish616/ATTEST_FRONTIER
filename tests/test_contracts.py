"""Contract tests for A0's deliverables.

Every model must round-trip through JSON, the Verdict/MutationType enums
must match PLAN.md exactly, and attest.llm.complete must work end-to-end
against a mocked provider (no network, no real API key needed).
"""

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import BaseModel

from attest import llm
from attest.models import (
    AttestConfig,
    Chunk,
    Claim,
    ClaimDetail,
    Evidence,
    MutationType,
    ObserveRequest,
    ObserveResponse,
    Probe,
    RetrievedChunk,
    RunDetail,
    RunSummary,
    Verdict,
    Verification,
    VerifierProtocol,
    VerifyContext,
)


def _round_trip(model: BaseModel) -> None:
    cls = type(model)
    restored = cls.model_validate_json(model.model_dump_json())
    assert restored == model


# ---------------------------------------------------------------------------
# Verdict taxonomy (PLAN.md §3) — FROZEN
# ---------------------------------------------------------------------------


def test_verdict_taxonomy_matches_plan_exactly():
    assert {v.value for v in Verdict} == {
        "GROUNDED",
        "FRAGILE",
        "UNSUPPORTED",
        "CONTRADICTED",
        "STALE",
        "UNVERIFIABLE",
    }
    assert len(Verdict) == 6


def test_mutation_type_matches_plan_exactly():
    assert {m.value for m in MutationType} == {
        "negation",
        "entity_swap",
        "quantifier_shift",
    }
    assert len(MutationType) == 3


# ---------------------------------------------------------------------------
# Model round-trips
# ---------------------------------------------------------------------------


def test_retrieved_chunk_round_trip():
    rc = RetrievedChunk(
        chunk_index=0, source_id="doc-12", source_url="https://x", text="t", score=0.5
    )
    _round_trip(rc)


def test_chunk_round_trip():
    chunk = Chunk(
        id=uuid4(),
        run_id=uuid4(),
        chunk_index=0,
        source_id="doc-12",
        source_url="https://example.com/doc",
        text="chunk text",
        score=0.82,
    )
    _round_trip(chunk)


def test_claim_round_trip_pending_and_final():
    pending = Claim(
        id=uuid4(), run_id=uuid4(), claim_index=0, text="Paris is the capital of France."
    )
    assert pending.verdict is None  # verifiers see claims before reconciliation
    _round_trip(pending)

    final = pending.model_copy(
        update={
            "verdict": Verdict.GROUNDED,
            "confidence": 0.9,
            "disagreement": 0.0,
            "rationale": "matches context",
        }
    )
    _round_trip(final)


def test_evidence_round_trip_chunk_and_url():
    ev_chunk = Evidence(chunk_id=uuid4(), quote_span=(10, 42), stance="support")
    _round_trip(ev_chunk)
    ev_url = Evidence(url="https://example.com", quote_span=(0, 5), stance="refute")
    _round_trip(ev_url)


def test_verification_round_trip():
    v = Verification(
        id=uuid4(),
        claim_id=uuid4(),
        verifier="entailment",
        verdict=Verdict.CONTRADICTED,
        rationale="context says otherwise",
        evidence=[Evidence(chunk_id=uuid4(), quote_span=(0, 10), stance="refute")],
        latency_ms=120,
        cost_usd=0.0004,
    )
    _round_trip(v)


def test_probe_round_trip():
    p = Probe(
        id=uuid4(),
        claim_id=uuid4(),
        mutation_type=MutationType.NEGATION,
        mutated_text="Paris is not the capital of France.",
        expected_flip=True,
        observed_verdict=Verdict.CONTRADICTED,
        flipped=True,
    )
    _round_trip(p)


def test_claim_detail_round_trip():
    cd = ClaimDetail(
        id=uuid4(),
        run_id=uuid4(),
        claim_index=0,
        text="claim",
        verdict=Verdict.FRAGILE,
        confidence=0.6,
        disagreement=0.4,
        rationale="probe failed to flip",
        verifications=[
            Verification(id=uuid4(), claim_id=uuid4(), verifier="prober", verdict=Verdict.GROUNDED)
        ],
        probes=[
            Probe(
                id=uuid4(),
                claim_id=uuid4(),
                mutation_type=MutationType.ENTITY_SWAP,
                mutated_text="x",
                expected_flip=True,
                observed_verdict=Verdict.GROUNDED,
                flipped=False,
            )
        ],
    )
    _round_trip(cd)


def test_run_summary_and_detail_round_trip():
    rs = RunSummary(
        id=uuid4(),
        created_at=datetime.now(UTC),
        pipeline_name="demo-rag",
        query="q",
        answer="a",
        model="claude-sonnet-5",
        status="complete",
        grounding_score=0.8,
        fragility_score=0.1,
        total_claims=5,
        latency_ms=2400,
        cost_usd=0.01,
    )
    _round_trip(rs)

    rd = RunDetail(**rs.model_dump(), retrieved_chunks=[], claims=[])
    _round_trip(rd)


def test_observe_request_matches_plan_example():
    payload = {
        "pipeline_name": "demo-rag",
        "query": "string",
        "answer": "string",
        "retrieved_chunks": [
            {
                "chunk_index": 0,
                "source_id": "doc-12",
                "source_url": "https://example.com",
                "text": "…",
                "score": 0.82,
            }
        ],
        "model": "claude-sonnet-4-6",
        "config": {
            "sample_rate": 1.0,
            "enable_independent": True,
            "enable_prober": True,
            "budget_usd": 0.05,
        },
    }
    req = ObserveRequest.model_validate(payload)
    assert req.pipeline_name == "demo-rag"
    assert req.retrieved_chunks[0].chunk_index == 0
    assert req.config.budget_usd == 0.05
    _round_trip(req)


def test_observe_request_defaults():
    req = ObserveRequest(pipeline_name="p", query="q", answer="a")
    assert req.retrieved_chunks == []
    assert req.config == AttestConfig()


def test_observe_response_round_trip():
    resp = ObserveResponse(run_id=uuid4(), status="pending")
    _round_trip(resp)


def test_verify_context_round_trip():
    ctx = VerifyContext(run_id=uuid4(), query="q", answer="a")
    _round_trip(ctx)


def test_verifier_protocol_is_structural():
    class DummyVerifier:
        async def verify(self, claim: Claim, ctx: VerifyContext) -> Verification:
            return Verification(
                id=uuid4(), claim_id=claim.id, verifier="entailment", verdict=Verdict.GROUNDED
            )

    class NotAVerifier:
        pass

    assert isinstance(DummyVerifier(), VerifierProtocol)
    assert not isinstance(NotAVerifier(), VerifierProtocol)


# ---------------------------------------------------------------------------
# attest.llm.complete against a mocked provider
# ---------------------------------------------------------------------------


class _JudgeSchema(BaseModel):
    verdict: str
    rationale: str


async def test_complete_plain_text_no_schema(monkeypatch):
    async def fake_call(prompt: str, model: str) -> tuple[str, int, int]:
        return "hello world", 10, 5

    monkeypatch.setattr(llm, "_PROVIDER_CHAIN", [("anthropic", fake_call)])

    result = await llm.complete("say hi", model_tier="fast")

    assert result.text == "hello world"
    assert result.provider == "anthropic"
    assert result.model_tier == "fast"
    assert result.tokens_in == 10
    assert result.tokens_out == 5
    assert result.cost_usd > 0
    assert result.parsed is None


async def test_complete_with_schema_parses_first_try(monkeypatch):
    async def fake_call(prompt: str, model: str) -> tuple[str, int, int]:
        return '{"verdict": "GROUNDED", "rationale": "ok"}', 20, 8

    monkeypatch.setattr(llm, "_PROVIDER_CHAIN", [("anthropic", fake_call)])

    result = await llm.complete("verify this", model_tier="judge", schema=_JudgeSchema)

    assert isinstance(result.parsed, _JudgeSchema)
    assert result.parsed.verdict == "GROUNDED"


async def test_complete_with_schema_local_repair_strips_fences(monkeypatch):
    calls = []

    async def fake_call(prompt: str, model: str) -> tuple[str, int, int]:
        calls.append(prompt)
        return '```json\n{"verdict": "STALE", "rationale": "old source"}\n```', 15, 6

    monkeypatch.setattr(llm, "_PROVIDER_CHAIN", [("anthropic", fake_call)])

    result = await llm.complete("verify", model_tier="judge", schema=_JudgeSchema)

    assert result.parsed.verdict == "STALE"
    assert len(calls) == 1  # local repair succeeded — no extra LLM call needed


async def test_complete_with_schema_retries_once_with_nudge(monkeypatch):
    responses = iter(
        [
            ("not json at all, sorry", 10, 5),
            ('{"verdict": "UNSUPPORTED", "rationale": "no evidence"}', 12, 6),
        ]
    )
    prompts_seen = []

    async def fake_call(prompt: str, model: str) -> tuple[str, int, int]:
        prompts_seen.append(prompt)
        return next(responses)

    monkeypatch.setattr(llm, "_PROVIDER_CHAIN", [("anthropic", fake_call)])

    result = await llm.complete("verify", model_tier="judge", schema=_JudgeSchema)

    assert result.parsed.verdict == "UNSUPPORTED"
    assert len(prompts_seen) == 2
    assert "valid JSON only" in prompts_seen[1]
    assert result.tokens_in == 22  # accumulated across both calls
    assert result.tokens_out == 11


async def test_complete_falls_back_to_next_provider_on_error(monkeypatch):
    async def fail_call(prompt: str, model: str) -> tuple[str, int, int]:
        raise RuntimeError("rate limited")

    async def ok_call(prompt: str, model: str) -> tuple[str, int, int]:
        return "fallback response", 3, 2

    monkeypatch.setattr(llm.settings, "groq_api_key", "fake-groq-key")
    monkeypatch.setattr(llm, "_PROVIDER_CHAIN", [("anthropic", fail_call), ("groq", ok_call)])

    result = await llm.complete("say hi", model_tier="fast")

    assert result.provider == "groq"
    assert result.text == "fallback response"


async def test_complete_raises_when_all_providers_fail(monkeypatch):
    async def fail_call(prompt: str, model: str) -> tuple[str, int, int]:
        raise RuntimeError("down")

    monkeypatch.setattr(llm, "_PROVIDER_CHAIN", [("anthropic", fail_call)])

    with pytest.raises(llm.LLMError):
        await llm.complete("x", model_tier="fast")


async def test_complete_raises_when_no_provider_configured(monkeypatch):
    monkeypatch.setattr(llm, "_PROVIDER_CHAIN", [])

    with pytest.raises(llm.LLMError):
        await llm.complete("x", model_tier="fast")


# ---------------------------------------------------------------------------
# Provider configuration: any ONE provider key is sufficient to boot, so a
# Gemini-only (or Groq-only) setup is valid. Regression: anthropic_api_key was
# a hard-required field, so a free-tier Gemini demo could not start at all.
# ---------------------------------------------------------------------------


def test_gemini_only_configuration_is_valid():
    from attest.config import Settings

    settings = Settings(
        supabase_url="https://example.supabase.co",
        supabase_key="k",
        anthropic_api_key=None,
        groq_api_key=None,
        gemini_api_key="gem-key",
        _env_file=None,
    )

    assert settings.gemini_api_key == "gem-key"
    assert settings.anthropic_api_key is None


def test_groq_only_configuration_is_valid():
    from attest.config import Settings

    settings = Settings(
        supabase_url="https://example.supabase.co",
        supabase_key="k",
        anthropic_api_key=None,
        gemini_api_key=None,
        groq_api_key="groq-key",
        _env_file=None,
    )

    assert settings.groq_api_key == "groq-key"


def test_provider_chain_skips_providers_without_a_key(monkeypatch):
    """With only Gemini keyed, Gemini is the effective primary."""
    from attest import llm

    called: list[str] = []

    async def fake_anthropic(prompt: str, model: str):
        called.append("anthropic")
        return "{}", 1, 1

    async def fake_gemini(prompt: str, model: str):
        called.append("gemini")
        return "{}", 1, 1

    monkeypatch.setattr(
        llm,
        "_PROVIDER_CHAIN",
        [("anthropic", fake_anthropic), ("gemini", fake_gemini)],
    )
    monkeypatch.setattr(llm.settings, "anthropic_api_key", None)
    monkeypatch.setattr(llm.settings, "gemini_api_key", "gem-key")

    import asyncio

    asyncio.run(llm.complete(prompt="hi", model_tier="fast"))

    assert called == ["gemini"], "keyless provider must be skipped entirely"


# ---------------------------------------------------------------------------
# Transient-failure retry: a 429/503 is the provider telling us to wait, not
# that the request was wrong. With a single provider key configured (the free
# tier demo case) an immediate give-up kills the run.
# ---------------------------------------------------------------------------


async def test_transient_error_is_retried_on_the_same_provider(monkeypatch):
    from attest import llm

    attempts: list[int] = []

    async def flaky(prompt: str, model: str):
        attempts.append(1)
        if len(attempts) < 3:
            raise RuntimeError("429 RESOURCE_EXHAUSTED ... Please retry in 0.01s.")
        return "ok", 1, 1

    monkeypatch.setattr(llm, "_PROVIDER_CHAIN", [("gemini", flaky)])
    monkeypatch.setattr(llm.settings, "gemini_api_key", "k")
    monkeypatch.setattr(llm.settings, "llm_transient_retries", 2)
    monkeypatch.setattr(llm.settings, "llm_retry_base_delay_s", 0.01)

    result = await llm.complete(prompt="hi", model_tier="fast")

    assert result.text == "ok"
    assert len(attempts) == 3  # two failures, then success


async def test_non_transient_error_is_not_retried(monkeypatch):
    """A 401 is a real error — retrying it just wastes demo time."""
    from attest import llm

    attempts: list[int] = []

    async def bad_key(prompt: str, model: str):
        attempts.append(1)
        raise RuntimeError("401 authentication_error: invalid x-api-key")

    monkeypatch.setattr(llm, "_PROVIDER_CHAIN", [("gemini", bad_key)])
    monkeypatch.setattr(llm.settings, "gemini_api_key", "k")
    monkeypatch.setattr(llm.settings, "llm_transient_retries", 2)
    monkeypatch.setattr(llm.settings, "llm_retry_base_delay_s", 0.01)

    with pytest.raises(llm.LLMError):
        await llm.complete(prompt="hi", model_tier="fast")

    assert len(attempts) == 1  # no retry


async def test_retry_honours_provider_supplied_delay(monkeypatch):
    from attest import llm

    assert llm._suggested_delay(RuntimeError("Please retry in 7.5s."), 2.0) == 7.5
    assert llm._suggested_delay(RuntimeError("boom"), 2.0) == 2.0
    # capped so a 300s hint can't stall a live demo
    assert llm._suggested_delay(RuntimeError("Please retry in 300s."), 2.0) == 30.0


async def test_exhausted_retries_still_fall_through_to_next_provider(monkeypatch):
    from attest import llm

    async def always_429(prompt: str, model: str):
        raise RuntimeError("503 UNAVAILABLE high demand")

    async def healthy(prompt: str, model: str):
        return "from-groq", 1, 1

    monkeypatch.setattr(
        llm, "_PROVIDER_CHAIN", [("gemini", always_429), ("groq", healthy)]
    )
    monkeypatch.setattr(llm.settings, "gemini_api_key", "k")
    monkeypatch.setattr(llm.settings, "groq_api_key", "k")
    monkeypatch.setattr(llm.settings, "llm_transient_retries", 1)
    monkeypatch.setattr(llm.settings, "llm_retry_base_delay_s", 0.01)

    result = await llm.complete(prompt="hi", model_tier="fast")
    assert result.text == "from-groq"
