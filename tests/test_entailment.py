"""Tests for attest.verifiers.entailment.EntailmentVerifier.

Mocked LLM only (monkeypatch attest.llm._PROVIDER_CHAIN directly — see
tests/test_contracts.py for the established pattern). No network.
"""

import json
from uuid import uuid4

from attest import llm
from attest.models import Chunk, Claim, Verdict, VerifierProtocol, VerifyContext
from attest.verifiers.entailment import EntailmentVerifier


def _chunk(text: str, run_id, index: int = 0) -> Chunk:
    return Chunk(id=uuid4(), run_id=run_id, chunk_index=index, text=text)


def _claim(text: str, run_id) -> Claim:
    return Claim(id=uuid4(), run_id=run_id, claim_index=0, text=text)


def _ctx(run_id, chunks: list[Chunk], query="q", answer="a") -> VerifyContext:
    return VerifyContext(run_id=run_id, query=query, answer=answer, retrieved_chunks=chunks)


def _mock_judge_response(monkeypatch, payload: dict) -> list[str]:
    prompts_seen: list[str] = []

    async def fake_call(prompt: str, model: str) -> tuple[str, int, int]:
        prompts_seen.append(prompt)
        return json.dumps(payload), 40, 20

    monkeypatch.setattr(llm, "_PROVIDER_CHAIN", [("anthropic", fake_call)])
    return prompts_seen


async def test_supported_claim_is_grounded(monkeypatch):
    run_id = uuid4()
    chunk = _chunk("The Eiffel Tower was completed in 1889 in Paris.", run_id)
    claim = _claim("The Eiffel Tower was completed in 1889.", run_id)
    _mock_judge_response(
        monkeypatch,
        {
            "verdict": "GROUNDED",
            "rationale": "Context states the same completion year.",
            "evidence": [{"chunk_index": 0, "span_start": 22, "span_end": 41, "stance": "support"}],
        },
    )

    verifier = EntailmentVerifier()
    verification = await verifier.verify(claim, _ctx(run_id, [chunk]))

    assert verification.verdict == Verdict.GROUNDED
    assert verification.verifier == "entailment"
    assert len(verification.evidence) == 1
    assert verification.evidence[0].chunk_id == chunk.id
    assert verification.evidence[0].stance == "support"
    assert verification.evidence[0].quote_span == (22, 41)


async def test_unsupported_claim_when_context_silent(monkeypatch):
    run_id = uuid4()
    chunk = _chunk("The Eiffel Tower is a wrought-iron lattice tower in Paris.", run_id)
    claim = _claim("The Eiffel Tower is 330 meters tall.", run_id)
    _mock_judge_response(
        monkeypatch,
        {
            "verdict": "UNSUPPORTED",
            "rationale": "Context never mentions height.",
            "evidence": [],
        },
    )

    verifier = EntailmentVerifier()
    verification = await verifier.verify(claim, _ctx(run_id, [chunk]))

    assert verification.verdict == Verdict.UNSUPPORTED
    assert verification.evidence == []


async def test_contradicted_claim(monkeypatch):
    run_id = uuid4()
    chunk_text = "The bridge was closed to traffic in 2020 and has not reopened."
    chunk = _chunk(chunk_text, run_id)
    claim = _claim("The bridge is currently open to traffic.", run_id)
    _mock_judge_response(
        monkeypatch,
        {
            "verdict": "CONTRADICTED",
            "rationale": "Context says the bridge has not reopened.",
            "evidence": [
                {"chunk_index": 0, "span_start": 13, "span_end": len(chunk_text), "stance": "refute"}
            ],
        },
    )

    verifier = EntailmentVerifier()
    verification = await verifier.verify(claim, _ctx(run_id, [chunk]))

    assert verification.verdict == Verdict.CONTRADICTED
    assert verification.evidence[0].stance == "refute"


async def test_numeric_quantitative_claim(monkeypatch):
    run_id = uuid4()
    chunk = _chunk("The city's population was 2,148,000 as of the 2023 census.", run_id)
    claim = _claim("The city has a population of about 2.1 million.", run_id)
    _mock_judge_response(
        monkeypatch,
        {
            "verdict": "GROUNDED",
            "rationale": "2,148,000 rounds to approximately 2.1 million.",
            "evidence": [{"chunk_index": 0, "span_start": 27, "span_end": 36, "stance": "support"}],
        },
    )

    verifier = EntailmentVerifier()
    verification = await verifier.verify(claim, _ctx(run_id, [chunk]))

    assert verification.verdict == Verdict.GROUNDED
    assert verification.evidence[0].quote_span == (27, 36)


async def test_negated_claim(monkeypatch):
    run_id = uuid4()
    chunk = _chunk("Vaccine X does not require refrigeration during transport.", run_id)
    claim = _claim("Vaccine X does not require refrigeration during transport.", run_id)
    _mock_judge_response(
        monkeypatch,
        {
            "verdict": "GROUNDED",
            "rationale": "Context states the same negation verbatim.",
            "evidence": [{"chunk_index": 0, "span_start": 0, "span_end": 60, "stance": "support"}],
        },
    )

    verifier = EntailmentVerifier()
    verification = await verifier.verify(claim, _ctx(run_id, [chunk]))

    assert verification.verdict == Verdict.GROUNDED


async def test_multi_hop_claim_combines_two_chunks(monkeypatch):
    run_id = uuid4()
    chunk_a = _chunk("Dr. Alice Chen led the 2019 fusion reactor project.", run_id, index=0)
    chunk_b = _chunk("Alice Chen later became the director of the national lab in 2022.", run_id, index=1)
    claim = _claim(
        "The person who led the 2019 fusion reactor project became a national lab director in 2022.",
        run_id,
    )
    _mock_judge_response(
        monkeypatch,
        {
            "verdict": "GROUNDED",
            "rationale": "Chunk 0 identifies Alice Chen as project lead; chunk 1 confirms her later directorship.",
            "evidence": [
                {"chunk_index": 0, "span_start": 3, "span_end": 13, "stance": "support"},
                {"chunk_index": 1, "span_start": 0, "span_end": 10, "stance": "support"},
            ],
        },
    )

    verifier = EntailmentVerifier()
    verification = await verifier.verify(claim, _ctx(run_id, [chunk_a, chunk_b]))

    assert verification.verdict == Verdict.GROUNDED
    assert len(verification.evidence) == 2
    assert verification.evidence[0].chunk_id == chunk_a.id
    assert verification.evidence[1].chunk_id == chunk_b.id


async def test_empty_retrieved_chunks_returns_unsupported_without_llm_call(monkeypatch):
    calls: list[str] = []

    async def fake_call(prompt: str, model: str) -> tuple[str, int, int]:
        calls.append(prompt)
        return json.dumps({"verdict": "GROUNDED", "rationale": "x", "evidence": []}), 1, 1

    monkeypatch.setattr(llm, "_PROVIDER_CHAIN", [("anthropic", fake_call)])
    run_id = uuid4()
    claim = _claim("Some claim.", run_id)

    verifier = EntailmentVerifier()
    verification = await verifier.verify(claim, _ctx(run_id, []))

    assert verification.verdict == Verdict.UNSUPPORTED
    assert calls == []  # short-circuited, no judge call spent on empty context


async def test_invalid_evidence_entries_are_dropped_but_verdict_kept(monkeypatch):
    run_id = uuid4()
    chunk = _chunk("Short chunk text.", run_id)
    claim = _claim("Some claim.", run_id)
    _mock_judge_response(
        monkeypatch,
        {
            "verdict": "GROUNDED",
            "rationale": "ok",
            "evidence": [
                {"chunk_index": 5, "span_start": 0, "span_end": 5, "stance": "support"},  # bad index
                {"chunk_index": 0, "span_start": 10, "span_end": 3, "stance": "support"},  # bad span
                {"chunk_index": 0, "span_start": 0, "span_end": 5, "stance": "support"},  # valid
            ],
        },
    )

    verifier = EntailmentVerifier()
    verification = await verifier.verify(claim, _ctx(run_id, [chunk]))

    assert verification.verdict == Verdict.GROUNDED
    assert len(verification.evidence) == 1
    assert verification.evidence[0].quote_span == (0, 5)


async def test_llm_failure_degrades_to_unsupported(monkeypatch):
    async def fail_call(prompt: str, model: str) -> tuple[str, int, int]:
        raise RuntimeError("rate limited")

    monkeypatch.setattr(llm, "_PROVIDER_CHAIN", [("anthropic", fail_call)])
    run_id = uuid4()
    claim = _claim("Some claim.", run_id)
    chunk = _chunk("Some chunk.", run_id)

    verifier = EntailmentVerifier()
    verification = await verifier.verify(claim, _ctx(run_id, [chunk]))

    assert verification.verdict == Verdict.UNSUPPORTED
    assert "degraded" in verification.rationale.lower()


async def test_unparsable_judge_output_degrades_to_unsupported(monkeypatch):
    async def fake_call(prompt: str, model: str) -> tuple[str, int, int]:
        return "not json, sorry, and no schema fields at all", 10, 5

    monkeypatch.setattr(llm, "_PROVIDER_CHAIN", [("anthropic", fake_call)])
    run_id = uuid4()
    claim = _claim("Some claim.", run_id)
    chunk = _chunk("Some chunk.", run_id)

    verifier = EntailmentVerifier()
    verification = await verifier.verify(claim, _ctx(run_id, [chunk]))

    assert verification.verdict == Verdict.UNSUPPORTED


async def test_model_attempting_fragile_or_stale_is_rejected(monkeypatch):
    # FRAGILE/STALE are reserved for the prober/independent verifier. The
    # private response schema's Literal type rejects them at parse time, so
    # this must degrade rather than ever return FRAGILE/STALE from entailment.
    async def fake_call(prompt: str, model: str) -> tuple[str, int, int]:
        return json.dumps({"verdict": "FRAGILE", "rationale": "x", "evidence": []}), 10, 5

    monkeypatch.setattr(llm, "_PROVIDER_CHAIN", [("anthropic", fake_call)])
    run_id = uuid4()
    claim = _claim("Some claim.", run_id)
    chunk = _chunk("Some chunk.", run_id)

    verifier = EntailmentVerifier()
    verification = await verifier.verify(claim, _ctx(run_id, [chunk]))

    assert verification.verdict in (Verdict.UNSUPPORTED,)
    assert verification.verdict not in (Verdict.FRAGILE, Verdict.STALE)


def test_entailment_verifier_satisfies_protocol():
    assert isinstance(EntailmentVerifier(), VerifierProtocol)
