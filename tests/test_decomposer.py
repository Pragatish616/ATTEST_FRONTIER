"""Tests for attest.verifiers.decomposer.decompose.

Mocked LLM only (monkeypatch attest.llm._PROVIDER_CHAIN directly — see
tests/test_contracts.py for the established pattern and PROGRESS.md for why
patching attest.llm._call_anthropic by name would not work). No network.
"""

import json
from uuid import uuid4

from attest import llm
from attest.models import Verdict
from attest.verifiers import decomposer
from attest.verifiers.decomposer import decompose
from attest.verifiers.entailment import EntailmentVerifier


def _mock_decompose_response(monkeypatch, payload: dict) -> list[str]:
    """Patch the provider chain to return `payload` as the decomposer's JSON
    response. Returns the list of prompts seen (for call-count assertions).
    """
    prompts_seen: list[str] = []

    async def fake_call(prompt: str, model: str) -> tuple[str, int, int]:
        prompts_seen.append(prompt)
        return json.dumps(payload), 50, 30

    monkeypatch.setattr(llm, "_PROVIDER_CHAIN", [("anthropic", fake_call)])
    return prompts_seen


async def test_decompose_basic_multi_claim(monkeypatch):
    answer = "Paris is the capital of France. It has a population of about 2.1 million."
    payload = {
        "claims": [
            {
                "text": "Paris is the capital of France.",
                "span_start": 0,
                "span_end": 32,
                "is_unverifiable": False,
                "reason": None,
            },
            {
                "text": "Paris has a population of about 2.1 million.",
                "span_start": 33,
                "span_end": 76,
                "is_unverifiable": False,
                "reason": None,
            },
        ]
    }
    _mock_decompose_response(monkeypatch, payload)
    run_id = uuid4()

    claims = await decompose(answer, "Tell me about Paris.", run_id=run_id)

    assert len(claims) == 2
    assert [c.claim_index for c in claims] == [0, 1]
    for claim in claims:
        assert claim.run_id == run_id
        assert claim.verdict is None
        assert answer[claim.span_start : claim.span_end]  # non-empty, in bounds


async def test_decompose_span_correctness_exact_offsets(monkeypatch):
    answer = "The Eiffel Tower was completed in 1889. It is located in Paris."
    first_sentence = "The Eiffel Tower was completed in 1889."
    second_sentence = "It is located in Paris."
    start1 = answer.index(first_sentence)
    end1 = start1 + len(first_sentence)
    start2 = answer.index(second_sentence)
    end2 = start2 + len(second_sentence)
    payload = {
        "claims": [
            {
                "text": first_sentence,
                "span_start": start1,
                "span_end": end1,
                "is_unverifiable": False,
                "reason": None,
            },
            {
                "text": "The Eiffel Tower is located in Paris.",
                "span_start": start2,
                "span_end": end2,
                "is_unverifiable": False,
                "reason": None,
            },
        ]
    }
    _mock_decompose_response(monkeypatch, payload)

    claims = await decompose(answer, "Tell me about the Eiffel Tower.", run_id=uuid4())

    assert len(claims) == 2
    assert answer[claims[0].span_start : claims[0].span_end] == first_sentence
    assert answer[claims[1].span_start : claims[1].span_end] == second_sentence


async def test_decompose_unverifiable_claim_not_sent_to_entailment(monkeypatch):
    answer = "The library opens at 9am. This is clearly the best library in the city."
    payload = {
        "claims": [
            {
                "text": "The library opens at 9am.",
                "span_start": 0,
                "span_end": 26,
                "is_unverifiable": False,
                "reason": None,
            },
            {
                "text": "This is clearly the best library in the city.",
                "span_start": 27,
                "span_end": 73,
                "is_unverifiable": True,
                "reason": "Subjective opinion, not checkable against evidence.",
            },
        ]
    }
    _mock_decompose_response(monkeypatch, payload)

    claims = await decompose(answer, "Tell me about the library.", run_id=uuid4())

    assert len(claims) == 2
    verifiable = [c for c in claims if c.verdict is None]
    unverifiable = [c for c in claims if c.verdict is not None]
    assert len(verifiable) == 1
    assert len(unverifiable) == 1
    assert unverifiable[0].verdict == Verdict.UNVERIFIABLE
    assert unverifiable[0].rationale

    # Simulate the orchestrator: only claims with verdict is None go to a verifier.
    entailment_calls: list[str] = []

    async def fake_entailment_call(prompt: str, model: str) -> tuple[str, int, int]:
        entailment_calls.append(prompt)
        return (
            json.dumps({"verdict": "GROUNDED", "rationale": "matches", "evidence": []}),
            10,
            5,
        )

    monkeypatch.setattr(llm, "_PROVIDER_CHAIN", [("anthropic", fake_entailment_call)])
    from attest.models import Chunk, VerifyContext

    ctx = VerifyContext(
        run_id=verifiable[0].run_id,
        query="Tell me about the library.",
        answer=answer,
        retrieved_chunks=[
            Chunk(
                id=uuid4(),
                run_id=verifiable[0].run_id,
                chunk_index=0,
                text="The library opens at 9am on weekdays.",
            )
        ],
    )
    verifier = EntailmentVerifier()
    for claim in claims:
        if claim.verdict is None:
            await verifier.verify(claim, ctx)

    assert len(entailment_calls) == 1  # only the verifiable claim triggered a judge call


async def test_decompose_unanchored_claim_skipped_without_fallback(monkeypatch):
    answer = "Short answer."
    payload = {
        "claims": [
            {
                "text": "Completely unrelated fabricated claim about spacecraft propulsion systems.",
                "span_start": 999,
                "span_end": 1050,
                "is_unverifiable": False,
                "reason": None,
            }
        ]
    }
    _mock_decompose_response(monkeypatch, payload)

    claims = await decompose(answer, "q", run_id=uuid4())

    assert claims == []


async def test_decompose_recovers_via_fallback_span(monkeypatch):
    answer = "The mitochondria is the powerhouse of the cell."
    payload = {
        "claims": [
            {
                "text": "Mitochondria produce energy for the cell.",
                # Deliberately bad offsets (out of range) — should trigger fallback search.
                "span_start": 500,
                "span_end": 600,
                "is_unverifiable": False,
                "reason": None,
            }
        ]
    }
    _mock_decompose_response(monkeypatch, payload)

    claims = await decompose(answer, "q", run_id=uuid4())

    assert len(claims) == 1
    span_text = answer[claims[0].span_start : claims[0].span_end]
    assert span_text.strip()
    assert "mitochondria" in span_text.lower()


async def test_decompose_llm_failure_returns_empty_list(monkeypatch):
    async def fail_call(prompt: str, model: str) -> tuple[str, int, int]:
        raise RuntimeError("provider down")

    monkeypatch.setattr(llm, "_PROVIDER_CHAIN", [("anthropic", fail_call)])

    claims = await decompose("Some answer.", "q", run_id=uuid4())

    assert claims == []


async def test_decompose_parse_failure_returns_empty_list(monkeypatch):
    async def fake_call(prompt: str, model: str) -> tuple[str, int, int]:
        return "this is not json and never will be", 10, 5

    monkeypatch.setattr(llm, "_PROVIDER_CHAIN", [("anthropic", fake_call)])

    claims = await decompose("Some answer.", "q", run_id=uuid4())

    assert claims == []


async def test_decompose_claim_indices_stay_contiguous_after_skip(monkeypatch):
    answer = "Fact one is true. Fact two is true."
    payload = {
        "claims": [
            {
                "text": "Fact one is true.",
                "span_start": 0,
                "span_end": 18,
                "is_unverifiable": False,
                "reason": None,
            },
            {
                "text": "Unanchorable fabricated nonsense about quasars.",
                "span_start": 9999,
                "span_end": 10050,
                "is_unverifiable": False,
                "reason": None,
            },
            {
                "text": "Fact two is true.",
                "span_start": 19,
                "span_end": 36,
                "is_unverifiable": False,
                "reason": None,
            },
        ]
    }
    _mock_decompose_response(monkeypatch, payload)

    claims = await decompose(answer, "q", run_id=uuid4())

    assert len(claims) == 2
    assert [c.claim_index for c in claims] == [0, 1]
    assert claims[0].text == "Fact one is true."
    assert claims[1].text == "Fact two is true."


def test_span_is_valid_rejects_out_of_bounds():
    assert decomposer._span_is_valid("short", 0, 100, "short") is False
    assert decomposer._span_is_valid("short", 3, 1, "short") is False


def test_span_is_valid_accepts_overlapping_subject():
    answer = "The mitochondria is the powerhouse of the cell."
    assert decomposer._span_is_valid(answer, 4, 16, "Mitochondria produce energy.") is True
