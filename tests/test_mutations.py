"""Tests for attest.verifiers.mutations.

Mocked LLM only (monkeypatch attest.llm._PROVIDER_CHAIN directly — see
tests/test_contracts.py for the established pattern). No network.
"""

import json

from attest import llm
from attest.models import MutationType
from attest.verifiers.mutations import (
    build_entity_swap,
    build_negation,
    build_quantifier_shift,
    generate_mutations,
)


def _mock_provider(monkeypatch, responder):
    """Patch the provider chain so `responder(prompt) -> dict` decides the
    JSON payload returned for each call. Returns the list of prompts seen.
    """
    prompts_seen: list[str] = []

    async def fake_call(prompt: str, model: str) -> tuple[str, int, int]:
        prompts_seen.append(prompt)
        return json.dumps(responder(prompt)), 20, 10

    monkeypatch.setattr(llm, "_PROVIDER_CHAIN", [("anthropic", fake_call)])
    return prompts_seen


# ---------------------------------------------------------------------------
# negation
# ---------------------------------------------------------------------------


async def test_build_negation_applicable(monkeypatch):
    def responder(prompt: str) -> dict:
        assert "negate" in prompt.lower()
        return {
            "applicable": True,
            "mutated_text": "Paris is not the capital of France.",
            "reason": None,
        }

    _mock_provider(monkeypatch, responder)

    mutation = await build_negation("Paris is the capital of France.")

    assert mutation is not None
    assert mutation.mutation_type == MutationType.NEGATION
    assert mutation.expected_flip is True
    assert mutation.text == "Paris is not the capital of France."


async def test_build_negation_llm_reports_not_applicable(monkeypatch):
    def responder(prompt: str) -> dict:
        return {"applicable": False, "mutated_text": None, "reason": "not a well-formed assertion"}

    _mock_provider(monkeypatch, responder)

    mutation = await build_negation("Please.")

    assert mutation is None


async def test_build_negation_identical_text_is_treated_as_inapplicable(monkeypatch):
    def responder(prompt: str) -> dict:
        return {
            "applicable": True,
            "mutated_text": "Paris is the capital of France.",
            "reason": None,
        }

    _mock_provider(monkeypatch, responder)

    mutation = await build_negation("Paris is the capital of France.")

    assert mutation is None


async def test_build_negation_unparseable_response_is_skipped(monkeypatch):
    async def fake_call(prompt: str, model: str) -> tuple[str, int, int]:
        return "not json at all", 5, 5

    monkeypatch.setattr(llm, "_PROVIDER_CHAIN", [("anthropic", fake_call)])

    mutation = await build_negation("Paris is the capital of France.")

    assert mutation is None


async def test_build_negation_llm_failure_returns_none(monkeypatch):
    async def fail_call(prompt: str, model: str) -> tuple[str, int, int]:
        raise RuntimeError("provider down")

    monkeypatch.setattr(llm, "_PROVIDER_CHAIN", [("anthropic", fail_call)])

    mutation = await build_negation("Paris is the capital of France.")

    assert mutation is None


# ---------------------------------------------------------------------------
# entity_swap
# ---------------------------------------------------------------------------


async def test_build_entity_swap_applicable(monkeypatch):
    def responder(prompt: str) -> dict:
        assert "entity" in prompt.lower()
        assert "Paris" in prompt or "France" in prompt  # candidates surfaced in prompt
        return {
            "applicable": True,
            "mutated_text": "Lyon is the capital of France.",
            "reason": None,
        }

    _mock_provider(monkeypatch, responder)

    mutation = await build_entity_swap(
        "Paris is the capital of France.", "Paris is the capital of France, per the atlas."
    )

    assert mutation is not None
    assert mutation.mutation_type == MutationType.ENTITY_SWAP
    assert mutation.expected_flip is True
    assert mutation.text == "Lyon is the capital of France."


async def test_build_entity_swap_no_candidate_skips_without_llm_call(monkeypatch):
    calls = []

    async def fake_call(prompt: str, model: str) -> tuple[str, int, int]:
        calls.append(prompt)
        return json.dumps({"applicable": True, "mutated_text": "x", "reason": None}), 5, 5

    monkeypatch.setattr(llm, "_PROVIDER_CHAIN", [("anthropic", fake_call)])

    # No capitalized words, numbers, or dates -> nothing to swap.
    mutation = await build_entity_swap("it happened yesterday somehow", "some context")

    assert mutation is None
    assert calls == []  # never called the LLM at all


async def test_build_entity_swap_llm_reports_not_applicable(monkeypatch):
    def responder(prompt: str) -> dict:
        return {
            "applicable": False,
            "mutated_text": None,
            "reason": "no safe decoy absent from context",
        }

    _mock_provider(monkeypatch, responder)

    mutation = await build_entity_swap("Paris is the capital of France.", "context text")

    assert mutation is None


# ---------------------------------------------------------------------------
# quantifier_shift
# ---------------------------------------------------------------------------


async def test_build_quantifier_shift_applicable(monkeypatch):
    def responder(prompt: str) -> dict:
        assert "quantifier" in prompt.lower()
        return {
            "applicable": True,
            "mutated_text": "Sales increased by 60% in 2023.",
            "reason": None,
        }

    _mock_provider(monkeypatch, responder)

    mutation = await build_quantifier_shift("Sales increased by 40% in 2023.")

    assert mutation is not None
    assert mutation.mutation_type == MutationType.QUANTIFIER_SHIFT
    assert mutation.expected_flip is True
    assert mutation.text == "Sales increased by 60% in 2023."


async def test_build_quantifier_shift_no_quantifier_skips_without_llm_call(monkeypatch):
    """The explicit 'no applicable mutation' case: a claim with no
    quantifier, number, or date must skip quantifier_shift, not fabricate
    one.
    """
    calls = []

    async def fake_call(prompt: str, model: str) -> tuple[str, int, int]:
        calls.append(prompt)
        return json.dumps({"applicable": True, "mutated_text": "x", "reason": None}), 5, 5

    monkeypatch.setattr(llm, "_PROVIDER_CHAIN", [("anthropic", fake_call)])

    mutation = await build_quantifier_shift("Paris is the capital of France.")

    assert mutation is None
    assert calls == []  # structural skip — no LLM call made at all


# ---------------------------------------------------------------------------
# generate_mutations (concurrent orchestration of all three)
# ---------------------------------------------------------------------------


async def test_generate_mutations_returns_only_applicable(monkeypatch):
    def responder(prompt: str) -> dict:
        lowered = prompt.lower()
        if "negate" in lowered:
            return {
                "applicable": True,
                "mutated_text": "Sales did not increase by 40% in 2023.",
                "reason": None,
            }
        if "entity" in lowered:
            return {
                "applicable": True,
                "mutated_text": "Widgets increased by 40% in 2023.",
                "reason": None,
            }
        if "quantifier" in lowered:
            return {
                "applicable": True,
                "mutated_text": "Sales increased by 60% in 2023.",
                "reason": None,
            }
        raise AssertionError(f"unexpected prompt: {prompt}")

    _mock_provider(monkeypatch, responder)

    mutations = await generate_mutations(
        "Sales increased by 40% in 2023.", "Sales increased by 40% in 2023, per the report."
    )

    kinds = {m.mutation_type for m in mutations}
    assert kinds == {
        MutationType.NEGATION,
        MutationType.ENTITY_SWAP,
        MutationType.QUANTIFIER_SHIFT,
    }
    assert len(mutations) == 3


async def test_generate_mutations_skips_quantifier_when_no_quantifier_present(monkeypatch):
    def responder(prompt: str) -> dict:
        lowered = prompt.lower()
        if "negate" in lowered:
            return {
                "applicable": True,
                "mutated_text": "Paris is not the capital of France.",
                "reason": None,
            }
        if "entity" in lowered:
            return {
                "applicable": True,
                "mutated_text": "Lyon is the capital of France.",
                "reason": None,
            }
        # quantifier_shift must never reach here for this claim.
        raise AssertionError(f"unexpected quantifier_shift call: {prompt}")

    _mock_provider(monkeypatch, responder)

    mutations = await generate_mutations(
        "Paris is the capital of France.", "Paris is the capital of France."
    )

    kinds = {m.mutation_type for m in mutations}
    assert kinds == {MutationType.NEGATION, MutationType.ENTITY_SWAP}
    assert MutationType.QUANTIFIER_SHIFT not in kinds
