"""Tests for attest.verifiers.prober.AdversarialProber.

The injected entailment verifier is always a small fake/stub implementing
VerifierProtocol (attest.models.VerifierProtocol) — never a concrete
attest.verifiers.entailment.EntailmentVerifier, per this component's design:
the prober depends on the *interface*, not A1's implementation, which may
not exist yet.

Mutation generation still goes through attest.llm.complete under the hood,
so the provider chain is mocked the same way tests/test_contracts.py does
(monkeypatch attest.llm._PROVIDER_CHAIN itself). No network, no real LLM
calls.
"""

import json
import re
from uuid import uuid4

from attest import llm
from attest.models import (
    Chunk,
    Claim,
    Verdict,
    Verification,
    VerifierProtocol,
    VerifyContext,
)
from attest.verifiers.prober import AdversarialProber

RUN_ID = uuid4()


def _claim(text: str) -> Claim:
    return Claim(id=uuid4(), run_id=RUN_ID, claim_index=0, text=text)


def _ctx(chunk_text: str) -> VerifyContext:
    return VerifyContext(
        run_id=RUN_ID,
        query="q",
        answer="a",
        retrieved_chunks=[
            Chunk(id=uuid4(), run_id=RUN_ID, chunk_index=0, text=chunk_text)
        ],
    )


def _mock_mutation_llm(monkeypatch) -> None:
    """Deterministic, distinguishable mutated text per mutation kind, driven
    off the prompt content (mirrors tests/test_mutations.py's mocking
    pattern) so the fake entailment verifiers below can key off the text.
    """

    async def fake_call(prompt: str, model: str) -> tuple[str, int, int]:
        lowered = prompt.lower()
        if "negate" in lowered:
            payload = {
                "applicable": True,
                "mutated_text": "Paris is not the capital of France.",
                "reason": None,
            }
        elif "entity" in lowered:
            payload = {
                "applicable": True,
                "mutated_text": "Lyon is the capital of France.",
                "reason": None,
            }
        elif "quantifier" in lowered:
            payload = {
                "applicable": True,
                "mutated_text": "France has 60 capitals.",
                "reason": None,
            }
        else:
            payload = {"applicable": False, "mutated_text": None, "reason": "unrecognized"}
        return json.dumps(payload), 10, 5

    monkeypatch.setattr(llm, "_PROVIDER_CHAIN", [("anthropic", fake_call)])


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class LazyEntailment:
    """Returns GROUNDED unconditionally — never actually reads the claim
    text or context. The prober should catch this as maximally FRAGILE.
    """

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def verify(self, claim: Claim, ctx: VerifyContext) -> Verification:
        self.calls.append(claim.text)
        return Verification(
            id=uuid4(), claim_id=claim.id, verifier="entailment", verdict=Verdict.GROUNDED
        )


_NEGATION_RE = re.compile(r"\bnot\b|n't\b", re.IGNORECASE)


class GoodEntailment:
    """A simple but honest heuristic verifier: GROUNDED only if the claim
    text appears verbatim in the retrieved context, CONTRADICTED if it looks
    negated, else UNSUPPORTED. Good enough to prove a verifier that actually
    reads the context produces no false FRAGILE.
    """

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def verify(self, claim: Claim, ctx: VerifyContext) -> Verification:
        self.calls.append(claim.text)
        context_text = "\n".join(chunk.text for chunk in ctx.retrieved_chunks)
        if claim.text.strip() in context_text:
            verdict = Verdict.GROUNDED
        elif _NEGATION_RE.search(claim.text):
            verdict = Verdict.CONTRADICTED
        else:
            verdict = Verdict.UNSUPPORTED
        return Verification(id=uuid4(), claim_id=claim.id, verifier="entailment", verdict=verdict)


class StaticVerdictEntailment:
    """Always returns a fixed non-GROUNDED verdict, ignoring text."""

    def __init__(self, verdict: Verdict) -> None:
        self._verdict = verdict

    async def verify(self, claim: Claim, ctx: VerifyContext) -> Verification:
        return Verification(id=uuid4(), claim_id=claim.id, verifier="entailment", verdict=self._verdict)


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_prober_satisfies_verifier_protocol():
    prober = AdversarialProber(entailment_verifier=LazyEntailment())
    assert isinstance(prober, VerifierProtocol)


# ---------------------------------------------------------------------------
# Lazy verifier -> FRAGILE
# ---------------------------------------------------------------------------


async def test_lazy_verifier_produces_fragile(monkeypatch):
    _mock_mutation_llm(monkeypatch)
    lazy = LazyEntailment()
    prober = AdversarialProber(entailment_verifier=lazy)
    claim = _claim("Paris is the capital of France.")
    ctx = _ctx("Paris is the capital of France.")

    verification, probes = await prober.verify_with_probes(claim, ctx)

    assert verification.verifier == "prober"
    assert verification.verdict == Verdict.FRAGILE
    assert len(probes) >= 1
    assert all(not probe.flipped for probe in probes)
    assert all(probe.claim_id == claim.id for probe in probes)
    # rationale should call out fragility
    assert "did not appear to re-read" in verification.rationale

    # verify() (the plain VerifierProtocol entrypoint) agrees, and populates
    # last_probes as the documented side-effect exposure mechanism.
    verification_via_protocol = await prober.verify(claim, ctx)
    assert verification_via_protocol.verdict == Verdict.FRAGILE
    assert prober.last_probes  # side-effect exposure populated


# ---------------------------------------------------------------------------
# Good verifier -> no false FRAGILE
# ---------------------------------------------------------------------------


async def test_good_verifier_produces_no_false_fragile(monkeypatch):
    _mock_mutation_llm(monkeypatch)
    good = GoodEntailment()
    prober = AdversarialProber(entailment_verifier=good)
    claim = _claim("Paris is the capital of France.")
    ctx = _ctx("Paris is the capital of France.")

    verification, probes = await prober.verify_with_probes(claim, ctx)

    assert verification.verdict == Verdict.GROUNDED  # baseline passes through, no FRAGILE
    assert len(probes) >= 1
    assert all(probe.flipped for probe in probes)
    assert "no fragility detected" in verification.rationale


# ---------------------------------------------------------------------------
# No applicable mutations at all
# ---------------------------------------------------------------------------


async def test_no_applicable_mutations_defaults_fragility_zero(monkeypatch):
    async def fake_call(prompt: str, model: str) -> tuple[str, int, int]:
        return json.dumps({"applicable": False, "mutated_text": None, "reason": "n/a"}), 5, 5

    monkeypatch.setattr(llm, "_PROVIDER_CHAIN", [("anthropic", fake_call)])

    lazy = LazyEntailment()
    prober = AdversarialProber(entailment_verifier=lazy)
    claim = _claim("Paris is the capital of France.")
    ctx = _ctx("Paris is the capital of France.")

    verification, probes = await prober.verify_with_probes(claim, ctx)

    assert probes == []
    assert verification.verdict == Verdict.GROUNDED  # baseline unchanged, can't assess fragility
    assert "cannot be assessed" in verification.rationale


async def test_quantifier_shift_skipped_when_no_quantifier_in_claim(monkeypatch):
    """Claim with no number/date/quantifier: quantifier_shift must be
    skipped rather than fabricated, even inside the full prober flow.
    """
    _mock_mutation_llm(monkeypatch)
    lazy = LazyEntailment()
    prober = AdversarialProber(entailment_verifier=lazy)
    claim = _claim("Paris is the capital of France.")  # no quantifier/number
    ctx = _ctx("Paris is the capital of France.")

    _, probes = await prober.verify_with_probes(claim, ctx)

    mutation_kinds = {probe.mutation_type.value for probe in probes}
    assert "quantifier_shift" not in mutation_kinds
    assert {"negation", "entity_swap"} == mutation_kinds


# ---------------------------------------------------------------------------
# Non-GROUNDED baselines pass through unchanged
# ---------------------------------------------------------------------------


async def test_non_grounded_baseline_passes_through_unchanged(monkeypatch):
    _mock_mutation_llm(monkeypatch)
    static = StaticVerdictEntailment(Verdict.UNSUPPORTED)
    prober = AdversarialProber(entailment_verifier=static)
    claim = _claim("Paris is the capital of France.")
    ctx = _ctx("some unrelated context")

    verification, probes = await prober.verify_with_probes(claim, ctx)

    # Every probe also reports UNSUPPORTED (static verifier), so nothing
    # "flips" — but since baseline isn't GROUNDED, verdict must stay
    # UNSUPPORTED, not become FRAGILE.
    assert verification.verdict == Verdict.UNSUPPORTED
    assert len(probes) >= 1
    assert all(not probe.flipped for probe in probes)


# ---------------------------------------------------------------------------
# Baseline uses direct injection, not ctx.prior_entailment
# ---------------------------------------------------------------------------


async def test_baseline_ignores_prior_entailment_field(monkeypatch):
    """The prober must always re-derive its own baseline via the injected
    verifier, never read ctx.prior_entailment (that field is reserved for
    the independent verifier only).
    """
    _mock_mutation_llm(monkeypatch)
    lazy = LazyEntailment()
    prober = AdversarialProber(entailment_verifier=lazy)
    claim = _claim("Paris is the capital of France.")
    ctx = _ctx("Paris is the capital of France.").model_copy(
        update={"prior_entailment": Verdict.CONTRADICTED}
    )

    verification, _ = await prober.verify_with_probes(claim, ctx)

    # LazyEntailment always returns GROUNDED regardless of input, and the
    # prober must have actually called it for the baseline (not trusted
    # ctx.prior_entailment=CONTRADICTED) — fragility logic proves that: if
    # the prober had used CONTRADICTED as baseline it would pass through
    # unchanged as CONTRADICTED, not FRAGILE.
    assert verification.verdict == Verdict.FRAGILE
    assert claim.text in lazy.calls  # baseline call happened with original text


# ---------------------------------------------------------------------------
# Cost/latency aggregation
# ---------------------------------------------------------------------------


async def test_latency_and_cost_are_summed_from_entailment_calls(monkeypatch):
    _mock_mutation_llm(monkeypatch)

    class PricedEntailment:
        async def verify(self, claim: Claim, ctx: VerifyContext) -> Verification:
            return Verification(
                id=uuid4(),
                claim_id=claim.id,
                verifier="entailment",
                verdict=Verdict.GROUNDED,
                latency_ms=100,
                cost_usd=0.001,
            )

    prober = AdversarialProber(entailment_verifier=PricedEntailment())
    claim = _claim("Paris is the capital of France.")
    ctx = _ctx("Paris is the capital of France.")

    verification, probes = await prober.verify_with_probes(claim, ctx)

    n_calls = 1 + len(probes)  # baseline + one per probe
    assert verification.latency_ms == 100 * n_calls
    assert round(verification.cost_usd, 6) == round(0.001 * n_calls, 6)


# ---------------------------------------------------------------------------
# Expected flip targets (regression: any verdict change used to count as a
# "correct" flip, so a verifier that degrades to UNVERIFIABLE on every mutation
# scored as perfectly healthy and hollowed out the FRAGILE signal)
# ---------------------------------------------------------------------------


class DegradingEntailment:
    """GROUNDED on the original claim, UNVERIFIABLE on anything mutated.

    This is the failure mode the expected-flip-target check exists to catch: the
    verdict *changes* under every mutation, but it changes to "can't tell",
    which is not evidence that the verifier re-read the context.
    """

    def __init__(self, original: str) -> None:
        self._original = original

    async def verify(self, claim: Claim, ctx: VerifyContext) -> Verification:
        verdict = Verdict.GROUNDED if claim.text.strip() == self._original else Verdict.UNVERIFIABLE
        return Verification(id=uuid4(), claim_id=claim.id, verifier="entailment", verdict=verdict)


async def test_change_to_unverifiable_does_not_count_as_a_flip(monkeypatch):
    _mock_mutation_llm(monkeypatch)
    text = "Paris is the capital of France."
    prober = AdversarialProber(entailment_verifier=DegradingEntailment(text))

    verification, probes = await prober.verify_with_probes(_claim(text), _ctx(text))

    assert probes, "expected at least one probe"
    # Verdict changed on every probe, but never to a verdict the mutation
    # should produce -> not a real flip -> still FRAGILE.
    assert all(probe.observed_verdict == Verdict.UNVERIFIABLE for probe in probes)
    assert all(not probe.flipped for probe in probes)
    assert verification.verdict == Verdict.FRAGILE


async def test_expected_flip_targets_cover_every_mutation_type():
    """A new MutationType must not silently fall back to 'no target'."""
    from attest.models import MutationType
    from attest.verifiers.prober import EXPECTED_FLIP_TARGETS

    assert set(EXPECTED_FLIP_TARGETS) == set(MutationType)
    for targets in EXPECTED_FLIP_TARGETS.values():
        assert Verdict.UNVERIFIABLE not in targets
        assert Verdict.GROUNDED not in targets
