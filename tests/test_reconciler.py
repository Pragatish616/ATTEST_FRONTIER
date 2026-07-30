"""Reconciler precedence table — the crux of the whole system (CLAUDE.md).

Exercises `reconcile()` directly with plain `Verification` inputs, no
orchestrator involved, per the A4 brief.
"""

from uuid import uuid4

import pytest

from attest.models import Verdict, Verification, VerifierName
from attest.reconciler import reconcile


def _verification(verifier: VerifierName, verdict: Verdict) -> Verification:
    return Verification(id=uuid4(), claim_id=uuid4(), verifier=verifier, verdict=verdict)


# ---------------------------------------------------------------------------
# Precedence: CONTRADICTED > STALE > UNSUPPORTED > FRAGILE > GROUNDED > UNVERIFIABLE
# ---------------------------------------------------------------------------

_PRECEDENCE_ORDER = [
    Verdict.CONTRADICTED,
    Verdict.STALE,
    Verdict.UNSUPPORTED,
    Verdict.FRAGILE,
    Verdict.GROUNDED,
    Verdict.UNVERIFIABLE,
]


@pytest.mark.parametrize(
    ("higher", "lower"),
    [
        (a, b)
        for i, a in enumerate(_PRECEDENCE_ORDER)
        for b in _PRECEDENCE_ORDER[i + 1 :]
    ],
)
def test_precedence_pairwise(higher, lower):
    """For every pair in the precedence table, whichever verifier reports the
    higher-precedence verdict wins, regardless of which slot (entailment /
    prober / independent) it came from.
    """
    entailment = _verification("entailment", lower)
    prober = _verification("prober", higher)
    independent = _verification("independent", lower)

    verdict, _confidence, _disagreement, rationale = reconcile(entailment, prober, independent)

    assert verdict == higher
    assert "prober" in rationale


def test_all_agree_full_confidence_zero_disagreement():
    entailment = _verification("entailment", Verdict.GROUNDED)
    prober = _verification("prober", Verdict.GROUNDED)
    independent = _verification("independent", Verdict.GROUNDED)

    verdict, confidence, disagreement, rationale = reconcile(entailment, prober, independent)

    assert verdict == Verdict.GROUNDED
    assert confidence == 1.0
    assert disagreement == 0.0
    assert "entailment" in rationale or "GROUNDED" in rationale


def test_contradicted_beats_everything_even_as_minority():
    entailment = _verification("entailment", Verdict.GROUNDED)
    prober = _verification("prober", Verdict.GROUNDED)
    independent = _verification("independent", Verdict.CONTRADICTED)

    verdict, confidence, disagreement, rationale = reconcile(entailment, prober, independent)

    assert verdict == Verdict.CONTRADICTED
    assert confidence == pytest.approx(1 / 3, abs=1e-4)
    assert disagreement == pytest.approx(0.5)  # 2 distinct verdicts among 3 -> (2-1)/(3-1)
    assert "independent" in rationale


def test_stale_requires_prior_entailment_context_but_reconciler_is_mechanical():
    """The reconciler applies the precedence table mechanically — it doesn't
    re-derive STALE's semantic precondition (entailed-by-context) itself;
    that's the independent verifier's job when it sets its own verdict,
    using ctx.prior_entailment (CONTRACT_CHANGE_REQUEST.md).
    """
    entailment = _verification("entailment", Verdict.GROUNDED)
    independent = _verification("independent", Verdict.STALE)

    verdict, _confidence, _disagreement, rationale = reconcile(entailment, None, independent)

    assert verdict == Verdict.STALE
    assert "independent" in rationale


def test_fragile_beats_grounded():
    entailment = _verification("entailment", Verdict.GROUNDED)
    prober = _verification("prober", Verdict.FRAGILE)

    verdict, confidence, disagreement, rationale = reconcile(entailment, prober, None)

    assert verdict == Verdict.FRAGILE
    assert confidence == 0.5
    assert disagreement == 1.0  # 2 distinct verdicts among 2 ran -> (2-1)/(2-1)
    assert "prober" in rationale


# ---------------------------------------------------------------------------
# Missing verifiers (None) — budget cutoff, disabled config, or a verifier
# that raised and was turned into None upstream by the orchestrator.
# ---------------------------------------------------------------------------


def test_only_entailment_ran_confidence_is_full_disagreement_zero():
    entailment = _verification("entailment", Verdict.UNSUPPORTED)

    verdict, confidence, disagreement, rationale = reconcile(entailment, None, None)

    assert verdict == Verdict.UNSUPPORTED
    assert confidence == 1.0
    assert disagreement == 0.0
    assert "entailment" in rationale
    assert "only signal" in rationale


def test_no_verifier_ran_defaults_to_unverifiable():
    verdict, confidence, disagreement, rationale = reconcile(None, None, None)

    assert verdict == Verdict.UNVERIFIABLE
    assert confidence == 0.0
    assert disagreement == 0.0
    assert "No verifier" in rationale


def test_two_verifiers_disagree_full_disagreement():
    entailment = _verification("entailment", Verdict.GROUNDED)
    independent = _verification("independent", Verdict.UNSUPPORTED)

    verdict, confidence, disagreement, rationale = reconcile(entailment, None, independent)

    assert verdict == Verdict.UNSUPPORTED
    assert disagreement == 1.0
    assert confidence == 0.5


def test_three_verifiers_two_distinct_verdicts_partial_disagreement():
    entailment = _verification("entailment", Verdict.GROUNDED)
    prober = _verification("prober", Verdict.GROUNDED)
    independent = _verification("independent", Verdict.UNSUPPORTED)

    verdict, confidence, disagreement, rationale = reconcile(entailment, prober, independent)

    assert verdict == Verdict.UNSUPPORTED  # UNSUPPORTED outranks GROUNDED
    assert confidence == pytest.approx(1 / 3, abs=1e-4)
    assert disagreement == pytest.approx(0.5)
