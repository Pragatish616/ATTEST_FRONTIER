"""Resolves three independent `Verification` opinions into one final verdict.

This is the crux of ATTEST (PLAN.md §11, "who verifies the verifier?"):
disagreement between verifiers is a first-class signal to surface, not
something to average away. See `reconcile()` for the precedence rule and
`tests/test_reconciler.py` for the exhaustive precedence-table tests.
"""

from __future__ import annotations

from attest.models import Verdict, Verification

# Precedence when verdicts disagree (PLAN.md §3 + A4 brief). The final
# verdict is whichever verifier's opinion ranks highest here, regardless of
# how many verifiers agree with it — a single CONTRADICTED outranks two
# GROUNDED opinions, on purpose: precision over consensus.
_PRECEDENCE: tuple[Verdict, ...] = (
    Verdict.CONTRADICTED,
    Verdict.STALE,
    Verdict.UNSUPPORTED,
    Verdict.FRAGILE,
    Verdict.GROUNDED,
    Verdict.UNVERIFIABLE,
)
_RANK: dict[Verdict, int] = {verdict: index for index, verdict in enumerate(_PRECEDENCE)}


def reconcile(
    entailment: Verification | None,
    prober: Verification | None,
    independent: Verification | None,
) -> tuple[Verdict, float, float, str]:
    """Reconcile up to three independent `Verification`s for one claim.

    Any of the three may be `None` — a verifier can be absent because
    `config.enable_prober` / `config.enable_independent` was `False`, because
    the run's budget was exhausted before it could be launched, or because it
    raised and `asyncio.gather(..., return_exceptions=True)` in the
    orchestrator turned that into `None` rather than failing the run.

    Returns `(verdict, confidence, disagreement, rationale)`:
      - `verdict`: the highest-precedence verdict among whichever verifiers
        ran (see `_PRECEDENCE`). `UNVERIFIABLE` if none ran at all.
      - `confidence`: fraction of the verifiers that ran which agree with
        the final verdict (0..1). 1.0 when only one verifier ran — there is
        nothing for it to disagree with.
      - `disagreement`: normalized pairwise divergence across whichever
        verifiers ran — `(distinct_verdicts - 1) / (ran - 1)`, so 0.0 when
        they all agree (or only one ran) and 1.0 when every verifier that
        ran produced a different verdict. This is reported as-is, not
        folded into `confidence` — surfacing disagreement is the point.
      - `rationale`: one plain sentence naming which verifier drove the
        final verdict, for display when a judge clicks a claim.
    """
    ran = [v for v in (entailment, prober, independent) if v is not None]

    if not ran:
        return (
            Verdict.UNVERIFIABLE,
            0.0,
            0.0,
            "No verifier produced a result for this claim; defaulting to UNVERIFIABLE.",
        )

    driver = min(ran, key=lambda v: _RANK[v.verdict])
    final_verdict = driver.verdict

    agreeing = [v for v in ran if v.verdict == final_verdict]
    confidence = round(len(agreeing) / len(ran), 4)

    distinct_verdicts = {v.verdict for v in ran}
    disagreement = (
        round((len(distinct_verdicts) - 1) / (len(ran) - 1), 4) if len(ran) > 1 else 0.0
    )

    others = [v for v in ran if v is not driver]
    if others:
        other_desc = ", ".join(f"{v.verifier}={v.verdict.value}" for v in others)
        rationale = (
            f"{driver.verifier} verifier drove the verdict ({final_verdict.value}) "
            f"by precedence; other verifier(s) reported {other_desc}."
        )
    else:
        rationale = (
            f"{driver.verifier} verifier was the only signal available: {final_verdict.value}."
        )

    return final_verdict, confidence, disagreement, rationale


__all__ = ["reconcile"]
