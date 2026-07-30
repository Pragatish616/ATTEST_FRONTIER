# Owned by A2 (Prober). Not implemented by A0.
#
# Implements VerifierProtocol (attest.models.VerifierProtocol): mutates the
# claim via mutations.py, re-verifies, and flags FRAGILE when the verdict
# fails to flip when it logically must.
"""Adversarial prober — the project's novel contribution.

`AdversarialProber` does not judge entailment itself. It wraps an injected
`VerifierProtocol` (typically A1's `EntailmentVerifier`, but never imported
concretely here — see below) and asks: does that verifier's verdict survive
adversarial mutation of the claim it just judged? A verifier that says
GROUNDED for both a claim and its logical negation wasn't reading the
context; that claim is FRAGILE even though the first pass said "supported."

Why the entailment verifier is injected, not imported:
    A1's `attest.verifiers.entailment` is being written in parallel and may
    not exist with real content at import time. Coupling to it directly
    would also violate independence: the prober treats the entailment
    verifier as an opaque `VerifierProtocol`, gets its own baseline by
    calling it directly (never via `ctx.prior_entailment`, which is reserved
    for the independent verifier per CONTRACT_CHANGE_REQUEST.md), and is
    fully testable against a fake/stub without any real verifier existing.

Probe records and the frozen `Verification` contract:
    `Verification` (attest.models) has no field for individual probe
    attempts — its `evidence` field is `list[Evidence]` (chunk/url + span +
    stance), not shaped like a `Probe` row, and the contract is frozen
    (CLAUDE.md). The `probes` table is a separate persisted artifact the
    orchestrator (A4) needs. Since `VerifierProtocol.verify` cannot be
    widened without breaking the frozen interface, individual `Probe`
    objects are exposed two ways, both backed by the same underlying call:

    1. `verify_with_probes(claim, ctx) -> tuple[Verification, list[Probe]]`
       — the preferred path for any caller (i.e. the orchestrator) that
       knows it's talking to an `AdversarialProber` specifically and wants
       the probes inline, without a second attribute read.
    2. `self.last_probes: list[Probe]` — a side-effect attribute, updated at
       the end of every `verify()` (and `verify_with_probes()`) call, for
       any caller that only holds a `VerifierProtocol` reference and can't
       call the extra method.

    `Probe.claim_id` is always set to the *original* claim's id passed into
    `verify()` — never a synthetic id for the mutated text, since the
    mutation is a vehicle for re-verification, not a new persisted claim.

Cost/latency visibility (CLAUDE.md hard rule: every LLM call is logged):
    The returned `Verification.latency_ms` / `cost_usd` are the sum of the
    baseline entailment call plus every successful mutation re-verify call
    (whatever the injected verifier reports on its own `Verification`
    objects). The mutation-*generation* calls made inside
    `attest.verifiers.mutations` are not included in that sum — see the open
    question in this component's final report to A4.
"""

from __future__ import annotations

import asyncio
from uuid import uuid4

import structlog

from attest.models import (
    Claim,
    MutationType,
    Probe,
    Verdict,
    Verification,
    VerifierProtocol,
    VerifyContext,
)
from attest.verifiers.mutations import MutatedClaim, generate_mutations

logger = structlog.get_logger(__name__)

# What each mutation kind should logically turn a GROUNDED claim into.
#
# `verdict != baseline.verdict` alone is too weak: a verifier that degrades to
# UNVERIFIABLE for every mutated claim — a real failure mode, not a
# hypothetical — would score as "correctly flipped" on every probe and silently
# hollow out the FRAGILE signal, which is the headline contribution. A negated
# assertion or a swap to a decoy that is guaranteed absent from the context
# (see mutations.py's prompt constraint) should land on disagreement, not on
# "can't tell".
EXPECTED_FLIP_TARGETS: dict[MutationType, frozenset[Verdict]] = {
    MutationType.NEGATION: frozenset({Verdict.CONTRADICTED, Verdict.UNSUPPORTED}),
    MutationType.ENTITY_SWAP: frozenset({Verdict.UNSUPPORTED, Verdict.CONTRADICTED}),
    MutationType.QUANTIFIER_SHIFT: frozenset({Verdict.UNSUPPORTED, Verdict.CONTRADICTED}),
}

_MAX_PROBES = 3  # one per MutationType; explicit even though there are only 3 kinds


class AdversarialProber:
    """Mutates a claim in ways whose correct verdict is known in advance,
    re-verifies each mutation against the same context via the injected
    entailment verifier, and flags the claim FRAGILE when the verifier fails
    to flip its verdict on a mutation that necessarily should have flipped
    it.
    """

    def __init__(self, entailment_verifier: VerifierProtocol) -> None:
        self._entailment = entailment_verifier
        self.last_probes: list[Probe] = []

    async def verify(self, claim: Claim, ctx: VerifyContext) -> Verification:
        """`VerifierProtocol` entrypoint. See module docstring for how to
        also retrieve the individual `Probe` records this call produces.
        """
        verification, _ = await self.verify_with_probes(claim, ctx)
        return verification

    async def verify_with_probes(
        self, claim: Claim, ctx: VerifyContext
    ) -> tuple[Verification, list[Probe]]:
        """Same result as `verify()`, plus the individual `Probe` records.

        Preferred entrypoint for the orchestrator, which needs to persist
        `Probe` rows and knows it's holding an `AdversarialProber`, not just
        a `VerifierProtocol`.
        """
        baseline = await self._entailment.verify(claim, ctx)

        context_text = "\n".join(chunk.text for chunk in ctx.retrieved_chunks)
        mutations = (await generate_mutations(claim.text, context_text))[:_MAX_PROBES]

        probes, total_latency_ms, total_cost_usd = await self._probe_all(
            claim, ctx, baseline, mutations
        )

        applicable = len(probes)
        not_flipped = sum(1 for probe in probes if not probe.flipped)
        fragility = (not_flipped / applicable) if applicable else 0.0

        if baseline.verdict == Verdict.GROUNDED and fragility > 0.0:
            final_verdict = Verdict.FRAGILE
        else:
            # Non-GROUNDED baselines pass through unchanged: a claim that's
            # already UNSUPPORTED/CONTRADICTED/UNVERIFIABLE isn't "fragile,"
            # re-probing doesn't change what it already is.
            final_verdict = baseline.verdict

        rationale = self._build_rationale(baseline, probes, applicable, fragility, final_verdict)

        verification = Verification(
            id=uuid4(),
            claim_id=claim.id,
            verifier="prober",
            verdict=final_verdict,
            rationale=rationale,
            evidence=baseline.evidence,
            latency_ms=total_latency_ms,
            cost_usd=total_cost_usd,
        )
        self.last_probes = probes
        return verification, probes

    async def _probe_all(
        self,
        claim: Claim,
        ctx: VerifyContext,
        baseline: Verification,
        mutations: list[MutatedClaim],
    ) -> tuple[list[Probe], int | None, float | None]:
        """Re-verify each mutation concurrently and build `Probe` rows.

        Returns the probes plus latency/cost totals seeded from `baseline`
        and accumulated across every mutation re-verify call that didn't
        raise.
        """
        total_latency_ms = baseline.latency_ms
        total_cost_usd = baseline.cost_usd

        if not mutations:
            return [], total_latency_ms, total_cost_usd

        results = await asyncio.gather(
            *[
                self._entailment.verify(
                    claim.model_copy(update={"text": mutation.text, "verdict": None}), ctx
                )
                for mutation in mutations
            ],
            return_exceptions=True,
        )

        probes: list[Probe] = []
        for mutation, result in zip(mutations, results, strict=True):
            if isinstance(result, BaseException):
                logger.warning(
                    "prober_mutation_verify_failed",
                    claim_id=str(claim.id),
                    mutation_type=mutation.mutation_type.value,
                    error=str(result),
                )
                continue

            if result.latency_ms is not None:
                total_latency_ms = (total_latency_ms or 0) + result.latency_ms
            if result.cost_usd is not None:
                total_cost_usd = (total_cost_usd or 0.0) + result.cost_usd

            changed = result.verdict != baseline.verdict
            expected_targets = EXPECTED_FLIP_TARGETS.get(mutation.mutation_type, frozenset())
            flipped = changed and result.verdict in expected_targets
            if changed and not flipped:
                # Verdict moved, but not to a verdict this mutation should
                # produce (most commonly UNVERIFIABLE). Counted as NOT flipped
                # -> contributes to fragility, and logged so the distinction is
                # visible when tuning prompts.
                logger.info(
                    "prober_unexpected_flip_target",
                    claim_id=str(claim.id),
                    mutation_type=mutation.mutation_type.value,
                    baseline_verdict=baseline.verdict.value,
                    observed_verdict=result.verdict.value,
                    expected_any_of=sorted(v.value for v in expected_targets),
                )
            probes.append(
                Probe(
                    id=uuid4(),
                    claim_id=claim.id,
                    mutation_type=mutation.mutation_type,
                    mutated_text=mutation.text,
                    expected_flip=mutation.expected_flip,
                    observed_verdict=result.verdict,
                    flipped=flipped,
                )
            )
        return probes, total_latency_ms, total_cost_usd

    @staticmethod
    def _build_rationale(
        baseline: Verification,
        probes: list[Probe],
        applicable: int,
        fragility: float,
        final_verdict: Verdict,
    ) -> str:
        if applicable == 0:
            return (
                f"Baseline entailment verdict was {baseline.verdict.value}. No "
                "applicable mutations could be generated for this claim (no "
                "swappable entity, no quantifier/number, and/or negation was "
                "not applicable) — fragility cannot be assessed and defaults "
                "to 0.0."
            )

        failed = [probe for probe in probes if not probe.flipped]
        if not failed:
            return (
                f"Baseline entailment verdict was {baseline.verdict.value}. All "
                f"{applicable} probe(s) correctly flipped the verdict under "
                "mutation — no fragility detected."
            )

        failed_kinds = ", ".join(probe.mutation_type.value for probe in failed)
        if final_verdict == Verdict.FRAGILE:
            outcome = "downgrading the verdict to FRAGILE"
        else:
            outcome = (
                f"baseline verdict {baseline.verdict.value} is passed through "
                "unchanged (only GROUNDED baselines are downgraded)"
            )
        return (
            f"Baseline entailment verdict was {baseline.verdict.value}. "
            f"{len(failed)}/{applicable} probe(s) failed to flip the verdict "
            f"({failed_kinds}) even though the mutation necessarily changes "
            "the claim's truth value — the entailment verifier did not "
            f"appear to re-read the context under mutation; {outcome}."
        )


__all__ = ["AdversarialProber"]
