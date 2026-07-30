"""Fan-out orchestrator: sampling, budget enforcement, per-claim verifier
fan-out, and handoff to `reconciler.py` + `store.py`.

A1/A2/A3's decomposer and verifiers are injected as a callable / objects
implementing `attest.models.VerifierProtocol` — never imported at module
load time, per CLAUDE.md's dependency-injection rule. That keeps this
module importable and unit-testable in isolation (see
`tests/test_orchestrator.py`, which exercises it entirely with fakes) even
before those files have real content. The real wiring happens in
`attest/api/routes.py` at request time.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

import structlog

from attest.models import (
    Chunk,
    Claim,
    ClaimDetail,
    ObserveRequest,
    Probe,
    RunDetail,
    RunStatus,
    Verdict,
    Verification,
    VerifierProtocol,
    VerifyContext,
)
from attest.reconciler import reconcile
from attest.store import StoreProtocol

logger = structlog.get_logger(__name__)

# Injected-callable shapes. `decompose` and `get_probes` are the two hooks
# this module needs from other agents' work that aren't captured by
# `VerifierProtocol` alone:
#   - `decompose(request, run_id) -> list[Claim]` is A1's decomposer. It
#     owns claim id/run_id/claim_index/span assignment, and may resolve a
#     claim's verdict directly at decomposition time (e.g. UNVERIFIABLE for
#     a subjective/predictive claim per CLAUDE.md) — those claims are
#     persisted as-is and never sent to the verifiers (see the brief).
#   - `get_probes(claim, prober_verification) -> list[Probe]` is an optional
#     hook for A2's prober to hand back the individual mutation probes it
#     ran (negation/entity_swap/quantifier_shift) so they can be persisted
#     into `probes` and surfaced as `probe.completed` events. This is NOT
#     part of the frozen `VerifierProtocol` (`verify()` only returns a
#     `Verification`) — see the handoff note in A4's final report for why
#     this is a real gap between the protocol and the `probes` table, and
#     what's expected once A2's real prober lands. If omitted, orchestrator
#     still emits one summary `probe.completed` event per claim the prober
#     ran on, just without per-mutation detail.
DecomposeFn = Callable[[ObserveRequest, UUID], Awaitable[list[Claim]]]
GetProbesFn = Callable[[Claim, Verification], Awaitable[list[Probe]]]
EmitFn = Callable[[str, dict[str, Any]], Awaitable[None]]
SampleFn = Callable[[float], bool]

_BUDGET_EXCEEDED_RATIONALE = "budget_exceeded"


async def _noop_emit(event: str, data: dict[str, Any]) -> None:
    return None


def _default_sample(sample_rate: float) -> bool:
    return random.random() < sample_rate


async def _gather_verifications(
    *coros: Awaitable[Verification],
) -> list[Verification | None]:
    """`asyncio.gather(..., return_exceptions=True)`: one dead verifier must
    never kill the run (CLAUDE.md hard rule). A raised exception is logged
    and turned into `None` in the result list so the caller can proceed with
    whatever verifiers *did* succeed.
    """
    if not coros:
        return []
    results = await asyncio.gather(*coros, return_exceptions=True)
    out: list[Verification | None] = []
    for result in results:
        if isinstance(result, BaseException):
            logger.warning(
                "verifier_failed", error=str(result), error_type=type(result).__name__
            )
            out.append(None)
        else:
            out.append(result)
    return out


def _cost_of(verification: Verification | None) -> float:
    return (verification.cost_usd or 0.0) if verification is not None else 0.0


def _score(claim_details: list[ClaimDetail]) -> tuple[float | None, float | None]:
    """grounding_score / fragility_score: share of *verifiable* claims
    (verdict != UNVERIFIABLE) landing on GROUNDED / FRAGILE respectively.
    `None` if every claim was UNVERIFIABLE (nothing to score).
    """
    verifiable = [c for c in claim_details if c.verdict != Verdict.UNVERIFIABLE]
    if not verifiable:
        return None, None
    grounded = sum(1 for c in verifiable if c.verdict == Verdict.GROUNDED)
    fragile = sum(1 for c in verifiable if c.verdict == Verdict.FRAGILE)
    return grounded / len(verifiable), fragile / len(verifiable)


def _empty_detail(
    run_id: UUID,
    created_at: datetime,
    request: ObserveRequest,
    *,
    status: RunStatus,
    chunks: list[Chunk] | None = None,
) -> RunDetail:
    return RunDetail(
        id=run_id,
        created_at=created_at,
        pipeline_name=request.pipeline_name,
        query=request.query,
        answer=request.answer,
        model=request.model,
        status=status,
        retrieved_chunks=chunks or [],
        claims=[],
    )


async def _abort_run(
    run_id: UUID,
    stage: str,
    exc: Exception,
    *,
    store: StoreProtocol,
    emit: EmitFn,
) -> None:
    """Best-effort: mark the run errored and tell any SSE subscriber.

    A store failure is exactly the case where the *next* store call may also
    fail, so `update_run_status` is itself guarded — the one thing that must
    always happen is emitting `run.error`, otherwise the dashboard hangs on a
    run that will never finish.
    """
    logger.error("run_failed", run_id=str(run_id), stage=stage, error=str(exc))
    try:
        await store.update_run_status(run_id, "error")
    except Exception as status_exc:  # noqa: BLE001 - best-effort, already failing
        logger.error(
            "run_status_update_failed", run_id=str(run_id), stage=stage, error=str(status_exc)
        )
    await emit("run.error", {"run_id": str(run_id), "error": str(exc), "stage": stage})


async def run(
    request: ObserveRequest,
    *,
    decompose: DecomposeFn,
    entailment: VerifierProtocol,
    prober: VerifierProtocol,
    independent: VerifierProtocol,
    store: StoreProtocol,
    emit: EmitFn = _noop_emit,
    run_id: UUID | None = None,
    sample: SampleFn = _default_sample,
    get_probes: GetProbesFn | None = None,
) -> RunDetail:
    """Run one full ATTEST verification pass for `request`.

    Never raises: decomposition failures and per-verifier exceptions are
    caught, logged, and reflected in the run's persisted status / claim
    rationale instead of propagating — a background task running this must
    not die silently, and a partial result always beats no result
    (CLAUDE.md, A4 brief).

    Returns the `RunDetail` reflecting exactly what was persisted via
    `store`. Emits SSE-shaped events via `emit(event_name, data)` in this
    order: `run.started` -> (`claims.decomposed` ->)* `claim.verified` (+
    `probe.completed` when the prober ran) -> `run.completed` /
    `run.error`. Event names and the `{run_id, data}` envelope match
    PLAN.md §5.2 exactly; `emit` itself just needs to accept `(name, data)`
    and return an awaitable — `attest/api/routes.py` wires it to
    `attest.api.stream.EventBus.publish`.
    """
    run_id = run_id or uuid4()
    created_at = datetime.now(UTC)
    started = time.monotonic()

    await emit("run.started", {"run_id": str(run_id), "pipeline_name": request.pipeline_name})

    try:
        await store.create_run(run_id, request)

        if not sample(request.config.sample_rate):
            await store.update_run_status(run_id, "skipped")
            logger.info(
                "run_skipped_by_sampling",
                run_id=str(run_id),
                sample_rate=request.config.sample_rate,
            )
            await emit("run.completed", {"run_id": str(run_id), "status": "skipped"})
            return _empty_detail(run_id, created_at, request, status="skipped")

        await store.update_run_status(run_id, "running")
        chunks = await store.save_chunks(run_id, request.retrieved_chunks)
    except Exception as exc:  # noqa: BLE001 - a store outage must not hang the stream
        await _abort_run(run_id, "persist_setup", exc, store=store, emit=emit)
        return _empty_detail(run_id, created_at, request, status="error")

    try:
        claims = await decompose(request, run_id)
    except Exception as exc:  # noqa: BLE001 - decomposer failure must not crash the run
        logger.error("decompose_failed", run_id=str(run_id), error=str(exc))
        await store.update_run_status(run_id, "error")
        await emit("run.error", {"run_id": str(run_id), "error": str(exc), "stage": "decompose"})
        return _empty_detail(run_id, created_at, request, status="error", chunks=chunks)

    await emit(
        "claims.decomposed",
        {"run_id": str(run_id), "count": len(claims), "claim_ids": [str(c.id) for c in claims]},
    )

    base_ctx = VerifyContext(
        run_id=run_id,
        query=request.query,
        answer=request.answer,
        retrieved_chunks=chunks,
        config=request.config,
    )

    budget = request.config.budget_usd
    total_cost = 0.0
    claim_details: list[ClaimDetail] = []

    try:
        for claim in claims:
            if claim.verdict is not None:
                # Resolved at decomposition time (e.g. UNVERIFIABLE for a
                # subjective/predictive claim) — never sent to the verifiers.
                await store.save_claim(claim)
                claim_details.append(ClaimDetail(**claim.model_dump()))
                await emit(
                    "claim.verified",
                    {
                        "run_id": str(run_id),
                        "claim_id": str(claim.id),
                        "verdict": claim.verdict.value,
                        "source": "decomposer",
                    },
                )
                continue

            if total_cost >= budget:
                final_claim = claim.model_copy(
                    update={
                        "verdict": Verdict.UNVERIFIABLE,
                        "confidence": 0.0,
                        "disagreement": 0.0,
                        "rationale": _BUDGET_EXCEEDED_RATIONALE,
                    }
                )
                await store.save_claim(final_claim)
                claim_details.append(ClaimDetail(**final_claim.model_dump()))
                await emit(
                    "claim.verified",
                    {
                        "run_id": str(run_id),
                        "claim_id": str(claim.id),
                        "verdict": "UNVERIFIABLE",
                        "reason": _BUDGET_EXCEEDED_RATIONALE,
                    },
                )
                continue

            # Entailment runs first; its verdict seeds prior_entailment for the
            # independent verifier (CONTRACT_CHANGE_REQUEST.md) and is needed to
            # decide whether budget survives to launch the second pass.
            (entailment_result,) = await _gather_verifications(entailment.verify(claim, base_ctx))
            total_cost += _cost_of(entailment_result)

            prior_ctx = base_ctx.model_copy(
                update={"prior_entailment": entailment_result.verdict if entailment_result else None}
            )

            run_prober = request.config.enable_prober and total_cost < budget
            run_independent = request.config.enable_independent and total_cost < budget

            second_pass: list[Awaitable[Verification]] = []
            labels: list[str] = []
            if run_prober:
                # prior_entailment is scoped to the independent verifier only
                # (CONTRACT_CHANGE_REQUEST.md) — the prober re-derives its own
                # baseline directly, so it gets the plain base context.
                second_pass.append(prober.verify(claim, base_ctx))
                labels.append("prober")
            if run_independent:
                second_pass.append(independent.verify(claim, prior_ctx))
                labels.append("independent")

            second_results = await _gather_verifications(*second_pass)
            by_label = dict(zip(labels, second_results, strict=True))
            prober_result = by_label.get("prober")
            independent_result = by_label.get("independent")

            total_cost += _cost_of(prober_result) + _cost_of(independent_result)

            verdict, confidence, disagreement, rationale = reconcile(
                entailment_result, prober_result, independent_result
            )
            final_claim = claim.model_copy(
                update={
                    "verdict": verdict,
                    "confidence": confidence,
                    "disagreement": disagreement,
                    "rationale": rationale,
                }
            )
            await store.save_claim(final_claim)
            for verification in (entailment_result, prober_result, independent_result):
                if verification is not None:
                    await store.save_verification(verification)

            probes: list[Probe] = []
            if prober_result is not None:
                if get_probes is not None:
                    try:
                        probes = await get_probes(claim, prober_result)
                    except Exception as exc:  # noqa: BLE001 - a probe-extraction bug can't kill the run
                        logger.warning("get_probes_failed", claim_id=str(claim.id), error=str(exc))
                        probes = []
                    for probe in probes:
                        await store.save_probe(probe)
                        await emit(
                            "probe.completed",
                            {
                                "run_id": str(run_id),
                                "claim_id": str(claim.id),
                                "mutation_type": probe.mutation_type.value,
                                "flipped": probe.flipped,
                            },
                        )
                else:
                    # No per-mutation detail available from this prober
                    # implementation yet — still signal that probing happened.
                    await emit(
                        "probe.completed",
                        {
                            "run_id": str(run_id),
                            "claim_id": str(claim.id),
                            "verdict": prober_result.verdict.value,
                        },
                    )

            claim_details.append(ClaimDetail(**final_claim.model_dump(), probes=probes))
            await emit(
                "claim.verified",
                {
                    "run_id": str(run_id),
                    "claim_id": str(claim.id),
                    "verdict": verdict.value,
                    "confidence": confidence,
                    "disagreement": disagreement,
                },
            )

        grounding_score, fragility_score = _score(claim_details)
        latency_ms = int((time.monotonic() - started) * 1000)
        cost_usd = round(total_cost, 6)

        await store.finalize_run(
            run_id,
            status="complete",
            grounding_score=grounding_score,
            fragility_score=fragility_score,
            total_claims=len(claim_details),
            latency_ms=latency_ms,
            cost_usd=cost_usd,
        )
    except Exception as exc:  # noqa: BLE001 - partial results beat a run stuck 'running'
        await _abort_run(run_id, "persist_claims", exc, store=store, emit=emit)
        grounding_score, fragility_score = _score(claim_details)
        return RunDetail(
            id=run_id,
            created_at=created_at,
            pipeline_name=request.pipeline_name,
            query=request.query,
            answer=request.answer,
            model=request.model,
            status="error",
            grounding_score=grounding_score,
            fragility_score=fragility_score,
            total_claims=len(claim_details),
            latency_ms=int((time.monotonic() - started) * 1000),
            cost_usd=round(total_cost, 6),
            retrieved_chunks=chunks,
            claims=claim_details,
        )
    await emit(
        "run.completed",
        {
            "run_id": str(run_id),
            "status": "complete",
            "grounding_score": grounding_score,
            "fragility_score": fragility_score,
            "total_claims": len(claim_details),
            "latency_ms": latency_ms,
            "cost_usd": cost_usd,
        },
    )

    return RunDetail(
        id=run_id,
        created_at=created_at,
        pipeline_name=request.pipeline_name,
        query=request.query,
        answer=request.answer,
        model=request.model,
        status="complete",
        grounding_score=grounding_score,
        fragility_score=fragility_score,
        total_claims=len(claim_details),
        latency_ms=latency_ms,
        cost_usd=cost_usd,
        retrieved_chunks=chunks,
        claims=claim_details,
    )


__all__ = ["run", "DecomposeFn", "GetProbesFn", "EmitFn", "SampleFn"]
