"""Fan-out runner for the four benchmark configurations (PLAN.md §10).

Wires the *real* `attest` pipeline components -- `attest.verifiers.decomposer.
decompose`, `attest.verifiers.entailment.EntailmentVerifier`,
`attest.verifiers.prober.AdversarialProber`,
`attest.verifiers.independent.IndependentVerifier`, `attest.reconciler.
reconcile` -- not a reimplementation of their logic (CLAUDE.md, task brief).
Deliberately does NOT go through `attest.orchestrator.run`: the orchestrator
owns sampling, per-run budget cutoffs, Supabase persistence, and SSE
emission, none of which the benchmark wants (it needs raw per-claim
verdicts across four *different* verifier combinations for the identical
claim set, not one persisted trace). `_run_attest_config` below reproduces
the orchestrator's entailment-then-second-pass sequencing exactly (see
`attest/orchestrator.py` and `HANDOFF.md`'s note on `prior_entailment`), so
each ATTEST configuration here is a faithful simulation of actually running
the orchestrator with that configuration's `enable_prober`/
`enable_independent` flags -- not a shortcut.

Decomposition happens exactly once per example, shared across all four
configs (PLAN.md §10: "the independent variable is only the verification
strategy") -- see `run_example`.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

import structlog
from pydantic import BaseModel

from attest.models import (
    AttestConfig,
    Chunk,
    Claim,
    Verdict,
    Verification,
    VerifierProtocol,
    VerifyContext,
)
from attest.reconciler import reconcile
from bench.configs import ALL_CONFIGS, USES_INDEPENDENT, USES_PROBER, BenchConfig
from bench.datasets import BenchExample

if TYPE_CHECKING:
    # Deferred: `bench.baseline` imports `attest.llm`, which imports
    # `attest.config`, which fails loudly at import time if `.env` isn't
    # configured (CLAUDE.md, by design). Data types in this module
    # (`ClaimRecord`) and pure-mapping/metrics/report code that only needs
    # `ClaimRecord`'s shape must stay importable with zero attest.config
    # dependency -- see `bench/report.py`'s pending-template path, which has
    # to build a valid report with no real credentials anywhere. `from
    # __future__ import annotations` (above) makes this a string annotation
    # at runtime, so the real import only happens for type-checkers.
    from bench.baseline import SinglePassJudge

logger = structlog.get_logger(__name__)

# Matches `attest.verifiers.decomposer.decompose`'s real signature:
#   async def decompose(answer: str, query: str, *, run_id: UUID) -> list[Claim]
DecomposeFn = Callable[..., Awaitable[list[Claim]]]

# Generous per-run budget: the benchmark measures real cost/latency, it
# should never have a verifier skipped because of the orchestrator-style
# budget cutoff `attest.orchestrator.run` applies in production (that
# machinery isn't used here at all -- see module docstring).
_BENCH_CONFIG = AttestConfig(
    sample_rate=1.0, enable_prober=True, enable_independent=True, budget_usd=1_000.0
)

# Any fixed namespace works here; only used to make each example's run_id
# deterministic (reproducible traces/logs across repeated benchmark runs of
# the same example set), never for cross-run identity guarantees.
_RUN_ID_NAMESPACE = NAMESPACE_URL


class ClaimRecord(BaseModel):
    """Bench-local per-claim, per-config result record. Not part of the
    frozen `attest.models` contract -- see `bench/baseline.py`'s docstring
    for why a bench-only result type is necessary here."""

    example_id: str
    dataset: str
    hallucination_type: str | None
    claim_id: UUID
    claim_text: str
    span_start: int | None
    span_end: int | None
    config: str  # BenchConfig.value
    predicted_verdict: Verdict
    cost_usd: float
    latency_ms: int
    decomposer_resolved: bool = False  # True: decomposer itself resolved this claim (e.g. UNVERIFIABLE); never sent to any verifier


@dataclass
class RunnerDeps:
    """Injected dependencies. In production these are the real
    `attest.verifiers.*` classes and `attest.verifiers.decomposer.decompose`
    (see `bench/run_benchmark.py`); tests inject fakes implementing the same
    shapes (`VerifierProtocol` / `DecomposeFn`), exactly like
    `tests/test_orchestrator.py` does for the orchestrator itself."""

    decompose: DecomposeFn
    entailment: VerifierProtocol
    prober: VerifierProtocol
    independent: VerifierProtocol
    baseline: SinglePassJudge


def _cost_of(v: Verification | None) -> float:
    return (v.cost_usd or 0.0) if v is not None else 0.0


def _latency_of(v: Verification | None) -> int:
    return (v.latency_ms or 0) if v is not None else 0


async def _gather_optional(*coros: Awaitable[Verification]) -> list[Verification | None]:
    """Same contract as `attest.orchestrator`'s internal gather helper: one
    dead verifier must never kill the benchmark run. Reimplemented locally
    (rather than importing orchestrator's underscore-prefixed helper) since
    `attest/orchestrator.py` is outside this agent's lane and its private
    functions aren't a surface bench/ should depend on."""
    if not coros:
        return []
    results = await asyncio.gather(*coros, return_exceptions=True)
    out: list[Verification | None] = []
    for result in results:
        if isinstance(result, BaseException):
            logger.warning(
                "bench_verifier_failed", error=str(result), error_type=type(result).__name__
            )
            out.append(None)
        else:
            out.append(result)
    return out


async def _run_attest_config(
    claim: Claim,
    base_ctx: VerifyContext,
    config: BenchConfig,
    entailment: VerifierProtocol,
    prober: VerifierProtocol,
    independent: VerifierProtocol,
) -> tuple[Verdict, float, int]:
    """Reproduces `attest.orchestrator.run`'s per-claim sequencing exactly
    for one configuration: entailment runs first (its verdict seeds
    `prior_entailment` for the independent verifier only, per
    `CONTRACT_CHANGE_REQUEST.md`), then whichever of prober/independent this
    config enables run concurrently, then `attest.reconciler.reconcile`
    resolves the final verdict from whichever verifiers actually ran.
    """
    (entailment_result,) = await _gather_optional(entailment.verify(claim, base_ctx))
    prior_ctx = base_ctx.model_copy(
        update={"prior_entailment": entailment_result.verdict if entailment_result else None}
    )

    second_pass: list[Awaitable[Verification]] = []
    labels: list[str] = []
    if USES_PROBER[config]:
        second_pass.append(prober.verify(claim, base_ctx))
        labels.append("prober")
    if USES_INDEPENDENT[config]:
        second_pass.append(independent.verify(claim, prior_ctx))
        labels.append("independent")

    second_results = await _gather_optional(*second_pass)
    by_label = dict(zip(labels, second_results, strict=True))
    prober_result = by_label.get("prober")
    independent_result = by_label.get("independent")

    verdict, _confidence, _disagreement, _rationale = reconcile(
        entailment_result, prober_result, independent_result
    )
    cost = _cost_of(entailment_result) + _cost_of(prober_result) + _cost_of(independent_result)
    latency = (
        _latency_of(entailment_result) + _latency_of(prober_result) + _latency_of(independent_result)
    )
    return verdict, cost, latency


def _build_ctx(example: BenchExample, run_id: UUID) -> VerifyContext:
    chunks = [Chunk(id=uuid4(), run_id=run_id, **rc.model_dump()) for rc in example.retrieved_chunks]
    return VerifyContext(
        run_id=run_id,
        query=example.query,
        answer=example.answer,
        retrieved_chunks=chunks,
        config=_BENCH_CONFIG,
    )


async def run_example(example: BenchExample, deps: RunnerDeps) -> list[ClaimRecord]:
    """Decompose `example.answer` exactly once, then run all four
    configurations over the identical resulting claim set."""
    run_id = uuid5(_RUN_ID_NAMESPACE, example.example_id)
    claims = await deps.decompose(example.answer, example.query, run_id=run_id)
    base_ctx = _build_ctx(example, run_id)

    records: list[ClaimRecord] = []
    for claim in claims:
        common = {
            "example_id": example.example_id,
            "dataset": example.dataset,
            "hallucination_type": example.hallucination_type,
            "claim_id": claim.id,
            "claim_text": claim.text,
            "span_start": claim.span_start,
            "span_end": claim.span_end,
        }

        if claim.verdict is not None:
            # Decomposer already resolved this claim (e.g. UNVERIFIABLE for
            # a subjective/predictive claim) -- skip verification entirely
            # and share that verdict across all four configs (task brief:
            # "skip claims the decomposer already resolved as UNVERIFIABLE").
            for config in ALL_CONFIGS:
                records.append(
                    ClaimRecord(
                        **common,
                        config=config.value,
                        predicted_verdict=claim.verdict,
                        cost_usd=0.0,
                        latency_ms=0,
                        decomposer_resolved=True,
                    )
                )
            continue

        baseline_result = await deps.baseline.judge(claim, base_ctx)
        records.append(
            ClaimRecord(
                **common,
                config=BenchConfig.BASELINE.value,
                predicted_verdict=baseline_result.verdict,
                cost_usd=baseline_result.cost_usd,
                latency_ms=baseline_result.latency_ms,
            )
        )

        for config in (
            BenchConfig.ATTEST_MINUS_PROBER,
            BenchConfig.ATTEST_MINUS_INDEPENDENT,
            BenchConfig.ATTEST_FULL,
        ):
            verdict, cost, latency = await _run_attest_config(
                claim, base_ctx, config, deps.entailment, deps.prober, deps.independent
            )
            records.append(
                ClaimRecord(
                    **common,
                    config=config.value,
                    predicted_verdict=verdict,
                    cost_usd=cost,
                    latency_ms=latency,
                )
            )

    return records


async def run_benchmark(
    examples: list[BenchExample],
    deps: RunnerDeps,
    *,
    concurrency: int = 4,
) -> list[ClaimRecord]:
    """Run every example through `run_example`, bounded by `concurrency`
    examples in flight at once (real LLM/search calls -- unbounded
    concurrency would just trip provider rate limits). One example's
    failure never kills the whole benchmark run."""
    semaphore = asyncio.Semaphore(concurrency)

    async def _bounded(example: BenchExample) -> list[ClaimRecord]:
        async with semaphore:
            try:
                return await run_example(example, deps)
            except Exception as exc:  # noqa: BLE001 - one bad example must not kill the whole run
                logger.error("bench_example_failed", example_id=example.example_id, error=str(exc))
                return []

    results = await asyncio.gather(*[_bounded(ex) for ex in examples])
    return [record for batch in results for record in batch]


__all__ = ["ClaimRecord", "RunnerDeps", "run_example", "run_benchmark", "DecomposeFn"]
