"""Tests for bench.runner: wiring correctness with every attest component
faked (no real LLM, no network) -- proves the four-configuration fan-out
runs to completion and produces the expected per-claim records, mirroring
tests/test_orchestrator.py's approach for the production orchestrator."""

from __future__ import annotations

from uuid import UUID, uuid4

from attest.models import Claim, Verdict, Verification, VerifyContext
from bench.baseline import BaselineResult
from bench.configs import ALL_CONFIGS, BenchConfig
from bench.datasets import BenchExample
from bench.runner import RunnerDeps, run_benchmark, run_example


class FakeVerifier:
    """Implements `attest.models.VerifierProtocol` structurally."""

    def __init__(
        self, name: str, verdict: Verdict = Verdict.GROUNDED, cost: float = 0.001, latency: int = 10
    ):
        self.name = name
        self.verdict = verdict
        self.cost = cost
        self.latency = latency
        self.calls: list[tuple[UUID, VerifyContext]] = []

    async def verify(self, claim: Claim, ctx: VerifyContext) -> Verification:
        self.calls.append((claim.id, ctx))
        return Verification(
            id=uuid4(),
            claim_id=claim.id,
            verifier=self.name,
            verdict=self.verdict,
            cost_usd=self.cost,
            latency_ms=self.latency,
        )


class FakeBaselineJudge:
    """Duck-types `bench.baseline.SinglePassJudge`'s `.judge()` interface."""

    def __init__(self, verdict: Verdict = Verdict.UNSUPPORTED):
        self.verdict = verdict
        self.calls = 0

    async def judge(self, claim: Claim, ctx: VerifyContext) -> BaselineResult:
        self.calls += 1
        return BaselineResult(
            claim_id=claim.id, verdict=self.verdict, rationale="fake", cost_usd=0.0005, latency_ms=50
        )


def _example(example_id: str = "ex-1", dataset: str = "ragtruth") -> BenchExample:
    return BenchExample(
        example_id=example_id,
        dataset=dataset,
        hallucination_type="Evident Conflict",
        query="q",
        retrieved_chunks=[],
        answer="Paris is the capital of France. It has the Eiffel Tower.",
        ground_truth_spans=[(0, 32)],
        ground_truth_hallucinated=True,
    )


async def test_run_example_shares_decomposition_and_runs_all_four_configs():
    decompose_calls = {"count": 0}

    async def decompose(answer, query, *, run_id):
        decompose_calls["count"] += 1
        return [
            Claim(
                id=uuid4(),
                run_id=run_id,
                claim_index=0,
                text="Paris is the capital of France.",
                span_start=0,
                span_end=32,
            )
        ]

    entailment = FakeVerifier("entailment", verdict=Verdict.GROUNDED)
    prober = FakeVerifier("prober", verdict=Verdict.FRAGILE)
    independent = FakeVerifier("independent", verdict=Verdict.GROUNDED)
    baseline = FakeBaselineJudge(verdict=Verdict.GROUNDED)

    deps = RunnerDeps(
        decompose=decompose, entailment=entailment, prober=prober, independent=independent, baseline=baseline
    )
    records = await run_example(_example(), deps)

    assert decompose_calls["count"] == 1  # decomposed exactly once, shared across all 4 configs
    configs_seen = {r.config for r in records}
    assert configs_seen == {c.value for c in ALL_CONFIGS}
    assert len(records) == 4  # one claim x four configs
    assert baseline.calls == 1
    assert len(entailment.calls) == 3  # entailment runs once per ATTEST config (not baseline)


async def test_prober_and_independent_called_only_by_their_configs():
    async def decompose(answer, query, *, run_id):
        return [Claim(id=uuid4(), run_id=run_id, claim_index=0, text="claim text")]

    entailment = FakeVerifier("entailment")
    prober = FakeVerifier("prober")
    independent = FakeVerifier("independent")
    deps = RunnerDeps(
        decompose=decompose,
        entailment=entailment,
        prober=prober,
        independent=independent,
        baseline=FakeBaselineJudge(),
    )

    await run_example(_example(), deps)

    # prober runs for ATTEST_MINUS_INDEPENDENT and ATTEST_FULL only
    assert len(prober.calls) == 2
    # independent runs for ATTEST_MINUS_PROBER and ATTEST_FULL only
    assert len(independent.calls) == 2
    # entailment runs for all three ATTEST configs
    assert len(entailment.calls) == 3


async def test_independent_receives_prior_entailment_context():
    async def decompose(answer, query, *, run_id):
        return [Claim(id=uuid4(), run_id=run_id, claim_index=0, text="claim text")]

    entailment = FakeVerifier("entailment", verdict=Verdict.GROUNDED)
    prober = FakeVerifier("prober")
    independent = FakeVerifier("independent")
    deps = RunnerDeps(
        decompose=decompose,
        entailment=entailment,
        prober=prober,
        independent=independent,
        baseline=FakeBaselineJudge(),
    )

    await run_example(_example(), deps)

    assert independent.calls  # ran at least once (ATTEST_MINUS_PROBER, ATTEST_FULL)
    for _claim_id, ctx in independent.calls:
        assert ctx.prior_entailment == Verdict.GROUNDED
    # prober is never handed prior_entailment -- it re-derives its own baseline
    for _claim_id, ctx in prober.calls:
        assert ctx.prior_entailment is None


async def test_decomposer_resolved_claim_skips_verification_for_all_configs():
    async def decompose(answer, query, *, run_id):
        return [
            Claim(
                id=uuid4(),
                run_id=run_id,
                claim_index=0,
                text="This is probably the best approach.",
                verdict=Verdict.UNVERIFIABLE,
                rationale="subjective",
            )
        ]

    entailment = FakeVerifier("entailment")
    prober = FakeVerifier("prober")
    independent = FakeVerifier("independent")
    baseline = FakeBaselineJudge()
    deps = RunnerDeps(
        decompose=decompose,
        entailment=entailment,
        prober=prober,
        independent=independent,
        baseline=baseline,
    )

    records = await run_example(_example(), deps)

    assert len(records) == 4
    assert all(r.predicted_verdict == Verdict.UNVERIFIABLE for r in records)
    assert all(r.decomposer_resolved for r in records)
    assert all(r.cost_usd == 0.0 and r.latency_ms == 0 for r in records)
    assert baseline.calls == 0
    assert entailment.calls == []


async def test_attest_full_cost_and_latency_sum_all_three_verifiers():
    async def decompose(answer, query, *, run_id):
        return [Claim(id=uuid4(), run_id=run_id, claim_index=0, text="claim")]

    entailment = FakeVerifier("entailment", cost=0.01, latency=100)
    prober = FakeVerifier("prober", cost=0.02, latency=200)
    independent = FakeVerifier("independent", cost=0.03, latency=300)
    deps = RunnerDeps(
        decompose=decompose,
        entailment=entailment,
        prober=prober,
        independent=independent,
        baseline=FakeBaselineJudge(),
    )

    records = await run_example(_example(), deps)
    full = next(r for r in records if r.config == BenchConfig.ATTEST_FULL.value)
    assert abs(full.cost_usd - 0.06) < 1e-9
    assert full.latency_ms == 600


async def test_reconcile_precedence_is_actually_applied():
    async def decompose(answer, query, *, run_id):
        return [Claim(id=uuid4(), run_id=run_id, claim_index=0, text="claim")]

    entailment = FakeVerifier("entailment", verdict=Verdict.GROUNDED)
    prober = FakeVerifier("prober", verdict=Verdict.GROUNDED)
    independent = FakeVerifier("independent", verdict=Verdict.CONTRADICTED)
    deps = RunnerDeps(
        decompose=decompose,
        entailment=entailment,
        prober=prober,
        independent=independent,
        baseline=FakeBaselineJudge(),
    )

    records = await run_example(_example(), deps)
    full = next(r for r in records if r.config == BenchConfig.ATTEST_FULL.value)
    # CONTRADICTED outranks GROUNDED per attest.reconciler's real precedence
    # rule -- this is the real reconcile() function, not a reimplementation.
    assert full.predicted_verdict == Verdict.CONTRADICTED


async def test_run_benchmark_runs_multiple_examples_and_is_resilient_to_one_failure():
    good_example = _example("ex-good")
    bad_example = _example("ex-bad").model_copy(update={"query": "explode"})

    async def decompose(answer, query, *, run_id):
        if query == "explode":
            raise RuntimeError("boom")
        return [Claim(id=uuid4(), run_id=run_id, claim_index=0, text="claim")]

    entailment = FakeVerifier("entailment")
    prober = FakeVerifier("prober")
    independent = FakeVerifier("independent")
    deps = RunnerDeps(
        decompose=decompose,
        entailment=entailment,
        prober=prober,
        independent=independent,
        baseline=FakeBaselineJudge(),
    )

    records = await run_benchmark([good_example, bad_example], deps, concurrency=2)

    assert all(r.example_id == "ex-good" for r in records)
    assert len(records) == 4  # one claim x four configs, only for the surviving example
