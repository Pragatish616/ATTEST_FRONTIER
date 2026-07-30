"""End-to-end orchestrator tests with the decomposer, all three verifiers,
and the store fully faked — no real LLM, no real Supabase, no network
(CLAUDE.md; A4 brief).
"""

from __future__ import annotations

from uuid import UUID, uuid4

from attest import orchestrator
from attest.models import (
    AttestConfig,
    Chunk,
    Claim,
    ObserveRequest,
    Verdict,
    Verification,
)

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeStore:
    """Records every write in memory; implements StoreProtocol structurally."""

    def __init__(self) -> None:
        self.runs: dict[UUID, dict] = {}
        self.status_history: list[str] = []
        self.saved_claims: list[Claim] = []
        self.saved_verifications: list[Verification] = []
        self.saved_probes: list = []

    async def create_run(self, run_id: UUID, request: ObserveRequest) -> None:
        self.runs[run_id] = {"status": "pending", "request": request}
        self.status_history.append("pending")

    async def update_run_status(self, run_id: UUID, status: str, **fields) -> None:
        self.runs[run_id]["status"] = status
        self.runs[run_id].update(fields)
        self.status_history.append(status)

    async def save_chunks(self, run_id: UUID, chunks) -> list[Chunk]:
        return [Chunk(id=uuid4(), run_id=run_id, **c.model_dump()) for c in chunks]

    async def save_claim(self, claim: Claim) -> None:
        assert claim.verdict is not None, "store must never receive a claim with verdict=None"
        self.saved_claims.append(claim)

    async def save_verification(self, verification: Verification) -> None:
        self.saved_verifications.append(verification)

    async def save_probe(self, probe) -> None:
        self.saved_probes.append(probe)

    async def finalize_run(self, run_id: UUID, **fields) -> None:
        self.runs[run_id].update(fields)
        self.status_history.append(fields.get("status", self.runs[run_id]["status"]))


class FakeVerifier:
    def __init__(self, name: str, verdict: Verdict = Verdict.GROUNDED, cost: float = 0.001, exc: Exception | None = None):
        self.name = name
        self.verdict = verdict
        self.cost = cost
        self.exc = exc
        self.calls: list[tuple] = []

    async def verify(self, claim: Claim, ctx) -> Verification:
        self.calls.append((claim.id, ctx))
        if self.exc is not None:
            raise self.exc
        return Verification(
            id=uuid4(),
            claim_id=claim.id,
            verifier=self.name,
            verdict=self.verdict,
            cost_usd=self.cost,
        )


def make_claim(run_id: UUID, idx: int = 0, verdict: Verdict | None = None) -> Claim:
    return Claim(id=uuid4(), run_id=run_id, claim_index=idx, text=f"claim {idx}", verdict=verdict)


def make_request(**config_kwargs) -> ObserveRequest:
    return ObserveRequest(
        pipeline_name="test-pipeline",
        query="what is the capital of France?",
        answer="Paris is the capital of France.",
        retrieved_chunks=[],
        config=AttestConfig(**config_kwargs),
    )


def decompose_factory(n: int, verdicts: list[Verdict | None] | None = None):
    async def decompose(request: ObserveRequest, run_id: UUID) -> list[Claim]:
        vs = verdicts or [None] * n
        return [make_claim(run_id, i, v) for i, v in enumerate(vs)]

    return decompose


class EventRecorder:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    async def __call__(self, event: str, data: dict) -> None:
        self.events.append((event, data))

    def names(self) -> list[str]:
        return [name for name, _ in self.events]


# ---------------------------------------------------------------------------
# Event ordering
# ---------------------------------------------------------------------------


async def test_event_ordering():
    store = FakeStore()
    recorder = EventRecorder()
    entailment = FakeVerifier("entailment")
    prober = FakeVerifier("prober")
    independent = FakeVerifier("independent")

    request = make_request()
    detail = await orchestrator.run(
        request,
        decompose=decompose_factory(2),
        entailment=entailment,
        prober=prober,
        independent=independent,
        store=store,
        emit=recorder,
    )

    names = recorder.names()
    assert names[0] == "run.started"
    assert names[1] == "claims.decomposed"
    assert names[-1] == "run.completed"
    assert names.index("claims.decomposed") < names.index("claim.verified")
    assert names.index("claim.verified") < names.index("run.completed")
    assert detail.status == "complete"
    assert detail.total_claims == 2
    assert len(store.saved_claims) == 2


# ---------------------------------------------------------------------------
# Budget cutoff
# ---------------------------------------------------------------------------


async def test_budget_cutoff_marks_remaining_claims_unverifiable_and_run_completes():
    store = FakeStore()
    recorder = EventRecorder()
    # Each entailment call costs 0.002, budget is 0.001 -> exceeded after the
    # very first entailment call, before prober/independent for that claim,
    # and before entailment is even attempted for subsequent claims.
    entailment = FakeVerifier("entailment", cost=0.002)
    prober = FakeVerifier("prober", cost=0.002)
    independent = FakeVerifier("independent", cost=0.002)

    request = make_request(budget_usd=0.001)
    detail = await orchestrator.run(
        request,
        decompose=decompose_factory(3),
        entailment=entailment,
        prober=prober,
        independent=independent,
        store=store,
        emit=recorder,
    )

    assert detail.status == "complete"  # partial results beat no results
    assert len(entailment.calls) == 1  # only the first claim got an entailment call
    assert len(prober.calls) == 0  # budget was already blown by entailment's cost
    assert len(independent.calls) == 0

    budget_claims = [c for c in store.saved_claims if c.rationale == "budget_exceeded"]
    assert len(budget_claims) == 2
    assert all(c.verdict == Verdict.UNVERIFIABLE for c in budget_claims)
    assert len(store.saved_claims) == 3


# ---------------------------------------------------------------------------
# One dead verifier must not kill the run
# ---------------------------------------------------------------------------


async def test_one_verifier_raising_does_not_fail_the_run():
    store = FakeStore()
    recorder = EventRecorder()
    entailment = FakeVerifier("entailment", verdict=Verdict.GROUNDED)
    prober = FakeVerifier("prober", exc=RuntimeError("prober blew up"))
    independent = FakeVerifier("independent", verdict=Verdict.GROUNDED)

    request = make_request()
    detail = await orchestrator.run(
        request,
        decompose=decompose_factory(2),
        entailment=entailment,
        prober=prober,
        independent=independent,
        store=store,
        emit=recorder,
    )

    assert detail.status == "complete"
    assert len(store.saved_claims) == 2
    # entailment + independent both ran and agreed GROUNDED; prober's crash
    # became None, not a run failure.
    assert all(c.verdict == Verdict.GROUNDED for c in detail.claims)
    # entailment + independent verifications persisted per claim; prober's
    # crashed call produced nothing to persist.
    assert len(store.saved_verifications) == 4  # 2 claims x (entailment + independent)


async def test_entailment_raising_does_not_fail_the_run():
    store = FakeStore()
    entailment = FakeVerifier("entailment", exc=RuntimeError("entailment down"))
    prober = FakeVerifier("prober", verdict=Verdict.GROUNDED)
    independent = FakeVerifier("independent", verdict=Verdict.GROUNDED)

    request = make_request()
    detail = await orchestrator.run(
        request,
        decompose=decompose_factory(1),
        entailment=entailment,
        prober=prober,
        independent=independent,
        store=store,
    )

    assert detail.status == "complete"
    assert len(store.saved_claims) == 1
    # entailment produced no Verification (None) but prober/independent still
    # ran and drove the final verdict.
    assert detail.claims[0].verdict == Verdict.GROUNDED


# ---------------------------------------------------------------------------
# Claims resolved at decomposition time (e.g. UNVERIFIABLE) skip fan-out
# ---------------------------------------------------------------------------


async def test_decomposer_resolved_claims_skip_verifier_fanout():
    store = FakeStore()
    entailment = FakeVerifier("entailment")
    prober = FakeVerifier("prober")
    independent = FakeVerifier("independent")

    request = make_request()
    detail = await orchestrator.run(
        request,
        decompose=decompose_factory(2, verdicts=[Verdict.UNVERIFIABLE, None]),
        entailment=entailment,
        prober=prober,
        independent=independent,
        store=store,
    )

    assert detail.status == "complete"
    assert len(entailment.calls) == 1  # only the unresolved claim went to the verifiers
    assert len(store.saved_claims) == 2
    unverifiable = [c for c in store.saved_claims if c.verdict == Verdict.UNVERIFIABLE]
    assert len(unverifiable) == 1
    assert unverifiable[0].rationale is None  # untouched decomposer claim, not budget cutoff


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


async def test_sample_rate_zero_persists_run_as_skipped():
    store = FakeStore()
    recorder = EventRecorder()
    decompose_called = False

    async def decompose(request, run_id):
        nonlocal decompose_called
        decompose_called = True
        return []

    request = make_request(sample_rate=0.0)
    detail = await orchestrator.run(
        request,
        decompose=decompose,
        entailment=FakeVerifier("entailment"),
        prober=FakeVerifier("prober"),
        independent=FakeVerifier("independent"),
        store=store,
        emit=recorder,
    )

    assert detail.status == "skipped"
    assert store.status_history == ["pending", "skipped"]
    assert not decompose_called
    assert recorder.names() == ["run.started", "run.completed"]
    assert len(store.saved_claims) == 0


async def test_sample_rate_one_always_runs():
    store = FakeStore()
    request = make_request(sample_rate=1.0)

    detail = await orchestrator.run(
        request,
        decompose=decompose_factory(1),
        entailment=FakeVerifier("entailment"),
        prober=FakeVerifier("prober"),
        independent=FakeVerifier("independent"),
        store=store,
    )

    assert detail.status == "complete"


# ---------------------------------------------------------------------------
# Decompose failure
# ---------------------------------------------------------------------------


async def test_decompose_failure_marks_run_error_and_does_not_raise():
    store = FakeStore()
    recorder = EventRecorder()

    async def broken_decompose(request, run_id):
        raise ValueError("decomposer exploded")

    request = make_request()
    detail = await orchestrator.run(
        request,
        decompose=broken_decompose,
        entailment=FakeVerifier("entailment"),
        prober=FakeVerifier("prober"),
        independent=FakeVerifier("independent"),
        store=store,
        emit=recorder,
    )

    assert detail.status == "error"
    assert "run.error" in recorder.names()
    assert store.runs[detail.id]["status"] == "error"


# ---------------------------------------------------------------------------
# enable_prober / enable_independent config toggles
# ---------------------------------------------------------------------------


async def test_disabling_prober_and_independent_only_calls_entailment():
    store = FakeStore()
    entailment = FakeVerifier("entailment", verdict=Verdict.GROUNDED)
    prober = FakeVerifier("prober")
    independent = FakeVerifier("independent")

    request = make_request(enable_prober=False, enable_independent=False)
    detail = await orchestrator.run(
        request,
        decompose=decompose_factory(1),
        entailment=entailment,
        prober=prober,
        independent=independent,
        store=store,
    )

    assert len(entailment.calls) == 1
    assert len(prober.calls) == 0
    assert len(independent.calls) == 0
    assert detail.claims[0].verdict == Verdict.GROUNDED


# ---------------------------------------------------------------------------
# Store failures (regression: a Supabase blip used to leave the run stuck
# "running" with no terminal SSE event, hanging the dashboard forever)
# ---------------------------------------------------------------------------


class FailingClaimStore(FakeStore):
    """Fails on the Nth save_claim call, like a transient Supabase error."""

    def __init__(self, fail_on_call: int = 2) -> None:
        super().__init__()
        self.fail_on_call = fail_on_call
        self.save_claim_calls = 0

    async def save_claim(self, claim: Claim) -> None:
        self.save_claim_calls += 1
        if self.save_claim_calls == self.fail_on_call:
            raise RuntimeError("supabase unavailable")
        await super().save_claim(claim)


class FailingSetupStore(FakeStore):
    async def save_chunks(self, run_id: UUID, chunks):
        raise RuntimeError("supabase unavailable")


async def test_store_failure_mid_run_marks_error_and_emits_terminal_event():
    store = FailingClaimStore(fail_on_call=2)
    recorder = EventRecorder()

    detail = await orchestrator.run(
        make_request(),
        decompose=decompose_factory(3),
        entailment=FakeVerifier("entailment"),
        prober=FakeVerifier("prober"),
        independent=FakeVerifier("independent"),
        store=store,
        emit=recorder,
    )

    # No unhandled exception, run is explicitly errored, not left "running".
    assert detail.status == "error"
    assert store.runs[detail.id]["status"] == "error"
    # A terminal event must reach the dashboard.
    assert recorder.names()[-1] == "run.error"
    stage = dict(recorder.events)["run.error"]["stage"]
    assert stage == "persist_claims"


async def test_store_failure_during_setup_marks_error_and_emits_terminal_event():
    store = FailingSetupStore()
    recorder = EventRecorder()

    detail = await orchestrator.run(
        make_request(),
        decompose=decompose_factory(2),
        entailment=FakeVerifier("entailment"),
        prober=FakeVerifier("prober"),
        independent=FakeVerifier("independent"),
        store=store,
        emit=recorder,
    )

    assert detail.status == "error"
    assert recorder.names()[-1] == "run.error"
    assert dict(recorder.events)["run.error"]["stage"] == "persist_setup"


async def test_run_error_still_emitted_when_status_update_also_fails():
    """Even a totally dead store must not swallow the terminal SSE event."""

    class TotallyDeadStore(FailingClaimStore):
        async def update_run_status(self, run_id: UUID, status: str, **fields) -> None:
            if status == "error":
                raise RuntimeError("supabase still down")
            await super().update_run_status(run_id, status, **fields)

    recorder = EventRecorder()
    detail = await orchestrator.run(
        make_request(),
        decompose=decompose_factory(2),
        entailment=FakeVerifier("entailment"),
        prober=FakeVerifier("prober"),
        independent=FakeVerifier("independent"),
        store=TotallyDeadStore(fail_on_call=1),
        emit=recorder,
    )

    assert detail.status == "error"
    assert recorder.names()[-1] == "run.error"
