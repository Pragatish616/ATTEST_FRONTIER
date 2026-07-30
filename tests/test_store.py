"""Store tests against a fake in-memory Supabase client — no network, no
real Supabase project (CLAUDE.md; A4 brief).
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from attest.models import (
    Claim,
    Evidence,
    MutationType,
    Probe,
    RetrievedChunk,
    Verdict,
    Verification,
)
from attest.store import Store

# ---------------------------------------------------------------------------
# Fake Supabase client — mimics the .table(x).insert(...)/.update(...)/
# .select(...)/.eq(...)/.order(...)/.range(...).execute() chain closely
# enough for Store's usage, recording everything in memory.
# ---------------------------------------------------------------------------


class _FakeResult:
    def __init__(self, data: list[dict]) -> None:
        self.data = data


class _FakeQuery:
    def __init__(self, tables: dict[str, list[dict]], table_name: str) -> None:
        self._tables = tables
        self._table_name = table_name
        self._op: str | None = None
        self._payload: list[dict] | dict | None = None
        self._filters: list[tuple[str, str]] = []
        self._order: tuple[str, bool] | None = None
        self._range: tuple[int, int] | None = None

    def insert(self, payload):
        self._op = "insert"
        self._payload = payload if isinstance(payload, list) else [payload]
        return self

    def update(self, payload):
        self._op = "update"
        self._payload = payload
        return self

    def upsert(self, payload):
        self._op = "insert"
        self._payload = payload if isinstance(payload, list) else [payload]
        return self

    def select(self, *_args, **_kwargs):
        if self._op is None:
            self._op = "select"
        return self

    def eq(self, column, value):
        self._filters.append((column, value))
        return self

    def order(self, column, desc: bool = False):
        self._order = (column, desc)
        return self

    def range(self, start: int, end: int):
        self._range = (start, end)
        return self

    def _matches(self, row: dict) -> bool:
        return all(row.get(col) == val for col, val in self._filters)

    def execute(self) -> _FakeResult:
        table = self._tables.setdefault(self._table_name, [])
        if self._op == "insert":
            table.extend(self._payload)
            return _FakeResult(list(self._payload))
        if self._op == "update":
            matched = [row for row in table if self._matches(row)]
            for row in matched:
                row.update(self._payload)
            return _FakeResult(matched)
        if self._op == "select":
            rows = [row for row in table if self._matches(row)]
            if self._order:
                column, desc = self._order
                rows = sorted(rows, key=lambda r: r[column], reverse=desc)
            if self._range:
                start, end = self._range
                rows = rows[start : end + 1]
            return _FakeResult(rows)
        raise AssertionError(f"no operation configured for table {self._table_name}")


class FakeSupabaseClient:
    def __init__(self) -> None:
        self.tables: dict[str, list[dict]] = {}

    def table(self, name: str) -> _FakeQuery:
        return _FakeQuery(self.tables, name)


@pytest.fixture
def fake_client() -> FakeSupabaseClient:
    return FakeSupabaseClient()


@pytest.fixture
def store(fake_client: FakeSupabaseClient) -> Store:
    return Store(client=fake_client)


# ---------------------------------------------------------------------------
# runs
# ---------------------------------------------------------------------------


async def test_create_run_inserts_row(store: Store, fake_client: FakeSupabaseClient):
    from attest.models import AttestConfig, ObserveRequest

    run_id = uuid4()
    request = ObserveRequest(pipeline_name="p", query="q", answer="a", config=AttestConfig())

    await store.create_run(run_id, request)

    rows = fake_client.tables["runs"]
    assert len(rows) == 1
    assert rows[0]["id"] == str(run_id)
    assert rows[0]["status"] == "pending"
    assert rows[0]["pipeline_name"] == "p"


async def test_update_run_status_updates_matching_row(store: Store, fake_client: FakeSupabaseClient):
    from attest.models import ObserveRequest

    run_id = uuid4()
    await store.create_run(run_id, ObserveRequest(pipeline_name="p", query="q", answer="a"))

    await store.update_run_status(run_id, "running")

    row = fake_client.tables["runs"][0]
    assert row["status"] == "running"


async def test_finalize_run_sets_aggregate_fields(store: Store, fake_client: FakeSupabaseClient):
    from attest.models import ObserveRequest

    run_id = uuid4()
    await store.create_run(run_id, ObserveRequest(pipeline_name="p", query="q", answer="a"))

    await store.finalize_run(
        run_id,
        status="complete",
        grounding_score=0.8,
        fragility_score=0.1,
        total_claims=5,
        latency_ms=1200,
        cost_usd=0.002,
    )

    row = fake_client.tables["runs"][0]
    assert row["status"] == "complete"
    assert row["grounding_score"] == 0.8
    assert row["total_claims"] == 5


# ---------------------------------------------------------------------------
# retrieved_chunks
# ---------------------------------------------------------------------------


async def test_save_chunks_persists_and_returns_chunks_with_ids(
    store: Store, fake_client: FakeSupabaseClient
):
    run_id = uuid4()
    chunks_in = [RetrievedChunk(chunk_index=0, text="hello"), RetrievedChunk(chunk_index=1, text="world")]

    result = await store.save_chunks(run_id, chunks_in)

    assert len(result) == 2
    assert all(c.run_id == run_id for c in result)
    assert all(c.id is not None for c in result)
    rows = fake_client.tables["retrieved_chunks"]
    assert len(rows) == 2
    assert rows[0]["text"] == "hello"


async def test_save_chunks_empty_list_is_a_noop(store: Store, fake_client: FakeSupabaseClient):
    result = await store.save_chunks(uuid4(), [])
    assert result == []
    assert "retrieved_chunks" not in fake_client.tables


# ---------------------------------------------------------------------------
# claims — verdict NOT NULL
# ---------------------------------------------------------------------------


async def test_save_claim_rejects_none_verdict(store: Store):
    claim = Claim(id=uuid4(), run_id=uuid4(), claim_index=0, text="x")
    assert claim.verdict is None

    with pytest.raises(ValueError, match="verdict=None"):
        await store.save_claim(claim)


async def test_save_claim_persists_resolved_claim(store: Store, fake_client: FakeSupabaseClient):
    claim = Claim(
        id=uuid4(),
        run_id=uuid4(),
        claim_index=0,
        text="Paris is the capital of France.",
        verdict=Verdict.GROUNDED,
        confidence=0.9,
        disagreement=0.0,
        rationale="entailment verifier drove the verdict",
    )

    await store.save_claim(claim)

    rows = fake_client.tables["claims"]
    assert len(rows) == 1
    assert rows[0]["verdict"] == "GROUNDED"
    assert rows[0]["id"] == str(claim.id)


# ---------------------------------------------------------------------------
# verifications / probes
# ---------------------------------------------------------------------------


async def test_save_verification_serializes_evidence(store: Store, fake_client: FakeSupabaseClient):
    verification = Verification(
        id=uuid4(),
        claim_id=uuid4(),
        verifier="entailment",
        verdict=Verdict.CONTRADICTED,
        evidence=[Evidence(chunk_id=uuid4(), quote_span=(0, 10), stance="refute")],
    )

    await store.save_verification(verification)

    row = fake_client.tables["verifications"][0]
    assert row["verdict"] == "CONTRADICTED"
    assert row["evidence"][0]["stance"] == "refute"
    assert row["evidence"][0]["quote_span"] == [0, 10]


async def test_save_probe_persists_row(store: Store, fake_client: FakeSupabaseClient):
    probe = Probe(
        id=uuid4(),
        claim_id=uuid4(),
        mutation_type=MutationType.NEGATION,
        mutated_text="Paris is not the capital of France.",
        expected_flip=True,
        observed_verdict=Verdict.GROUNDED,
        flipped=False,
    )

    await store.save_probe(probe)

    row = fake_client.tables["probes"][0]
    assert row["mutation_type"] == "negation"
    assert row["flipped"] is False


# ---------------------------------------------------------------------------
# reads
# ---------------------------------------------------------------------------


async def test_list_runs_orders_by_created_at_desc(store: Store, fake_client: FakeSupabaseClient):
    from attest.models import ObserveRequest

    older, newer = uuid4(), uuid4()
    await store.create_run(older, ObserveRequest(pipeline_name="old", query="q", answer="a"))
    await store.create_run(newer, ObserveRequest(pipeline_name="new", query="q", answer="a"))
    # created_at defaults are identical (fake client doesn't set them) —
    # patch them so ordering is meaningful.
    fake_client.tables["runs"][0]["created_at"] = "2026-01-01T00:00:00Z"
    fake_client.tables["runs"][1]["created_at"] = "2026-01-02T00:00:00Z"
    for row in fake_client.tables["runs"]:
        row.setdefault("model", None)
        row.setdefault("grounding_score", None)
        row.setdefault("fragility_score", None)
        row.setdefault("total_claims", None)
        row.setdefault("latency_ms", None)
        row.setdefault("cost_usd", None)

    result = await store.list_runs(limit=10, offset=0)

    assert [r.pipeline_name for r in result] == ["new", "old"]


async def test_get_run_detail_returns_none_when_missing(store: Store):
    assert await store.get_run_detail(uuid4()) is None


async def test_get_run_detail_assembles_nested_trace(store: Store, fake_client: FakeSupabaseClient):
    from attest.models import ObserveRequest

    run_id = uuid4()
    await store.create_run(run_id, ObserveRequest(pipeline_name="p", query="q", answer="a"))
    row = fake_client.tables["runs"][0]
    row.setdefault("created_at", "2026-01-01T00:00:00Z")
    for field in ("grounding_score", "fragility_score", "total_claims", "latency_ms", "cost_usd"):
        row.setdefault(field, None)

    claim = Claim(
        id=uuid4(), run_id=run_id, claim_index=0, text="claim", verdict=Verdict.GROUNDED
    )
    await store.save_claim(claim)
    verification = Verification(
        id=uuid4(), claim_id=claim.id, verifier="entailment", verdict=Verdict.GROUNDED
    )
    await store.save_verification(verification)

    detail = await store.get_run_detail(run_id)

    assert detail is not None
    assert detail.id == run_id
    assert len(detail.claims) == 1
    assert detail.claims[0].id == claim.id
    assert len(detail.claims[0].verifications) == 1
    assert detail.claims[0].verifications[0].verifier == "entailment"
