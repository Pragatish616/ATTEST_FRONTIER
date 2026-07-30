"""Supabase read/write layer.

Persists `runs`, `retrieved_chunks`, `claims`, `verifications`, and `probes`
exactly per the frozen schema in `migrations/001_init.sql` (PLAN.md §5.1) —
same table/column names. Every ATTEST model in `attest.models` was designed
to mirror its table row 1:1, so `model.model_dump(mode="json")` already
produces a JSON-safe payload with the right column names (UUIDs and enums
become plain strings, datetimes become ISO strings).

Writes land incrementally, one row at a time, as results come in (CLAUDE.md:
"write incrementally... so Realtime subscribers actually see live
progress") — never batched at the end.

`claims.verdict` is `NOT NULL` in the DB. `Store.save_claim` refuses to
insert a claim whose `verdict` is still `None` rather than let Postgres
reject it with a less useful error — that should only ever happen after the
reconciler runs, or when the decomposer resolves a claim directly (e.g.
`UNVERIFIABLE` for a subjective/predictive claim).
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable
from uuid import UUID, uuid4

import structlog
from supabase import Client, create_client

from attest.config import settings
from attest.models import (
    Chunk,
    Claim,
    ClaimDetail,
    ObserveRequest,
    Probe,
    RetrievedChunk,
    RunDetail,
    RunStatus,
    RunSummary,
    Verification,
)

logger = structlog.get_logger(__name__)


@runtime_checkable
class StoreProtocol(Protocol):
    """Structural interface `attest.orchestrator.run` depends on.

    Lets orchestrator tests inject a lightweight in-memory fake instead of a
    real Supabase-backed `Store` — see `tests/test_orchestrator.py`. `Store`
    below implements this; nothing about the Protocol is Supabase-specific.
    """

    async def create_run(self, run_id: UUID, request: ObserveRequest) -> None: ...

    async def update_run_status(self, run_id: UUID, status: RunStatus, **fields: Any) -> None: ...

    async def save_chunks(self, run_id: UUID, chunks: list[RetrievedChunk]) -> list[Chunk]: ...

    async def save_claim(self, claim: Claim) -> None: ...

    async def save_verification(self, verification: Verification) -> None: ...

    async def save_probe(self, probe: Probe) -> None: ...

    async def finalize_run(self, run_id: UUID, **fields: Any) -> None: ...


class Store:
    """Supabase-backed implementation of `StoreProtocol`.

    `supabase-py`'s default client (`create_client`) is synchronous. Rather
    than depend on its optional async client shape (which has moved around
    across versions), every call is pushed through `asyncio.to_thread` so
    this module still honors "all I/O is async" (CLAUDE.md) without adding a
    second Supabase client implementation to keep in sync.

    Inject a fake client in tests: anything exposing the same
    `.table(name).insert(...)/.update(...)/.select(...)/.execute()` chain
    that `supabase-py` does. See `tests/test_store.py` for the fake used
    here — it just records what would have been written, no network.
    """

    def __init__(self, client: Client | None = None) -> None:
        self._client = client or create_client(settings.supabase_url, settings.supabase_key)

    async def _exec(self, fn: Callable[[], Any]) -> Any:
        return await asyncio.to_thread(fn)

    # -- runs ----------------------------------------------------------

    async def create_run(self, run_id: UUID, request: ObserveRequest) -> None:
        payload = {
            "id": str(run_id),
            "pipeline_name": request.pipeline_name,
            "query": request.query,
            "answer": request.answer,
            "model": request.model,
            "status": "pending",
        }
        await self._exec(lambda: self._client.table("runs").insert(payload).execute())
        logger.info("run_created", run_id=str(run_id), pipeline_name=request.pipeline_name)

    async def update_run_status(self, run_id: UUID, status: RunStatus, **fields: Any) -> None:
        payload: dict[str, Any] = {"status": status, **fields}
        await self._exec(
            lambda: self._client.table("runs").update(payload).eq("id", str(run_id)).execute()
        )
        logger.info("run_status_updated", run_id=str(run_id), status=status)

    async def finalize_run(self, run_id: UUID, **fields: Any) -> None:
        await self._exec(
            lambda: self._client.table("runs").update(fields).eq("id", str(run_id)).execute()
        )
        logger.info("run_finalized", run_id=str(run_id), **fields)

    # -- retrieved_chunks ------------------------------------------------

    async def save_chunks(self, run_id: UUID, chunks: list[RetrievedChunk]) -> list[Chunk]:
        persisted = [Chunk(id=uuid4(), run_id=run_id, **rc.model_dump()) for rc in chunks]
        if persisted:
            rows = [c.model_dump(mode="json") for c in persisted]
            await self._exec(lambda: self._client.table("retrieved_chunks").insert(rows).execute())
        return persisted

    # -- claims ----------------------------------------------------------

    async def save_claim(self, claim: Claim) -> None:
        if claim.verdict is None:
            raise ValueError(
                f"refusing to persist claim {claim.id} with verdict=None "
                "(claims.verdict is NOT NULL) — reconcile or resolve it first"
            )
        payload = claim.model_dump(mode="json")
        await self._exec(lambda: self._client.table("claims").insert(payload).execute())

    # -- verifications -----------------------------------------------------

    async def save_verification(self, verification: Verification) -> None:
        payload = verification.model_dump(mode="json")
        await self._exec(lambda: self._client.table("verifications").insert(payload).execute())

    # -- probes ------------------------------------------------------------

    async def save_probe(self, probe: Probe) -> None:
        payload = probe.model_dump(mode="json")
        await self._exec(lambda: self._client.table("probes").insert(payload).execute())

    # -- reads (GET /runs, GET /runs/{run_id}) ------------------------------

    async def list_runs(self, *, limit: int = 20, offset: int = 0) -> list[RunSummary]:
        result = await self._exec(
            lambda: self._client.table("runs")
            .select("*")
            .order("created_at", desc=True)
            .range(offset, offset + limit - 1)
            .execute()
        )
        return [RunSummary.model_validate(row) for row in result.data]

    async def get_run_detail(self, run_id: UUID) -> RunDetail | None:
        run_result = await self._exec(
            lambda: self._client.table("runs").select("*").eq("id", str(run_id)).execute()
        )
        if not run_result.data:
            return None
        run_row = run_result.data[0]

        chunks_result = await self._exec(
            lambda: self._client.table("retrieved_chunks")
            .select("*")
            .eq("run_id", str(run_id))
            .order("chunk_index")
            .execute()
        )
        claims_result = await self._exec(
            lambda: self._client.table("claims")
            .select("*")
            .eq("run_id", str(run_id))
            .order("claim_index")
            .execute()
        )

        claim_details: list[dict[str, Any]] = []
        for claim_row in claims_result.data:
            claim_id = claim_row["id"]
            verifications_result = await self._exec(
                lambda cid=claim_id: self._client.table("verifications")
                .select("*")
                .eq("claim_id", cid)
                .execute()
            )
            probes_result = await self._exec(
                lambda cid=claim_id: self._client.table("probes")
                .select("*")
                .eq("claim_id", cid)
                .execute()
            )
            claim_details.append(
                ClaimDetail.model_validate(
                    {
                        **claim_row,
                        "verifications": verifications_result.data,
                        "probes": probes_result.data,
                    }
                ).model_dump(mode="json")
            )

        return RunDetail.model_validate(
            {
                **run_row,
                "retrieved_chunks": chunks_result.data,
                "claims": claim_details,
            }
        )


__all__ = ["Store", "StoreProtocol"]
