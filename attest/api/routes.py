"""REST routes per PLAN.md §5.2: `/observe`, `/runs`, `/runs/{run_id}`,
`/evaluate`, `/health`.

Dependency wiring for the pieces A1/A2/A3 own (the decomposer and the three
verifiers) lives here, lazily, behind small `get_*()` functions instead of
module-level imports — those files may still be empty stubs (see
CLAUDE.md's dependency-injection rule). A missing/incomplete module turns
into a `503` naming exactly which class and file is missing, instead of an
`ImportError` traceback crashing the whole app at import time. `/health`,
`/runs`, and `/runs/{run_id}` never depend on the verifiers at all, so they
keep working even before A1/A2/A3 land.
"""

from __future__ import annotations

import importlib
from collections.abc import Awaitable, Callable
from functools import lru_cache
from typing import Any
from uuid import UUID, uuid4

import structlog
from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from pydantic import BaseModel, Field

from attest import orchestrator
from attest.api import stream
from attest.config import settings
from attest.models import (
    Claim,
    ObserveRequest,
    ObserveResponse,
    Probe,
    RetrievedChunk,
    RunDetail,
    RunSummary,
    Verification,
)
from attest.store import Store, StoreProtocol

logger = structlog.get_logger(__name__)

router = APIRouter()


class DependencyNotReady(RuntimeError):
    """Raised when a component owned by another in-progress agent (the
    decomposer or one of the three verifiers) hasn't landed with real
    content yet. Route handlers catch this and turn it into an HTTP 503
    with the same message, naming exactly what's missing.
    """


def _load_attr(module_path: str, attr_name: str) -> Any:
    try:
        module = importlib.import_module(module_path)
        return getattr(module, attr_name)
    except (ImportError, AttributeError) as exc:
        raise DependencyNotReady(
            f"{attr_name} is not available yet in {module_path} ({exc}). "
            "This route depends on another agent's file landing with real "
            "content — see PLAN.md §7."
        ) from exc


@lru_cache(maxsize=1)
def get_store() -> StoreProtocol:
    return Store()


@lru_cache(maxsize=1)
def get_entailment_verifier() -> Any:
    return _load_attr("attest.verifiers.entailment", "EntailmentVerifier")()


@lru_cache(maxsize=1)
def get_prober_verifier() -> Any:
    # A2 shipped `AdversarialProber(entailment_verifier: VerifierProtocol)`,
    # not the no-arg `ProberVerifier` originally assumed here — it depends on
    # the entailment verifier via dependency injection rather than importing
    # it, so it gets the same cached singleton `get_entailment_verifier()`
    # already returns.
    prober_cls = _load_attr("attest.verifiers.prober", "AdversarialProber")
    return prober_cls(entailment_verifier=get_entailment_verifier())


@lru_cache(maxsize=1)
def get_independent_verifier() -> Any:
    return _load_attr("attest.verifiers.independent", "IndependentVerifier")()


@lru_cache(maxsize=1)
def get_decompose() -> orchestrator.DecomposeFn:
    # A1 shipped `decompose(answer: str, query: str, *, run_id: UUID)`, not
    # the `(request: ObserveRequest, run_id: UUID)` shape `orchestrator.run`
    # calls it with — adapt here rather than in either agent's file.
    real_decompose = _load_attr("attest.verifiers.decomposer", "decompose")

    async def _decompose_adapter(request: ObserveRequest, run_id: UUID) -> list[Claim]:
        return await real_decompose(request.answer, request.query, run_id=run_id)

    return _decompose_adapter


def get_probes_hook() -> Callable[[Claim, Verification], Awaitable[list[Probe]]]:
    """Bridges A2's probe-exposure mechanism to `orchestrator.run`'s optional
    `get_probes` hook.

    `AdversarialProber` has no way to return `Probe` rows through the plain
    `VerifierProtocol.verify()` call `orchestrator.run` makes — it exposes
    them as a `last_probes` side-effect attribute instead (see
    `attest/verifiers/prober.py`'s module docstring). This is safe to read
    right after the fact because `orchestrator.run` processes claims in a
    strictly sequential `for` loop — `prober.verify(claim, ctx)` and this
    hook's call for that same claim always complete before the next claim's
    `verify()` call begins, so there is no cross-claim race on the shared
    prober instance's `last_probes`.
    """

    async def _get_probes(claim: Claim, verification: Verification) -> list[Probe]:
        prober = get_prober_verifier()
        return list(getattr(prober, "last_probes", None) or [])

    return _get_probes


# ---------------------------------------------------------------------------
# POST /observe
# ---------------------------------------------------------------------------


async def _run_in_background(
    request: ObserveRequest,
    run_id: UUID,
    decompose: orchestrator.DecomposeFn,
    entailment: Any,
    prober: Any,
    independent: Any,
    store: StoreProtocol,
    emit: orchestrator.EmitFn,
    get_probes: Callable[[Claim, Verification], Awaitable[list[Probe]]],
) -> None:
    try:
        await orchestrator.run(
            request,
            run_id=run_id,
            decompose=decompose,
            entailment=entailment,
            prober=prober,
            independent=independent,
            store=store,
            emit=emit,
            get_probes=get_probes,
        )
    except Exception:  # noqa: BLE001 - a background task must never die silently
        logger.exception("observe_background_task_failed", run_id=str(run_id))


@router.post("/observe", response_model=ObserveResponse, status_code=202)
async def observe(request: ObserveRequest, background_tasks: BackgroundTasks) -> ObserveResponse:
    try:
        store = get_store()
        entailment = get_entailment_verifier()
        prober = get_prober_verifier()
        independent = get_independent_verifier()
        decompose = get_decompose()
    except DependencyNotReady as exc:
        logger.error("observe_dependency_not_ready", error=str(exc))
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    # Clamp the caller-supplied spend ceiling.
    #
    # AttestConfig.budget_usd validates only `> 0`, and it arrives in the
    # request body — so a caller can ask us to spend arbitrarily much of our own
    # provider credit on one request. The model itself is part of the frozen
    # §5.3 SDK surface and may not gain a validator, so the ceiling is applied
    # here, at the trust boundary, instead. Clamping (rather than rejecting)
    # keeps well-behaved SDK callers working untouched.
    if request.config.budget_usd > settings.max_budget_usd:
        logger.warning(
            "budget_clamped",
            requested_usd=request.config.budget_usd,
            allowed_usd=settings.max_budget_usd,
        )
        request = request.model_copy(
            update={
                "config": request.config.model_copy(
                    update={"budget_usd": settings.max_budget_usd}
                )
            }
        )

    run_id = uuid4()
    emit = stream.make_emitter(run_id)
    get_probes = get_probes_hook()

    background_tasks.add_task(
        _run_in_background,
        request,
        run_id,
        decompose,
        entailment,
        prober,
        independent,
        store,
        emit,
        get_probes,
    )

    return ObserveResponse(run_id=run_id, status="pending")


# ---------------------------------------------------------------------------
# GET /runs, GET /runs/{run_id}
# ---------------------------------------------------------------------------


class RunsListResponse(BaseModel):
    runs: list[RunSummary]


def _store_unavailable(operation: str, exc: Exception) -> HTTPException:
    """Turn any Supabase/transport failure into a 503 with no internals leaked.

    Without this, a network blip or an unreachable Supabase project propagates
    the raw `httpx.ConnectError` out of the handler and FastAPI returns a 500
    with a full stack trace in the response body. A judge refreshing the runs
    list mid-demo should see "trace store unavailable", not our traceback. The
    real exception goes to the structured log, where it belongs.
    """
    logger.error("store_unavailable", operation=operation, error=str(exc))
    return HTTPException(
        status_code=503,
        detail="Trace store unavailable; the run history could not be read.",
    )


@router.get("/runs", response_model=RunsListResponse)
async def list_runs(
    limit: int = Query(default=20, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> RunsListResponse:
    store = get_store()
    try:
        runs = await store.list_runs(limit=limit, offset=offset)
    except Exception as exc:  # noqa: BLE001 - see _store_unavailable
        raise _store_unavailable("list_runs", exc) from exc
    return RunsListResponse(runs=runs)


@router.get("/runs/{run_id}", response_model=RunDetail)
async def get_run(run_id: UUID) -> RunDetail:
    store = get_store()
    try:
        detail = await store.get_run_detail(run_id)
    except Exception as exc:  # noqa: BLE001 - see _store_unavailable
        raise _store_unavailable("get_run_detail", exc) from exc
    if detail is None:
        raise HTTPException(status_code=404, detail=f"run {run_id} not found")
    return detail


# ---------------------------------------------------------------------------
# POST /evaluate
# ---------------------------------------------------------------------------


class EvaluateRequest(BaseModel):
    dataset: str
    n: int = 100
    ablation: str | None = None


class EvaluateResult(BaseModel):
    system: str
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    cost_per_claim: float | None = None
    p95_latency_ms: float | None = None


class EvaluateResponse(BaseModel):
    dataset: str
    n: int
    ablation: str | None
    results: list[EvaluateResult] = Field(default_factory=list)
    status: str = "not_implemented"


@router.post("/evaluate", response_model=EvaluateResponse)
async def evaluate(request: EvaluateRequest) -> EvaluateResponse:
    """Route shape + response contract for the benchmark table (PLAN.md
    §5.2, §10). This is a thin stub: it returns the correct shape with an
    empty result set until A6's `bench/` harness exists to call into. Wire
    it up as `from bench... import run_benchmark` (lazily, same pattern as
    the verifier getters above) once that lands — see final report.
    """
    return EvaluateResponse(
        dataset=request.dataset,
        n=request.n,
        ablation=request.ablation,
        results=[],
        status="not_implemented",
    )


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    ok: bool
    version: str


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(ok=True, version=settings.app_version)


__all__ = [
    "router",
    "DependencyNotReady",
    "get_store",
    "get_entailment_verifier",
    "get_prober_verifier",
    "get_independent_verifier",
    "get_decompose",
    "get_probes_hook",
]


# ---------------------------------------------------------------------------
# POST /demo/query — DEMO-ONLY convenience route
#
# Additive to PLAN.md §5.2 (see CONTRACT_CHANGE_REQUEST.md): it changes no
# existing shape, so consumers built against /observe, /runs and the SSE
# stream are unaffected. It exists so the dashboard can accept a free-text
# question instead of only replaying stored runs — ATTEST itself verifies,
# it does not generate, so *something* has to run a RAG pipeline first.
#
# The demo package is imported lazily and its absence is a clean 503, so the
# `attest` package never hard-depends on `demo/`. Drop this router before any
# production deployment.
# ---------------------------------------------------------------------------

DEMO_PIPELINE_NAME = "northwind-demo-rag"


class DemoQueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    k: int = Field(default=4, ge=1, le=12)


class DemoQueryChunk(BaseModel):
    chunk_index: int
    source_id: str | None = None
    source_url: str | None = None
    text: str
    score: float | None = None


class DemoQueryResponse(BaseModel):
    """The answer is returned synchronously; verdicts arrive asynchronously.

    Callers render `answer` immediately, then poll `GET /runs/{run_id}` (or
    subscribe to the SSE stream) for the verification trail.
    """

    run_id: UUID
    query: str
    answer: str
    retrieved_chunks: list[DemoQueryChunk]


@router.post("/demo/query", response_model=DemoQueryResponse, status_code=202)
async def demo_query(
    body: DemoQueryRequest, background_tasks: BackgroundTasks
) -> DemoQueryResponse:
    """Run the bundled demo RAG over `body.query`, then verify it with ATTEST."""
    try:
        from demo.rag_pipeline import answer_query
    except Exception as exc:  # noqa: BLE001 - missing demo package is a 503, not a crash
        logger.error("demo_pipeline_unavailable", error=str(exc))
        raise HTTPException(
            status_code=503,
            detail=(
                "Demo RAG pipeline unavailable. Run `python demo/build_corpus.py` "
                f"and start the API from the repo root. Underlying error: {exc}"
            ),
        ) from exc

    try:
        store = get_store()
        entailment = get_entailment_verifier()
        prober = get_prober_verifier()
        independent = get_independent_verifier()
        decompose = get_decompose()
    except DependencyNotReady as exc:
        logger.error("demo_query_dependency_not_ready", error=str(exc))
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    try:
        answer, docs = await answer_query(body.query, k=body.k)
    except Exception as exc:  # noqa: BLE001 - upstream LLM/corpus failure -> 502
        logger.error("demo_query_generation_failed", error=str(exc))
        raise HTTPException(
            status_code=502,
            detail=(
                "The demo RAG pipeline could not generate an answer (usually an "
                f"LLM provider error or an unbuilt corpus). Underlying error: {exc}"
            ),
        ) from exc

    chunks = [
        DemoQueryChunk(
            chunk_index=doc.chunk_index,
            source_id=doc.source_id,
            source_url=doc.source_url,
            text=doc.text,
            score=doc.score,
        )
        for doc in docs
    ]

    request = ObserveRequest(
        pipeline_name=DEMO_PIPELINE_NAME,
        query=body.query,
        answer=answer,
        retrieved_chunks=[RetrievedChunk(**c.model_dump()) for c in chunks],
    )

    run_id = uuid4()
    background_tasks.add_task(
        _run_in_background,
        request,
        run_id,
        decompose,
        entailment,
        prober,
        independent,
        store,
        stream.make_emitter(run_id),
        get_probes_hook(),
    )

    logger.info(
        "demo_query_submitted", run_id=str(run_id), query=body.query[:120], k=body.k
    )
    return DemoQueryResponse(
        run_id=run_id, query=body.query, answer=answer, retrieved_chunks=chunks
    )
