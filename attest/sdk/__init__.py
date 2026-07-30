"""Public SDK surface per PLAN.md §5.3.

    import attest

    attest.init(api_url=..., api_key=..., sample_rate=0.05)

    @attest.observe(pipeline_name="support-bot")
    def answer(query: str) -> attest.Output:
        ...
        return attest.Output(answer=text, retrieved_chunks=chunks)

    with attest.trace(pipeline_name="support-bot", query=q) as t:
        t.record_chunks(chunks)
        t.record_answer(text)

    chain = attest.wrap(chain, pipeline_name="rag-v2")

Non-negotiable (CLAUDE.md hard rule): **the SDK never raises into the host
pipeline.** Every path — init, the decorator, the trace context manager, the
wrap() integration, and everything below that they call — is wrapped so
that on any internal failure it logs a structlog warning and returns control
to the caller as if ATTEST weren't there. The one thing that is deliberately
*not* swallowed is the host's own function body / chain call: if the wrapped
function raises, that's the host's exception and it propagates untouched —
only ATTEST's own instrumentation around it is defused.

This module holds the shared plumbing (the `Output` sugar type, chunk
normalization, the bounded fire-and-forget submission queue, and the
`_submit_observation` choke point). `attest.sdk.decorator` and
`attest.sdk.wrappers` build the public `observe` / `trace` / `wrap` names on
top of it and are imported at the bottom of this file, after everything
they depend on is defined (safe against the circular import that would
otherwise create, since by then `attest.sdk` already exists in
`sys.modules` with these names bound).
"""

from __future__ import annotations

import asyncio
import functools
import random
import threading
from collections.abc import Callable, Iterable
from typing import Any, ParamSpec, TypeVar

import structlog
from pydantic import BaseModel, ConfigDict, Field

from attest.models import AttestConfig, ObserveRequest, RetrievedChunk

logger = structlog.get_logger(__name__)

P = ParamSpec("P")
T = TypeVar("T")

DEFAULT_QUEUE_MAXSIZE = 1000
DEFAULT_TIMEOUT_S = 5.0
API_PREFIX = "/v1"


# ---------------------------------------------------------------------------
# attest.Output — SDK-facing sugar, not a wire contract (that's RetrievedChunk
# / ObserveRequest in attest.models). Ergonomic to construct from whatever
# shape a host pipeline already has lying around: RetrievedChunk instances,
# plain dicts (LangChain-Document-ish: page_content/content + metadata),
# bare strings, or duck-typed objects with .page_content/.text attributes.
# ---------------------------------------------------------------------------


class Output(BaseModel):
    """Return this from a function decorated with `@attest.observe(...)`."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    answer: str
    retrieved_chunks: list[Any] = Field(default_factory=list)


def _safe(func: Callable[P, T]) -> Callable[P, T | None]:
    """Wrap `func` so any exception is logged and swallowed, never raised.

    This is the workhorse behind the "never raises into the host pipeline"
    guarantee: apply it to every ATTEST-internal code path that runs around
    (not: instead of) the host's own call.
    """

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T | None:
        try:
            return func(*args, **kwargs)
        except Exception:  # noqa: BLE001 - deliberate: never raise into the host
            logger.warning("attest_sdk_internal_error", where=func.__qualname__, exc_info=True)
            return None

    return wrapper


def _normalize_chunks(chunks: Iterable[Any] | None) -> list[RetrievedChunk]:
    """Best-effort conversion of whatever `retrieved_chunks` shape a host
    pipeline hands us into `attest.models.RetrievedChunk`. Any single bad
    entry is skipped (logged) rather than aborting the whole batch — a
    malformed chunk must never take down submission of the rest of the run.
    """
    normalized: list[RetrievedChunk] = []
    for idx, chunk in enumerate(chunks or []):
        try:
            if isinstance(chunk, RetrievedChunk):
                normalized.append(chunk)
                continue
            if isinstance(chunk, dict):
                data: dict[str, Any] = dict(chunk)
                data.setdefault("chunk_index", idx)
                if "text" not in data:
                    for alt in ("page_content", "content", "chunk_text"):
                        if alt in data:
                            data["text"] = data.pop(alt)
                            break
                # LangChain Document-as-dict shape: {"page_content": ..., "metadata": {...}}
                metadata = data.pop("metadata", None) or {}
                data.setdefault("source_id", metadata.get("source_id") or metadata.get("id"))
                data.setdefault("source_url", metadata.get("source_url") or metadata.get("source"))
                normalized.append(RetrievedChunk(**data))
                continue
            if isinstance(chunk, str):
                normalized.append(RetrievedChunk(chunk_index=idx, text=chunk))
                continue
            # Duck-typed object, e.g. a LangChain Document (.page_content,
            # .metadata) or LlamaIndex NodeWithScore (.text/.node, .score).
            text = (
                getattr(chunk, "page_content", None)
                or getattr(chunk, "text", None)
                or str(chunk)
            )
            metadata = getattr(chunk, "metadata", None) or {}
            score = getattr(chunk, "score", None)
            normalized.append(
                RetrievedChunk(
                    chunk_index=idx,
                    text=text,
                    source_id=metadata.get("source_id") or metadata.get("id"),
                    source_url=metadata.get("source_url") or metadata.get("source"),
                    score=score if isinstance(score, int | float) else None,
                )
            )
        except Exception:  # noqa: BLE001
            logger.warning("attest_sdk_chunk_normalize_failed", index=idx, exc_info=True)
            continue
    return normalized


async def _put_drop_oldest(queue: asyncio.Queue, item: Any) -> None:
    """Put `item` on `queue`; if full, drop the oldest queued item instead of
    blocking. A host app's request latency must never depend on ATTEST's
    backend being reachable or fast (CLAUDE.md / PLAN.md §5.3), so overflow
    must never be handled by waiting for room.

    Extracted as a pure, directly-testable function: given a plain
    `asyncio.Queue`, it can be unit tested without touching threads, HTTP, or
    the rest of the client.
    """
    if queue.full():
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            pass
    queue.put_nowait(item)


class _AttestClient:
    """Owns the background worker thread, its own event loop, the bounded
    `asyncio.Queue`, and the HTTP submission of queued `ObserveRequest`s.

    A dedicated thread + loop (rather than piggy-backing on whatever loop the
    host happens to be running, or requiring one) is what lets `attest.init`
    and the decorator/context-manager/wrap paths work uniformly whether the
    host pipeline is sync (Flask-style `def answer(query): ...`) or async —
    submission is always a non-blocking hand-off via
    `asyncio.run_coroutine_threadsafe`, never an `await` on the caller's own
    event loop.
    """

    def __init__(
        self,
        *,
        api_url: str,
        api_key: str | None,
        sample_rate: float,
        queue_maxsize: int,
        timeout_s: float,
    ) -> None:
        self.api_url = (api_url or "").rstrip("/")
        self.api_key = api_key
        self.sample_rate = sample_rate
        self.timeout_s = timeout_s
        self.dropped_count = 0

        self._queue_maxsize = queue_maxsize
        self._loop: asyncio.AbstractEventLoop | None = None
        self._queue: asyncio.Queue[ObserveRequest] | None = None
        self._ready = threading.Event()
        self._thread = threading.Thread(
            target=self._run_loop, name="attest-sdk-worker", daemon=True
        )
        self._thread.start()
        # Bounded wait for the worker loop to come up; if it doesn't (e.g. no
        # threads available in this environment), submit() below just no-ops
        # forever rather than raising.
        self._ready.wait(timeout=2.0)

    def _run_loop(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        self._queue = asyncio.Queue(maxsize=self._queue_maxsize)
        loop.create_task(self._consume())
        self._ready.set()
        try:
            loop.run_forever()
        finally:
            loop.close()

    async def _consume(self) -> None:
        import httpx

        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            while True:
                assert self._queue is not None
                request = await self._queue.get()
                try:
                    await self._send(client, request)
                except Exception as exc:  # noqa: BLE001 - one dead submit must not kill the worker
                    logger.warning("attest_sdk_submit_failed", error=str(exc))

    async def _send(self, client: Any, request: ObserveRequest) -> None:
        headers = {"content-type": "application/json"}
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"
        url = f"{self.api_url}{API_PREFIX}/observe"
        response = await client.post(url, content=request.model_dump_json(), headers=headers)
        response.raise_for_status()

    def submit(self, request: ObserveRequest) -> None:
        if self._loop is None or self._queue is None:
            logger.warning("attest_sdk_worker_not_ready")
            return
        if self.sample_rate < 1.0 and random.random() > self.sample_rate:
            return

        queue = self._queue

        async def _enqueue() -> None:
            if queue.full():
                self.dropped_count += 1
                logger.warning(
                    "attest_sdk_queue_overflow_drop_oldest", pipeline=request.pipeline_name
                )
            await _put_drop_oldest(queue, request)

        try:
            asyncio.run_coroutine_threadsafe(_enqueue(), self._loop)
        except Exception:  # noqa: BLE001
            logger.warning("attest_sdk_enqueue_failed", exc_info=True)


_client: _AttestClient | None = None


def init(
    *,
    api_url: str,
    api_key: str | None = None,
    sample_rate: float = 1.0,
    queue_maxsize: int = DEFAULT_QUEUE_MAXSIZE,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> None:
    """`attest.init(api_url=..., api_key=..., sample_rate=0.05)` — PLAN.md §5.3.

    Configures the process-global SDK client and starts its background
    submission worker. Safe to call multiple times (each call replaces the
    previous client). Never raises: a bad `api_url`, unreachable host, or any
    other setup failure is logged and leaves the SDK in a no-op state.
    """
    global _client
    try:
        _client = _AttestClient(
            api_url=api_url,
            api_key=api_key,
            sample_rate=sample_rate,
            queue_maxsize=queue_maxsize,
            timeout_s=timeout_s,
        )
        logger.info("attest_sdk_initialized", api_url=api_url, sample_rate=sample_rate)
    except Exception:  # noqa: BLE001
        logger.warning("attest_sdk_init_failed", exc_info=True)
        _client = None


def _submit_observation(
    *,
    pipeline_name: str,
    query: str,
    answer: str,
    chunks: Iterable[Any] | None,
    model: str | None = None,
    config: AttestConfig | None = None,
) -> None:
    """Single choke point every SDK entrypoint funnels through. Builds the
    frozen `ObserveRequest` shape (PLAN.md §5.2) and hands it to the client's
    fire-and-forget queue. Never raises.
    """
    try:
        if _client is None:
            logger.warning("attest_sdk_not_initialized", pipeline_name=pipeline_name)
            return
        request = ObserveRequest(
            pipeline_name=pipeline_name,
            query=query or "",
            answer=answer or "",
            retrieved_chunks=_normalize_chunks(chunks),
            model=model,
            config=config or AttestConfig(),
        )
        _client.submit(request)
    except Exception:  # noqa: BLE001
        logger.warning(
            "attest_sdk_submit_observation_failed", pipeline_name=pipeline_name, exc_info=True
        )


# Imported at the bottom, deliberately: `decorator.py` and `wrappers.py` do
# `from attest.sdk import Output, _safe, _submit_observation` — by the time
# Python executes those imports (triggered by the two lines below), this
# module is already in `sys.modules` with all the names above bound, so the
# lookup succeeds even though this file isn't finished executing yet.
from attest.sdk.decorator import observe, trace  # noqa: E402
from attest.sdk.wrappers import wrap  # noqa: E402

__all__ = ["init", "observe", "trace", "wrap", "Output"]
