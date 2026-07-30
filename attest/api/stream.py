"""SSE stream for `GET /runs/{run_id}/stream`.

Event names are frozen (PLAN.md §5.2): `run.started`, `claims.decomposed`,
`claim.verified`, `probe.completed`, `run.completed`, `run.error`. Every
payload is `{"run_id": ..., "data": {...}}`.

`EventBus` is the pub/sub glue between `attest.orchestrator.run` (the
publisher — see its `emit` parameter) and this module's SSE endpoint (the
subscriber). It's deliberately just an `asyncio.Queue` per run_id, in
process memory, so it's trivial to test without a running HTTP server (see
`tests/test_stream.py`): publish directly and read back off the generator.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from collections.abc import AsyncIterator
from typing import Any
from uuid import UUID

import structlog
from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse

logger = structlog.get_logger(__name__)

HEARTBEAT_SECONDS = 15
TERMINAL_EVENTS = frozenset({"run.completed", "run.error"})

# Per-run replay buffer. /observe returns 202 and starts the orchestrator as a
# BackgroundTask immediately, so events routinely fire before the dashboard has
# opened its EventSource. Without replay those events are dropped forever and a
# fast run that finishes pre-subscribe leaves the stream heartbeating with no
# terminal event. HISTORY_LIMIT caps memory per run; HISTORY_TTL_SECONDS is the
# grace period after a terminal event before a finished run's history is evicted
# (GET /runs/{run_id} is the source of truth for long-finished runs).
HISTORY_LIMIT = 1000
HISTORY_TTL_SECONDS = 300.0

# Cap on concurrent SSE subscribers per run.
#
# Every subscriber is an open connection holding an unbounded asyncio.Queue that
# publish() fans out to, so without a cap anyone can open connections to one
# run_id until the process runs out of memory — and each extra queue also
# multiplies the work every publish does. Legitimate use is one or two dashboard
# tabs per run, so 8 is generous. Rate limiting deliberately exempts the stream
# path (a normal client holds one long-lived connection), which makes this cap
# the only thing standing between the bus and a slow memory exhaustion.
MAX_SUBSCRIBERS_PER_RUN = 8


class TooManySubscribers(RuntimeError):
    """Raised by `EventBus.subscribe` when a run is already at its cap."""


class EventBus:
    """In-memory pub/sub keyed by run_id.

    Supports multiple subscribers per run (e.g. more than one dashboard tab
    watching the same run) by fanning each publish out to every subscribed
    queue, and replays already-published events to late subscribers.

    SINGLE-PROCESS ONLY. This is plain process memory with no shared backing
    store. Do not run with `--workers > 1`, behind a load balancer, or as
    multiple replicas: `/observe`'s background task and a client's
    `/runs/{run_id}/stream` connection would land on different processes and
    never see each other's events — the stream would just heartbeat forever
    with no data and no error. Scaling out requires a shared pub/sub backend
    (Postgres LISTEN/NOTIFY is the natural fit given the existing Supabase
    dependency; Redis pub/sub otherwise) behind this same
    publish/subscribe/unsubscribe interface. That is a deployment-contract
    change — write it up in CONTRACT_CHANGE_REQUEST.md first.
    """

    def __init__(self) -> None:
        self._subscribers: dict[UUID, list[asyncio.Queue]] = {}
        # run_id -> (events, finished_at). `finished_at` is None until a
        # terminal event is published; after that the entry is eligible for
        # eviction once HISTORY_TTL_SECONDS has elapsed.
        self._history: dict[UUID, tuple[deque, float | None]] = {}

    def _evict_expired(self) -> None:
        now = time.monotonic()
        for run_id, (_events, finished_at) in list(self._history.items()):
            if finished_at is not None and now - finished_at > HISTORY_TTL_SECONDS:
                self._history.pop(run_id, None)

    async def publish(self, run_id: UUID, event: str, data: dict[str, Any]) -> None:
        payload = {"run_id": str(run_id), "data": data}
        self._evict_expired()
        events, finished_at = self._history.get(run_id, (deque(maxlen=HISTORY_LIMIT), None))
        events.append((event, payload))
        if event in TERMINAL_EVENTS:
            finished_at = time.monotonic()
        self._history[run_id] = (events, finished_at)
        for queue in list(self._subscribers.get(run_id, [])):
            await queue.put((event, payload))

    def subscribe(self, run_id: UUID) -> asyncio.Queue:
        """Subscribe to `run_id`, replaying any already-published events first.

        The replay is what makes a late subscriber correct: the queue is
        pre-loaded, in order, with everything published so far, including a
        terminal event if the run already finished.
        """
        existing = self._subscribers.get(run_id, [])
        if len(existing) >= MAX_SUBSCRIBERS_PER_RUN:
            logger.warning(
                "subscriber_cap_reached", run_id=str(run_id), cap=MAX_SUBSCRIBERS_PER_RUN
            )
            raise TooManySubscribers(
                f"run {run_id} already has {MAX_SUBSCRIBERS_PER_RUN} stream subscribers"
            )

        queue: asyncio.Queue = asyncio.Queue()
        self._evict_expired()
        events, _finished_at = self._history.get(run_id, (deque(maxlen=HISTORY_LIMIT), None))
        for event, payload in list(events):
            queue.put_nowait((event, payload))
        self._subscribers.setdefault(run_id, []).append(queue)
        return queue

    def unsubscribe(self, run_id: UUID, queue: asyncio.Queue) -> None:
        subs = self._subscribers.get(run_id)
        if not subs:
            return
        if queue in subs:
            subs.remove(queue)
        if not subs:
            self._subscribers.pop(run_id, None)


# Process-wide bus. A single FastAPI process serves both /observe (the
# publisher, via make_emitter) and /runs/{run_id}/stream (the subscriber, via
# sse_event_generator) — see attest/api/routes.py.
event_bus = EventBus()


def make_emitter(run_id: UUID, bus: EventBus = event_bus):
    """Build an `orchestrator.EmitFn` bound to one run, publishing onto `bus`."""

    async def emit(event: str, data: dict[str, Any]) -> None:
        await bus.publish(run_id, event, data)

    return emit


async def sse_event_generator(
    run_id: UUID,
    bus: EventBus = event_bus,
    *,
    heartbeat_seconds: float = HEARTBEAT_SECONDS,
) -> AsyncIterator[dict[str, Any]]:
    """Yield sse-starlette-shaped dicts (`{"event": ..., "data": <json str>}`)
    for one run, closing after a terminal event or heartbeating every 15s.
    """
    try:
        queue = bus.subscribe(run_id)
    except TooManySubscribers:
        # Report it inside the SSE contract rather than as a broken stream: the
        # dashboard already renders `run.error`, whereas a 500 mid-handshake just
        # looks like the backend died.
        yield {
            "event": "run.error",
            "data": json.dumps(
                {
                    "run_id": str(run_id),
                    "data": {"error": "too many concurrent stream subscribers for this run"},
                }
            ),
        }
        return

    try:
        while True:
            try:
                event, payload = await asyncio.wait_for(queue.get(), timeout=heartbeat_seconds)
            except TimeoutError:
                yield {"event": "heartbeat", "data": json.dumps({"run_id": str(run_id)})}
                continue
            yield {"event": event, "data": json.dumps(payload)}
            if event in TERMINAL_EVENTS:
                break
    finally:
        bus.unsubscribe(run_id, queue)


router = APIRouter()


@router.get("/runs/{run_id}/stream")
async def stream_run(run_id: UUID) -> EventSourceResponse:
    return EventSourceResponse(sse_event_generator(run_id))


__all__ = ["EventBus", "event_bus", "make_emitter", "sse_event_generator", "router"]
