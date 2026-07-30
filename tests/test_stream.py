"""EventBus / SSE generator tests — exercised directly, without a running
HTTP server, per the A4 brief ("whatever's simplest to test without a real
HTTP server in the loop").
"""

from __future__ import annotations

import asyncio
import json
from uuid import uuid4

from attest.api.stream import EventBus, make_emitter, sse_event_generator


async def test_publish_subscribe_round_trip():
    bus = EventBus()
    run_id = uuid4()
    queue = bus.subscribe(run_id)

    await bus.publish(run_id, "run.started", {"pipeline_name": "demo"})

    event, payload = await asyncio.wait_for(queue.get(), timeout=1)
    assert event == "run.started"
    assert payload == {"run_id": str(run_id), "data": {"pipeline_name": "demo"}}


async def test_publish_with_no_subscribers_does_not_raise():
    bus = EventBus()
    run_id = uuid4()
    await bus.publish(run_id, "run.started", {})  # no subscribers yet — no-op


async def test_make_emitter_publishes_onto_bus():
    bus = EventBus()
    run_id = uuid4()
    queue = bus.subscribe(run_id)
    emit = make_emitter(run_id, bus)

    await emit("claims.decomposed", {"count": 3})

    event, payload = await asyncio.wait_for(queue.get(), timeout=1)
    assert event == "claims.decomposed"
    assert payload["data"] == {"count": 3}


async def test_sse_generator_yields_events_in_order_and_stops_on_terminal_event():
    bus = EventBus()
    run_id = uuid4()

    async def producer():
        await bus.publish(run_id, "run.started", {})
        await bus.publish(run_id, "claims.decomposed", {"count": 1})
        await bus.publish(run_id, "claim.verified", {"verdict": "GROUNDED"})
        await bus.publish(run_id, "run.completed", {"status": "complete"})

    producer_task = asyncio.create_task(producer())
    received = []
    async for chunk in sse_event_generator(run_id, bus, heartbeat_seconds=5):
        received.append(chunk["event"])
    await producer_task

    assert received == ["run.started", "claims.decomposed", "claim.verified", "run.completed"]


async def test_sse_generator_heartbeats_when_idle():
    bus = EventBus()
    run_id = uuid4()

    events = []

    async def consume():
        async for chunk in sse_event_generator(run_id, bus, heartbeat_seconds=0.05):
            events.append(chunk["event"])
            if len(events) >= 2:
                await bus.publish(run_id, "run.completed", {})

    await asyncio.wait_for(consume(), timeout=2)

    assert events[0] == "heartbeat"
    assert events[1] == "heartbeat"
    assert events[-1] == "run.completed"


async def test_sse_generator_payload_is_json_with_run_id_and_data_envelope():
    bus = EventBus()
    run_id = uuid4()

    async def producer():
        await bus.publish(run_id, "run.error", {"error": "boom"})

    producer_task = asyncio.create_task(producer())
    chunk = await anext(aiter(sse_event_generator(run_id, bus, heartbeat_seconds=5)))
    await producer_task

    payload = json.loads(chunk["data"])
    assert payload == {"run_id": str(run_id), "data": {"error": "boom"}}


# --- Late-subscriber replay (regression: run finishing before the dashboard
# --- connects used to hang the stream forever on heartbeats) ---


async def test_events_published_before_subscribe_are_replayed():
    """A subscriber that connects after publishing still sees every event."""
    bus = EventBus()
    run_id = uuid4()

    await bus.publish(run_id, "run.started", {"pipeline_name": "demo"})
    await bus.publish(run_id, "claims.decomposed", {"count": 2})

    queue = bus.subscribe(run_id)

    event, payload = await asyncio.wait_for(queue.get(), timeout=1)
    assert event == "run.started"
    assert payload == {"run_id": str(run_id), "data": {"pipeline_name": "demo"}}
    event, _payload = await asyncio.wait_for(queue.get(), timeout=1)
    assert event == "claims.decomposed"


async def test_generator_terminates_when_run_completed_before_subscribe():
    """The whole point: a fast run must not leave the stream heartbeating."""
    bus = EventBus()
    run_id = uuid4()

    await bus.publish(run_id, "run.started", {"pipeline_name": "demo"})
    await bus.publish(run_id, "run.completed", {"status": "completed"})

    events = []
    gen = sse_event_generator(run_id, bus, heartbeat_seconds=30)
    async for message in gen:
        events.append((message["event"], json.loads(message["data"])))

    assert [event for event, _ in events] == ["run.started", "run.completed"]


async def test_history_is_evicted_after_terminal_event_ttl(monkeypatch):
    """A finished run's replay buffer must not be retained forever."""
    from attest.api import stream as stream_module

    bus = EventBus()
    run_id = uuid4()

    await bus.publish(run_id, "run.started", {})
    await bus.publish(run_id, "run.completed", {"status": "completed"})

    monkeypatch.setattr(stream_module, "HISTORY_TTL_SECONDS", -1.0)
    queue = bus.subscribe(run_id)

    assert queue.empty()  # history evicted; GET /runs/{run_id} is the source of truth
