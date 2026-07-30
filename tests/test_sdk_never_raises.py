"""Proves the SDK's non-negotiable guarantee (CLAUDE.md): it never raises
into the host pipeline, under any failure mode.

Every test here points `attest.init` at a genuinely broken/unreachable
backend (bad port, non-routable IP for timeouts, garbage URL) or feeds
malformed input, then asserts the wrapped host function still returns its
normal value with no exception propagating — and, for the network cases,
that the call returns promptly (a host request's latency must never depend
on ATTEST being slow or down).

The one thing that must NOT be swallowed is the host's own exception; one
test asserts that explicitly so a future change can't "fix" the never-raise
guarantee by over-broadly catching everything.
"""

import time

import pytest

import attest.sdk as sdk


@pytest.fixture(autouse=True)
def _reset_sdk_client():
    """Each test manages its own `attest.init(...)`; don't leak the client
    (or its background thread) across tests."""
    original = sdk._client
    yield
    sdk._client = original


# ---------------------------------------------------------------------------
# Unreachable / refused connection
# ---------------------------------------------------------------------------


def test_observe_decorator_never_raises_with_unreachable_backend():
    sdk.init(api_url="http://127.0.0.1:1", api_key="whatever", sample_rate=1.0, timeout_s=0.5)

    @sdk.observe(pipeline_name="test-pipeline")
    def answer(query: str) -> sdk.Output:
        return sdk.Output(answer="the answer", retrieved_chunks=[{"text": "chunk"}])

    start = time.perf_counter()
    result = answer("what is x?")
    elapsed = time.perf_counter() - start

    assert result.answer == "the answer"
    assert elapsed < 2.0  # never blocks the caller on the (dead) network call


def test_trace_context_manager_never_raises_with_unreachable_backend():
    sdk.init(api_url="http://127.0.0.1:1", timeout_s=0.5)

    with sdk.trace(pipeline_name="test-pipeline", query="what is x?") as t:
        t.record_chunks([{"text": "chunk"}])
        t.record_answer("the answer")
    # reaching here means _finalize() (which submits to the dead backend)
    # did not raise.


# ---------------------------------------------------------------------------
# Malformed / invalid api_url and api_key
# ---------------------------------------------------------------------------


def test_init_never_raises_with_malformed_api_url():
    sdk.init(api_url="not a url at all :::: ???", api_key=None, sample_rate=1.0, timeout_s=0.5)

    @sdk.observe(pipeline_name="test-pipeline")
    def answer(query: str) -> sdk.Output:
        return sdk.Output(answer="ok", retrieved_chunks=[])

    result = answer("q")
    assert result.answer == "ok"


def test_init_never_raises_with_empty_api_url():
    sdk.init(api_url="", api_key="x", sample_rate=1.0)

    @sdk.observe(pipeline_name="test-pipeline")
    def answer(query: str) -> sdk.Output:
        return sdk.Output(answer="ok", retrieved_chunks=[])

    assert answer("q").answer == "ok"


def test_decorator_never_raises_with_non_string_api_key():
    sdk.init(api_url="http://127.0.0.1:1", api_key=12345, timeout_s=0.5)  # type: ignore[arg-type]

    @sdk.observe(pipeline_name="test-pipeline")
    def answer(query: str) -> sdk.Output:
        return sdk.Output(answer="ok", retrieved_chunks=[])

    assert answer("q").answer == "ok"


# ---------------------------------------------------------------------------
# Timeout (non-routable address so the connection hangs instead of refusing)
# ---------------------------------------------------------------------------


def test_decorator_never_raises_on_network_timeout():
    # 10.255.255.1 is a non-routable private address commonly used to
    # simulate a hanging/unreachable-but-not-immediately-refused connection.
    sdk.init(api_url="http://10.255.255.1", timeout_s=0.2)

    @sdk.observe(pipeline_name="test-pipeline")
    def answer(query: str) -> sdk.Output:
        return sdk.Output(answer="ok", retrieved_chunks=[])

    start = time.perf_counter()
    result = answer("q")
    elapsed = time.perf_counter() - start

    assert result.answer == "ok"
    assert elapsed < 2.0  # submission is fire-and-forget; the timeout happens in the background


# ---------------------------------------------------------------------------
# Malformed chunks / inputs
# ---------------------------------------------------------------------------


def test_trace_record_chunks_with_object_that_raises_on_iteration_never_raises():
    sdk.init(api_url="http://127.0.0.1:1", timeout_s=0.5)

    class BadIterable:
        def __iter__(self):
            raise RuntimeError("boom: this iterable is broken")

    with sdk.trace(pipeline_name="test", query="q") as t:
        t.record_chunks(BadIterable())
        t.record_answer("ok")
    # no exception escaped record_chunks() or the context manager's finalize.


def test_decorator_handles_malformed_output_chunks_without_raising():
    sdk.init(api_url="http://127.0.0.1:1", timeout_s=0.5)

    @sdk.observe(pipeline_name="test")
    def answer(query: str) -> sdk.Output:
        return sdk.Output(
            answer="ok",
            retrieved_chunks=[{"no_text_field_here": True}, object(), None, 42],
        )

    result = answer("q")
    assert result.answer == "ok"


# ---------------------------------------------------------------------------
# No init() called / client reset mid-run
# ---------------------------------------------------------------------------


def test_decorator_works_even_if_init_was_never_called():
    sdk._client = None

    @sdk.observe(pipeline_name="test")
    def answer(query: str) -> sdk.Output:
        return sdk.Output(answer="ok", retrieved_chunks=[])

    assert answer("q").answer == "ok"


def test_decorator_handles_non_output_return_value_gracefully():
    sdk.init(api_url="http://127.0.0.1:1", timeout_s=0.5)

    @sdk.observe(pipeline_name="test")
    def answer(query: str) -> str:
        return "not an attest.Output at all"

    result = answer("q")
    assert result == "not an attest.Output at all"


# ---------------------------------------------------------------------------
# Async host functions
# ---------------------------------------------------------------------------


async def test_observe_decorator_works_with_async_host_functions():
    sdk.init(api_url="http://127.0.0.1:1", timeout_s=0.5)

    @sdk.observe(pipeline_name="test")
    async def answer(query: str) -> sdk.Output:
        return sdk.Output(answer="async ok", retrieved_chunks=[])

    result = await answer("q")
    assert result.answer == "async ok"


# ---------------------------------------------------------------------------
# The one thing that must NOT be swallowed: the host's own exception
# ---------------------------------------------------------------------------


def test_trace_propagates_the_hosts_own_exception_unchanged():
    sdk.init(api_url="http://127.0.0.1:1", timeout_s=0.5)

    class HostPipelineError(Exception):
        pass

    with pytest.raises(HostPipelineError, match="host pipeline broke"):
        with sdk.trace(pipeline_name="test", query="q") as t:
            t.record_chunks([{"text": "chunk"}])
            t.record_answer("partial answer")
            raise HostPipelineError("host pipeline broke")


def test_observe_decorator_propagates_the_hosts_own_exception_unchanged():
    sdk.init(api_url="http://127.0.0.1:1", timeout_s=0.5)

    class HostPipelineError(Exception):
        pass

    @sdk.observe(pipeline_name="test")
    def answer(query: str) -> sdk.Output:
        raise HostPipelineError("the wrapped function itself is broken")

    with pytest.raises(HostPipelineError, match="the wrapped function itself is broken"):
        answer("q")
