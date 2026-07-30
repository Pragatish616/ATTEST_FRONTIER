"""Tests for `attest.wrap(chain, pipeline_name=...)` (attest/sdk/wrappers.py).

Uses fake, minimal LangChain-shaped objects rather than a real LangChain
install (not a project dependency — see the final report). Submission is
verified against a fake `_client` (monkeypatched onto `attest.sdk._client`)
that just records what it was handed, so these tests don't touch threads or
the network.
"""

import pytest

import attest.sdk as sdk


class _FakeClient:
    def __init__(self):
        self.submitted = []

    def submit(self, request):
        self.submitted.append(request)


@pytest.fixture
def fake_client(monkeypatch):
    client = _FakeClient()
    monkeypatch.setattr(sdk, "_client", client)
    return client


# ---------------------------------------------------------------------------
# Fake LangChain-shaped objects
# ---------------------------------------------------------------------------


class FakeRunnableChain:
    """Mimics a modern LangChain `Runnable`: `.invoke(input) -> dict`, the
    dict shaped like `RetrievalQA`'s return value."""

    def invoke(self, input):
        return {
            "result": "The refund window is 30 days.",
            "source_documents": [
                {
                    "page_content": "Refunds are accepted within 30 days of purchase.",
                    "metadata": {"source_id": "doc-1", "source_url": "https://kb/doc-1"},
                },
            ],
        }


class FakeLegacyRunChain:
    """Mimics an older LangChain `Chain`: `.run(query) -> str`."""

    def run(self, query):
        return "plain string answer"


class FakeBrokenChain:
    def invoke(self, input):
        raise ValueError("chain blew up")


class FakeImmutableChain:
    """Raises on any attribute assignment — simulates a chain object that
    can't be monkeypatched (e.g. a frozen dataclass or `__slots__` class
    with a custom `__setattr__`)."""

    def invoke(self, input):
        return "answer"

    def __setattr__(self, name, value):
        raise AttributeError("this object is immutable")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_wrap_captures_answer_and_chunks_via_invoke(fake_client):
    chain = FakeRunnableChain()
    wrapped = sdk.wrap(chain, pipeline_name="rag-v2")

    result = wrapped.invoke({"query": "what is the refund window?"})

    assert result["result"] == "The refund window is 30 days."
    assert len(fake_client.submitted) == 1
    request = fake_client.submitted[0]
    assert request.pipeline_name == "rag-v2"
    assert request.query == "what is the refund window?"
    assert request.answer == "The refund window is 30 days."
    assert len(request.retrieved_chunks) == 1
    assert request.retrieved_chunks[0].text == "Refunds are accepted within 30 days of purchase."
    assert request.retrieved_chunks[0].source_id == "doc-1"


def test_wrap_falls_back_to_run_and_plain_string_answer(fake_client):
    chain = FakeLegacyRunChain()
    wrapped = sdk.wrap(chain, pipeline_name="rag-v2")

    result = wrapped.run("some query")

    assert result == "plain string answer"
    assert len(fake_client.submitted) == 1
    request = fake_client.submitted[0]
    assert request.query == "some query"
    assert request.answer == "plain string answer"
    assert request.retrieved_chunks == []


def test_wrap_prefers_invoke_over_run_when_both_exist(fake_client):
    class Both:
        def invoke(self, input):
            return "via invoke"

        def run(self, input):
            return "via run"

    chain = Both()
    wrapped = sdk.wrap(chain, pipeline_name="rag-v2")
    assert wrapped.invoke({"query": "q"}) == "via invoke"
    assert fake_client.submitted[0].answer == "via invoke"


def test_wrap_returns_original_chain_unchanged_when_no_entrypoint_found():
    class NotAChain:
        pass

    chain = NotAChain()
    wrapped = sdk.wrap(chain, pipeline_name="rag-v2")
    assert wrapped is chain


def test_wrap_never_raises_when_instrumentation_setattr_fails():
    chain = FakeImmutableChain()
    wrapped = sdk.wrap(chain, pipeline_name="rag-v2")

    assert wrapped is chain  # wrap() gave up and returned the original, un-instrumented chain
    assert wrapped.invoke({"query": "q"}) == "answer"  # still works normally


def test_wrap_propagates_the_chains_own_exception_untouched(fake_client):
    chain = FakeBrokenChain()
    wrapped = sdk.wrap(chain, pipeline_name="rag-v2")

    with pytest.raises(ValueError, match="chain blew up"):
        wrapped.invoke({"query": "q"})

    assert fake_client.submitted == []  # no observation submitted for a failed call


def test_wrap_never_raises_with_unreachable_backend_end_to_end():
    """Integration-style: no fake client — the real `_AttestClient` pointed
    at a dead backend. Proves `wrap()` end-to-end honors the "never raises"
    guarantee, not just the mocked seam."""
    import attest.sdk as sdk_module

    original_client = sdk_module._client
    try:
        sdk_module.init(api_url="http://127.0.0.1:1", timeout_s=0.5)
        chain = FakeRunnableChain()
        wrapped = sdk_module.wrap(chain, pipeline_name="rag-v2")
        result = wrapped.invoke({"query": "what is the refund window?"})
        assert result["result"] == "The refund window is 30 days."
    finally:
        sdk_module._client = original_client
