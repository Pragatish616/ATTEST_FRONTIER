"""Query extraction for @attest.observe across call shapes.

Regression: `args[0]` was assumed to be the query, so decorating a bound
method recorded `str(self)` — e.g. "<Bot object at 0x...>" — as the query for
every observation, silently, since str(self) never raises.
"""

from __future__ import annotations

import attest
from attest.sdk import decorator as decorator_module
from attest.sdk.decorator import _make_query_extractor


def _submitted_queries(monkeypatch) -> list[str]:
    """Capture at decorator.py's own reference — it imported the symbol
    directly, so patching attest.sdk._submit_observation would not intercept.
    """
    captured: list[str] = []

    def fake_submit(**kwargs) -> None:
        captured.append(kwargs["query"])

    monkeypatch.setattr(decorator_module, "_submit_observation", fake_submit)
    return captured


def test_plain_function_positional(monkeypatch):
    captured = _submitted_queries(monkeypatch)

    @attest.observe(pipeline_name="p")
    def answer(query: str) -> attest.Output:
        return attest.Output(answer="a", retrieved_chunks=[])

    answer("what is the refund window?")
    assert captured == ["what is the refund window?"]


def test_plain_function_keyword(monkeypatch):
    captured = _submitted_queries(monkeypatch)

    @attest.observe(pipeline_name="p")
    def answer(query: str) -> attest.Output:
        return attest.Output(answer="a", retrieved_chunks=[])

    answer(query="kwarg query")
    assert captured == ["kwarg query"]


def test_bound_method_skips_self(monkeypatch):
    captured = _submitted_queries(monkeypatch)

    class Bot:
        @attest.observe(pipeline_name="p")
        def answer(self, query: str) -> attest.Output:
            return attest.Output(answer="a", retrieved_chunks=[])

    Bot().answer("class-based query")

    assert captured == ["class-based query"]
    assert not captured[0].startswith("<")  # not repr(self)


def test_classmethod_skips_cls(monkeypatch):
    captured = _submitted_queries(monkeypatch)

    class Bot:
        @classmethod
        @attest.observe(pipeline_name="p")
        def answer(cls, query: str) -> attest.Output:
            return attest.Output(answer="a", retrieved_chunks=[])

    Bot.answer("classmethod query")
    assert captured == ["classmethod query"]


def test_query_param_not_first(monkeypatch):
    """The parameter actually named `query` wins over positional order."""
    captured = _submitted_queries(monkeypatch)

    class Bot:
        @attest.observe(pipeline_name="p")
        def answer(self, k: int, query: str) -> attest.Output:
            return attest.Output(answer="a", retrieved_chunks=[])

    Bot().answer(4, "named-param query")
    assert captured == ["named-param query"]


async def test_async_bound_method_skips_self(monkeypatch):
    captured = _submitted_queries(monkeypatch)

    class Bot:
        @attest.observe(pipeline_name="p")
        async def answer(self, query: str) -> attest.Output:
            return attest.Output(answer="a", retrieved_chunks=[])

    await Bot().answer("async class query")
    assert captured == ["async class query"]


def test_uninspectable_callable_never_raises():
    """Whether a C builtin exposes a signature is CPython-version dependent
    (3.11+ exposes many via Argument Clinic, 3.10 does not). Either way the
    extractor must not raise and must return a string — a wrong query is
    recoverable, an exception into the host pipeline is not.
    """
    for fn in (print, len, dict):
        extract = _make_query_extractor(fn)
        assert isinstance(extract((), {}), str)
        assert isinstance(extract(("positional",), {}), str)
