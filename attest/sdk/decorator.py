"""`@attest.observe(pipeline_name=...)` decorator and `attest.trace(...)`
context manager (PLAN.md §5.3).

Both wrap arbitrary sync/async host functions. The host function's own body
is called untouched — if it raises, that's the host's exception and it
propagates exactly as it would without ATTEST. Everything ATTEST does around
that call (extracting the query, building the `ObserveRequest`, handing it
to the background queue) is wrapped via `_safe`/`_submit_observation` so it
can never raise into the host pipeline (CLAUDE.md hard rule).
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import inspect
from collections.abc import Callable, Generator
from typing import Any, ParamSpec, TypeVar

from attest.models import AttestConfig
from attest.sdk import Output, _safe, _submit_observation

P = ParamSpec("P")
T = TypeVar("T")


def _extract_query(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    """Signature-free fallback: a `query` kwarg, else the first positional arg.

    Used only when the wrapped callable's signature can't be inspected (C
    builtins, some callable objects). Prefer `_make_query_extractor`, which
    knows about `self`/`cls`.
    """
    if "query" in kwargs:
        return str(kwargs["query"])
    if args:
        return str(args[0])
    return ""


_IMPLICIT_FIRST_ARGS = frozenset({"self", "cls"})


def _make_query_extractor(func: Callable[..., Any]) -> Callable[..., str]:
    """Build a query extractor for `func`, resolved once at decoration time.

    `args[0]` is only the query for a plain function. On a bound method —
    `class Bot: def answer(self, query): ...`, a completely normal way to
    structure a RAG pipeline — `args[0]` is `self`, and blindly stringifying
    it records `<Bot object at 0x...>` as the query for every observation.
    That corrupts the run silently: `str(self)` doesn't raise, so `_safe`
    never sees a problem.

    Resolution order: the parameter actually named `query`, else the first
    parameter that isn't `self`/`cls`. Signature inspection happens once here,
    not per call.
    """
    try:
        params = list(inspect.signature(func).parameters.values())
    except (TypeError, ValueError):  # C builtins etc. — keep the old behaviour
        return _extract_query

    positional = [
        param
        for param in params
        if param.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    named = [param for param in params if param.kind is not inspect.Parameter.VAR_POSITIONAL]

    target: inspect.Parameter | None = next(
        (param for param in named if param.name == "query"), None
    )
    if target is None:
        target = next(
            (param for param in positional if param.name not in _IMPLICIT_FIRST_ARGS), None
        )
    if target is None:
        return lambda args, kwargs: ""

    name = target.name
    index = positional.index(target) if target in positional else None

    def extract(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
        if name in kwargs:
            return str(kwargs[name])
        if index is not None and len(args) > index:
            return str(args[index])
        return ""

    return extract


def _handle_output(
    pipeline_name: str,
    query: str,
    result: Any,
    model: str | None,
    config: AttestConfig | None,
) -> None:
    if not isinstance(result, Output):
        return
    _submit_observation(
        pipeline_name=pipeline_name,
        query=query,
        answer=result.answer,
        chunks=result.retrieved_chunks,
        model=model,
        config=config,
    )


_safe_handle_output = _safe(_handle_output)
_safe_extract_query = _safe(_extract_query)


def observe(
    *,
    pipeline_name: str,
    model: str | None = None,
    config: AttestConfig | None = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """`@attest.observe(pipeline_name=...)` — PLAN.md §5.3.

    Wraps a function that returns `attest.Output`. Works on both sync and
    async functions. The wrapped function's return value is always exactly
    what the underlying function returned; ATTEST's submission happens as a
    side effect after that return value is captured, never in place of it.
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        # Resolved once per decorated function, not per call.
        safe_extract_query = _safe(_make_query_extractor(func))

        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
                result = await func(*args, **kwargs)
                query = safe_extract_query(args, kwargs) or ""
                _safe_handle_output(pipeline_name, query, result, model, config)
                return result

            return async_wrapper  # type: ignore[return-value]

        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            result = func(*args, **kwargs)
            query = safe_extract_query(args, kwargs) or ""
            _safe_handle_output(pipeline_name, query, result, model, config)
            return result

        return sync_wrapper

    return decorator


class Trace:
    """Handle yielded by `attest.trace(...)`. See PLAN.md §5.3."""

    def __init__(
        self,
        *,
        pipeline_name: str,
        query: str,
        model: str | None = None,
        config: AttestConfig | None = None,
    ) -> None:
        self.pipeline_name = pipeline_name
        self.query = query
        self.model = model
        self.config = config
        self._answer: str | None = None
        self._chunks: list[Any] = []

    def record_chunks(self, chunks: Any) -> None:
        _safe(self._record_chunks)(chunks)

    def _record_chunks(self, chunks: Any) -> None:
        self._chunks = list(chunks) if chunks else []

    def record_answer(self, answer: str) -> None:
        _safe(self._record_answer)(answer)

    def _record_answer(self, answer: str) -> None:
        self._answer = answer

    def _finalize(self) -> None:
        _safe(self._do_finalize)()

    def _do_finalize(self) -> None:
        if self._answer is None:
            # record_answer() was never called (or failed); nothing to submit.
            return
        _submit_observation(
            pipeline_name=self.pipeline_name,
            query=self.query,
            answer=self._answer,
            chunks=self._chunks,
            model=self.model,
            config=self.config,
        )


@contextlib.contextmanager
def trace(
    *,
    pipeline_name: str,
    query: str,
    model: str | None = None,
    config: AttestConfig | None = None,
) -> Generator[Trace, None, None]:
    """`with attest.trace(pipeline_name=..., query=...) as t: ...` — PLAN.md §5.3.

    Submission happens on exit, whether the `with` block succeeded or
    raised — the host's own exception (if any) always propagates unchanged
    after the (always-safe) finalize runs.
    """
    t = Trace(pipeline_name=pipeline_name, query=query, model=model, config=config)
    try:
        yield t
    finally:
        t._finalize()
