"""`attest.wrap(chain, pipeline_name=...)` — LangChain integration (PLAN.md §5.3).

Scope (see the final report for the full rationale): this gets LangChain
working solidly without adding a hard dependency on the `langchain` package
— everything here is duck-typed against the shapes LangChain objects
commonly expose, not an `isinstance` check against a real LangChain class.
LlamaIndex is a known gap: `_extract_answer_and_chunks` happens to handle
LlamaIndex's `Response`-shaped return value too (`.response` / `.source_nodes`
attributes), but that path is untested against a real LlamaIndex install and
should be treated as unsupported until it is.

Mechanism: wrap whichever call entrypoint the chain object exposes —
`invoke` (LangChain's modern `Runnable` interface) preferred, falling back
to `__call__` then `run` (older `Chain` API) — and, after the real call
returns, best-effort extract the final answer and retrieved
documents/chunks from the return value. The host's own call is never
touched: `original(*args, **kwargs)` runs unguarded, so any exception the
chain itself raises propagates exactly as it would unwrapped. Only the
post-call extraction + submission is wrapped (CLAUDE.md "never raises"
rule).

Known limitation: instance-level `setattr(chain, "__call__", wrapped)` does
NOT intercept Python's implicit `chain(...)` call syntax (dunder methods are
looked up on the type, not the instance) — it only intercepts explicit
`chain.__call__(...)` calls. `invoke` and `run` are ordinary methods and are
not affected by this. `invoke` is tried first for exactly this reason.
"""

from __future__ import annotations

import functools
from typing import Any

import structlog

from attest.models import AttestConfig
from attest.sdk import _safe, _submit_observation

logger = structlog.get_logger(__name__)

_ANSWER_KEYS = ("answer", "result", "output", "output_text", "text", "response")
_CHUNK_KEYS = ("source_documents", "context", "documents", "source_nodes")
_ENTRYPOINTS = ("invoke", "__call__", "run")
_QUERY_KEYS = ("query", "question", "input")


def _extract_query(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    if args:
        candidate = args[0]
        if isinstance(candidate, str):
            return candidate
        if isinstance(candidate, dict):
            for key in _QUERY_KEYS:
                if key in candidate:
                    return str(candidate[key])
            return str(candidate)
    for key in _QUERY_KEYS:
        if key in kwargs:
            return str(kwargs[key])
    return ""


def _extract_answer_and_chunks(result: Any) -> tuple[str, list[Any]]:
    if isinstance(result, str):
        return result, []
    if isinstance(result, dict):
        answer = ""
        for key in _ANSWER_KEYS:
            if key in result:
                answer = result[key]
                break
        chunks: list[Any] = []
        for key in _CHUNK_KEYS:
            if key in result and result[key]:
                chunks = list(result[key])
                break
        return str(answer), chunks
    # Generic object (e.g. a LlamaIndex Response): best-effort attribute probe.
    answer = ""
    for key in _ANSWER_KEYS:
        value = getattr(result, key, None)
        if value:
            answer = value
            break
    chunks: list[Any] = []
    for key in _CHUNK_KEYS:
        value = getattr(result, key, None)
        if value:
            chunks = list(value)
            break
    return str(answer), chunks


def wrap(
    chain: Any,
    *,
    pipeline_name: str,
    model: str | None = None,
    config: AttestConfig | None = None,
) -> Any:
    """`attest.wrap(chain, pipeline_name=...)` — instruments a LangChain chain
    in one line. Returns `chain` unchanged (but instrumented in place) on
    success; on any failure to wrap it, logs a warning and returns the
    original, un-instrumented `chain` so the host pipeline keeps working.
    """
    try:
        return _wrap_impl(chain, pipeline_name, model, config)
    except Exception:  # noqa: BLE001
        logger.warning("attest_wrap_failed", chain_type=type(chain).__name__, exc_info=True)
        return chain


def _wrap_impl(
    chain: Any, pipeline_name: str, model: str | None, config: AttestConfig | None
) -> Any:
    entrypoint_name = next((name for name in _ENTRYPOINTS if hasattr(chain, name)), None)
    if entrypoint_name is None:
        logger.warning("attest_wrap_no_entrypoint", chain_type=type(chain).__name__)
        return chain

    original = getattr(chain, entrypoint_name)

    @functools.wraps(original)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)  # host's own call: never guarded
        _safe(_capture_and_submit)(pipeline_name, args, kwargs, result, model, config)
        return result

    setattr(chain, entrypoint_name, wrapped)
    return chain


def _capture_and_submit(
    pipeline_name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    result: Any,
    model: str | None,
    config: AttestConfig | None,
) -> None:
    query = _extract_query(args, kwargs)
    answer, chunks = _extract_answer_and_chunks(result)
    _submit_observation(
        pipeline_name=pipeline_name,
        query=query,
        answer=answer,
        chunks=chunks,
        model=model,
        config=config,
    )
