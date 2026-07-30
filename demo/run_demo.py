#!/usr/bin/env python3
"""One-command demo runner (PLAN.md §8/§9): runs the Northwind Devices RAG
pipeline over the seeded demo corpus, submits the result to ATTEST via the
SDK, and reports back a run id/URL to open in the dashboard.

Usage:
    uv run python demo/build_corpus.py          # once, to materialize demo/corpus/
    uv run python demo/run_demo.py               # runs both seeded queries
    uv run python demo/run_demo.py --query "..."  # runs one custom query instead
    uv run python demo/run_demo.py --api-url http://localhost:8000 --api-key ...

Note on `import attest.sdk as attest`: PLAN.md §5.3 shows `import attest;
attest.init(...)`, which requires the top-level `attest/__init__.py` to
re-export the SDK's public names. That file is outside this agent's lane
(repo skeleton / A0) and currently only exposes `__version__` — see the
final report for the exact one-line fix needed there. Importing
`attest.sdk` directly here keeps this script correct today regardless of
when that re-export lands.

Fail-loudly contract: this script is an operator-facing CLI, not a host
pipeline — unlike the SDK it drives (which never raises, by design), this
script prints a clear, specific, non-stack-trace error message and exits
non-zero if `.env` isn't configured, the corpus hasn't been built, the LLM
provider isn't configured, or the ATTEST API is unreachable.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time

import httpx

DEFAULT_API_URL = "http://localhost:8000"
PIPELINE_NAME = "northwind-demo-rag"

# The two deliberately-seeded queries — see demo/SEED_NOTES.md for exactly
# which doc / verdict each one targets and why.
SEED_QUERIES = [
    "If I install SupportBot Desktop on Windows 10, will it keep getting "
    "Microsoft security updates?",
    "Does the Northwind Aria Earbuds charging case support Qi wireless charging?",
]


def _fail(message: str) -> None:
    print(f"\n[ATTEST demo] ERROR: {message}\n", file=sys.stderr)
    sys.exit(1)


def _import_dependencies():
    """Deferred, guarded import.

    `attest.config` (transitively imported via `demo.rag_pipeline` ->
    `attest.llm`) fails loudly *at import time* if required settings are
    missing (CLAUDE.md: fail loud at boot, by design — that's the right
    behavior for the app as a whole). For this operator-facing CLI script
    we still want a clean, specific message instead of a raw traceback, so
    the import is deferred to here and guarded rather than done at module
    level.
    """
    try:
        import attest.sdk as attest_sdk
        from demo.rag_pipeline import CORPUS_DIR, answer_query
    except RuntimeError as exc:
        _fail(
            "ATTEST configuration is missing or invalid (raised by "
            "attest.config at import time, by design).\n"
            "Copy .env.example to .env and fill in the required keys "
            "(SUPABASE_URL, SUPABASE_KEY, ANTHROPIC_API_KEY), then re-run "
            f"this script.\nUnderlying error: {exc}"
        )
        raise  # unreachable — _fail() exits — kept for type-checkers/linters
    except Exception as exc:  # noqa: BLE001 - this script fails loudly on purpose
        _fail(f"Unexpected error importing the demo RAG pipeline: {exc}")
        raise
    return attest_sdk, CORPUS_DIR, answer_query


def _check_corpus_built(corpus_dir) -> None:
    if not corpus_dir.exists() or not any(corpus_dir.glob("*.md")):
        _fail(
            f"Demo corpus not found at {corpus_dir}. Run "
            "`uv run python demo/build_corpus.py` first, then re-run this script."
        )


def _check_backend(api_url: str) -> None:
    try:
        with httpx.Client(timeout=3.0) as client:
            response = client.get(f"{api_url}/v1/health")
            response.raise_for_status()
    except httpx.ConnectError as exc:
        _fail(
            f"Could not reach the ATTEST API at {api_url} (connection refused).\n"
            "Is the API server running? Start it (see the API team's README), "
            "or pass --api-url to point at the right host.\n"
            f"Underlying error: {exc}"
        )
    except httpx.HTTPStatusError as exc:
        _fail(
            f"ATTEST API at {api_url} responded, but GET /v1/health returned "
            f"HTTP {exc.response.status_code}.\nUnderlying error: {exc}"
        )
    except httpx.TimeoutException as exc:
        _fail(f"ATTEST API at {api_url} timed out on GET /v1/health.\nUnderlying error: {exc}")
    except Exception as exc:  # noqa: BLE001 - this script fails loudly on purpose
        _fail(f"Unexpected error checking ATTEST API health at {api_url}: {exc}")


def _poll_for_latest_run(api_url: str, attempts: int = 6, delay_s: float = 0.5) -> str | None:
    """`attest.trace(...)` is deliberately fire-and-forget (the SDK never
    blocks the host on submission — CLAUDE.md hard rule), so it has no
    run_id to hand back. This is a script-level convenience only: poll
    GET /v1/runs a few times so the person running the demo gets an id/URL
    to click into. Any failure here is non-fatal to the demo."""
    with httpx.Client(timeout=3.0) as client:
        for _ in range(attempts):
            try:
                response = client.get(f"{api_url}/v1/runs", params={"limit": 1})
                response.raise_for_status()
                runs = response.json().get("runs", [])
                if runs:
                    return runs[0].get("id")
            except Exception:
                pass
            time.sleep(delay_s)
    return None


async def _run_one_query(attest, answer_query, query: str, api_url: str, k: int) -> None:
    print(f"\n=== Query: {query}")
    try:
        answer, chunks = await answer_query(query, k=k)
    except Exception as exc:
        _fail(
            "The RAG pipeline's answer-generation step failed. This usually "
            "means no LLM provider is configured — check ANTHROPIC_API_KEY "
            "(and optionally GROQ_API_KEY / GEMINI_API_KEY) in your .env.\n"
            f"Underlying error: {exc}"
        )
        return  # unreachable (`_fail` exits); keeps type-checkers happy

    print(f"Answer: {answer}")
    print(f"Retrieved {len(chunks)} chunk(s): " + ", ".join(c.source_id for c in chunks))

    with attest.trace(pipeline_name=PIPELINE_NAME, query=query) as t:
        t.record_chunks(
            [
                {
                    "chunk_index": c.chunk_index,
                    "source_id": c.source_id,
                    "source_url": c.source_url,
                    "text": c.text,
                    "score": c.score,
                }
                for c in chunks
            ]
        )
        t.record_answer(answer)

    run_id = _poll_for_latest_run(api_url)
    if run_id:
        print(f"Submitted. View it at: {api_url}/v1/runs/{run_id}")
    else:
        print(
            "Submitted to ATTEST (fire-and-forget, per the SDK's non-blocking "
            "design), but couldn't confirm a run id back from GET /v1/runs "
            "within a few retries — either it hasn't landed yet, or that "
            "route isn't wired up on the API side yet. Check the API "
            "server's own logs / GET /v1/runs directly."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ATTEST Northwind Devices demo.")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="ATTEST API base URL.")
    parser.add_argument("--api-key", default=None, help="ATTEST API key, if configured.")
    parser.add_argument(
        "--query",
        default=None,
        help="Run one custom query instead of the two seeded demo queries.",
    )
    parser.add_argument("--k", type=int, default=4, help="Top-k chunks to retrieve.")
    parser.add_argument("--sample-rate", type=float, default=1.0)
    args = parser.parse_args()

    attest, corpus_dir, answer_query = _import_dependencies()

    _check_corpus_built(corpus_dir)
    _check_backend(args.api_url)

    attest.init(api_url=args.api_url, api_key=args.api_key, sample_rate=args.sample_rate)

    queries = [args.query] if args.query else SEED_QUERIES
    for query in queries:
        asyncio.run(_run_one_query(attest, answer_query, query, args.api_url, args.k))


if __name__ == "__main__":
    main()
