#!/usr/bin/env python3
"""One-command benchmark runner (PLAN.md §10):

    uv run python -m bench.run_benchmark --n 250 --seed 42 [--dataset ragtruth|halueval]

Wires the real `attest` pipeline (decomposer, entailment verifier, prober,
independent verifier, reconciler -- see `bench/runner.py`) against a
real-network-fetched RAGTruth or HaluEval sample (see `bench/datasets.py`),
runs all four benchmark configurations, and writes `bench/results.md` +
`bench/results.json`.

Fail-loudly contract, matching `demo/run_demo.py`'s precedent (A5): this is
an operator-facing CLI, not a host pipeline -- it prints a clear, specific,
non-stack-trace error message and exits non-zero if `.env` isn't configured
or no real LLM provider key is present, rather than making the user read a
traceback (or worse, silently producing garbage).
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bench.runner import RunnerDeps

DEFAULT_MD_OUT = "bench/results.md"
DEFAULT_JSON_OUT = "bench/results.json"


def _fail(message: str) -> None:
    print(f"\n[ATTEST bench] ERROR: {message}\n", file=sys.stderr)
    sys.exit(1)


def _import_pipeline() -> RunnerDeps:
    """Deferred, guarded import.

    `attest.config` (transitively imported via every real verifier) fails
    loudly *at import time* if required settings are missing (CLAUDE.md:
    fail loud at boot, by design -- correct for the app as a whole). For
    this operator-facing CLI, the import is deferred to here and guarded so
    the failure becomes a clean, specific message instead of a raw
    traceback -- mirrors `demo/run_demo.py`'s `_import_dependencies`.
    """
    try:
        from attest.verifiers.decomposer import decompose
        from attest.verifiers.entailment import EntailmentVerifier
        from attest.verifiers.independent import IndependentVerifier
        from attest.verifiers.prober import AdversarialProber
        from bench.baseline import SinglePassJudge
        from bench.runner import RunnerDeps
    except RuntimeError as exc:
        _fail(
            "ATTEST configuration is missing or invalid (raised by "
            "attest.config at import time, by design).\n"
            "Copy .env.example to .env and fill in the required keys "
            "(SUPABASE_URL, SUPABASE_KEY, ANTHROPIC_API_KEY) -- a real "
            "ANTHROPIC_API_KEY is what's actually needed to produce real "
            "numbers from this benchmark.\n"
            f"Underlying error: {exc}"
        )
        raise  # unreachable -- _fail() exits -- kept for type-checkers/linters
    except Exception as exc:  # noqa: BLE001 - this CLI fails loudly on purpose
        _fail(f"Unexpected error importing the ATTEST pipeline: {exc}")
        raise

    entailment = EntailmentVerifier()
    return RunnerDeps(
        decompose=decompose,
        entailment=entailment,
        prober=AdversarialProber(entailment_verifier=entailment),
        independent=IndependentVerifier(),
        baseline=SinglePassJudge(),
    )


def _check_api_key_looks_real() -> None:
    """Belt-and-suspenders on top of `_import_pipeline`'s guard.

    `attest.config.settings` already requires `anthropic_api_key` to be a
    non-empty string, so a genuinely *missing* key fails at import time
    above with a clear message. This catches the "pasted a placeholder into
    .env" case before any real (billed) LLM call happens.
    """
    from attest.config import settings

    placeholder_markers = ("your-", "changeme", "sk-placeholder", "xxxx", "test-")
    key = (settings.anthropic_api_key or "").strip().lower()
    if not key or any(marker in key for marker in placeholder_markers):
        _fail(
            "ANTHROPIC_API_KEY in .env looks unset or unfilled (empty, or "
            "matches a known placeholder pattern). This benchmark makes "
            "real, billed LLM calls against RAGTruth/HaluEval examples -- "
            "fill in a real key before running it for real."
        )


async def _run(args: argparse.Namespace, deps: RunnerDeps) -> None:
    from bench.configs import ALL_CONFIGS
    from bench.datasets import load_dataset
    from bench.report import build_report, write_results
    from bench.runner import run_benchmark

    print(f"[ATTEST bench] Loading dataset={args.dataset} n={args.n} seed={args.seed} ...")
    examples = load_dataset(args.dataset, n=args.n, seed=args.seed)
    print(
        f"[ATTEST bench] Loaded {len(examples)} examples. Running "
        f"{len(ALL_CONFIGS)} configurations per claim (this makes real LLM "
        "calls and will take a while + cost real money) ..."
    )

    records = await run_benchmark(examples, deps, concurrency=args.concurrency)
    print(f"[ATTEST bench] Collected {len(records)} claim x config records. Building report ...")

    command = (
        f"uv run python -m bench.run_benchmark --n {args.n} --seed {args.seed} "
        f"--dataset {args.dataset}"
    )
    report = build_report(
        records, examples, dataset=args.dataset, n=args.n, seed=args.seed, command=command
    )
    write_results(report, md_path=args.md_out, json_path=args.json_out)
    print(f"[ATTEST bench] Wrote {args.md_out} and {args.json_out}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ATTEST benchmark (PLAN.md §10).")
    parser.add_argument("--n", type=int, default=250, help="Number of examples to sample.")
    parser.add_argument(
        "--seed", type=int, default=42, help="Sampling seed (fixed, for reproducibility)."
    )
    parser.add_argument("--dataset", choices=["ragtruth", "halueval"], default="ragtruth")
    parser.add_argument(
        "--concurrency", type=int, default=4, help="Concurrent examples in flight."
    )
    parser.add_argument("--md-out", default=DEFAULT_MD_OUT)
    parser.add_argument("--json-out", default=DEFAULT_JSON_OUT)
    args = parser.parse_args()

    deps = _import_pipeline()
    _check_api_key_looks_real()

    asyncio.run(_run(args, deps))


if __name__ == "__main__":
    main()
