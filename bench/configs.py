"""The four benchmark configurations (PLAN.md §10).

Same model, same temperature (hardcoded to 0 in `attest.llm.complete`), same
examples, same one-time decomposition per example (see `bench/runner.py`) --
the only thing that varies across these four is the verification strategy,
which is exactly what makes the ablation rows meaningful.
"""

from __future__ import annotations

from enum import StrEnum


class BenchConfig(StrEnum):
    """Not part of `attest.models` -- this is a bench-only axis, never
    persisted through the frozen Supabase/API contracts."""

    BASELINE = "single_pass_baseline"
    ATTEST_MINUS_PROBER = "attest_minus_prober"
    ATTEST_MINUS_INDEPENDENT = "attest_minus_independent"
    ATTEST_FULL = "attest_full"


# Exact row labels for bench/report.py's results.md table (PLAN.md §10).
DISPLAY_NAMES: dict[BenchConfig, str] = {
    BenchConfig.BASELINE: "Single-pass LLM judge (baseline)",
    BenchConfig.ATTEST_MINUS_PROBER: "ATTEST − prober (ablation)",
    BenchConfig.ATTEST_MINUS_INDEPENDENT: "ATTEST − independent (ablation)",
    BenchConfig.ATTEST_FULL: "ATTEST full",
}

# Table order (PLAN.md §10): baseline first, then the two ablations, then
# the full system.
ALL_CONFIGS: tuple[BenchConfig, ...] = (
    BenchConfig.BASELINE,
    BenchConfig.ATTEST_MINUS_PROBER,
    BenchConfig.ATTEST_MINUS_INDEPENDENT,
    BenchConfig.ATTEST_FULL,
)

# Which of the two second-pass verifiers run under each ATTEST configuration.
# Entailment always runs (PLAN.md §4 -- it's the baseline signal every other
# verifier is defined relative to). Consulted by `bench/runner.py`.
USES_PROBER: dict[BenchConfig, bool] = {
    BenchConfig.BASELINE: False,
    BenchConfig.ATTEST_MINUS_PROBER: False,
    BenchConfig.ATTEST_MINUS_INDEPENDENT: True,
    BenchConfig.ATTEST_FULL: True,
}
USES_INDEPENDENT: dict[BenchConfig, bool] = {
    BenchConfig.BASELINE: False,
    BenchConfig.ATTEST_MINUS_PROBER: True,
    BenchConfig.ATTEST_MINUS_INDEPENDENT: False,
    BenchConfig.ATTEST_FULL: True,
}


__all__ = ["BenchConfig", "DISPLAY_NAMES", "ALL_CONFIGS", "USES_PROBER", "USES_INDEPENDENT"]
