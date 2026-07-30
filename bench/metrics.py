"""Precision / recall / F1 (with bootstrap 95% CIs), cost-per-claim, and p95
latency (PLAN.md §10), plus the two secondary diagnostics called out in the
task brief: the FRAGILE precision-risk rate and the UNVERIFIABLE abstention
rate.

Bootstrap CIs are hand-rolled with the standard library (`random`) rather
than adding `numpy`/`scipy`: the resampling loop here is a few lines of pure
Python over a list of a few hundred booleans/floats per config, run once per
report -- nowhere near the scale where a vectorized implementation would
matter, and it keeps the benchmark's dependency footprint at zero beyond
what the rest of the repo already needs.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from typing import NamedTuple

from bench.mapping import EvalUnit
from bench.runner import ClaimRecord

_DEFAULT_N_RESAMPLES = 1000
_DEFAULT_SEED = 42


class ConfusionCounts(NamedTuple):
    tp: int
    fp: int
    fn: int
    tn: int
    abstained: int


class CI(NamedTuple):
    low: float
    high: float


def confusion_counts(units: list[EvalUnit]) -> ConfusionCounts:
    tp = fp = fn = tn = abstained = 0
    for u in units:
        if u.predicted is None:
            abstained += 1
            continue
        if u.predicted and u.ground_truth_positive:
            tp += 1
        elif u.predicted and not u.ground_truth_positive:
            fp += 1
        elif not u.predicted and u.ground_truth_positive:
            fn += 1
        else:
            tn += 1
    return ConfusionCounts(tp=tp, fp=fp, fn=fn, tn=tn, abstained=abstained)


def precision(units: list[EvalUnit]) -> float:
    c = confusion_counts(units)
    denom = c.tp + c.fp
    return c.tp / denom if denom else 0.0


def recall(units: list[EvalUnit]) -> float:
    c = confusion_counts(units)
    denom = c.tp + c.fn
    return c.tp / denom if denom else 0.0


def f1(units: list[EvalUnit]) -> float:
    p, r = precision(units), recall(units)
    return (2 * p * r) / (p + r) if (p + r) else 0.0


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Linear-interpolation percentile (matches `numpy.percentile`'s default
    method), implemented without numpy."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    idx = (pct / 100) * (len(sorted_values) - 1)
    lo, hi = math.floor(idx), math.ceil(idx)
    if lo == hi:
        return sorted_values[int(idx)]
    frac = idx - lo
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * frac


def bootstrap_ci(
    units: list[EvalUnit],
    metric_fn: Callable[[list[EvalUnit]], float],
    *,
    n_resamples: int = _DEFAULT_N_RESAMPLES,
    seed: int = _DEFAULT_SEED,
) -> CI:
    """Resample `units` with replacement `n_resamples` times, recompute
    `metric_fn` each time, and report the [2.5th, 97.5th] percentile band.

    `units` empty -> `CI(0.0, 0.0)` (nothing to resample). `len(units) == 1`
    is a valid, non-crashing degenerate case: every resample is that same
    single unit, so the "interval" collapses to a point -- that is the
    mathematically correct interval for n=1, not a bug to paper over.
    """
    if not units:
        return CI(0.0, 0.0)

    rng = random.Random(seed)
    n = len(units)
    estimates: list[float] = []
    for _ in range(n_resamples):
        resample = [units[rng.randrange(n)] for _ in range(n)]
        estimates.append(metric_fn(resample))
    estimates.sort()
    return CI(low=_percentile(estimates, 2.5), high=_percentile(estimates, 97.5))


def mean_cost_per_claim(records: list[ClaimRecord]) -> float:
    if not records:
        return 0.0
    return sum(r.cost_usd for r in records) / len(records)


def p95_latency_ms(records: list[ClaimRecord]) -> float:
    if not records:
        return 0.0
    latencies = sorted(r.latency_ms for r in records)
    return _percentile([float(v) for v in latencies], 95)


# ---------------------------------------------------------------------------
# Secondary diagnostics (task brief: "not folded into the main table")
# ---------------------------------------------------------------------------


def fragile_precision_risk_rate(
    records: list[ClaimRecord],
    ground_truth_fn: Callable[[ClaimRecord], bool],
) -> tuple[float, int]:
    """Among claims this config marked FRAGILE, what fraction were actually
    ground-truth-faithful (i.e. would have been a false positive)? Makes
    the precision cost of counting FRAGILE as a positive detection visible
    rather than hidden inside a rosier headline F1 (task brief).

    Returns `(rate, n_fragile)` -- `n_fragile == 0` means the rate is
    undefined (reported as 0.0 with `n_fragile` making that explicit,
    rather than raising or silently reporting a misleading 0%).
    """
    from attest.models import Verdict

    fragile = [r for r in records if r.predicted_verdict == Verdict.FRAGILE]
    if not fragile:
        return 0.0, 0
    false_positives = sum(1 for r in fragile if not ground_truth_fn(r))
    return false_positives / len(fragile), len(fragile)


def unverifiable_abstention_rate(records: list[ClaimRecord]) -> float:
    """Share of claims this config abstained on (UNVERIFIABLE), including
    ones the decomposer itself resolved as UNVERIFIABLE before any verifier
    ran. CLAUDE.md: "UNVERIFIABLE is not a failure -- it protects
    precision" -- reported on its own, not counted as an error."""
    from attest.models import Verdict

    if not records:
        return 0.0
    abstained = sum(1 for r in records if r.predicted_verdict == Verdict.UNVERIFIABLE)
    return abstained / len(records)


def stale_fire_rate(records: list[ClaimRecord]) -> float:
    """Share of claims this config marked STALE. Purely diagnostic:
    RAGTruth/HaluEval have no notion of temporal staleness, so any
    non-zero rate here means the independent verifier is invoking STALE
    against a dataset that structurally cannot confirm or refute it --
    worth flagging explicitly, per the task brief, rather than silently
    folding STALE into the positive-class count and moving on."""
    from attest.models import Verdict

    if not records:
        return 0.0
    stale = sum(1 for r in records if r.predicted_verdict == Verdict.STALE)
    return stale / len(records)


__all__ = [
    "ConfusionCounts",
    "CI",
    "confusion_counts",
    "precision",
    "recall",
    "f1",
    "bootstrap_ci",
    "mean_cost_per_claim",
    "p95_latency_ms",
    "fragile_precision_risk_rate",
    "unverifiable_abstention_rate",
    "stale_fire_rate",
]
