"""Tests for bench.metrics: precision/recall/F1 against a hand-computed
example, bootstrap CI sanity (wider than zero for variable data, doesn't
crash on n=1), and the secondary diagnostics (FRAGILE precision-risk,
UNVERIFIABLE abstention, STALE fire rate)."""

from __future__ import annotations

import random
from uuid import uuid4

from attest.models import Verdict
from bench.mapping import EvalUnit
from bench.metrics import (
    bootstrap_ci,
    confusion_counts,
    f1,
    fragile_precision_risk_rate,
    mean_cost_per_claim,
    p95_latency_ms,
    precision,
    recall,
    stale_fire_rate,
    unverifiable_abstention_rate,
)
from bench.runner import ClaimRecord


def _unit(gt: bool, predicted: bool | None) -> EvalUnit:
    return EvalUnit(
        example_id="e",
        dataset="ragtruth",
        hallucination_type="none",
        config="attest_full",
        ground_truth_positive=gt,
        predicted=predicted,
    )


def test_precision_recall_f1_hand_computed():
    # 3 TP, 1 FP, 1 FN, 1 TN, 1 abstain (excluded)
    units = [
        _unit(True, True),
        _unit(True, True),
        _unit(True, True),
        _unit(False, True),
        _unit(True, False),
        _unit(False, False),
        _unit(True, None),
    ]
    counts = confusion_counts(units)
    assert counts.tp == 3
    assert counts.fp == 1
    assert counts.fn == 1
    assert counts.tn == 1
    assert counts.abstained == 1

    assert precision(units) == 3 / 4
    assert recall(units) == 3 / 4
    p, r = 3 / 4, 3 / 4
    assert abs(f1(units) - (2 * p * r) / (p + r)) < 1e-9


def test_precision_recall_f1_zero_when_no_positive_predictions():
    units = [_unit(True, False), _unit(False, False)]
    assert precision(units) == 0.0
    assert recall(units) == 0.0
    assert f1(units) == 0.0


def test_precision_recall_f1_empty_units():
    assert precision([]) == 0.0
    assert recall([]) == 0.0
    assert f1([]) == 0.0


def test_bootstrap_ci_empty_list_does_not_crash():
    ci = bootstrap_ci([], precision)
    assert ci.low == 0.0
    assert ci.high == 0.0


def test_bootstrap_ci_n_equals_1_does_not_crash_and_is_degenerate():
    units = [_unit(True, True)]
    ci = bootstrap_ci(units, precision, n_resamples=200, seed=1)
    # every resample of a single-element list is that same element -> a
    # single point, not a bug to paper over.
    assert ci.low == ci.high == 1.0


def test_bootstrap_ci_wider_than_zero_for_variable_data():
    rng = random.Random(0)
    units = []
    for _ in range(60):
        gt = rng.random() < 0.5
        pred = gt if rng.random() < 0.7 else (not gt)
        units.append(_unit(gt, pred))

    ci = bootstrap_ci(units, f1, n_resamples=500, seed=42)
    assert ci.high > ci.low


def _claim_record(
    cost: float,
    latency: int,
    verdict: Verdict = Verdict.GROUNDED,
    dataset: str = "ragtruth",
    example_id: str = "e",
    span: tuple[int, int] = (0, 5),
) -> ClaimRecord:
    return ClaimRecord(
        example_id=example_id,
        dataset=dataset,
        hallucination_type="none",
        claim_id=uuid4(),
        claim_text="c",
        span_start=span[0],
        span_end=span[1],
        config="attest_full",
        predicted_verdict=verdict,
        cost_usd=cost,
        latency_ms=latency,
    )


def test_mean_cost_per_claim():
    records = [_claim_record(0.01, 100), _claim_record(0.02, 200), _claim_record(0.03, 300)]
    assert abs(mean_cost_per_claim(records) - 0.02) < 1e-9


def test_mean_cost_per_claim_empty():
    assert mean_cost_per_claim([]) == 0.0


def test_p95_latency_ms_known_values():
    # 100 values, 1..100. p95 with linear interpolation (numpy's default
    # method) = 95.05.
    records = [_claim_record(0.0, i) for i in range(1, 101)]
    result = p95_latency_ms(records)
    assert 94.5 <= result <= 95.55


def test_p95_latency_ms_empty():
    assert p95_latency_ms([]) == 0.0


def test_fragile_precision_risk_rate():
    records = [
        _claim_record(0.0, 0, verdict=Verdict.FRAGILE, example_id="a"),
        _claim_record(0.0, 0, verdict=Verdict.FRAGILE, example_id="b"),
        _claim_record(0.0, 0, verdict=Verdict.GROUNDED, example_id="c"),
    ]
    gt_map = {"a": False, "b": True, "c": False}  # "a" is a false-positive risk case
    rate, n_fragile = fragile_precision_risk_rate(records, lambda r: gt_map[r.example_id])
    assert n_fragile == 2
    assert rate == 0.5


def test_fragile_precision_risk_rate_no_fragile_claims():
    records = [_claim_record(0.0, 0, verdict=Verdict.GROUNDED)]
    rate, n = fragile_precision_risk_rate(records, lambda r: False)
    assert n == 0
    assert rate == 0.0


def test_unverifiable_abstention_rate():
    records = [
        _claim_record(0.0, 0, verdict=Verdict.UNVERIFIABLE),
        _claim_record(0.0, 0, verdict=Verdict.GROUNDED),
        _claim_record(0.0, 0, verdict=Verdict.UNVERIFIABLE),
        _claim_record(0.0, 0, verdict=Verdict.UNSUPPORTED),
    ]
    assert unverifiable_abstention_rate(records) == 0.5


def test_unverifiable_abstention_rate_empty():
    assert unverifiable_abstention_rate([]) == 0.0


def test_stale_fire_rate():
    records = [
        _claim_record(0.0, 0, verdict=Verdict.STALE),
        _claim_record(0.0, 0, verdict=Verdict.GROUNDED),
        _claim_record(0.0, 0, verdict=Verdict.GROUNDED),
        _claim_record(0.0, 0, verdict=Verdict.GROUNDED),
    ]
    assert stale_fire_rate(records) == 0.25
