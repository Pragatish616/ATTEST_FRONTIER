"""Tests for bench.mapping: verdict -> predicted label, ground truth
resolution (RAGTruth span-overlap / HaluEval response-level), and eval-unit
construction including HaluEval's OR-aggregation. See bench/MAPPING.md for
the reasoning these tests hold to a fixed behavior."""

from uuid import uuid4

import pytest

from attest.models import Verdict
from bench.configs import BenchConfig
from bench.datasets import BenchExample
from bench.mapping import (
    ABSTAIN_VERDICTS,
    NEGATIVE_VERDICTS,
    POSITIVE_VERDICTS,
    build_eval_units,
    claim_ground_truth,
    claim_ground_truth_ragtruth,
    predicted_label,
    response_ground_truth_halueval,
    spans_overlap,
)
from bench.runner import ClaimRecord


def test_predicted_label_covers_every_verdict_exhaustively():
    for v in Verdict:
        predicted_label(v)  # must not raise for any of the 6 frozen verdicts
    assert POSITIVE_VERDICTS | NEGATIVE_VERDICTS | ABSTAIN_VERDICTS == set(Verdict)


def test_predicted_label_mapping_matches_documented_table():
    assert predicted_label(Verdict.GROUNDED) is False
    for v in (Verdict.UNSUPPORTED, Verdict.CONTRADICTED, Verdict.STALE, Verdict.FRAGILE):
        assert predicted_label(v) is True
    assert predicted_label(Verdict.UNVERIFIABLE) is None


@pytest.mark.parametrize(
    "a_start,a_end,b_start,b_end,expected",
    [
        (0, 10, 5, 15, True),  # partial overlap
        (0, 10, 10, 20, False),  # touching, half-open -> no overlap
        (5, 8, 0, 20, True),  # fully inside
        (0, 20, 5, 8, True),  # fully contains
        (0, 5, 10, 15, False),  # disjoint
        (0, 10, 0, 10, True),  # identical
    ],
)
def test_spans_overlap(a_start, a_end, b_start, b_end, expected):
    assert spans_overlap(a_start, a_end, b_start, b_end) is expected


def _ragtruth_example(spans, example_id="ex-1"):
    return BenchExample(
        example_id=example_id,
        dataset="ragtruth",
        hallucination_type="Evident Conflict",
        query="q",
        retrieved_chunks=[],
        answer="a" * 100,
        ground_truth_spans=spans,
        ground_truth_hallucinated=bool(spans),
    )


def test_claim_ground_truth_ragtruth_overlap_true():
    example = _ragtruth_example([(10, 20)])
    assert claim_ground_truth_ragtruth(example, 15, 25) is True


def test_claim_ground_truth_ragtruth_no_overlap_false():
    example = _ragtruth_example([(10, 20)])
    assert claim_ground_truth_ragtruth(example, 30, 40) is False


def test_claim_ground_truth_ragtruth_no_spans_at_all_is_false():
    example = _ragtruth_example([])
    assert claim_ground_truth_ragtruth(example, 0, 5) is False


def test_claim_ground_truth_ragtruth_none_span_is_false():
    example = _ragtruth_example([(10, 20)])
    assert claim_ground_truth_ragtruth(example, None, None) is False


def test_response_ground_truth_halueval_passthrough():
    example = BenchExample(
        example_id="he-1",
        dataset="halueval",
        hallucination_type="qa",
        query="q",
        retrieved_chunks=[],
        answer="a",
        ground_truth_spans=[],
        ground_truth_hallucinated=True,
    )
    assert response_ground_truth_halueval(example) is True


def _record(example_id, dataset, config, verdict, span=None):
    return ClaimRecord(
        example_id=example_id,
        dataset=dataset,
        hallucination_type=None,
        claim_id=uuid4(),
        claim_text="claim",
        span_start=span[0] if span else None,
        span_end=span[1] if span else None,
        config=config,
        predicted_verdict=verdict,
        cost_usd=0.001,
        latency_ms=100,
    )


def test_build_eval_units_ragtruth_one_unit_per_claim():
    example = _ragtruth_example([(10, 20)])
    examples_by_id = {example.example_id: example}
    config = BenchConfig.ATTEST_FULL

    r1 = _record(example.example_id, "ragtruth", config.value, Verdict.CONTRADICTED, span=(12, 18))
    r2 = _record(example.example_id, "ragtruth", config.value, Verdict.GROUNDED, span=(50, 60))

    units = build_eval_units([r1, r2], examples_by_id, config)
    assert len(units) == 2
    by_claim = {u.claim_id: u for u in units}
    assert by_claim[r1.claim_id].ground_truth_positive is True
    assert by_claim[r1.claim_id].predicted is True
    assert by_claim[r2.claim_id].ground_truth_positive is False
    assert by_claim[r2.claim_id].predicted is False


def test_build_eval_units_ignores_other_configs():
    example = _ragtruth_example([])
    examples_by_id = {example.example_id: example}
    r = _record(example.example_id, "ragtruth", BenchConfig.BASELINE.value, Verdict.GROUNDED, span=(0, 5))
    units = build_eval_units([r], examples_by_id, BenchConfig.ATTEST_FULL)
    assert units == []


def test_build_eval_units_halueval_or_aggregation_positive():
    example = BenchExample(
        example_id="he-2",
        dataset="halueval",
        hallucination_type="qa",
        query="q",
        retrieved_chunks=[],
        answer="a",
        ground_truth_spans=[],
        ground_truth_hallucinated=True,
    )
    examples_by_id = {example.example_id: example}
    config = BenchConfig.ATTEST_FULL
    records = [
        _record(example.example_id, "halueval", config.value, Verdict.GROUNDED),
        _record(example.example_id, "halueval", config.value, Verdict.UNSUPPORTED),
    ]
    units = build_eval_units(records, examples_by_id, config)
    assert len(units) == 1
    assert units[0].predicted is True  # OR-aggregation: one positive claim -> positive response


def test_build_eval_units_halueval_all_abstain_is_abstain():
    example = BenchExample(
        example_id="he-3",
        dataset="halueval",
        hallucination_type="qa",
        query="q",
        retrieved_chunks=[],
        answer="a",
        ground_truth_spans=[],
        ground_truth_hallucinated=False,
    )
    examples_by_id = {example.example_id: example}
    config = BenchConfig.ATTEST_FULL
    records = [
        _record(example.example_id, "halueval", config.value, Verdict.UNVERIFIABLE),
        _record(example.example_id, "halueval", config.value, Verdict.UNVERIFIABLE),
    ]
    units = build_eval_units(records, examples_by_id, config)
    assert len(units) == 1
    assert units[0].predicted is None


def test_build_eval_units_halueval_negative_when_no_positive_and_not_all_abstain():
    example = BenchExample(
        example_id="he-4",
        dataset="halueval",
        hallucination_type="qa",
        query="q",
        retrieved_chunks=[],
        answer="a",
        ground_truth_spans=[],
        ground_truth_hallucinated=False,
    )
    examples_by_id = {example.example_id: example}
    config = BenchConfig.ATTEST_FULL
    records = [
        _record(example.example_id, "halueval", config.value, Verdict.GROUNDED),
        _record(example.example_id, "halueval", config.value, Verdict.UNVERIFIABLE),
    ]
    units = build_eval_units(records, examples_by_id, config)
    assert len(units) == 1
    assert units[0].predicted is False


def test_claim_ground_truth_dispatches_on_dataset():
    ragtruth_example = _ragtruth_example([(0, 10)])
    r = _record(ragtruth_example.example_id, "ragtruth", "x", Verdict.GROUNDED, span=(2, 5))
    assert claim_ground_truth(r, ragtruth_example) is True

    halueval_example = BenchExample(
        example_id="he-5",
        dataset="halueval",
        hallucination_type="qa",
        query="q",
        retrieved_chunks=[],
        answer="a",
        ground_truth_spans=[],
        ground_truth_hallucinated=True,
    )
    r2 = _record(halueval_example.example_id, "halueval", "x", Verdict.GROUNDED)
    assert claim_ground_truth(r2, halueval_example) is True
