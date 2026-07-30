"""Tests for bench.report: the pending template's shape (never a placeholder
number), and real report generation from fake ClaimRecords -- proves
results.md/results.json's schema is valid markdown/JSON in both states."""

from __future__ import annotations

import json
from uuid import uuid4

from attest.models import Verdict
from bench.configs import ALL_CONFIGS, BenchConfig
from bench.datasets import BenchExample
from bench.report import build_pending_report, build_report, render_json, render_markdown
from bench.runner import ClaimRecord


def test_pending_report_has_pending_status_and_all_cells_pending():
    report = build_pending_report(dataset="ragtruth", n=250, seed=42)
    assert report.status == "pending"
    assert report.n_examples == 250
    assert report.seed == 42
    assert len(report.systems) == len(ALL_CONFIGS)
    for s in report.systems:
        assert s.precision.status == "pending"
        assert s.precision.value is None
        assert s.recall.status == "pending"
        assert s.f1.status == "pending"
        assert s.cost_per_claim_usd.status == "pending"
        assert s.p95_latency_ms.status == "pending"


def test_pending_report_markdown_never_leaks_a_placeholder_number():
    report = build_pending_report()
    md = render_markdown(report)
    assert "PENDING" in md

    main_table_section = md.split("## Main table")[1].split("## Prober gain finding")[0]
    for line in main_table_section.splitlines():
        if line.startswith("| Single-pass") or line.startswith("| ATTEST"):
            # every metric cell in the main table must render as PENDING,
            # never a plausible-looking number like 0.0 or 0.87
            cells = [c.strip() for c in line.strip("|").split("|")][1:]
            assert all(cell == "PENDING" for cell in cells), line


def test_pending_report_json_round_trips_and_matches_schema():
    report = build_pending_report(dataset="halueval", n=100, seed=7)
    payload = json.loads(render_json(report))
    assert payload["status"] == "pending"
    assert payload["dataset"] == "halueval"
    assert len(payload["systems"]) == len(ALL_CONFIGS)
    for system in payload["systems"]:
        assert system["precision"]["status"] == "pending"
        assert system["precision"]["value"] is None
    assert payload["disagreement_analysis"] == []
    assert "fragile_precision_risk_rate" in payload["secondary"]


def _example(example_id: str, hallucinated: bool) -> BenchExample:
    return BenchExample(
        example_id=example_id,
        dataset="ragtruth",
        hallucination_type="Evident Conflict" if hallucinated else "none",
        query="q",
        retrieved_chunks=[],
        answer="x" * 50,
        ground_truth_spans=[(0, 10)] if hallucinated else [],
        ground_truth_hallucinated=hallucinated,
    )


def _record(example_id: str, config: str, verdict: Verdict, span=(0, 5), claim_id=None) -> ClaimRecord:
    return ClaimRecord(
        example_id=example_id,
        dataset="ragtruth",
        hallucination_type="Evident Conflict",
        claim_id=claim_id or uuid4(),
        claim_text="c",
        span_start=span[0],
        span_end=span[1],
        config=config,
        predicted_verdict=verdict,
        cost_usd=0.001,
        latency_ms=50,
    )


def test_build_report_computes_real_numbers():
    examples = [_example("e1", True), _example("e2", False)]
    records = []
    for config in ALL_CONFIGS:
        records.append(_record("e1", config.value, Verdict.CONTRADICTED, span=(2, 6)))  # correctly caught
        records.append(_record("e2", config.value, Verdict.GROUNDED, span=(0, 5)))  # correctly passed

    report = build_report(records, examples, dataset="ragtruth", n=2, seed=1)
    assert report.status == "complete"

    full = next(s for s in report.systems if s.config == BenchConfig.ATTEST_FULL.value)
    assert full.precision.status == "computed"
    assert full.precision.value == 1.0
    assert full.recall.value == 1.0
    assert full.f1.value == 1.0
    assert full.n_claims == 2

    md = render_markdown(report)
    main_table_section = md.split("## Main table")[1].split("## Prober gain finding")[0]
    assert "PENDING" not in main_table_section


def test_build_report_prober_gain_finding_flags_overlap_when_configs_tie():
    examples = [_example("e1", True)]
    records = [_record("e1", config.value, Verdict.CONTRADICTED) for config in ALL_CONFIGS]

    report = build_report(records, examples, dataset="ragtruth", n=1, seed=1)
    assert report.prober_gain_finding.status == "computed"
    # identical F1 for every config (all CONTRADICTED, all correct) -> the
    # bootstrap CIs for 'ATTEST - prober' and 'ATTEST full' must overlap.
    assert report.prober_gain_finding.ci_overlap is True
    assert "NO MEASURABLE" in report.prober_gain_finding.note


def test_build_report_disagreement_analysis_finds_universally_wrong_claim():
    examples = [_example("e1", True)]
    # Same claim_id across all four configs -- reflecting how bench/runner.py
    # actually works (one Claim, decomposed once, reused across configs).
    # Every config says GROUNDED (predicted negative) on a claim whose span
    # overlaps the ground-truth hallucination span -> wrong for all four.
    shared_claim_id = uuid4()
    records = [
        _record("e1", config.value, Verdict.GROUNDED, span=(2, 6), claim_id=shared_claim_id)
        for config in ALL_CONFIGS
    ]

    report = build_report(records, examples, dataset="ragtruth", n=1, seed=1)
    assert len(report.disagreement_analysis) == 1
    case = report.disagreement_analysis[0]
    assert case.example_id == "e1"
    assert case.ground_truth_positive is True
    assert set(case.verdicts_by_config) == {c.value for c in ALL_CONFIGS}


def test_build_report_json_serializes_cleanly():
    examples = [_example("e1", True)]
    records = [_record("e1", config.value, Verdict.CONTRADICTED) for config in ALL_CONFIGS]
    report = build_report(records, examples, dataset="ragtruth", n=1, seed=1)

    payload = json.loads(render_json(report))
    assert payload["status"] == "complete"
    assert payload["systems"][0]["precision"]["status"] == "computed"
