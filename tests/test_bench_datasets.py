"""Tests for bench.datasets: parsing real (small, saved) RAGTruth/HaluEval
samples into BenchExample, and stratified sampling. No network -- reads the
saved fixture files under tests/fixtures/ (fetched for real while building
this loader; see bench/MAPPING.md and this agent's final report) instead of
live-fetching.
"""

from pathlib import Path

import pytest

from bench.datasets import (
    BenchExample,
    _parse_halueval_task,
    _parse_jsonl,
    _parse_ragtruth,
    load_dataset,
    stratified_sample,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _load_ragtruth_fixture():
    response_rows = _parse_jsonl(
        (FIXTURES / "ragtruth_sample_response.jsonl").read_text(encoding="utf-8")
    )
    source_rows = _parse_jsonl(
        (FIXTURES / "ragtruth_sample_source_info.jsonl").read_text(encoding="utf-8")
    )
    return response_rows, source_rows


def test_parse_ragtruth_produces_benchexample_shape():
    response_rows, source_rows = _load_ragtruth_fixture()
    examples = _parse_ragtruth(response_rows, source_rows)

    assert len(examples) == len(response_rows)
    for ex in examples:
        assert isinstance(ex, BenchExample)
        assert ex.dataset == "ragtruth"
        assert ex.example_id.startswith("ragtruth-")
        assert ex.retrieved_chunks and ex.retrieved_chunks[0].text
        assert isinstance(ex.answer, str) and ex.answer


def test_parse_ragtruth_non_hallucinated_example_has_no_spans():
    response_rows, source_rows = _load_ragtruth_fixture()
    examples = {ex.example_id: ex for ex in _parse_ragtruth(response_rows, source_rows)}
    non_hallucinated = examples["ragtruth-0"]
    assert non_hallucinated.ground_truth_hallucinated is False
    assert non_hallucinated.ground_truth_spans == []
    assert non_hallucinated.hallucination_type == "none"


def test_parse_ragtruth_hallucinated_example_has_spans_and_type():
    response_rows, source_rows = _load_ragtruth_fixture()
    examples = {ex.example_id: ex for ex in _parse_ragtruth(response_rows, source_rows)}
    hallucinated = examples["ragtruth-2"]
    assert hallucinated.ground_truth_hallucinated is True
    assert len(hallucinated.ground_truth_spans) >= 1
    assert hallucinated.hallucination_type in {"Evident Conflict", "Evident Baseless Info"}


def test_parse_ragtruth_single_label_types_preserved():
    response_rows, source_rows = _load_ragtruth_fixture()
    examples = {ex.example_id: ex for ex in _parse_ragtruth(response_rows, source_rows)}
    assert examples["ragtruth-21"].hallucination_type == "Evident Baseless Info"
    assert examples["ragtruth-79"].hallucination_type == "Subtle Conflict"
    assert examples["ragtruth-3"].hallucination_type == "Subtle Baseless Info"


def test_parse_ragtruth_data2txt_source_serialized_as_json_string():
    # ragtruth-5658 / source 13599 in the fixture has a structured (dict)
    # source_info -- RAGTruth's Data2txt task type, discovered while
    # actually downloading and inspecting the real dataset (not assumed
    # from memory) -- see bench/datasets.py::_source_info_text.
    response_rows, source_rows = _load_ragtruth_fixture()
    examples = {ex.example_id: ex for ex in _parse_ragtruth(response_rows, source_rows)}
    data2txt_example = examples["ragtruth-5658"]
    chunk_text = data2txt_example.retrieved_chunks[0].text
    assert isinstance(chunk_text, str)
    assert chunk_text.strip().startswith("{")  # serialized JSON object, not a Python dict repr


def test_parse_ragtruth_skips_rows_with_no_matching_source():
    response_rows, source_rows = _load_ragtruth_fixture()
    extra_row = dict(response_rows[0])
    extra_row["id"] = "no-such-source"
    extra_row["source_id"] = "does-not-exist"
    examples = _parse_ragtruth([*response_rows, extra_row], source_rows)
    assert "ragtruth-no-such-source" not in {ex.example_id for ex in examples}
    assert len(examples) == len(response_rows)  # the bad row was skipped, not crashed on


def test_parse_halueval_qa_yields_two_examples_per_row():
    rows = _parse_jsonl((FIXTURES / "halueval_qa_sample.jsonl").read_text(encoding="utf-8"))
    examples = _parse_halueval_task(rows, "qa")

    assert len(examples) == 2 * len(rows)
    right = [e for e in examples if not e.ground_truth_hallucinated]
    hallucinated = [e for e in examples if e.ground_truth_hallucinated]
    assert len(right) == len(rows)
    assert len(hallucinated) == len(rows)
    for ex in examples:
        assert ex.dataset == "halueval"
        assert ex.hallucination_type == "qa"
        assert ex.ground_truth_spans == []  # no span-level ground truth for HaluEval
        assert ex.retrieved_chunks and ex.retrieved_chunks[0].text


def test_stratified_sample_covers_multiple_buckets_deterministically():
    examples = []
    for bucket, count in [("a", 100), ("b", 10), ("c", 3)]:
        for i in range(count):
            examples.append(
                BenchExample(
                    example_id=f"{bucket}-{i}",
                    dataset="ragtruth",
                    hallucination_type=bucket,
                    query="q",
                    retrieved_chunks=[],
                    answer="a",
                    ground_truth_spans=[],
                    ground_truth_hallucinated=False,
                )
            )

    sample = stratified_sample(examples, n=12, seed=42)
    assert len(sample) == 12
    buckets_in_sample = {ex.hallucination_type for ex in sample}
    assert buckets_in_sample == {"a", "b", "c"}  # not dominated by the largest bucket

    sample_again = stratified_sample(examples, n=12, seed=42)
    assert [ex.example_id for ex in sample] == [ex.example_id for ex in sample_again]


def test_stratified_sample_returns_everything_when_n_exceeds_population():
    examples = [
        BenchExample(
            example_id=f"x-{i}",
            dataset="ragtruth",
            hallucination_type="none",
            query="q",
            retrieved_chunks=[],
            answer="a",
            ground_truth_spans=[],
            ground_truth_hallucinated=False,
        )
        for i in range(3)
    ]
    sample = stratified_sample(examples, n=100, seed=1)
    assert len(sample) == 3


def test_load_dataset_rejects_unknown_name():
    with pytest.raises(ValueError):
        load_dataset("unknown", n=5, seed=1)
