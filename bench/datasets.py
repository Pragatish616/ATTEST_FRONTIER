"""Benchmark dataset loader (PLAN.md §10): RAGTruth primary, HaluEval fallback.

Both datasets are plain JSON/JSONL on GitHub, no auth required:

- RAGTruth (https://github.com/ParticleMedia/RAGTruth): `dataset/response.jsonl`
  (one row per generated answer, `labels: [{start, end, text, label_type, ...}]`
  span-annotated hallucinations) joined against `dataset/source_info.jsonl`
  (one row per source document, keyed by `source_id`, carrying the original
  `prompt` and `source_info` text). Verified against the real files (not
  guessed from memory) before writing the parser below -- see the field
  names used in `_parse_ragtruth`.
- HaluEval (https://github.com/RUCAIBox/HaluEval): `data/qa_data.json` (and
  siblings `dialogue_data.json` / `summarization_data.json`) are JSON-*lines*
  files despite the `.json` extension -- one JSON object per line, each
  carrying a `knowledge`/context field, a query-ish field, and a
  `right_*` / `hallucinated_*` answer pair. Each row yields two
  `BenchExample`s (one faithful, one hallucinated) since HaluEval has no
  span-level annotation, only this response-level pairing.

Every fetch is cached to disk under `bench/data_cache/` (mirrors the
disk-cache pattern in `attest/search.py`) so repeat benchmark runs, and CI,
don't re-download multi-megabyte dataset files. This module makes real
network calls but never touches an LLM -- it can (and was, while building
this) be exercised for real with no API key configured at all.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Literal

import httpx
import structlog
from pydantic import BaseModel, Field

from attest.models import RetrievedChunk

logger = structlog.get_logger(__name__)

_CACHE_DIR = Path(__file__).parent / "data_cache"

_RAGTRUTH_RESPONSE_URL = (
    "https://raw.githubusercontent.com/ParticleMedia/RAGTruth/main/dataset/response.jsonl"
)
_RAGTRUTH_SOURCE_URL = (
    "https://raw.githubusercontent.com/ParticleMedia/RAGTruth/main/dataset/source_info.jsonl"
)

_HALUEVAL_BASE = "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data"
_HALUEVAL_TASK_FILES = {
    "qa": "qa_data.json",
    "dialogue": "dialogue_data.json",
    "summarization": "summarization_data.json",
}
# Field-name mapping per HaluEval task type, verified against the real files:
#   qa_data.json:            knowledge / question         / right_answer   / hallucinated_answer
#   dialogue_data.json:      knowledge / dialogue_history  / right_response / hallucinated_response
#   summarization_data.json: document  / (none)            / right_summary  / hallucinated_summary
_HALUEVAL_FIELD_MAP: dict[str, dict[str, str | None]] = {
    "qa": {
        "context": "knowledge",
        "query": "question",
        "right": "right_answer",
        "hallucinated": "hallucinated_answer",
    },
    "dialogue": {
        "context": "knowledge",
        "query": "dialogue_history",
        "right": "right_response",
        "hallucinated": "hallucinated_response",
    },
    "summarization": {
        "context": "document",
        "query": None,
        "right": "right_summary",
        "hallucinated": "hallucinated_summary",
    },
}

# RAGTruth's four annotation categories observed in the real dataset (PLAN.md
# §10 stratification), plus the sentinel bucket for non-hallucinated examples.
_NOT_HALLUCINATED = "none"


class BenchExample(BaseModel):
    """One example in the internal benchmark format, shared across datasets.

    `retrieved_chunks` feeds straight into the real ATTEST pipeline
    (`attest.models.RetrievedChunk`). `ground_truth_spans` are character
    offsets into `answer` -- RAGTruth's hallucinated-span annotations,
    always empty for HaluEval (no span-level ground truth exists there) and
    for non-hallucinated RAGTruth examples.
    """

    example_id: str
    dataset: Literal["ragtruth", "halueval"]
    hallucination_type: str | None
    query: str
    retrieved_chunks: list[RetrievedChunk]
    answer: str
    ground_truth_spans: list[tuple[int, int]] = Field(default_factory=list)
    ground_truth_hallucinated: bool


def _cache_path(name: str) -> Path:
    return _CACHE_DIR / name


def _fetch_text(url: str, cache_name: str, *, timeout: float = 60.0) -> str:
    """Fetch `url` as text, caching to disk. A cache hit short-circuits the
    network entirely (attest/search.py's disk-cache pattern)."""
    path = _cache_path(cache_name)
    if path.exists():
        logger.info("dataset_cache_hit", cache_name=cache_name, path=str(path))
        return path.read_text(encoding="utf-8")

    logger.info("dataset_fetching", url=url)
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        response = client.get(url)
        response.raise_for_status()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(response.text, encoding="utf-8")
    logger.info("dataset_fetched", url=url, cache_name=cache_name, bytes=len(response.text))
    return response.text


def _parse_jsonl(text: str) -> list[dict]:
    rows: list[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# RAGTruth
# ---------------------------------------------------------------------------


def _source_info_text(source_info: object) -> str:
    """RAGTruth's `source_info` field is plain text for the Summary and QA
    task types, but a structured JSON object for the Data2txt task type
    (discovered while fetching and inspecting the real dataset, not
    assumed from memory). `RetrievedChunk.text` is a plain string, so a
    structured source is serialized to a stable JSON string instead of
    being dropped or coerced with `str()` (which would produce Python
    repr-style quoting, not valid/readable JSON)."""
    if isinstance(source_info, str):
        return source_info
    return json.dumps(source_info, ensure_ascii=False, sort_keys=True)


def _ragtruth_hallucination_type(labels: list[dict]) -> str:
    """Stratification bucket for one response row.

    A response can carry multiple labels of *different* types (e.g. one
    Evident Conflict span and one Subtle Baseless Info span in the same
    answer) -- RAGTruth annotates at the span level, not the response
    level. Stratified sampling needs exactly one bucket per example, so
    this takes the first label's type; the underlying claim-level ground
    truth used for precision/recall (`ground_truth_spans`, checked via span
    overlap in `bench/mapping.py`) is unaffected by this simplification --
    only which sampling stratum the *example* counts toward. Documented
    again, with the alternative considered, in `bench/MAPPING.md`.
    """
    if not labels:
        return _NOT_HALLUCINATED
    return labels[0].get("label_type") or _NOT_HALLUCINATED


def _parse_ragtruth(response_rows: list[dict], source_rows: list[dict]) -> list[BenchExample]:
    sources_by_id = {row["source_id"]: row for row in source_rows}
    examples: list[BenchExample] = []
    skipped_no_source = 0
    for row in response_rows:
        source_id = row.get("source_id")
        source = sources_by_id.get(source_id)
        if source is None:
            skipped_no_source += 1
            continue

        labels = row.get("labels") or []
        spans = [
            (label["start"], label["end"])
            for label in labels
            if "start" in label and "end" in label
        ]
        chunk = RetrievedChunk(
            chunk_index=0,
            source_id=str(source_id),
            source_url=None,
            text=_source_info_text(source.get("source_info", "")),
        )
        examples.append(
            BenchExample(
                example_id=f"ragtruth-{row['id']}",
                dataset="ragtruth",
                hallucination_type=_ragtruth_hallucination_type(labels),
                query=source.get("prompt", ""),
                retrieved_chunks=[chunk],
                answer=row.get("response", ""),
                ground_truth_spans=spans,
                ground_truth_hallucinated=bool(spans),
            )
        )
    if skipped_no_source:
        logger.warning("ragtruth_rows_missing_source", count=skipped_no_source)
    return examples


def load_ragtruth_raw() -> list[BenchExample]:
    """Fetch + parse the full RAGTruth dataset. Real network call (or disk
    cache); no LLM involved."""
    response_text = _fetch_text(_RAGTRUTH_RESPONSE_URL, "ragtruth_response.jsonl")
    source_text = _fetch_text(_RAGTRUTH_SOURCE_URL, "ragtruth_source_info.jsonl")
    return _parse_ragtruth(_parse_jsonl(response_text), _parse_jsonl(source_text))


# ---------------------------------------------------------------------------
# HaluEval (fallback)
# ---------------------------------------------------------------------------


def _parse_halueval_task(rows: list[dict], task: str) -> list[BenchExample]:
    fields = _HALUEVAL_FIELD_MAP[task]
    examples: list[BenchExample] = []
    for i, row in enumerate(rows):
        context_text = row.get(fields["context"], "") if fields["context"] else ""
        query_field = fields["query"]
        query = (
            row.get(query_field, "")
            if query_field
            else f"Summarize the following: {context_text[:200]}"
        )
        chunk = RetrievedChunk(chunk_index=0, source_id=f"halueval-{task}-{i}", text=context_text)
        for kind, is_hallucinated in (("right", False), ("hallucinated", True)):
            answer = row.get(fields[kind])
            if not answer:
                continue
            examples.append(
                BenchExample(
                    example_id=f"halueval-{task}-{i}-{kind}",
                    dataset="halueval",
                    hallucination_type=task,
                    query=query,
                    retrieved_chunks=[chunk],
                    answer=answer,
                    ground_truth_spans=[],
                    ground_truth_hallucinated=is_hallucinated,
                )
            )
    return examples


def load_halueval_raw(*, tasks: tuple[str, ...] = ("qa",)) -> list[BenchExample]:
    """Fetch + parse HaluEval (the fallback dataset). Defaults to the QA
    split only; `dialogue`/`summarization` use the same parser (see
    `_HALUEVAL_FIELD_MAP`) and can be included via `tasks=(...)`, but QA
    alone is enough to exercise the fallback path for real."""
    examples: list[BenchExample] = []
    for task in tasks:
        filename = _HALUEVAL_TASK_FILES[task]
        text = _fetch_text(f"{_HALUEVAL_BASE}/{filename}", f"halueval_{filename}")
        examples.extend(_parse_halueval_task(_parse_jsonl(text), task))
    return examples


# ---------------------------------------------------------------------------
# Stratified sampling
# ---------------------------------------------------------------------------


def stratified_sample(examples: list[BenchExample], n: int, seed: int) -> list[BenchExample]:
    """Deterministic, stratified sample of `n` examples across
    `hallucination_type` buckets.

    Shuffles each bucket independently with `random.Random(seed)`, then
    round-robins across buckets (in sorted-name order, for determinism) so
    the sample spreads proportionally across categories rather than being
    dominated by whichever bucket is largest -- RAGTruth's non-hallucinated
    bucket outnumbers every hallucinated category by a wide margin, and
    PLAN.md §10 / the task brief are explicit that sampling "only the easy
    cases" isn't acceptable.
    """
    if n >= len(examples):
        return list(examples)

    rng = random.Random(seed)
    buckets: dict[str, list[BenchExample]] = defaultdict(list)
    for ex in examples:
        buckets[ex.hallucination_type or _NOT_HALLUCINATED].append(ex)
    for bucket in buckets.values():
        rng.shuffle(bucket)

    bucket_names = sorted(buckets)
    cursors = dict.fromkeys(bucket_names, 0)
    sample: list[BenchExample] = []
    while len(sample) < n:
        progressed = False
        for name in bucket_names:
            if len(sample) >= n:
                break
            cursor = cursors[name]
            bucket = buckets[name]
            if cursor < len(bucket):
                sample.append(bucket[cursor])
                cursors[name] = cursor + 1
                progressed = True
        if not progressed:
            break  # every bucket exhausted before reaching n
    return sample


def load_dataset(
    name: Literal["ragtruth", "halueval"],
    *,
    n: int = 250,
    seed: int = 42,
) -> list[BenchExample]:
    """Main entrypoint: fetch (or read from cache), parse, and draw a
    fixed-seed stratified sample of `n` examples. RAGTruth primary, HaluEval
    fallback (PLAN.md §10)."""
    if name == "ragtruth":
        examples = load_ragtruth_raw()
    elif name == "halueval":
        examples = load_halueval_raw()
    else:
        raise ValueError(f"unknown dataset {name!r}")

    if not examples:
        raise RuntimeError(f"loaded zero examples for dataset={name!r} -- check network/cache")

    return stratified_sample(examples, n, seed)


__all__ = [
    "BenchExample",
    "load_dataset",
    "load_ragtruth_raw",
    "load_halueval_raw",
    "stratified_sample",
]
