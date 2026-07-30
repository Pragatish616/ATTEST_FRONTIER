# ATTEST Benchmark Results

Generated: 2026-07-28T03:35:53.108808+00:00
Dataset: `ragtruth` | n=250 | seed=42 | status: **PENDING**

> No real ATTEST run has been executed against this dataset yet: this environment has no configured ANTHROPIC_API_KEY (or GROQ_API_KEY / GEMINI_API_KEY) and no .env file. Every metric cell above is an explicit template, not a placeholder number.
> To populate this report for real: uv run python -m bench.run_benchmark --n 250 --seed 42 --dataset ragtruth

## Main table (PLAN.md §10)

| System | Precision | Recall | F1 | Cost/claim | p95 latency |
|---|---|---|---|---|---|
| Single-pass LLM judge (baseline) | PENDING | PENDING | PENDING | PENDING | PENDING |
| ATTEST − prober (ablation) | PENDING | PENDING | PENDING | PENDING | PENDING |
| ATTEST − independent (ablation) | PENDING | PENDING | PENDING | PENDING | PENDING |
| ATTEST full | PENDING | PENDING | PENDING | PENDING | PENDING |

The ablation rows ('ATTEST - prober', 'ATTEST - independent') are the entire argument -- see the prober-gain finding below before reading anything else into the full-system row.

## Prober gain finding

PENDING -- requires a real run. Once computed: if the F1 confidence intervals for 'ATTEST - prober' and 'ATTEST full' overlap, that will be stated explicitly and prominently here, not buried or spun, per the task brief's unconditional instruction.

## Secondary metrics (PLAN.md §3, §10 / task brief)

Reported separately from the main table on purpose: folding these into a single headline number would hide exactly the tradeoffs a technical judge should be able to see.

| Config | FRAGILE precision-risk rate | UNVERIFIABLE abstention rate | STALE fire rate (diagnostic) |
|---|---|---|---|
| Single-pass LLM judge (baseline) | PENDING | PENDING | PENDING |
| ATTEST − prober (ablation) | PENDING | PENDING | PENDING |
| ATTEST − independent (ablation) | PENDING | PENDING | PENDING |
| ATTEST full | PENDING | PENDING | PENDING |

STALE fire rate is diagnostic only: RAGTruth/HaluEval have no notion of temporal staleness, so any non-zero rate here means the independent verifier is invoking STALE against a dataset that structurally cannot confirm or refute it -- see bench/MAPPING.md.

## Disagreement analysis

Once a real run exists: for every evaluation unit (one claim for RAGTruth, one response OR-aggregated across its claims for HaluEval -- see bench/mapping.py) where all four configurations' predicted label disagrees with ground truth, cases are grouped by `hallucination_type` and listed here with a 2-sentence pattern summary per group. Computed by `build_disagreement_analysis()` in bench/report.py.

(No cases yet -- populated once a real run exists. Empty is a valid outcome too: it would mean no evaluation unit fooled every single configuration.)

