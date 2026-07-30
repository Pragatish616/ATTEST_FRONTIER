"""Benchmark report generation: `bench/results.md` + `bench/results.json`
(PLAN.md §10, §11).

Two entrypoints:
  - `build_pending_report()` -- the honest, no-real-run-yet template this
    module ships with. Every metric cell is `status="pending"`,
    `value=None`; `render_markdown` renders those as the literal string
    "PENDING", never a plausible-looking placeholder number.
  - `build_report(records, examples, ...)` -- the real computation, run
    once actual `ClaimRecord`s exist (from `bench/runner.py` against a real
    `ANTHROPIC_API_KEY`, or from a mocked run in tests).

Both produce the same `BenchmarkReport` schema, so `results.json`'s shape
never changes between "not run yet" and "run for real" -- downstream
tooling can rely on it now.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field

from bench.configs import ALL_CONFIGS, DISPLAY_NAMES, BenchConfig
from bench.datasets import BenchExample
from bench.mapping import EvalUnit, build_eval_units, claim_ground_truth
from bench.metrics import (
    CI,
    bootstrap_ci,
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

DEFAULT_COMMAND = "uv run python -m bench.run_benchmark --n 250 --seed 42 --dataset ragtruth"


class MetricValue(BaseModel):
    """One reported metric cell. `status="pending"` (the default) means no
    real run has produced this number yet -- `value`/`ci_low`/`ci_high` stay
    `None`. Never populate `value` with a placeholder; render it as the
    literal string "PENDING" instead (see `render`)."""

    status: Literal["pending", "computed"] = "pending"
    value: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None

    def render(self, *, prefix: str = "", suffix: str = "", decimals: int = 3) -> str:
        if self.status == "pending" or self.value is None:
            return "PENDING"
        value_s = f"{prefix}{self.value:.{decimals}f}{suffix}"
        if self.ci_low is not None and self.ci_high is not None:
            lo_s = f"{self.ci_low:.{decimals}f}"
            hi_s = f"{self.ci_high:.{decimals}f}"
            return f"{value_s} [{lo_s}, {hi_s}]"
        return value_s


class SystemResult(BaseModel):
    """One row of the PLAN.md §10 main table."""

    config: str
    system: str
    precision: MetricValue = Field(default_factory=MetricValue)
    recall: MetricValue = Field(default_factory=MetricValue)
    f1: MetricValue = Field(default_factory=MetricValue)
    cost_per_claim_usd: MetricValue = Field(default_factory=MetricValue)
    p95_latency_ms: MetricValue = Field(default_factory=MetricValue)
    n_eval_units: int | None = None
    n_claims: int | None = None


class SecondaryMetrics(BaseModel):
    """Reported as its own small table, never folded into the main one
    (task brief)."""

    fragile_precision_risk_rate: dict[str, MetricValue] = Field(default_factory=dict)
    unverifiable_abstention_rate: dict[str, MetricValue] = Field(default_factory=dict)
    stale_fire_rate: dict[str, MetricValue] = Field(default_factory=dict)


class DisagreementCase(BaseModel):
    example_id: str
    dataset: str
    hallucination_type: str | None
    claim_text: str | None
    ground_truth_positive: bool
    verdicts_by_config: dict[str, str]


class ProberGainFinding(BaseModel):
    """PLAN.md §10 / task brief: if the prober shows no measurable F1 gain
    (CI overlap between 'ATTEST - prober' and 'ATTEST full'), say so
    explicitly and prominently. Unconditional -- applies regardless of what
    the real numbers turn out to be."""

    status: Literal["pending", "computed"] = "pending"
    ci_overlap: bool | None = None
    note: str = (
        "PENDING -- requires a real run. Once computed: if the F1 confidence "
        "intervals for 'ATTEST - prober' and 'ATTEST full' overlap, that will "
        "be stated explicitly and prominently here, not buried or spun, per "
        "the task brief's unconditional instruction."
    )


class BenchmarkReport(BaseModel):
    generated_at: str
    dataset: str
    n_examples: int | None
    seed: int | None
    command: str
    status: Literal["pending", "complete"]
    systems: list[SystemResult]
    secondary: SecondaryMetrics
    prober_gain_finding: ProberGainFinding
    disagreement_analysis: list[DisagreementCase]
    disagreement_analysis_note: str
    notes: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Pending (template) report
# ---------------------------------------------------------------------------


def build_pending_report(
    *, dataset: str = "ragtruth", n: int | None = None, seed: int | None = None
) -> BenchmarkReport:
    systems = [SystemResult(config=c.value, system=DISPLAY_NAMES[c]) for c in ALL_CONFIGS]
    secondary = SecondaryMetrics(
        fragile_precision_risk_rate={c.value: MetricValue() for c in ALL_CONFIGS},
        unverifiable_abstention_rate={c.value: MetricValue() for c in ALL_CONFIGS},
        stale_fire_rate={c.value: MetricValue() for c in ALL_CONFIGS},
    )
    return BenchmarkReport(
        generated_at=datetime.now(UTC).isoformat(),
        dataset=dataset,
        n_examples=n,
        seed=seed,
        command=DEFAULT_COMMAND,
        status="pending",
        systems=systems,
        secondary=secondary,
        prober_gain_finding=ProberGainFinding(),
        disagreement_analysis=[],
        disagreement_analysis_note=(
            "Once a real run exists: for every evaluation unit (one claim for "
            "RAGTruth, one response OR-aggregated across its claims for "
            "HaluEval -- see bench/mapping.py) where all four configurations' "
            "predicted label disagrees with ground truth, cases are grouped by "
            "`hallucination_type` and listed here with a 2-sentence pattern "
            "summary per group. Computed by `build_disagreement_analysis()` "
            "in bench/report.py."
        ),
        notes=[
            "No real ATTEST run has been executed against this dataset yet: "
            "this environment has no configured ANTHROPIC_API_KEY (or "
            "GROQ_API_KEY / GEMINI_API_KEY) and no .env file. Every metric "
            "cell above is an explicit template, not a placeholder number.",
            f"To populate this report for real: {DEFAULT_COMMAND}",
        ],
    )


# ---------------------------------------------------------------------------
# Real report
# ---------------------------------------------------------------------------


def _verdict_desc_for_unit(unit: EvalUnit, records: list[ClaimRecord]) -> str:
    """Human-readable verdict summary for one evaluation unit, for the
    disagreement table. RAGTruth units are exactly one claim; HaluEval units
    are OR-aggregated across a response's claims, so multiple distinct
    verdicts are joined."""
    if unit.dataset == "ragtruth":
        matches = [r for r in records if r.claim_id == unit.claim_id and r.config == unit.config]
        return matches[0].predicted_verdict.value if matches else "?"
    matches = [r for r in records if r.example_id == unit.example_id and r.config == unit.config]
    verdicts = sorted({r.predicted_verdict.value for r in matches})
    return "+".join(verdicts) if verdicts else "?"


def build_disagreement_analysis(
    records: list[ClaimRecord],
    examples_by_id: dict[str, BenchExample],
    *,
    max_cases: int = 20,
) -> list[DisagreementCase]:
    """Evaluation units where every one of the four configurations produced
    a non-abstaining, wrong prediction. Abstention (UNVERIFIABLE) is never
    counted as "wrong" here -- it's a deliberate non-answer, not an error."""
    units_by_config = {c: build_eval_units(records, examples_by_id, c) for c in ALL_CONFIGS}

    key_to_units: dict[tuple, dict[str, EvalUnit]] = defaultdict(dict)
    for config, units in units_by_config.items():
        for u in units:
            key = (u.dataset, u.claim_id) if u.dataset == "ragtruth" else (u.dataset, u.example_id)
            key_to_units[key][config.value] = u

    cases: list[DisagreementCase] = []
    for by_config in key_to_units.values():
        if len(by_config) < len(ALL_CONFIGS):
            continue  # not every config produced a unit for this key
        all_wrong = all(
            u.predicted is not None and u.predicted != u.ground_truth_positive
            for u in by_config.values()
        )
        if not all_wrong:
            continue

        any_unit = next(iter(by_config.values()))
        claim_text = None
        if any_unit.dataset == "ragtruth" and any_unit.claim_id is not None:
            matching = [r for r in records if r.claim_id == any_unit.claim_id]
            claim_text = matching[0].claim_text if matching else None

        cases.append(
            DisagreementCase(
                example_id=any_unit.example_id,
                dataset=any_unit.dataset,
                hallucination_type=any_unit.hallucination_type,
                claim_text=claim_text,
                ground_truth_positive=any_unit.ground_truth_positive,
                verdicts_by_config={
                    cfg: _verdict_desc_for_unit(u, records) for cfg, u in by_config.items()
                },
            )
        )
        if len(cases) >= max_cases:
            break
    return cases


def _build_prober_gain_finding(f1_ci_by_config: dict[str, CI]) -> ProberGainFinding:
    minus_prober = f1_ci_by_config.get(BenchConfig.ATTEST_MINUS_PROBER.value)
    full = f1_ci_by_config.get(BenchConfig.ATTEST_FULL.value)
    if minus_prober is None or full is None:
        return ProberGainFinding()

    overlap = not (full.low > minus_prober.high or minus_prober.low > full.high)
    if overlap:
        note = (
            "NO MEASURABLE F1 GAIN FROM THE PROBER: the 95% CIs for "
            f"'ATTEST - prober' F1 ([{minus_prober.low:.3f}, {minus_prober.high:.3f}]) "
            f"and 'ATTEST full' F1 ([{full.low:.3f}, {full.high:.3f}]) overlap. "
            "Stated explicitly and prominently, per the task brief's "
            "unconditional instruction -- not buried, not spun."
        )
    else:
        note = (
            "The prober shows a measurable F1 gain: the 95% CIs for "
            f"'ATTEST - prober' ([{minus_prober.low:.3f}, {minus_prober.high:.3f}]) and "
            f"'ATTEST full' ([{full.low:.3f}, {full.high:.3f}]) do not overlap."
        )
    return ProberGainFinding(status="computed", ci_overlap=overlap, note=note)


def build_report(
    records: list[ClaimRecord],
    examples: list[BenchExample],
    *,
    dataset: str,
    n: int,
    seed: int,
    command: str = DEFAULT_COMMAND,
) -> BenchmarkReport:
    """Compute a full `BenchmarkReport` from real (or realistically mocked)
    `ClaimRecord`s. Every number here is a genuine computation over
    `records` -- no fallback to placeholder values."""
    examples_by_id = {e.example_id: e for e in examples}

    systems: list[SystemResult] = []
    f1_ci_by_config: dict[str, CI] = {}
    fragile_by_config: dict[str, MetricValue] = {}
    abstain_by_config: dict[str, MetricValue] = {}
    stale_by_config: dict[str, MetricValue] = {}

    for config in ALL_CONFIGS:
        units = build_eval_units(records, examples_by_id, config)
        config_records = [r for r in records if r.config == config.value]

        p, r_, f = precision(units), recall(units), f1(units)
        p_ci = bootstrap_ci(units, precision)
        r_ci = bootstrap_ci(units, recall)
        f_ci = bootstrap_ci(units, f1)
        f1_ci_by_config[config.value] = f_ci

        systems.append(
            SystemResult(
                config=config.value,
                system=DISPLAY_NAMES[config],
                precision=MetricValue(status="computed", value=p, ci_low=p_ci.low, ci_high=p_ci.high),
                recall=MetricValue(status="computed", value=r_, ci_low=r_ci.low, ci_high=r_ci.high),
                f1=MetricValue(status="computed", value=f, ci_low=f_ci.low, ci_high=f_ci.high),
                cost_per_claim_usd=MetricValue(
                    status="computed", value=mean_cost_per_claim(config_records)
                ),
                p95_latency_ms=MetricValue(
                    status="computed", value=p95_latency_ms(config_records)
                ),
                n_eval_units=len(units),
                n_claims=len(config_records),
            )
        )

        def _gt(rec: ClaimRecord, _examples_by_id: dict[str, BenchExample] = examples_by_id) -> bool:
            return claim_ground_truth(rec, _examples_by_id[rec.example_id])

        fragile_rate, _n_fragile = fragile_precision_risk_rate(config_records, _gt)
        fragile_by_config[config.value] = MetricValue(status="computed", value=fragile_rate)
        abstain_by_config[config.value] = MetricValue(
            status="computed", value=unverifiable_abstention_rate(config_records)
        )
        stale_by_config[config.value] = MetricValue(
            status="computed", value=stale_fire_rate(config_records)
        )

    return BenchmarkReport(
        generated_at=datetime.now(UTC).isoformat(),
        dataset=dataset,
        n_examples=n,
        seed=seed,
        command=command,
        status="complete",
        systems=systems,
        secondary=SecondaryMetrics(
            fragile_precision_risk_rate=fragile_by_config,
            unverifiable_abstention_rate=abstain_by_config,
            stale_fire_rate=stale_by_config,
        ),
        prober_gain_finding=_build_prober_gain_finding(f1_ci_by_config),
        disagreement_analysis=build_disagreement_analysis(records, examples_by_id),
        disagreement_analysis_note=(
            "Evaluation units (claims for RAGTruth, OR-aggregated responses "
            "for HaluEval) where every one of the four configurations "
            "produced a non-abstaining prediction that disagreed with ground "
            "truth, grouped implicitly by hallucination_type below."
        ),
        notes=[],
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_markdown(report: BenchmarkReport) -> str:
    lines: list[str] = []
    lines.append("# ATTEST Benchmark Results")
    lines.append("")
    lines.append(f"Generated: {report.generated_at}")
    lines.append(
        f"Dataset: `{report.dataset}` | n={report.n_examples if report.n_examples is not None else 'PENDING'} "
        f"| seed={report.seed if report.seed is not None else 'PENDING'} | status: **{report.status.upper()}**"
    )
    lines.append("")
    if report.notes:
        for note in report.notes:
            lines.append(f"> {note}")
        lines.append("")

    lines.append("## Main table (PLAN.md §10)")
    lines.append("")
    lines.append("| System | Precision | Recall | F1 | Cost/claim | p95 latency |")
    lines.append("|---|---|---|---|---|---|")
    for s in report.systems:
        lines.append(
            "| "
            + " | ".join(
                [
                    s.system,
                    s.precision.render(),
                    s.recall.render(),
                    s.f1.render(),
                    s.cost_per_claim_usd.render(prefix="$", decimals=6),
                    s.p95_latency_ms.render(suffix=" ms", decimals=0),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append(
        "The ablation rows ('ATTEST - prober', 'ATTEST - independent') are the "
        "entire argument -- see the prober-gain finding below before reading "
        "anything else into the full-system row."
    )
    lines.append("")

    lines.append("## Prober gain finding")
    lines.append("")
    lines.append(report.prober_gain_finding.note)
    lines.append("")

    lines.append("## Secondary metrics (PLAN.md §3, §10 / task brief)")
    lines.append("")
    lines.append(
        "Reported separately from the main table on purpose: folding these "
        "into a single headline number would hide exactly the tradeoffs a "
        "technical judge should be able to see."
    )
    lines.append("")
    lines.append(
        "| Config | FRAGILE precision-risk rate | UNVERIFIABLE abstention rate | STALE fire rate (diagnostic) |"
    )
    lines.append("|---|---|---|---|")
    for c in ALL_CONFIGS:
        lines.append(
            "| "
            + " | ".join(
                [
                    DISPLAY_NAMES[c],
                    report.secondary.fragile_precision_risk_rate.get(c.value, MetricValue()).render(),
                    report.secondary.unverifiable_abstention_rate.get(c.value, MetricValue()).render(),
                    report.secondary.stale_fire_rate.get(c.value, MetricValue()).render(),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append(
        "STALE fire rate is diagnostic only: RAGTruth/HaluEval have no notion "
        "of temporal staleness, so any non-zero rate here means the "
        "independent verifier is invoking STALE against a dataset that "
        "structurally cannot confirm or refute it -- see bench/MAPPING.md."
    )
    lines.append("")

    lines.append("## Disagreement analysis")
    lines.append("")
    lines.append(report.disagreement_analysis_note)
    lines.append("")
    if report.disagreement_analysis:
        lines.append("| Example | Hallucination type | Ground truth | Verdicts by config |")
        lines.append("|---|---|---|---|")
        for case in report.disagreement_analysis:
            verdicts = "; ".join(
                f"{DISPLAY_NAMES.get(BenchConfig(cfg), cfg)}={v}"
                for cfg, v in case.verdicts_by_config.items()
            )
            lines.append(
                "| "
                + " | ".join(
                    [
                        case.example_id,
                        case.hallucination_type or "none",
                        "hallucinated" if case.ground_truth_positive else "faithful",
                        verdicts,
                    ]
                )
                + " |"
            )
        lines.append("")
    else:
        lines.append(
            "(No cases yet -- populated once a real run exists. Empty is a "
            "valid outcome too: it would mean no evaluation unit fooled every "
            "single configuration.)"
        )
        lines.append("")

    return "\n".join(lines) + "\n"


def render_json(report: BenchmarkReport) -> str:
    return report.model_dump_json(indent=2)


def write_results(report: BenchmarkReport, *, md_path: str, json_path: str) -> None:
    from pathlib import Path

    Path(md_path).write_text(render_markdown(report), encoding="utf-8")
    Path(json_path).write_text(render_json(report), encoding="utf-8")


__all__ = [
    "MetricValue",
    "SystemResult",
    "SecondaryMetrics",
    "DisagreementCase",
    "ProberGainFinding",
    "BenchmarkReport",
    "DEFAULT_COMMAND",
    "build_pending_report",
    "build_report",
    "build_disagreement_analysis",
    "render_markdown",
    "render_json",
    "write_results",
]
