"""Single-pass LLM judge -- the benchmark's baseline configuration
(PLAN.md §10, row 1).

This is a deliberately *different, simpler* prompt from
`attest.verifiers.entailment.EntailmentVerifier` -- one direct judgment per
claim against the full context, no adversarial probing, no independent
search, no reconciliation (there is only one signal, so nothing to
reconcile). Writing this fresh (rather than calling `EntailmentVerifier` and
relabeling its output) is what makes the baseline row a meaningful point of
comparison instead of a duplicate of "ATTEST − prober − independent" wearing
a different name -- see PLAN.md §10 and the task brief.

Deliberately not shaped as `attest.models.Verification`: this verdict is
never reconciled with anything else, and `Verification.verifier` is a frozen
`Literal["entailment", "independent", "prober"]` (attest.models) that has no
slot for "baseline" -- see `bench/MAPPING.md`. `BaselineResult` is this
module's own type instead.

Same rules as every other verifier in this codebase (CLAUDE.md): structured
output only (JSON via `attest.llm.complete`, parsed into a Pydantic model,
never regex'd prose), `temperature=0` (hardcoded upstream in
`attest.llm.complete` -- this module has no way to override it even if it
wanted to), and every call's cost/latency is captured on the result.
"""

from __future__ import annotations

from typing import Literal
from uuid import UUID

import structlog
from pydantic import BaseModel

from attest.llm import LLMError, complete
from attest.models import Claim, Verdict, VerifyContext

logger = structlog.get_logger(__name__)

_AllowedVerdict = Literal["GROUNDED", "UNSUPPORTED", "CONTRADICTED", "UNVERIFIABLE"]

_PROMPT_TEMPLATE = """\
You are fact-checking a single claim from an AI-generated answer. You have \
one pass and no other tools: judge the claim directly against the context \
below and commit to a verdict.

Claim: {claim}

Context:
{chunks}

Choose exactly one verdict:
- GROUNDED: the context supports the claim.
- UNSUPPORTED: the context does not address the claim, or only weakly implies it.
- CONTRADICTED: the context directly contradicts the claim.
- UNVERIFIABLE: the claim is subjective, predictive, an opinion, or otherwise \
not checkable against evidence.

Return ONLY JSON matching this schema, nothing else:
{{"verdict": "GROUNDED" | "UNSUPPORTED" | "CONTRADICTED" | "UNVERIFIABLE", "rationale": str}}
"""


class _BaselineResponse(BaseModel):
    verdict: _AllowedVerdict
    rationale: str


class BaselineResult(BaseModel):
    """Bench-local result shape for one baseline judgment. Not part of the
    frozen `attest.models` contract -- see module docstring."""

    claim_id: UUID
    verdict: Verdict
    rationale: str
    cost_usd: float
    latency_ms: int


def _build_prompt(claim: Claim, ctx: VerifyContext) -> str:
    chunks_block = "\n\n".join(f"[chunk {i}] {c.text}" for i, c in enumerate(ctx.retrieved_chunks))
    return _PROMPT_TEMPLATE.format(claim=claim.text, chunks=chunks_block or "(no context provided)")


class SinglePassJudge:
    """The baseline: one `attest.llm.complete(model_tier="judge", ...)` call
    per claim, nothing else."""

    async def judge(self, claim: Claim, ctx: VerifyContext) -> BaselineResult:
        prompt = _build_prompt(claim, ctx)
        try:
            result = await complete(prompt, model_tier="judge", schema=_BaselineResponse)
        except LLMError as exc:
            logger.warning("baseline_llm_failed", claim_id=str(claim.id), error=str(exc))
            return BaselineResult(
                claim_id=claim.id,
                verdict=Verdict.UNSUPPORTED,
                rationale=f"baseline judge degraded: LLM call failed ({exc})",
                cost_usd=0.0,
                latency_ms=0,
            )

        parsed = result.parsed
        if not isinstance(parsed, _BaselineResponse):
            logger.warning("baseline_parse_failed", claim_id=str(claim.id), text=result.text[:200])
            return BaselineResult(
                claim_id=claim.id,
                verdict=Verdict.UNSUPPORTED,
                rationale="baseline judge degraded: could not parse judge output",
                cost_usd=result.cost_usd,
                latency_ms=result.latency_ms,
            )

        return BaselineResult(
            claim_id=claim.id,
            verdict=Verdict(parsed.verdict),
            rationale=parsed.rationale,
            cost_usd=result.cost_usd,
            latency_ms=result.latency_ms,
        )


__all__ = ["SinglePassJudge", "BaselineResult"]
