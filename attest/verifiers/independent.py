"""Independent verifier (A3).

Everyone else checks faithfulness *to the retrieved context*. This verifier
re-searches the open web independently of whatever the pipeline retrieved
and cross-checks the *source* itself (PLAN.md §2) — surfacing `STALE`:
entailed by the retrieved context, but independent retrieval disagrees with
it (PLAN.md §3).

Implements `attest.models.VerifierProtocol` structurally:

    class IndependentVerifier:
        async def verify(self, claim: Claim, ctx: VerifyContext) -> Verification: ...

Two LLM calls per claim, both via `attest.llm.complete` (never a direct
provider call, never `temperature` override — that's hardcoded upstream):

1. `model_tier="fast"` — rewrite `claim.text` into a neutral search query.
   The claim's own framing is never passed straight through to `search()`,
   or the search would just retrieve confirmation of it.
2. `model_tier="judge"` — judge the claim against the fresh search results.
   This is a real verdict judgment, the same bar as the entailment verifier,
   and never sees the entailment verifier's output or rationale — the only
   channel for cross-verifier information is `ctx.prior_entailment`, and
   that is used only in the STALE-gating logic below, never fed into the
   judge prompt itself (CLAUDE.md: "never let a verifier see another
   verifier's output").
"""

from __future__ import annotations

from uuid import uuid4

import structlog
from pydantic import BaseModel, Field

from attest.llm import LLMError, complete
from attest.models import (
    Claim,
    Evidence,
    Stance,
    Verdict,
    Verification,
    VerifyContext,
)
from attest.search import SearchResult, search

logger = structlog.get_logger(__name__)

_MIN_CORROBORATING_SOURCES_FOR_STALE = 2

_ALLOWED_JUDGE_VERDICTS = {
    Verdict.GROUNDED,
    Verdict.UNSUPPORTED,
    Verdict.CONTRADICTED,
    Verdict.UNVERIFIABLE,
}

_REWRITE_PROMPT = """\
Rewrite the following claim as a short, neutral web search query.

Claim: "{claim}"

Rules:
- Strip loaded adjectives, hedges, and any assertion of certainty the claim makes.
- Do not repeat the claim's own framing or phrase it so as to confirm it — write a \
neutral query that would surface evidence for OR against it equally.
- Keep it short: a handful of keywords, not a full sentence.

Return JSON only, no prose: {{"query": "..."}}\
"""

_JUDGE_PROMPT = """\
You are verifying a factual claim against fresh, independently retrieved web search \
results. These sources were retrieved just now and were NOT part of whatever context \
originally produced the claim.

Claim: "{claim}"

Sources (index, url, published date, snippet text):
{sources}

Decide whether these sources support, contradict, or say nothing useful about the claim.

Rules:
- Choose exactly one verdict: GROUNDED (the sources support the claim), UNSUPPORTED (no \
source addresses the claim either way), CONTRADICTED (one or more sources directly \
contradict the claim), or UNVERIFIABLE (the claim is subjective, predictive, an opinion, \
or otherwise not checkable against evidence). Never output any verdict other than these four.
- If sources disagree with each other, prefer the more recent ones and say so in the rationale.
- For every source you rely on, add one evidence entry: its index, a character span \
(start, end) into THAT SOURCE's own snippet text — never a quoted string, spans are \
checkable and quotes hallucinate — and whether it supports or refutes the claim.
- Keep the rationale short.

Return JSON only, no prose:
{{"verdict": "...", "rationale": "...", \
"evidence": [{{"source_index": 0, "quote_span": [0, 10], "stance": "support"}}]}}\
"""


class _QueryRewrite(BaseModel):
    query: str


class _JudgeEvidence(BaseModel):
    source_index: int
    quote_span: tuple[int, int] | None = None
    stance: Stance


class _JudgeVerdict(BaseModel):
    verdict: str
    rationale: str
    evidence: list[_JudgeEvidence] = Field(default_factory=list)


class IndependentVerifier:
    """Fresh-web-search cross-check. See module docstring for the STALE logic."""

    def __init__(self, k: int = 5) -> None:
        self._k = k

    async def verify(self, claim: Claim, ctx: VerifyContext) -> Verification:
        try:
            return await self._verify(claim, ctx)
        except Exception as exc:  # noqa: BLE001 - a dead verifier must not kill a run
            logger.warning("independent_verify_failed", claim_id=str(claim.id), error=str(exc))
            return Verification(
                id=uuid4(),
                claim_id=claim.id,
                verifier="independent",
                verdict=Verdict.UNVERIFIABLE,
                rationale=f"independent verification crashed ({exc}); treating as unverifiable",
                evidence=[],
            )

    async def _verify(self, claim: Claim, ctx: VerifyContext) -> Verification:
        query = await self._neutral_query(claim.text)
        results = await search(query, k=self._k)

        if not results:
            return Verification(
                id=uuid4(),
                claim_id=claim.id,
                verifier="independent",
                verdict=Verdict.UNVERIFIABLE,
                rationale=(
                    "independent web search returned no results; evidence too thin to "
                    "judge the claim, deferring without asserting staleness"
                ),
                evidence=[],
            )

        judged_verdict, rationale, evidence, refute_sources = await self._judge(
            claim.text, results
        )

        final_verdict = judged_verdict
        if ctx.prior_entailment == Verdict.GROUNDED and judged_verdict == Verdict.CONTRADICTED:
            if len(refute_sources) >= _MIN_CORROBORATING_SOURCES_FOR_STALE:
                final_verdict = Verdict.STALE
            else:
                rationale = (
                    f"{rationale} Independent evidence thin/conflicting (only "
                    f"{len(refute_sources)} corroborating source(s) contradict the claim, "
                    f"{_MIN_CORROBORATING_SOURCES_FOR_STALE} required); deferring to "
                    "entailment's grounding assessment without asserting staleness."
                )

        return Verification(
            id=uuid4(),
            claim_id=claim.id,
            verifier="independent",
            verdict=final_verdict,
            rationale=rationale,
            evidence=evidence,
        )

    async def _neutral_query(self, claim_text: str) -> str:
        prompt = _REWRITE_PROMPT.format(claim=claim_text)
        try:
            result = await complete(prompt, model_tier="fast", schema=_QueryRewrite)
        except LLMError as exc:
            logger.warning("independent_query_rewrite_failed", error=str(exc))
            return claim_text

        if isinstance(result.parsed, _QueryRewrite) and result.parsed.query.strip():
            return result.parsed.query.strip()
        return claim_text

    async def _judge(
        self, claim_text: str, results: list[SearchResult]
    ) -> tuple[Verdict, str, list[Evidence], set[int]]:
        sources_block = "\n".join(
            f"[{i}] url={r.url!r} published_date={r.published_date!r}\n{r.snippet}"
            for i, r in enumerate(results)
        )
        prompt = _JUDGE_PROMPT.format(claim=claim_text, sources=sources_block)

        try:
            result = await complete(prompt, model_tier="judge", schema=_JudgeVerdict)
        except LLMError as exc:
            logger.warning("independent_judge_failed", error=str(exc))
            return (
                Verdict.UNVERIFIABLE,
                f"independent judge call failed ({exc}); evidence too thin to judge, "
                "deferring without asserting staleness",
                [],
                set(),
            )

        parsed = result.parsed
        if not isinstance(parsed, _JudgeVerdict):
            return (
                Verdict.UNVERIFIABLE,
                "independent judge response did not parse as valid JSON; evidence too thin "
                "to judge, deferring without asserting staleness",
                [],
                set(),
            )

        try:
            verdict = Verdict(parsed.verdict)
        except ValueError:
            verdict = Verdict.UNVERIFIABLE
        if verdict not in _ALLOWED_JUDGE_VERDICTS:
            verdict = Verdict.UNVERIFIABLE

        evidence: list[Evidence] = []
        refute_sources: set[int] = set()
        for item in parsed.evidence:
            if not 0 <= item.source_index < len(results):
                continue  # ignore hallucinated out-of-range source references
            span = item.quote_span
            if span is not None:
                snippet_len = len(results[item.source_index].snippet)
                span = (
                    max(0, min(span[0], snippet_len)),
                    max(0, min(span[1], snippet_len)),
                )
            evidence.append(
                Evidence(
                    url=results[item.source_index].url,
                    quote_span=span,
                    stance=item.stance,
                )
            )
            if item.stance == "refute":
                refute_sources.add(item.source_index)

        return verdict, parsed.rationale, evidence, refute_sources


__all__ = ["IndependentVerifier"]
