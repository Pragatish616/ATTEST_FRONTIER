# Owned by A1 (Decomposer + Entailment). Not implemented by A0.
#
# Implements VerifierProtocol (attest.models.VerifierProtocol): claim vs
# retrieved context, temperature=0, verdict from the frozen taxonomy only.
"""Entailment verifier: judges a claim strictly against retrieved context.

    class EntailmentVerifier:
        async def verify(self, claim: Claim, ctx: VerifyContext) -> Verification

Structurally satisfies `attest.models.VerifierProtocol`.

Design notes (CLAUDE.md "Prompt-engineering rules for the verifiers"):

- Judges `claim.text` against `ctx.retrieved_chunks` ONLY — no outside
  knowledge, no benefit of the doubt. Context silence is `UNSUPPORTED`, not
  "probably true".
- Returns exactly one of GROUNDED / UNSUPPORTED / CONTRADICTED /
  UNVERIFIABLE. The private response schema's `verdict` field is a
  `Literal` restricted to those four strings, so the parser itself rejects
  a model that tries to emit FRAGILE or STALE (those belong to the prober
  and independent verifier respectively) — such a response is treated as an
  unparsable judge output and degrades to UNSUPPORTED rather than ever
  reaching the reconciler.
- The LLM never sees or invents a UUID: it returns a `chunk_index` (an int
  into the prompt's enumerated chunk list) plus a span into that chunk's own
  text; this module translates `chunk_index` -> `ctx.retrieved_chunks[i].id`
  itself when building `Evidence`. Every evidence span is validated
  (`0 <= start < end <= len(chunk.text)`) and dropped (not raised) if it
  fails, same for an out-of-range `chunk_index`.
- `model_tier="judge"`; `temperature=0` is already hardcoded inside
  `attest.llm.complete`.
- Never reads `ctx.prior_entailment` — that field is reserved for the
  independent verifier only (CONTRACT_CHANGE_REQUEST.md).
- Never raises: an LLM failure or unparsable judge output degrades to a
  `Verification` with `verdict=Verdict.UNSUPPORTED` and an explanatory
  rationale, rather than propagating into the orchestrator's fan-out.
"""

from __future__ import annotations

from typing import Literal
from uuid import uuid4

import structlog
from pydantic import BaseModel, Field

from attest.llm import LLMError, complete
from attest.models import Claim, Evidence, Stance, Verdict, Verification, VerifyContext

logger = structlog.get_logger(__name__)

_AllowedVerdict = Literal["GROUNDED", "UNSUPPORTED", "CONTRADICTED", "UNVERIFIABLE"]

_PROMPT_TEMPLATE = """\
You are a strict entailment judge. Decide whether the Claim below is \
supported ONLY by the Retrieved Context. Use no outside knowledge and give \
no benefit of the doubt.

Claim: {claim}

Retrieved Context:
{chunks}

Rules:
- GROUNDED: the context entails the claim.
- CONTRADICTED: the context directly contradicts the claim.
- UNSUPPORTED: the context is silent on the claim, or too weak or indirect \
to entail it. Silence is UNSUPPORTED, never GROUNDED.
- UNVERIFIABLE: the claim is not the kind of statement evidence can settle \
(subjective, predictive, opinion) even if related words appear in the \
context.

For every piece of evidence you use, cite the chunk index (as labeled \
"[chunk N]" below) and a character span (span_start, span_end; 0-indexed, \
end exclusive) into THAT CHUNK's own text, plus whether it supports or \
refutes the claim. Do not quote text — give offsets only.

Return ONLY JSON matching this schema, nothing else:
{{"verdict": "GROUNDED" | "UNSUPPORTED" | "CONTRADICTED" | "UNVERIFIABLE", \
"rationale": str, "evidence": [{{"chunk_index": int, "span_start": int, \
"span_end": int, "stance": "support" | "refute"}}]}}
"""


class _RawEvidence(BaseModel):
    """Private response schema — one evidence entry as returned by the judge LLM call."""

    chunk_index: int
    span_start: int
    span_end: int
    stance: Stance


class _EntailmentResponse(BaseModel):
    """Private response schema for the entailment LLM call.

    `verdict` is intentionally restricted to the four values this verifier
    is allowed to emit — a model response using FRAGILE/STALE fails schema
    validation and is treated as unparsable, never reaches `verify()`'s
    caller as one of those two reserved verdicts.
    """

    verdict: _AllowedVerdict
    rationale: str
    evidence: list[_RawEvidence] = Field(default_factory=list)


def _degraded(claim: Claim, rationale: str) -> Verification:
    return Verification(
        id=uuid4(),
        claim_id=claim.id,
        verifier="entailment",
        verdict=Verdict.UNSUPPORTED,
        rationale=rationale,
    )


def _build_prompt(claim: Claim, ctx: VerifyContext) -> str:
    chunks_block = "\n\n".join(
        f"[chunk {i}] {chunk.text}" for i, chunk in enumerate(ctx.retrieved_chunks)
    )
    return _PROMPT_TEMPLATE.format(claim=claim.text, chunks=chunks_block)


def _build_evidence(parsed: _EntailmentResponse, ctx: VerifyContext, claim: Claim) -> list[Evidence]:
    evidence: list[Evidence] = []
    for raw_ev in parsed.evidence:
        if not (0 <= raw_ev.chunk_index < len(ctx.retrieved_chunks)):
            logger.warning(
                "entailment_evidence_bad_chunk_index",
                claim_id=str(claim.id),
                chunk_index=raw_ev.chunk_index,
            )
            continue
        chunk = ctx.retrieved_chunks[raw_ev.chunk_index]
        start, end = raw_ev.span_start, raw_ev.span_end
        if not (0 <= start < end <= len(chunk.text)):
            logger.warning(
                "entailment_evidence_bad_span",
                claim_id=str(claim.id),
                chunk_id=str(chunk.id),
                span=(start, end),
            )
            continue
        evidence.append(
            Evidence(chunk_id=chunk.id, quote_span=(start, end), stance=raw_ev.stance)
        )
    return evidence


class EntailmentVerifier:
    """Judges a claim strictly against `ctx.retrieved_chunks`. See module docstring."""

    async def verify(self, claim: Claim, ctx: VerifyContext) -> Verification:
        if not ctx.retrieved_chunks:
            # No context at all is maximal silence — UNSUPPORTED without
            # spending a judge call on an empty prompt.
            return _degraded(claim, "No retrieved context was available to check this claim against.")

        prompt = _build_prompt(claim, ctx)
        try:
            result = await complete(prompt, model_tier="judge", schema=_EntailmentResponse)
        except LLMError as exc:
            logger.warning("entailment_llm_failed", claim_id=str(claim.id), error=str(exc))
            return _degraded(claim, f"Verifier degraded: judge call failed ({exc}).")

        parsed = result.parsed
        if not isinstance(parsed, _EntailmentResponse):
            logger.warning(
                "entailment_parse_failed", claim_id=str(claim.id), text=result.text[:200]
            )
            return _degraded(claim, "Verifier degraded: could not parse judge output.")

        evidence = _build_evidence(parsed, ctx, claim)

        return Verification(
            id=uuid4(),
            claim_id=claim.id,
            verifier="entailment",
            verdict=Verdict(parsed.verdict),
            rationale=parsed.rationale,
            evidence=evidence,
            latency_ms=result.latency_ms,
            cost_usd=result.cost_usd,
        )


__all__ = ["EntailmentVerifier"]
