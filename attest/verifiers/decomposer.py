# Owned by A1 (Decomposer + Entailment). Not implemented by A0.
#
# answer -> atomic, self-contained Claim objects with char spans into the
# original answer text (PLAN.md §4).
"""Decomposer: split an answer into atomic, self-contained claims.

Public entrypoint:

    async def decompose(answer: str, query: str, *, run_id: UUID) -> list[Claim]

`run_id` is accepted as a required keyword parameter rather than left for the
caller to patch on afterwards: by the time the orchestrator (A4) calls this,
the `runs` row already exists (it needs a run_id to stream `run.started`
before decomposition finishes), so the id is available up front and every
`Claim` this function returns already carries the correct `run_id`. If a
caller ever needs to re-parent claims onto a different run, `.model_copy
(update={"run_id": ...})` still works since `Claim` is a normal Pydantic
model.

Design notes (see CLAUDE.md "Prompt-engineering rules for the verifiers" and
the A1 task brief):

- One LLM call per run (`model_tier="fast"`), private response schema below
  — nothing beyond {text, span_start, span_end, is_unverifiable, reason}.
- `span_start`/`span_end` are character offsets into the *original* `answer`
  string, never into the (possibly reworded, pronoun-resolved) claim text.
  Every span is validated before a Claim is emitted; if the model's own
  offsets don't check out, a best-effort fallback search is tried, and if
  that also fails the claim is dropped (logged, not raised) rather than
  emitting an unanchored span. It is fine to under-decompose (the prompt
  asks the model to merge sub-claims rather than guess at a split).
- Subjective/predictive/hedged/opinion claims come back from the model
  flagged `is_unverifiable=True` with a `reason`; those are emitted with
  `verdict=Verdict.UNVERIFIABLE` and `rationale=reason` set directly, which
  is how this module tells the orchestrator "don't send this one to the
  three verifiers". Every other claim keeps `verdict=None` (still pending).
"""

from __future__ import annotations

import re
from uuid import UUID, uuid4

import structlog
from pydantic import BaseModel, Field

from attest.llm import LLMError, complete
from attest.models import Claim, Verdict

logger = structlog.get_logger(__name__)

_MIN_OVERLAP_WORD_LEN = 3
_WORD_RE = re.compile(r"[A-Za-z0-9]+")

_PROMPT_TEMPLATE = """\
You are decomposing an AI-generated answer into atomic, self-contained \
factual claims for independent verification.

Query: {query}

Answer: {answer}

Task: split the Answer into atomic claims. Each claim must:
- Assert exactly one fact.
- Be self-contained: resolve every pronoun and anaphoric/temporal reference \
(e.g. "it", "this", "then", "the company") using the Query and Answer, so \
the claim reads correctly with no other context.
- Prefer merging two closely related sub-claims into one claim over \
guessing at a split you cannot anchor back to the Answer text.

For each claim, also give an anchor span: the character start/end offsets \
(0-indexed, end exclusive) INTO THE ANSWER STRING ABOVE of the substring in \
the original Answer that the claim was drawn from. The anchor is the \
literal original text location (it does not need to be reworded) — it is \
used to highlight the answer in a UI, so it must be accurate.

Mark a claim `is_unverifiable: true` (with a one-sentence `reason`) if it is \
subjective, predictive, an opinion, or otherwise not checkable against \
evidence (e.g. "this is the best approach", "prices will likely rise"). \
Otherwise set `is_unverifiable: false` and `reason: null`.

Return ONLY JSON matching this schema, nothing else:
{{"claims": [{{"text": str, "span_start": int, "span_end": int, \
"is_unverifiable": bool, "reason": str | null}}]}}
"""


class _RawClaim(BaseModel):
    """Private response schema — one claim as returned by the decomposer LLM call."""

    text: str
    span_start: int
    span_end: int
    is_unverifiable: bool = False
    reason: str | None = None


class _DecomposeResponse(BaseModel):
    """Private response schema for the decomposer LLM call."""

    claims: list[_RawClaim] = Field(default_factory=list)


def _significant_words(text: str) -> set[str]:
    return {w.lower() for w in _WORD_RE.findall(text) if len(w) >= _MIN_OVERLAP_WORD_LEN}


def _span_is_valid(answer: str, start: int, end: int, claim_text: str) -> bool:
    """True iff [start:end) is an in-bounds, non-empty span of `answer` that
    plausibly overlaps the claim's subject.

    "Overlaps" is a lightweight word-overlap heuristic, not exact matching —
    the anchor span is the original wording, the claim text may be reworded
    (pronouns resolved, etc.), so we only require some shared vocabulary.
    """
    if not (0 <= start < end <= len(answer)):
        return False
    span_text = answer[start:end]
    if not span_text.strip():
        return False
    span_words = _significant_words(span_text)
    claim_words = _significant_words(claim_text)
    if not span_words or not claim_words:
        # Nothing significant to compare (e.g. a short numeric span) — accept
        # rather than reject a claim we simply can't lexically check.
        return True
    return bool(span_words & claim_words)


def _fallback_span(answer: str, claim_text: str) -> tuple[int, int] | None:
    """Best-effort recovery when the model's own offsets fail validation.

    Looks for the longest significant word from the claim text literally
    present in the answer and anchors on that. Returns None if nothing can
    be found, signalling the caller to drop the claim entirely.
    """
    for word in sorted(_significant_words(claim_text), key=len, reverse=True):
        idx = answer.lower().find(word)
        if idx != -1:
            return idx, idx + len(word)
    return None


async def decompose(answer: str, query: str, *, run_id: UUID) -> list[Claim]:
    """Decompose `answer` into atomic, self-contained `Claim` objects.

    Returns claims with contiguous `claim_index` starting at 0, in emission
    order (which may skip indices from the model's raw output if a claim's
    span could not be anchored). Never raises: on LLM failure or unparsable
    output this logs a warning and returns an empty list rather than
    propagating, consistent with "one dead component must not kill a run".
    """
    prompt = _PROMPT_TEMPLATE.format(query=query, answer=answer)

    try:
        result = await complete(prompt, model_tier="fast", schema=_DecomposeResponse)
    except LLMError as exc:
        logger.warning("decompose_llm_failed", run_id=str(run_id), error=str(exc))
        return []

    parsed = result.parsed
    if not isinstance(parsed, _DecomposeResponse):
        logger.warning(
            "decompose_parse_failed", run_id=str(run_id), text=result.text[:200]
        )
        return []

    claims: list[Claim] = []
    for raw in parsed.claims:
        start, end = raw.span_start, raw.span_end
        if not _span_is_valid(answer, start, end, raw.text):
            fallback = _fallback_span(answer, raw.text)
            if fallback is None or not _span_is_valid(answer, *fallback, raw.text):
                logger.warning(
                    "decompose_unanchored_claim_skipped",
                    run_id=str(run_id),
                    claim_text=raw.text,
                )
                continue
            start, end = fallback

        verdict = Verdict.UNVERIFIABLE if raw.is_unverifiable else None
        rationale = raw.reason if raw.is_unverifiable else None

        claims.append(
            Claim(
                id=uuid4(),
                run_id=run_id,
                claim_index=len(claims),
                text=raw.text,
                span_start=start,
                span_end=end,
                verdict=verdict,
                rationale=rationale,
            )
        )

    return claims


__all__ = ["decompose"]
