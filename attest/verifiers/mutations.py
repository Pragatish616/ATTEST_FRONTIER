# Owned by A2 (Prober). Not implemented by A0.
#
# Mutation engine: negation | entity_swap | quantifier_shift (see
# attest.models.MutationType) plus expected-flip logic.
"""Adversarial mutation engine for the prober (attest/verifiers/prober.py).

Generates claim mutations whose *correct* verdict is known in advance
(`expected_flip=True` for all three kinds here — negating a claim, swapping
its key entity for an absent decoy, or shifting its quantifier/magnitude
should always change the true verdict). The prober re-verifies each mutation
against the same context and treats a verifier that fails to flip as
evidence it wasn't actually reading the context (CLAUDE.md: "A verifier that
says GROUNDED for both a claim and its negation wasn't reading the
context.").

`MutatedClaim` is defined here, not in `attest.models` — it's an internal
vehicle for the prober's own logic, not a frozen API/DB contract.

Applicability is decided structurally where possible (regex over the claim
text) rather than trusting an LLM's self-report, per CLAUDE.md: "never
fabricate applicability just to hit 3 probes." `quantifier_shift` is skipped
outright, with no LLM call at all, when the claim contains no number, date,
magnitude, or quantifier word. `entity_swap` is skipped outright when no
named-entity-shaped candidate is found in the claim text. `negation` has no
comparable structural precondition (any well-formed assertion can be
negated), so its applicability is left to the generation call itself, which
can still report `applicable: false` (e.g. a claim that's just a fragment)
without inventing a mutation.

Generation itself is a single `attest.llm.complete(model_tier="fast", ...)`
call per mutation kind, so a full probe pass costs at most 3 cheap calls
(plus whatever the injected entailment verifier costs to re-verify each
one) — kept on the fast tier deliberately, per CLAUDE.md's cost-control
guidance for this file.
"""

from __future__ import annotations

import asyncio
import re

import structlog
from pydantic import BaseModel

from attest.llm import LLMError, complete
from attest.models import MutationType

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Result type (internal to the prober — not a frozen contract)
# ---------------------------------------------------------------------------


class MutatedClaim(BaseModel):
    """One adversarial mutation of a claim's text.

    `expected_flip` is always True for the three mutation kinds implemented
    here: each is constructed so the correct verdict necessarily changes
    relative to the original claim. A verifier's failure to flip is what the
    prober treats as fragility, not this field itself.
    """

    text: str
    mutation_type: MutationType
    expected_flip: bool


# ---------------------------------------------------------------------------
# Structural applicability checks
# ---------------------------------------------------------------------------

_QUANTIFIER_RE = re.compile(
    r"\d+(?:\.\d+)?%?"
    r"|\b(?:all|some|many|few|several|most|none|no|every|each|both|any|"
    r"majority|minority|approximately|about|roughly|"
    r"increase[ds]?|decrease[ds]?|rose|rise[s]?|fell|fall[s]?|grew|grow[s]?|"
    r"shrank|shrink[s]?|more|less|fewer|greater|higher|lower|"
    r"always|never|often|rarely|sometimes|usually|"
    r"at\s+least|at\s+most)\b",
    re.IGNORECASE,
)

_ENTITY_CANDIDATE_RE = re.compile(
    r"\b\d{4}\b"  # 4-digit years
    r"|\b\d+(?:\.\d+)?%?\b"  # numbers / percentages
    r"|(?:[A-Z][\w.&-]*(?:\s+[A-Z][\w.&-]*)*)"  # capitalized word run
)

_ENTITY_STOPWORDS = {
    "the", "this", "that", "these", "those", "it", "there", "a", "an",
    "he", "she", "they", "we", "i", "you", "his", "her", "its", "their",
}


def _has_quantifier(text: str) -> bool:
    return bool(_QUANTIFIER_RE.search(text))


def _entity_candidates(text: str) -> list[str]:
    """Cheap heuristic named-entity-ish candidates: proper-noun word runs,
    years, and standalone numbers. Not real NER — good enough to decide
    whether a swap is *possible*, not to guarantee correctness.
    """
    seen: set[str] = set()
    candidates: list[str] = []
    for match in _ENTITY_CANDIDATE_RE.finditer(text):
        candidate = match.group(0).strip()
        if not candidate or candidate.lower() in _ENTITY_STOPWORDS:
            continue
        if candidate not in seen:
            seen.add(candidate)
            candidates.append(candidate)
    return candidates


# ---------------------------------------------------------------------------
# LLM-backed generation
# ---------------------------------------------------------------------------


class _MutationLLMResponse(BaseModel):
    applicable: bool
    mutated_text: str | None = None
    reason: str | None = None


_NEGATION_PROMPT = """You are generating an adversarial test case for a fact-verification system.

Given the claim below, produce a version that logically negates its core \
assertion, so the mutated claim asserts the opposite of what the original \
claim asserts. Change only what is needed to flip the truth value; keep \
everything else about the sentence the same.

Claim: {claim}

Respond with JSON only, matching exactly this shape:
{{"applicable": true, "mutated_text": "<negated claim>", "reason": null}}

If the claim has no clear assertion to negate (for example, it is a \
fragment with no verb), respond with:
{{"applicable": false, "mutated_text": null, "reason": "<why>"}}
"""

_ENTITY_SWAP_PROMPT = """You are generating an adversarial test case for a fact-verification system.

Given the claim and the retrieved context below, identify the single most \
identifiable named entity in the claim (a person, place, organization, \
date, or specific number) and replace it with a plausible decoy of the same \
type. The decoy must be clearly different from the original entity and must \
NOT appear anywhere in the retrieved context — a decoy that already appears \
in the context doesn't test anything.

Claim: {claim}

Candidate entities found in the claim: {candidates}

Retrieved context:
{context}

Respond with JSON only, matching exactly this shape:
{{"applicable": true, "mutated_text": "<claim with entity swapped for an absent decoy>", "reason": null}}

If none of the candidate entities can be swapped for a decoy that is absent \
from the context, respond with:
{{"applicable": false, "mutated_text": null, "reason": "<why>"}}
"""

_QUANTIFIER_SHIFT_PROMPT = """You are generating an adversarial test case for a fact-verification system.

Given the claim below, alter its quantifier or numeric magnitude — a \
number, date, percentage, or quantifier word (for example "some" -> "all", \
"increased" -> "decreased", 40% -> 60%) — so the mutated claim asserts a \
different magnitude or direction than the original.

Claim: {claim}

Respond with JSON only, matching exactly this shape:
{{"applicable": true, "mutated_text": "<claim with quantifier or number shifted>", "reason": null}}

If the claim contains no number, date, magnitude, or quantifier word to \
shift, respond with:
{{"applicable": false, "mutated_text": null, "reason": "<why>"}}
"""


async def _generate(
    prompt: str, mutation_type: MutationType, claim_text: str
) -> MutatedClaim | None:
    try:
        result = await complete(prompt, model_tier="fast", schema=_MutationLLMResponse)
    except LLMError as exc:
        logger.warning(
            "mutation_generation_failed",
            mutation_type=mutation_type.value,
            claim=claim_text,
            error=str(exc),
        )
        return None

    parsed = result.parsed
    if not isinstance(parsed, _MutationLLMResponse):
        logger.info(
            "mutation_skipped",
            mutation_type=mutation_type.value,
            claim=claim_text,
            reason="llm_response_did_not_parse",
        )
        return None
    if not parsed.applicable or not parsed.mutated_text:
        logger.info(
            "mutation_skipped",
            mutation_type=mutation_type.value,
            claim=claim_text,
            reason=parsed.reason or "model_reported_not_applicable",
        )
        return None
    if parsed.mutated_text.strip() == claim_text.strip():
        logger.info(
            "mutation_skipped",
            mutation_type=mutation_type.value,
            claim=claim_text,
            reason="mutated_text_identical_to_original",
        )
        return None

    return MutatedClaim(
        text=parsed.mutated_text.strip(),
        mutation_type=mutation_type,
        expected_flip=True,
    )


async def build_negation(claim_text: str) -> MutatedClaim | None:
    """Negate the claim's core assertion. `expected_flip=True`."""
    prompt = _NEGATION_PROMPT.format(claim=claim_text)
    return await _generate(prompt, MutationType.NEGATION, claim_text)


async def build_entity_swap(claim_text: str, context_text: str) -> MutatedClaim | None:
    """Swap the claim's key entity for a same-type decoy absent from context.

    `expected_flip=True`. Skipped (returns None) when no named-entity-shaped
    candidate is found in the claim text at all — no LLM call is made in
    that case.
    """
    candidates = _entity_candidates(claim_text)
    if not candidates:
        logger.info(
            "mutation_skipped",
            mutation_type=MutationType.ENTITY_SWAP.value,
            claim=claim_text,
            reason="no_entity_candidate_found",
        )
        return None
    prompt = _ENTITY_SWAP_PROMPT.format(
        claim=claim_text,
        candidates=", ".join(candidates),
        context=context_text[:4000] if context_text else "(no retrieved context)",
    )
    return await _generate(prompt, MutationType.ENTITY_SWAP, claim_text)


async def build_quantifier_shift(claim_text: str) -> MutatedClaim | None:
    """Shift a number, date, magnitude, or quantifier word in the claim.

    `expected_flip=True`. Skipped (returns None) when the claim contains no
    quantifier/number/date at all — no LLM call is made in that case; this
    is the "no applicable mutation" path CLAUDE.md calls out explicitly.
    """
    if not _has_quantifier(claim_text):
        logger.info(
            "mutation_skipped",
            mutation_type=MutationType.QUANTIFIER_SHIFT.value,
            claim=claim_text,
            reason="no_quantifier_or_number_found",
        )
        return None
    prompt = _QUANTIFIER_SHIFT_PROMPT.format(claim=claim_text)
    return await _generate(prompt, MutationType.QUANTIFIER_SHIFT, claim_text)


async def generate_mutations(claim_text: str, context_text: str) -> list[MutatedClaim]:
    """Generate all applicable mutations concurrently; drop inapplicable ones.

    Returns at most 3 entries (one per `MutationType`); fewer when a
    mutation kind is skipped as inapplicable.
    """
    results = await asyncio.gather(
        build_negation(claim_text),
        build_entity_swap(claim_text, context_text),
        build_quantifier_shift(claim_text),
    )
    return [mutation for mutation in results if mutation is not None]


__all__ = [
    "MutatedClaim",
    "build_negation",
    "build_entity_swap",
    "build_quantifier_shift",
    "generate_mutations",
]
