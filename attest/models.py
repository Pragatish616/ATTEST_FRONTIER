"""Shared Pydantic models for ATTEST.

This module is the single source of truth for every data shape that crosses a
component boundary: the Supabase schema (PLAN.md §5.1), the REST/SSE API
(§5.2), and the verifier interface all verifiers implement. It is a FROZEN
contract — see CLAUDE.md. Do not add fields or change the verdict taxonomy
without writing a CONTRACT_CHANGE_REQUEST.md first.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal, Protocol, runtime_checkable
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Verdict taxonomy (PLAN.md §3) — FROZEN. Exactly these six values, no more.
# ---------------------------------------------------------------------------


class Verdict(StrEnum):
    GROUNDED = "GROUNDED"
    FRAGILE = "FRAGILE"
    UNSUPPORTED = "UNSUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    STALE = "STALE"
    UNVERIFIABLE = "UNVERIFIABLE"


class MutationType(StrEnum):
    """Adversarial probe mutation kinds (PLAN.md §5.1 probes.mutation_type)."""

    NEGATION = "negation"
    ENTITY_SWAP = "entity_swap"
    QUANTIFIER_SHIFT = "quantifier_shift"


RunStatus = Literal["pending", "running", "complete", "error", "skipped"]
VerifierName = Literal["entailment", "independent", "prober"]
Stance = Literal["support", "refute"]


# ---------------------------------------------------------------------------
# Chunks (PLAN.md §5.1 retrieved_chunks)
# ---------------------------------------------------------------------------


class RetrievedChunk(BaseModel):
    """A chunk as supplied by the caller in ObserveRequest — no id/run_id yet."""

    model_config = ConfigDict(frozen=True)

    chunk_index: int
    source_id: str | None = None
    source_url: str | None = None
    text: str
    score: float | None = None


class Chunk(RetrievedChunk):
    """A chunk as persisted — matches the retrieved_chunks table row."""

    id: UUID
    run_id: UUID


# ---------------------------------------------------------------------------
# Claims (PLAN.md §5.1 claims)
# ---------------------------------------------------------------------------


class Claim(BaseModel):
    """Matches the `claims` table row.

    Decomposer produces one of these per atomic claim with `verdict=None`
    (pending). Verifiers receive it in that pending state — they must never
    see a verdict, since that would leak information between the independent
    verification passes. The reconciler is the only component that fills in
    verdict/confidence/disagreement/rationale before store.py persists the
    row (where the DB schema requires verdict NOT NULL).
    """

    id: UUID
    run_id: UUID
    claim_index: int
    text: str
    span_start: int | None = None
    span_end: int | None = None
    verdict: Verdict | None = None
    confidence: float | None = None
    disagreement: float | None = None
    rationale: str | None = None


# ---------------------------------------------------------------------------
# Verifications (PLAN.md §5.1 verifications)
# ---------------------------------------------------------------------------


class Evidence(BaseModel):
    """One entry in verifications.evidence jsonb.

    Evidence spans are char offsets into the *provided chunk*, never quoted
    text — quotes hallucinate, spans are checkable (CLAUDE.md prompt rules).
    Exactly one of chunk_id/url should be set depending on whether the
    evidence came from a retrieved chunk or the independent web search.
    """

    chunk_id: UUID | None = None
    url: str | None = None
    quote_span: tuple[int, int] | None = None
    stance: Stance


class Verification(BaseModel):
    """Matches the `verifications` table row."""

    id: UUID
    claim_id: UUID
    verifier: VerifierName
    verdict: Verdict
    rationale: str | None = None
    evidence: list[Evidence] = Field(default_factory=list)
    latency_ms: int | None = None
    cost_usd: float | None = None


# ---------------------------------------------------------------------------
# Probes (PLAN.md §5.1 probes)
# ---------------------------------------------------------------------------


class Probe(BaseModel):
    """Matches the `probes` table row."""

    id: UUID
    claim_id: UUID
    mutation_type: MutationType
    mutated_text: str
    expected_flip: bool
    observed_verdict: Verdict
    flipped: bool


# ---------------------------------------------------------------------------
# Nested / detail views for the API (PLAN.md §5.2)
# ---------------------------------------------------------------------------


class ClaimDetail(Claim):
    """A claim plus its full independent-verifier and probe trail."""

    verifications: list[Verification] = Field(default_factory=list)
    probes: list[Probe] = Field(default_factory=list)


class RunSummary(BaseModel):
    """Matches the `runs` table row. Returned by GET /runs."""

    id: UUID
    created_at: datetime
    pipeline_name: str
    query: str
    answer: str
    model: str | None = None
    status: RunStatus = "pending"
    grounding_score: float | None = None
    fragility_score: float | None = None
    total_claims: int | None = None
    latency_ms: int | None = None
    cost_usd: float | None = None


class RunDetail(RunSummary):
    """Full nested trace. Returned by GET /runs/{run_id}."""

    retrieved_chunks: list[Chunk] = Field(default_factory=list)
    claims: list[ClaimDetail] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# API request/response contracts (PLAN.md §5.2)
# ---------------------------------------------------------------------------


class AttestConfig(BaseModel):
    """Per-run config, nested in ObserveRequest.config.

    Distinct from the SDK's global `attest.init(...)` call (PLAN.md §5.3),
    which configures the client, not a single run.
    """

    sample_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    enable_independent: bool = True
    enable_prober: bool = True
    budget_usd: float = Field(default=0.05, gt=0.0)


class ObserveRequest(BaseModel):
    """POST /observe body."""

    pipeline_name: str
    query: str
    answer: str
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)
    model: str | None = None
    config: AttestConfig = Field(default_factory=AttestConfig)


class ObserveResponse(BaseModel):
    """POST /observe returns this with HTTP 202 — the run is processed async."""

    run_id: UUID
    status: RunStatus = "pending"


# ---------------------------------------------------------------------------
# Verifier interface — every verifier in attest/verifiers/ implements this.
# ---------------------------------------------------------------------------


class VerifyContext(BaseModel):
    """Everything a verifier is allowed to see.

    Deliberately excludes other verifiers' outputs and any reconciled
    verdict — independence is the whole design (CLAUDE.md prompt-engineering
    rules). A verifier only ever sees the claim, the retrieved context, and
    run-level config.

    `prior_entailment` is the one deliberate exception, and it is narrow on
    purpose: the STALE verdict is defined (PLAN.md §3) as "entailed by
    context, but independent retrieval disagrees with the context" — the
    independent verifier cannot tell STALE apart from plain CONTRADICTED
    without knowing whether entailment considered the claim grounded. The
    orchestrator runs entailment first and sets this field (via
    `ctx.model_copy(update={"prior_entailment": ...})`) only on the context
    it passes to the independent verifier. It defaults to None — meaning
    "unknown" / "run standalone" — and no other verifier should read it: the
    prober re-derives its own baseline by calling the entailment verifier
    directly on the unmutated claim, not through this field.
    """

    run_id: UUID
    query: str
    answer: str
    retrieved_chunks: list[Chunk] = Field(default_factory=list)
    prior_entailment: Verdict | None = None
    config: AttestConfig = Field(default_factory=AttestConfig)


@runtime_checkable
class VerifierProtocol(Protocol):
    """Structural interface implemented by entailment.py, prober.py, independent.py."""

    async def verify(self, claim: Claim, ctx: VerifyContext) -> Verification: ...


__all__ = [
    "Verdict",
    "MutationType",
    "RunStatus",
    "VerifierName",
    "Stance",
    "RetrievedChunk",
    "Chunk",
    "Claim",
    "Evidence",
    "Verification",
    "Probe",
    "ClaimDetail",
    "RunSummary",
    "RunDetail",
    "AttestConfig",
    "ObserveRequest",
    "ObserveResponse",
    "VerifyContext",
    "VerifierProtocol",
]
