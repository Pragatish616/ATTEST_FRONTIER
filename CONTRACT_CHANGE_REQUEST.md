# Contract change log

Frozen contracts (PLAN.md §5) may not change silently. Every change is
logged here with rationale before (or as) it's applied.

## 2026-07-27 — `VerifyContext.prior_entailment` (additive)

**What:** added `prior_entailment: Verdict | None = None` to
`attest.models.VerifyContext`.

**Why:** PLAN.md §3 defines `STALE` as "entailed by context, but independent
retrieval disagrees with the context." The independent verifier (A3) cannot
distinguish STALE from plain CONTRADICTED without knowing whether entailment
already found the claim grounded — that's not a design nicety, it's load-
bearing in the verdict's own definition. Without this field, A3 would have
to either duplicate entailment's judgment or the reconciler would have to
infer STALE indirectly from verdict combinations, which is fragile and hides
the actual signal.

**Scope of the exception:** this is the *one* place a verifier is allowed
to see another verifier's output, and it's narrow:
- Only the orchestrator sets it, only on the `VerifyContext` copy it passes
  to `IndependentVerifier.verify(...)` — never to the entailment verifier
  itself, and never to the prober.
- The prober does NOT use this field. It re-derives its own baseline by
  calling `EntailmentVerifier.verify(...)` directly on the unmutated claim
  (see A2's brief) — that keeps the prober self-contained and testable in
  isolation, and is consistent with "import the verifier via the protocol,
  don't duplicate its logic."
- Default is `None`, so every existing test/fixture that constructs a
  `VerifyContext` without it is unaffected (round-trip tests still pass).

**Backward compatibility:** additive, optional, default `None`. No existing
field renamed or removed. `tests/test_contracts.py` re-verified green after
the change.

## 2026-07-27 — `RunStatus` gains `"skipped"` (additive)

**What:** `RunStatus` (used by `RunSummary.status`, `RunDetail.status`,
`ObserveResponse.status`) is now `Literal["pending", "running", "complete",
"error", "skipped"]` — was missing `"skipped"`.

**Why:** A4's orchestrator brief requires honoring `config.sample_rate`: a
run that gets sampled out must still be persisted (so the dashboard can show
sampling coverage), and "skipped" is the only status that says that
truthfully — reusing "error" would corrupt any dashboard filtering/alerting
on real failures.

**Backward compatibility:** additive only; the SQL schema's `status` column
has no CHECK constraint (just a text column with a documented default), so
`migrations/001_init.sql` needs no change. Existing code checking for the
original four values is unaffected — this only widens what's accepted.

## Notes on informal field names in agent briefs

The A1-A5 task briefs use shorthand that doesn't match the frozen model
field names literally. These are NOT contract changes — just translations,
listed here so nobody "fixes" the mismatch by renaming the real fields:

| Brief says | Actual field |
|---|---|
| `ctx.chunks` | `ctx.retrieved_chunks` (`list[Chunk]`) |
| `ctx.budget` | `ctx.config.budget_usd` |
| `ctx.prior_entailment` | `ctx.prior_entailment` (added above — this one's literal) |

---

## Deferred: provider fallback on persistent JSON parse failure (`attest/llm.py`)

**Raised by:** post-A6 review (see `attest_fix_prompts.md` item 4).

**Observation.** `complete()`'s docstring claimed that a response which still
fails schema parsing after the one repair call causes fallback to the next
provider in `_PROVIDER_CHAIN`. The code never did this — the chain advances
only when a provider call *raises*.

**Resolved (2026-07-29):** documentation corrected to match the code, plus a
new `llm_schema_parse_failed_after_repair` warning so the degraded path is
visible in logs. No behavioural change.

**Rejected alternative (Option A).** Make the code match the old docs, i.e.
`continue` to the next provider when `parsed is None` after repair. Rejected
for now because:
- a provider that responds but can't emit valid JSON is unlikely to be fixed
  by a different model, so the extra call mostly buys latency and cost;
- re-asking a *different* provider for the same claim risks answer-content
  drift between providers, which is a correctness problem for a judge;
- it adds an unbounded-ish latency tail on the live demo path.

**Revisit if** benchmark runs show a non-trivial rate of
`llm_schema_parse_failed_after_repair` for any provider. That log line is the
trigger to reopen this.

---

## Additive: `POST /v1/demo/query` (demo-only convenience route)

**Raised by:** demo/dashboard work, 2026-07-29. **Status:** applied.

**What.** A new route that runs the bundled Northwind demo RAG over a free-text
question, returns the generated answer **synchronously**, and submits the result
to the normal orchestrator path so verdicts arrive asynchronously exactly as
they do for `POST /observe`.

```
POST /v1/demo/query   {"query": "...", "k": 4}   -> 202
{"run_id": "...", "query": "...", "answer": "...", "retrieved_chunks": [...]}
```

**Why.** ATTEST verifies; it does not generate. A dashboard that accepts a typed
question therefore needs *something* to run a RAG pipeline first. Without this
the dashboard can only replay stored runs, which reads as hard-coded to a judge.

**Impact on other tracks: none.** This is strictly additive — no existing
request/response shape, event name, SSE envelope, SDK signature, table or
verdict changed. Anything built against `/observe`, `/runs`, `/runs/{id}` or the
stream is unaffected. **Jyotish (frontend track) should know the route exists**
so the two dashboards don't diverge.

**Trade-offs accepted.**
- The `attest` package gains a *lazy, optional* dependency on `demo/`. Imported
  inside the handler; a missing demo package is a clean `503`, never an
  ImportError at app import time. The package still does not hard-depend on the
  demo folder.
- Generation happens inside the request, so the route is as slow as one LLM call
  (~4–13s) and returns `502` if every provider fails. Verification stays async.
- **This router must be dropped before any production deployment.** It exposes
  an unauthenticated LLM-spending endpoint, which is fine for a local demo and
  not fine anywhere else.

**Rejected alternative.** Have the browser call the LLM directly and POST to
`/observe`. Rejected: it would put provider API keys in client-side JavaScript.
