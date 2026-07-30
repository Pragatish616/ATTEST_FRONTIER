# DECISIONS

Technical decisions and the reasoning behind them, so they aren't relitigated
at 3am. Newest last within each section.

Changes to a **frozen** contract (PLAN.md §5) are not logged here — they go in
`CONTRACT_CHANGE_REQUEST.md`, which is the authoritative record for those.

---

## Architecture

### Verifiers are independent by construction, not by convention

No verifier may see another verifier's output. This is the entire premise: three
correlated judgments are one judgment with extra cost. Enforced structurally —
each verifier receives only the claim and a `VerifyContext`, and the orchestrator
fans out with `asyncio.gather(..., return_exceptions=True)` so one dead verifier
degrades the run instead of killing it.

**The one exception** is `VerifyContext.prior_entailment`, read only by
`IndependentVerifier`. `STALE` is *defined* as "entailed by context, but
independent retrieval disagrees with the context" — the independent verifier
cannot distinguish `STALE` from plain `CONTRADICTED` without knowing whether
entailment already found the claim grounded. The alternative (inferring `STALE`
in the reconciler from verdict combinations) hides the actual signal behind a
heuristic. Full argument and scope limits in `CONTRACT_CHANGE_REQUEST.md`.

### The prober takes the entailment verifier as a dependency

`AdversarialProber(entailment_verifier: VerifierProtocol)` rather than importing
`EntailmentVerifier` directly. The prober's job is "re-run *the same* check on a
mutated claim" — if it built its own checker, a probe failing to flip would be
ambiguous between "the verifier wasn't reading" and "the two checkers just
disagree," which destroys the meaning of `FRAGILE`. Dependency injection also
makes the prober testable in isolation with a fake verifier.

The prober re-derives its own baseline by calling the injected verifier on the
unmutated claim rather than reading `prior_entailment`. Slightly more cost, but
it keeps the prober self-contained.

### Probes surface through `last_probes`, not the return value

`VerifierProtocol.verify()` returns `Verification` and that return type is
frozen, so there is no way to hand `Probe` rows back through it. The prober
exposes them on a `last_probes` attribute, which the orchestrator reads
immediately after `verify()` returns.

**This is only safe because `orchestrator.run` processes claims in a strictly
sequential `for` loop.** If claim processing ever becomes concurrent, this hook
must move to `AdversarialProber.verify_with_probes()` — already implemented,
currently unused. Do not parallelize the claim loop without doing that first.

### Evidence is character spans, not quoted text

Verifiers return offsets into the provided chunk. Spans are checkable against the
source; quotes hallucinate, and a hallucinated quote in a *grounding checker* is
the single most embarrassing failure this project could ship.

---

## Models and cost

### Provider router with a fallback chain, not a single provider

Anthropic → Groq → Gemini, tried in order, skipping any provider whose key isn't
configured. A single-provider design has a single point of failure on demo day,
which is the one day it cannot fail. The side effect is useful: setting only
`GEMINI_API_KEY` runs the entire system on a free tier.

### `temperature=0` is hardcoded with no override

Non-determinism in a judge is a bug, not a tuning knob. There is no code path
that sets a different temperature for a verification call.

### JSON parsing has a free local repair before a paid retry

On a parse failure: strip fences and isolate the outermost `{}`/`[]` first (free),
and only then spend one more call on a "return valid JSON only" nudge to the
*same* provider. `result.parsed is None` is an expected outcome to be handled,
not a crash.

### Every LLM call is logged with model, tokens, latency, and cost

Cost visibility is a demo feature. A judge asking "what does this cost per run"
gets a real number off the trace, not an estimate.

---

## API and deployment

### Raw ASGI middleware, never `BaseHTTPMiddleware`

Starlette's `BaseHTTPMiddleware` (and the `@app.middleware("http")` sugar built
on it) wraps responses in a way that buffers and can stall `EventSourceResponse`.
Live SSE streaming is the entire demo. All hardening middleware in
`attest/api/security.py` inspects the scope, optionally short-circuits, and
otherwise passes the untouched send/receive channels straight through.

### Auth is off unless `ATTEST_API_KEY` is set

The SDK has always sent `Authorization: Bearer <api_key>`; the server simply
never read it. `security.py` completes that half-implemented contract rather
than adding a new one. Defaulting to off means the dashboard track and the Opal
agent keep working against frozen §5.2 exactly as before. **Turning it on is a
deployment decision that requires telling both other tracks first.**

### `MAX_BUDGET_USD` caps the caller-supplied budget server-side

`AttestConfig.budget_usd` arrives in the request body with only a `> 0`
validator, and `AttestConfig` is part of the frozen SDK surface so its shape
can't change. The ceiling is therefore enforced in `api/routes.py::observe`, not
in the model. Without it, an unauthenticated caller sets `budget_usd=1e9` and
drains the team's provider credits. **Credit burn, not data theft, is the
primary threat to this service.**

### `CORS_ALLOW_ORIGINS` is a comma-separated string, not `list[str]`

pydantic-settings requires JSON syntax for complex types in env vars.
`CORS_ALLOW_ORIGINS=https://a.vercel.app,https://b.vercel.app` is what a deploy
dashboard's env editor makes easy to get right at 3am; `["https://a...", ...]`
is not. Default stays `*` so no track is ever blocked by CORS locally.

### The API must run single-process

The SSE broker keeps per-run subscribers in process memory. Multiple uvicorn
workers silently drop events for runs they don't own. Documented in
`create_app()`, the README, and the deploy configs.

### Search queries are hashed in logs, never logged verbatim

The independent verifier builds its query from claim text, and claim text comes
from the host application's answer — i.e. in a real deployment, its users'
content. Logging it verbatim copies that content into our log stream where it
outlives the run. A 12-char SHA-256 digest still correlates cache misses and
provider failures across log lines, which is the only thing the field was for.

---

## Evaluation

### `FRAGILE` counts as a positive in the benchmark mapping

Contestable, and deliberately argued in the open rather than buried: see
`bench/MAPPING.md`, which also documents the rejected alternative. Expect a judge
to push on this — the honest answer is that `FRAGILE` asserts "this verdict is
untrustworthy," and on a hallucination-detection benchmark an untrustworthy
"supported" is a miss, so counting it as a positive is the reading consistent
with what the label means.

### `bench/results.md` says `PENDING`, not a number

No LLM-backed benchmark run has been executed. Every metric cell literally reads
`PENDING`/`null`. CLAUDE.md's "no mock data in the main path" extends to "no
fabricated results," and a panel that checks one number and finds it invented
discards everything else the project claims. Numbers land only after
`uv run python -m bench.run_benchmark` has actually been run with a real key.

---

## Repository

### Line endings are normalized to LF via `.gitattributes`

The working tree was checked out on Windows with CRLF, which made every tracked
file show as modified — 16,602 insertions / 15,391 deletions for ~1,465 lines of
real change. That buries actual diffs and makes a targeted revert impossible. LF
in the repo is also what the Linux container and every deploy host expect.

### Presentation material is gitignored

`SPEECH.md`, `PRESENTATION_NOTES.md`, and the slide images stay local. The public
repo is the engineering artifact.

### Republished as `ATTEST_FRONTIER`; the old repo was deleted (2026-07-30)

The project was originally pushed to `Pragatish616/attest` across six commits.
It now lives in `Pragatish616/ATTEST_FRONTIER`, seeded with a single initial
commit, and the old repository has been deleted.

**Rationale:** one canonical repo under the submission name, with no stale
duplicate for judges or teammates to land on by mistake, and one reviewable
initial state — no half-reset index, no stray files, no question about what was
ever committed.

**Cost, recorded honestly:** the six-commit build narrative and the
parallel-agent build sequence are no longer visible in `git log` anywhere
public. They survive in `PROGRESS.md`, `HANDOFF.md`,
`CONTRACT_CHANGE_REQUEST.md`, and a local `--mirror` clone taken before the
deletion. Anyone holding a clone of the old repo has an `origin` pointing at a
repository that no longer exists and must re-point it.

**This was cosmetic, not containment.** Every blob in all six original commits
was scanned and no credential was ever committed. If one had been, deleting the
repo would still not have been the fix — the value has to be rotated, because
a leaked key is compromised from the moment it is pushed, not from the moment
someone finds it.
