# ATTEST

**A runtime adversarial grounding layer for RAG and agentic systems.**

Built 27–30 July 2026 by team **Byte_pros** for **FRONTIER 2026** — AWS Student Builder
Groups, VIT Chennai — in the AI Safety & Observability track.

---

## The problem

A RAG system that cites a source is not the same as a RAG system that is *correct*. Two
failures survive every grounding checker we could find:

1. **The checker isn't reading.** Ask a verifier whether a claim is supported by a chunk and
   it says yes. Ask it whether the *negation* of that claim is supported by the same chunk
   and it also says yes. The verdict was never a function of the evidence — but you get a
   confident "grounded" either way.
2. **The source itself is stale.** The answer is faithfully grounded in the retrieved chunk.
   The chunk was written eighteen months ago and is now wrong. Every faithfulness metric
   scores this as a pass, because faithfulness is measured *against the chunk*.

ATTEST is built to catch exactly these two.

## What it does

It decomposes an AI system's answer into atomic claims and verifies each one **three
independent ways**, then reconciles the verdicts and streams the whole trace to a dashboard
live.

| Verifier | Question it asks | Catches |
|---|---|---|
| **Entailment** | Does the retrieved chunk entail this claim? | Ordinary hallucination |
| **Adversarial prober** | Does the verdict *survive mutation* of the claim? | **FRAGILE** |
| **Independent retrieval** | Does the open web still agree? | **STALE** |

No verifier ever sees another's output. That independence is the whole design — a majority
vote between three correlated judges tells you nothing.

### Verdict taxonomy

```
GROUNDED       supported by the retrieved context
FRAGILE        the verdict did not flip when logic says it must — the checker wasn't reading
UNSUPPORTED    the context neither supports nor contradicts it
CONTRADICTED   the context says the opposite
STALE          grounded in the chunk, but the chunk is outdated
UNVERIFIABLE   subjective, predictive, or otherwise not checkable
```

`FRAGILE` and `STALE` are the two verdicts we could not find in any competing tool.
Everything else is table stakes.

### How the prober works

For a claim the entailment verifier called `GROUNDED`, the prober generates mutations —
`negation`, `entity_swap`, `quantifier_shift` — and re-verifies each one through the *same*
verifier. A negated claim that is still `GROUNDED` against the same chunk is a verifier that
pattern-matched instead of reading. Every mutation, its expected flip, and its observed
verdict is persisted to the `probes` table and rendered per-claim in the dashboard, so the
judgement is auditable rather than asserted.

### Reconciliation

Disagreement between verifiers is a signal to surface, not to average away:

```
CONTRADICTED > STALE > UNSUPPORTED > FRAGILE > GROUNDED > UNVERIFIABLE
```

The highest-precedence verdict wins regardless of how many verifiers dissent — one
`CONTRADICTED` outranks two `GROUNDED`s, deliberately: precision over consensus. A separate
**disagreement score** is reported alongside, never folded into confidence.

## Architecture

```
       observed pipeline (any RAG/agent app)
                  │  attest SDK — @observe / wrap() / context manager
                  ▼
        POST /v1/observe ──► 202 Accepted, run_id
                  │
                  ▼
            decomposer  (answer ──► atomic claims)
                  │
        ┌─────────┼─────────┐        fan-out, asyncio.gather
        ▼         ▼         ▼        one dead verifier cannot kill a run
   entailment  prober  independent
        └─────────┼─────────┘
                  ▼
             reconciler  (verdict + confidence + disagreement)
                  │
        ┌─────────┴─────────┐
        ▼                   ▼
    Supabase           SSE event bus
   (trace store)   GET /v1/runs/{id}/stream ──► dashboard
```

The SDK **never raises into the host pipeline**. Every path is wrapped; on failure it logs
and returns control. A grounding checker that crashes the app it observes is worse than
useless.

## Quickstart

Requires Python 3.11 (`StrEnum`, `datetime.UTC`, and `TimeoutError`/`asyncio.TimeoutError`
aliasing) and [uv](https://docs.astral.sh/uv/).

```bash
cp .env.example .env          # fill in SUPABASE_* and at least one LLM provider key
uv sync --all-groups
psql "$SUPABASE_DB_URL" -f migrations/001_init.sql
psql "$SUPABASE_DB_URL" -f migrations/002_rls.sql

uv run python demo/build_corpus.py     # materialise the demo corpus
uv run uvicorn attest.api.main:app --reload
```

Then open `dashboard/attest-dashboard.html` and point it at `http://localhost:8000`.

```bash
uv run pytest        # 271 tests
uv run ruff check .
```

## API

All routes are prefixed `/v1`.

| Method | Path | Notes |
|---|---|---|
| `POST` | `/observe` | Submit a run. `202` + `run_id`; verification is a background task |
| `GET` | `/runs` | Recent runs |
| `GET` | `/runs/{run_id}` | Full trace: claims, verifications, probes |
| `GET` | `/runs/{run_id}/stream` | SSE — live trace, 15 s heartbeat, replays to late subscribers |
| `POST` | `/demo/query` | Convenience: run the reference RAG pipeline *and* verify it |
| `POST` | `/evaluate` | Benchmark / ablation sweep |
| `GET` | `/health` | Liveness. Unauthenticated by design (platform healthcheck) |

## Deployment

One container image, single replica.

```bash
docker build -t attest .
docker run -p 8000:8000 --env-file .env attest
```

`railway.json` and `render.yaml` are both checked in. **The service must run as exactly one
replica with one worker.** The SSE event bus is in-process memory, so a second worker or
replica does not error — it silently splits the bus, and a dashboard connected to replica B
never sees events published on replica A. Scaling out requires replacing the bus with
Postgres `LISTEN/NOTIFY` or Redis behind the same interface.

Set `TRUSTED_PROXY_HOPS=1` behind Railway or Render, or rate limiting keys every caller off
the platform proxy's address and collapses them into one bucket.

## Security posture

The threat that matters for this service is not data theft — it is **someone else spending
our LLM budget**. `/demo/query` and `/evaluate` each cost multiple model calls, and
`budget_usd` arrives in the request body. Controls, all in `attest/api/security.py`:

- **Bearer auth** — `ATTEST_API_KEY`, compared with `secrets.compare_digest`. Generic `401`
  that never distinguishes absent from malformed from wrong. Disabled when the variable is
  unset, so local development and the dashboard work out of the box; **set it in production.**
- **Three-tier rate limiting** — 5/min for LLM-heavy endpoints, 20/min writes, 120/min reads,
  keyed on the real caller IP counted from the right of `X-Forwarded-For` so the bucket
  cannot be forged with a header.
- **Server-side spend ceiling** — the caller's `budget_usd` is clamped to `MAX_BUDGET_USD` at
  the route boundary.
- **Request size cap** — enforced for chunked bodies too, not just `Content-Length`.
- **CORS allowlist** — `CORS_ALLOW_ORIGINS`; credentials are disabled automatically on
  wildcard, since `*` plus `Allow-Credentials: true` is spec-invalid.
- **Security headers** — `nosniff`, `DENY`, `no-referrer`; HSTS in production only.
- **Row Level Security** — `migrations/002_rls.sql` enables *and forces* RLS on all five
  tables and revokes the `anon`/`authenticated` grants. Supabase serves every table over
  PostgREST; without this the traces are readable directly with the publishable key,
  bypassing this API entirely. `SUPABASE_KEY` must be the **service_role** key and must never
  reach a browser.
- **Log hygiene** — search queries are logged as a truncated SHA-256 digest, never verbatim.
  They are derived from the observed pipeline's answer, which in a real deployment is someone
  else's user content.
- **Subscriber cap** — 8 concurrent SSE subscribers per run; the bus holds an unbounded queue
  per subscriber.
- Container runs as **uid 10001**, not root.

All middleware is **pure ASGI**. `BaseHTTPMiddleware` buffers responses in a way that can
stall `EventSourceResponse`, and live streaming is the point of the product.

## Repo layout

```
attest/
  config.py           pydantic-settings; fails loud at boot, never mid-request
  models.py           all shared Pydantic models
  llm.py              provider router: anthropic → groq → gemini, with retry
  search.py           web search client (Tavily → DuckDuckGo) with TTL cache
  store.py            Supabase read/write
  orchestrator.py     fan-out, budget enforcement, sampling
  reconciler.py       verdict resolution + disagreement score
  verifiers/          decomposer, entailment, prober, mutations, independent
  api/                main, routes, stream (SSE), security
  sdk/                @observe decorator, wrappers, async submit queue
demo/                 seeded corpus + reference RAG pipeline
bench/                eval harness + ablation runner
dashboard/            single-file trace dashboard
migrations/           SQL (001 schema, 002 RLS)
tests/                271 tests
```

## Engineering constraints we held to

- `temperature=0` for every verification call. Non-determinism in a judge is a bug.
- Verifiers return structured Pydantic objects parsed from JSON with a repair fallback —
  never regex over prose.
- Evidence is requested as **character spans into the provided chunk**, not quoted text.
  Spans are checkable; quotes hallucinate.
- Subjective, predictive, and opinion claims are forced to `UNVERIFIABLE`. Precision matters
  more than coverage.
- Every LLM call logs model, tokens, latency, and cost into `verifications`. Cost visibility
  is a feature, not an afterthought.
- No mock data in the main path. Fixtures live in `tests/fixtures/` only.

## Known limitations

Stated here rather than left for someone to find:

- **Single replica only** — the event bus is process memory. Documented above; not fixed.
- **Decomposition quality bounds everything.** A badly split claim gets three confident
  verdicts about the wrong proposition.
- **The prober inherits the entailment verifier's blind spots** for anything its three
  mutation types don't reach. It detects a specific failure mode, not unreliability in general.
- **`STALE` depends on web search quality.** A claim the search provider can't find evidence
  for lands as `UNVERIFIABLE`, not `STALE`.
- **Cost scales with claim count.** Sampling and a budget ceiling exist because verifying
  every claim of every request is not economic at real traffic.
- The reference RAG pipeline in `demo/` is deliberately naive — whole-document chunks, no
  reranking. It is a reproducible target for the seeded demo cases, not a RAG showcase.

## Team

Byte_pros — Jyotish (25BAI1176), Pragatish (25BAI1406), Ravi (25BAI1146).

The backend, SDK, and benchmark were built against a frozen contract (see `PLAN.md` §5) so
the dashboard and the Google Opal demo agent could be developed independently and in
parallel.
