# ATTEST

**A runtime adversarial grounding layer for RAG and agentic systems.**

ATTEST decomposes an AI system's answer into atomic claims and verifies each one
three independent ways — entailment against the retrieved context, an adversarial
prober that mutates the claim and re-checks it, and an independent web retrieval
that catches sources that were themselves wrong or outdated. A reconciler merges
the three verdicts and streams the full reasoning trail to a live dashboard.

Built for **FRONTIER 2026** (AWS Student Builder Groups, VIT Chennai) — AI Safety
& Observability track — by team **Byte_pros**.

---

## Live

| | |
|---|---|
| **Dashboard** | <https://attest-dashboard.vercel.app> |
| **API** | <https://attest-api-production.up.railway.app> — [interactive docs](https://attest-api-production.up.railway.app/docs) |

Two entry points. Run the bundled demo RAG and watch it get verified, or paste
output from **any** pipeline — RAG-Anything, LlamaIndex, LangChain, a bare LLM
call — into the *Verify an external pipeline* panel, which posts it to
`POST /v1/observe`. ATTEST never generates, it only observes: anything that can
produce a query, an answer, and the chunks it retrieved can be verified.

The deployment key is embedded in the dashboard so the link needs no setup. It is
therefore public and is deliberately not treated as a secret — spend is capped
server-side by `MAX_BUDGET_USD` and by a 5 requests/minute limit on the
LLM-heavy routes. See [Security posture](#security-posture).

---

## Why this is not another RAGAS

Existing faithfulness evaluators (RAGAS, TruLens, Patronus, Vectara HHEM) score
**passively, offline, in a single pass**: one LLM judgment per claim, run in a
notebook after the fact. Three failures follow, and ATTEST targets each one.

| | Existing evaluators | ATTEST |
|---|---|---|
| **Judgment** | One passive LLM pass. The judge inherits the generator's blind spots, so confidently-wrong claims pass. | A prober **mutates** each claim (negation, entity swap, quantifier shift) and re-verifies. If the verdict doesn't flip when it logically must, the checker wasn't reading the context → `FRAGILE`. |
| **Scope** | Answer vs. retrieved chunk. Nobody checks whether the *chunk itself* was wrong. | An independent retriever re-searches the open web and cross-checks the **source** → `STALE`. |
| **Timing** | Offline test harness. You find out in the postmortem. | SDK middleware in the live request path, with a sampling rate and a hard cost budget. Observability, not a test run. |

`FRAGILE` and `STALE` are the contributions. Everything else is table stakes,
executed carefully.

## Verdict taxonomy

| Verdict | Meaning |
|---|---|
| `GROUNDED` | Entailed by retrieved context; verdict stable under all probes |
| `FRAGILE` | Entailed, but ≥1 probe failed to flip the verdict when it logically should have |
| `UNSUPPORTED` | No entailment found in retrieved context |
| `CONTRADICTED` | Retrieved context directly contradicts the claim |
| `STALE` | Entailed by context, but independent retrieval disagrees with the context |
| `UNVERIFIABLE` | Subjective, predictive, or otherwise not checkable against evidence |

`UNVERIFIABLE` is not a failure mode — it is how the system protects precision.
Subjective, predictive, and opinion claims are forced into it by prompt design.

## Architecture

```
        Any RAG / agent pipeline
                  │
      attest.observe()   ← SDK captures query, chunks, tool calls, answer
                  │
                  ▼
          ┌───────────────┐
          │  DECOMPOSER   │  answer → atomic claims + char spans
          └───────┬───────┘
                  │  asyncio.gather, per claim
      ┌───────────┼───────────────┐
      ▼           ▼               ▼
  ENTAILMENT  ADVERSARIAL     INDEPENDENT
   claim vs    PROBER          RETRIEVER
  retrieved   mutate +        fresh web search,
   context    re-verify       cross-check source
              → FRAGILE       → STALE
      └───────────┼───────────────┘
                  ▼
          ┌───────────────┐
          │  RECONCILER   │  verdict + confidence + disagreement score
          └───────┬───────┘
                  ▼
       Supabase  →  SSE  →  live trace dashboard
```

Verifiers never see each other's output — independence is the whole design.
The one narrow, deliberate exception (`VerifyContext.prior_entailment`, required
by the definition of `STALE` itself) is argued in `CONTRACT_CHANGE_REQUEST.md`.

## Quickstart

Requires Python 3.11 and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env          # then fill in the keys below
uv run pytest                 # full suite, no network required
uv run ruff check .
```

Minimum viable `.env`: a Supabase project (`SUPABASE_URL`, `SUPABASE_KEY`) plus
**at least one** LLM key — `ANTHROPIC_API_KEY`, `GROQ_API_KEY`, or
`GEMINI_API_KEY`. Providers are tried in that order and any provider without a
key is skipped entirely, so a `GEMINI_API_KEY`-only setup runs the whole system
on a free tier. `TAVILY_API_KEY` is optional; the independent retriever falls
back to DuckDuckGo without it. See `.env.example` for every knob, each
documented inline.

Apply the schema before first run:

```bash
# paste migrations/001_init.sql into the Supabase SQL editor
# then migrations/002_rls.sql for row-level security
```

Run the API:

```bash
uv run uvicorn attest.api.main:app --reload
# http://localhost:8000/docs
```

> **Deployment constraint:** run single-process. The SSE broker keeps
> per-run subscribers in process memory, so multiple workers will drop events.

Run the seeded demo (needs a real LLM key — this path actually generates):

```bash
uv run python demo/build_corpus.py     # once, materializes demo/corpus/
uv run python demo/run_demo.py         # both seeded queries, end to end
```

The demo corpus is 30 fictional "Northwind Devices" documents seeded so that one
query lands on `STALE` and another on `UNSUPPORTED`/`FRAGILE`. Exactly which
document and why is written down in `demo/SEED_NOTES.md`.

## API

Base path `/v1`. Full request/response shapes are in `PLAN.md` §5.2 — **frozen**,
because the dashboard and Opal agent tracks build against them independently.

| Method | Path | Returns |
|---|---|---|
| `POST` | `/observe` | `{run_id, status}` (202, verification runs async) |
| `GET` | `/runs?limit=&offset=` | `{runs: RunSummary[]}` |
| `GET` | `/runs/{run_id}` | `RunDetail` — the full nested trace |
| `GET` | `/runs/{run_id}/stream` | SSE trace |
| `POST` | `/evaluate` | benchmark table |
| `GET` | `/health` | `{ok, version}` |

SSE event names, which the frontend depends on exactly:
`run.started` · `claims.decomposed` · `claim.verified` · `probe.completed` ·
`run.completed` · `run.error`

## SDK

The SDK **never raises into the host pipeline**. Every path is wrapped; on
failure it logs and returns control. A grounding checker that crashes the app it
observes is worse than useless — there is a dedicated test suite for exactly
this (`tests/test_sdk_never_raises.py`).

```python
import attest

attest.init(api_url="https://...", api_key="...", sample_rate=0.05)

@attest.observe(pipeline_name="support-bot")
def answer(query: str) -> attest.Output:
    chunks = retrieve(query)
    return attest.Output(answer=generate(query, chunks), retrieved_chunks=chunks)

# or, imperatively
with attest.trace(pipeline_name="support-bot", query=q) as t:
    t.record_chunks(chunks)
    t.record_answer(text)

# or wrap a LangChain chain
chain = attest.wrap(chain, pipeline_name="rag-v2")
```

## Benchmark

`bench/` is a real evaluation harness over RAGTruth and HaluEval, with four
ablation configs (single-pass baseline, ATTEST−prober, ATTEST−independent,
ATTEST full) sharing one decomposition pass per example for a fair comparison.
Metrics: precision/recall/F1 with bootstrap 95% CIs, cost, p95 latency, plus
FRAGILE precision-risk and UNVERIFIABLE abstention rates.

```bash
uv run python -m bench.run_benchmark --n 250 --seed 42 --dataset ragtruth
```

> **`bench/results.md` and `bench/results.json` currently read `PENDING`.**
> That is deliberate, not an oversight. No LLM-backed run has been executed yet,
> and fabricated numbers are worse than absent ones. The verdict → hallucination-
> label mapping, including the contestable decision to count `FRAGILE` as a
> positive, is argued in full in `bench/MAPPING.md`.

## Security posture

The primary threat to this service is not data theft, it is **credit burn**:
`budget_usd` arrives in the request body, so an open `/v1/observe` lets anyone
who finds the URL spend the team's LLM credits. `attest/api/security.py`
implements bearer auth, per-IP rate limiting, request-size caps, and security
headers as raw ASGI middleware (not `BaseHTTPMiddleware`, which buffers and
stalls SSE). `MAX_BUDGET_USD` caps the caller-supplied budget server-side.

Auth is **off** unless `ATTEST_API_KEY` is set, so local development and the
other two tracks keep working against the frozen contract untouched. Turning it
on is a deployment decision that requires telling those tracks first.

No secrets are committed. `.env` is gitignored, `.env.example` carries
placeholders only, and search queries are hashed rather than logged verbatim so
the host application's user content never enters our log retention.

## Repo layout

```
attest/
  config.py        pydantic-settings, fails loud at boot
  models.py        all shared Pydantic models (frozen contracts)
  llm.py           provider router: anthropic | groq | gemini, JSON repair
  search.py        web search client (Tavily → DuckDuckGo, disk-cached)
  store.py         Supabase read/write
  orchestrator.py  fan-out + budget + sampling
  reconciler.py    verdict resolution + disagreement score
  verifiers/       decomposer, entailment, prober, mutations, independent
  api/             main, routes, stream, security
  sdk/             decorator, wrappers
demo/              seeded corpus + reference RAG pipeline
bench/             eval harness + ablation runner
dashboard/         single-file trace dashboard
migrations/        SQL schema + RLS
tests/             pytest, network-free (LLM mocked)
```

## Project documents

| File | What it is |
|---|---|
| `PLAN.md` | Full spec. Single source of truth. §5 contracts are frozen. |
| `PROGRESS.md` | What is actually built, per component |
| `DECISIONS.md` | Technical decisions and their reasoning |
| `FAILURES.md` | What broke, what was tried, what fixed it |
| `HANDOFF.md` | Cross-agent integration notes and open items |
| `CONTRACT_CHANGE_REQUEST.md` | Every change to a frozen contract, with rationale |
| `CLAUDE.md` | Engineering rules for the agents building this |

## Team

Byte_pros — Jyotish (25BAI1176), Pragatish (25BAI1406), Ravi (25BAI1146),
Shyam (25BAI1774).

## License

MIT
