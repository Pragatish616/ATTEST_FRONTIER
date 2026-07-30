# ATTEST — Adversarial Grounding Layer for Agentic Systems

> Drop-in middleware that decomposes an AI system's output into atomic claims, verifies each one three independent ways — including actively trying to break its own verdict — and streams the full reasoning trail to a live dashboard.

**Hackathon track:** AI Safety & Observability
**Build window:** 36 hours
**Status of this doc:** single source of truth. Every agent (Claude Code, Antigravity, Opal) reads this. Contracts in §5 are FROZEN — no agent may change them unilaterally.

---

## 1. The problem in one paragraph

Every RAG and agentic pipeline in production ships answers that *sound* grounded. Existing evaluators (RAGAS, TruLens, Patronus, Vectara HHEM) score faithfulness **passively, offline, in a single pass** — one LLM judgment per claim, run in a notebook after the fact. Three failures follow: (a) the judge inherits the generator's blind spots, so confidently-wrong claims pass; (b) nobody checks whether the *retrieved source itself* was wrong or stale — only whether the answer matched it; (c) none of it runs at request time, so you find out in a postmortem.

## 2. What ATTEST does differently

Say these three, in this order, whenever anyone asks "how is this not RAGAS":

1. **Adversarial, not passive.** A prober mutates each claim (negation, entity swap, quantifier shift) and re-runs verification. If the verdict doesn't flip when it logically must, the verifier wasn't reading the context — the claim is marked `FRAGILE` even though pass #1 said "supported."
2. **Source-level, not just context-level.** An independent retriever re-searches the open web and cross-checks the *source*, surfacing `STALE`: grounded in the retrieved chunk, but the chunk is outdated or wrong.
3. **Runtime, not offline.** SDK middleware in the live path with a sampling budget, streaming to a dashboard. Observability, not a test harness.

`FRAGILE` and `STALE` are the novel contributions. Everything else is table stakes executed well.

## 3. Verdict taxonomy (FROZEN)

| Verdict | Definition | Colour in UI |
|---|---|---|
| `GROUNDED` | Entailed by retrieved context; verdict stable under all probes | green |
| `FRAGILE` | Entailed, but ≥1 probe failed to flip the verdict when it logically should have | amber |
| `UNSUPPORTED` | No entailment found in retrieved context | red |
| `CONTRADICTED` | Retrieved context directly contradicts the claim | red |
| `STALE` | Entailed by context, but independent retrieval disagrees with the context | purple |
| `UNVERIFIABLE` | Subjective, predictive, or otherwise not checkable against evidence | grey |

Never emit a verdict outside this set. `UNVERIFIABLE` is not a failure — it protects precision.

## 4. Architecture

```
       Any RAG / agent pipeline
                │
   attest.observe()  ← SDK captures query, retrieved chunks, tool calls, final answer
                │
                ▼
        ┌───────────────┐
        │  DECOMPOSER   │  answer → atomic, self-contained claims + char spans
        └───────┬───────┘
                │ fan-out, asyncio.gather, per claim
    ┌───────────┼────────────┬──────────────────┐
    ▼           ▼            ▼
ENTAILMENT  ADVERSARIAL   INDEPENDENT
VERIFIER    PROBER        RETRIEVER
claim vs    mutate +      fresh web search,
retrieved   re-verify     cross-check source
context     (3 mutations) → STALE
    └───────────┴────────────┴──────────────────┐
                                                ▼
                                    ┌───────────────────┐
                                    │    RECONCILER     │
                                    │ verdict + disagreement
                                    └─────────┬─────────┘
                                              ▼
                          Supabase (traces)  →  SSE  →  Dashboard
```

**Ownership split**
| Component | Built by |
|---|---|
| Everything in the box above, SDK, API, benchmark | **Claude Code** (§7 agents) |
| Dashboard, live trace viewer, marketing page | **Antigravity** (see `ANTIGRAVITY_FRONTEND_BRIEF.md`) |
| Demo target agent + judge-facing "try it" mini-app | **Google Opal** (see `OPAL_SPEC.md`) |

The three tracks are decoupled by the frozen contracts in §5. Nobody blocks anybody.

## 5. FROZEN CONTRACTS

### 5.1 Supabase schema

```sql
create table runs (
  id            uuid primary key default gen_random_uuid(),
  created_at    timestamptz not null default now(),
  pipeline_name text not null,
  query         text not null,
  answer        text not null,
  model         text,
  status        text not null default 'pending',   -- pending|running|complete|error
  grounding_score  real,       -- 0..1, share of verifiable claims that are GROUNDED
  fragility_score  real,       -- 0..1, share of verifiable claims that are FRAGILE
  total_claims     int,
  latency_ms       int,
  cost_usd         numeric(10,6)
);

create table retrieved_chunks (
  id          uuid primary key default gen_random_uuid(),
  run_id      uuid not null references runs(id) on delete cascade,
  chunk_index int not null,
  source_id   text,
  source_url  text,
  text        text not null,
  score       real
);

create table claims (
  id           uuid primary key default gen_random_uuid(),
  run_id       uuid not null references runs(id) on delete cascade,
  claim_index  int not null,
  text         text not null,
  span_start   int,            -- char offset into runs.answer
  span_end     int,
  verdict      text not null,  -- see taxonomy §3
  confidence   real,           -- 0..1
  disagreement real,           -- 0..1, how much the three verifiers disagreed
  rationale    text
);

create table verifications (
  id         uuid primary key default gen_random_uuid(),
  claim_id   uuid not null references claims(id) on delete cascade,
  verifier   text not null,    -- entailment|independent|prober
  verdict    text not null,
  rationale  text,
  evidence   jsonb,            -- [{chunk_id|url, quote_span:[s,e], stance:'support'|'refute'}]
  latency_ms int,
  cost_usd   numeric(10,6)
);

create table probes (
  id              uuid primary key default gen_random_uuid(),
  claim_id        uuid not null references claims(id) on delete cascade,
  mutation_type   text not null,   -- negation|entity_swap|quantifier_shift
  mutated_text    text not null,
  expected_flip   boolean not null,
  observed_verdict text not null,
  flipped         boolean not null
);

create index on claims(run_id);
create index on verifications(claim_id);
create index on probes(claim_id);
alter publication supabase_realtime add table runs, claims, verifications, probes;
```

### 5.2 REST + SSE API

Base: `/v1`

| Method | Path | Body / Returns |
|---|---|---|
| `POST` | `/observe` | `ObserveRequest` → `{run_id, status}` (202, async) |
| `GET` | `/runs?limit=&offset=` | `{runs: RunSummary[]}` |
| `GET` | `/runs/{run_id}` | `RunDetail` (full nested trace) |
| `GET` | `/runs/{run_id}/stream` | SSE, events below |
| `POST` | `/evaluate` | `{dataset, n, ablation}` → benchmark table |
| `GET` | `/health` | `{ok: true, version}` |

```jsonc
// ObserveRequest
{
  "pipeline_name": "demo-rag",
  "query": "string",
  "answer": "string",
  "retrieved_chunks": [
    {"chunk_index": 0, "source_id": "doc-12", "source_url": "https://…", "text": "…", "score": 0.82}
  ],
  "model": "claude-sonnet-4-6",
  "config": {"sample_rate": 1.0, "enable_independent": true, "enable_prober": true, "budget_usd": 0.05}
}
```

SSE event names (frontend depends on these exactly):
`run.started` · `claims.decomposed` · `claim.verified` · `probe.completed` · `run.completed` · `run.error`

Every event payload: `{"run_id": "...", "data": { … }}`.

### 5.3 Python SDK surface

```python
import attest

attest.init(api_url=..., api_key=..., sample_rate=0.05)

# decorator
@attest.observe(pipeline_name="support-bot")
def answer(query: str) -> attest.Output:
    ...
    return attest.Output(answer=text, retrieved_chunks=chunks)

# context manager
with attest.trace(pipeline_name="support-bot", query=q) as t:
    t.record_chunks(chunks)
    t.record_answer(text)

# LangChain / LlamaIndex
chain = attest.wrap(chain, pipeline_name="rag-v2")
```

Non-negotiable: **never raise into the host pipeline**. All SDK failures log and no-op.

## 6. Tech stack

| Layer | Choice | Why |
|---|---|---|
| Runtime | Python 3.11, `uv` | fast installs under time pressure |
| API | FastAPI + `sse-starlette` | async fan-out, streaming, free OpenAPI |
| Models | provider router: Anthropic primary, Groq/Gemini fallback | cost control + no single point of failure on demo day |
| Decomposer | small/fast model (Haiku-class) | called once per run, must be cheap |
| Verifiers | mid model (Sonnet-class), `temperature=0` | judgment quality matters here |
| Corpus store | ChromaDB, local, persisted to `./data/chroma` | zero setup |
| Trace store | Supabase Postgres + Realtime | frontend gets live updates free |
| Web search | Tavily (fallback: DuckDuckGo) | independent retriever |
| Frontend | Next.js 15 + Tailwind, on Vercel | Antigravity's track |
| Demo agent | Google Opal | Opal's track |

Config via `.env`, validated by Pydantic Settings at boot. Fail loudly at startup if a key is missing, never mid-demo.

## 7. Claude Code agent split

Run **A0 first, alone**. Then A1–A5 in parallel. Then A6.

| Agent | Owns | Touches only |
|---|---|---|
| **A0 Contracts** | Pydantic models, Supabase migration, settings, LLM router, repo skeleton | `attest/models.py`, `attest/config.py`, `attest/llm.py`, `migrations/`, `pyproject.toml` |
| **A1 Decomposer + Entailment** | claim extraction with spans, entailment verifier | `attest/verifiers/decomposer.py`, `attest/verifiers/entailment.py` |
| **A2 Prober** | mutation engine, expected-flip logic, fragility scoring | `attest/verifiers/prober.py`, `attest/verifiers/mutations.py` |
| **A3 Independent Retriever** | web search, source cross-check, STALE detection | `attest/verifiers/independent.py`, `attest/search.py` |
| **A4 Orchestrator + API** | fan-out, reconciler, FastAPI routes, SSE, Supabase writes | `attest/orchestrator.py`, `attest/reconciler.py`, `attest/api/`, `attest/store.py` |
| **A5 SDK + Demo corpus** | SDK, LangChain wrapper, seeded demo corpus + RAG pipeline | `attest/sdk/`, `demo/` |
| **A6 Benchmark** | eval harness, ablation runner, results table | `bench/` |

**File-ownership rule:** an agent may only write files in its own column. Cross-cutting change → stop, state the needed contract change, wait. This is what stops parallel agents from stomping each other.

## 8. Hour-by-hour

| Hours | Work | Gate |
|---|---|---|
| 0–2 | A0 alone. Schema pushed to Supabase, models importable, LLM router returns a completion. | `pytest tests/test_contracts.py` green |
| 2–8 | A1, A2, A3 in parallel. Each ships a `verify(claim, context) -> Verification` that runs standalone. | each verifier has a passing unit test on a fixture |
| 8–14 | A4 wires fan-out + reconciler + API. A5 builds SDK + demo corpus. | `POST /observe` returns a full trace end-to-end |
| 14–18 | SSE streaming live. Antigravity connects to real API. **Seed the demo corpus with one stale doc and one subtly-unsupported claim.** | dashboard shows a live run |
| 18–22 | Sleep in shifts. | — |
| 22–28 | A6 benchmark + ablation. Fix the worst verifier. | results table populated |
| 28–32 | Opal mini-app, CI mode, polish. | judge can run it themselves |
| 32–34 | Rehearse demo 5×. Record backup video. | video exists |
| 34–36 | **CODE FREEZE.** Deck only. | no commits |

Freeze at 32 even if something is unfinished. Teams still pushing at minute 5 of judging always look it.

## 9. Demo script (5 min)

**Act 1 — hook (60s).** Run the Opal demo agent on a seeded question. Answer is fluent, confident, well-cited-looking. Pause: *"Every claim here reads as grounded. Two are wrong."* Show the one-line SDK decorator. Re-run.

**Act 2 — reveal (2m).** Dashboard streams live. Sentences highlight as verdicts land. One goes **amber `FRAGILE`** — click it, open the probe panel: the original claim passed, the *negated* claim also passed, so the verifier never read the context. One goes **purple `STALE`** — click it: retrieved chunk says X, independent retrieval says the current answer is Y. **The amber click is the moment of the pitch.** No competing tool shows this.

**Act 3 — it's a product (90s).** Benchmark + ablation table. CI mode failing a build. Cost/sampling config. Close on the decorator again.

**Backup:** recorded video of the identical run, queued and ready. Hackathon wifi kills more demos than bad code.

## 10. Benchmark (this is what wins technical judges)

Dataset: **RAGTruth** or **HaluEval**, 200–300 examples. Report:

| System | Precision | Recall | F1 | Cost/claim | p95 latency |
|---|---|---|---|---|---|
| Single-pass LLM judge (baseline) | | | | | |
| ATTEST − prober (ablation) | | | | | |
| ATTEST − independent (ablation) | | | | | |
| **ATTEST full** | | | | | |

The **ablation rows are the entire argument**. If the prober adds nothing, you need to know that before the judges do.

## 11. Known flaws — state these before judges find them

| Flaw | Prepared answer |
|---|---|
| Latency & cost: 3 verifiers × N claims | Sampling + cheap decomposer + claim-hash cache. Observability layer, not an inline blocker, unless opted in. Have cost-per-1k-requests memorised. |
| Grounded ≠ true | Say it first. Faithfulness and truth are different axes; `STALE` is our partial reach toward truth and we don't overclaim past it. |
| Who verifies the verifier? | Ablation numbers, plus: *disagreement between independent verifiers is itself the signal*, not any single verdict. |
| Entailment is weak on numerics and negation | Known NLI failure mode — which is exactly what quantifier-shift probes target. Demo a numeric case working. |
| Decomposition is lossy on compound sentences | Acknowledged. Unresolvable claims surface as `UNVERIFIABLE`, never silently dropped. |
| "Isn't this just RAGAS?" | §2, memorised, in order. |

## 12. Judge Q&A

- *What's actually agentic here?* Three verifiers make independent judgments; the prober plans its own mutation strategy per claim type; a reconciler resolves genuine disagreement. It's a contested fan-out, not a chain.
- *Would you run this in production?* Yes — 5% sampling on the live path, 100% in CI. *[state cost number]*
- *What breaks first at scale?* Verifier cost and decomposition quality on long outputs. Roadmap: distil entailment into a small fine-tuned classifier, keep LLMs only for the prober.
- *Why you?* You run multi-agent systems in production at your internship and built this because you needed it. Almost nobody else in the room can say that truthfully.

## 13. Definition of done

- [ ] `POST /observe` returns a full trace with all five verdict types reachable
- [ ] Live SSE stream renders in the dashboard
- [ ] Seeded demo produces exactly one `FRAGILE` and one `STALE`
- [ ] Benchmark table filled with real numbers including ablations
- [ ] SDK installs and wraps a pipeline in one line
- [ ] Backup demo video recorded
- [ ] README with 60-second quickstart
