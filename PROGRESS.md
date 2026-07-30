# PROGRESS

## A0 — Contracts (complete)

**Shipped:** repo skeleton, `attest/models.py` (all shared Pydantic models),
`attest/config.py` (pydantic-settings, fails loud at import), `attest/llm.py`
(provider router with fallback + JSON-repair), `migrations/001_init.sql` +
rollback (verbatim PLAN.md §5.1), `tests/test_contracts.py` (22 tests,
green). `uv run pytest` and `uv run ruff check .` both pass clean.

**Stubbed:** every file owned by A1–A6 exists as an empty file with a
one-line ownership comment only (no logic) — see "Skeleton" below.

**What the next agent needs:** import everything from `attest.models`,
`attest.config`, and `attest.llm` exactly as documented below. These are
frozen — do not add fields, rename things, or change the Verdict taxonomy.
If something is missing, write it to `CONTRACT_CHANGE_REQUEST.md` instead of
editing these files.

---

## A1–A5 — verifiers, orchestrator, API, SDK, demo (complete)

Built in parallel against the A0 contracts above, then integrated. Full
suite: `uv run pytest` → **163 passed**. `uv run ruff check .` → clean.
`POST /observe` runs the real pipeline end-to-end — no verifier is a stub
anymore. `import attest; attest.init(...)` works.

- **A1** (`attest/verifiers/decomposer.py`, `entailment.py`): `decompose(answer, query, *, run_id) -> list[Claim]` and `EntailmentVerifier` (no-arg constructible). 22 tests.
- **A2** (`attest/verifiers/mutations.py`, `prober.py`): `AdversarialProber(entailment_verifier: VerifierProtocol)` — dependency-injected, not a hard import of A1's class. Exposes per-mutation `Probe` records via `verify_with_probes()` and the `last_probes` attribute (see HANDOFF.md for why). 20 tests.
- **A3** (`attest/search.py`, `attest/verifiers/independent.py`): `search(query, k) -> list[SearchResult]` (Tavily → DuckDuckGo, disk-cached) and `IndependentVerifier`, which is the only verifier that reads `ctx.prior_entailment`. 14 tests.
- **A4** (`attest/orchestrator.py`, `reconciler.py`, `store.py`, `attest/api/*`): `orchestrator.run(request, *, decompose, entailment, prober, independent, store, emit, get_probes=None, ...) -> RunDetail`, `reconciler.reconcile(entailment, prober, independent) -> (verdict, confidence, disagreement, rationale)` (precedence: CONTRADICTED > STALE > UNSUPPORTED > FRAGILE > GROUNDED > UNVERIFIABLE), `Store` (Supabase, incremental writes, refuses to persist a claim with `verdict=None`), and the full REST/SSE API. 57 tests.
- **A5** (`attest/sdk/*`, `demo/*`): `attest.init/observe/trace/wrap` per PLAN.md §5.3, proven to never raise into the host pipeline even against a dead backend; a 30-doc "Northwind Devices" demo corpus + ChromaDB RAG pipeline seeded for one `STALE` and one `UNSUPPORTED`/`FRAGILE` case (`demo/SEED_NOTES.md`). 28 tests.

**Integration work done centrally** (not by any single agent — see
`HANDOFF.md` for full detail): fixed a prober constructor mismatch
(`AdversarialProber` needs `entailment_verifier`, not the no-arg class
`routes.py` assumed), adapted A1's `decompose()` signature to what
`orchestrator.run` calls it with, wired the `get_probes` hook to A2's
`last_probes`, added the missing `attest/__init__.py` SDK re-export, and
added `duckduckgo-search` to `pyproject.toml`.

**What's still open** (non-blocking, tracked in `HANDOFF.md`): prober
mutation-generation cost isn't fully rolled into `Verification.cost_usd`;
`attest.wrap()`'s LlamaIndex path is unverified; `/evaluate` is a stub
pending A6; no CORS config yet; auth is sent by the SDK but not enforced by
the API; the demo's generation half needs one real end-to-end run with a
live `ANTHROPIC_API_KEY` before the actual demo.

---

## A6 — Benchmark + ablation harness (complete, unexecuted)

Built against the real A1–A5 pipeline (not reimplemented): `bench/datasets.py`
(real RAGTruth/HaluEval download + disk cache + stratified sampling, fixed
seed), `bench/mapping.py` + `bench/MAPPING.md` (verdict → hallucination-label
mapping, with the FRAGILE-as-positive judgment call argued explicitly and the
rejected alternative documented), `bench/configs.py` + `bench/runner.py`
(four configs — single-pass baseline, ATTEST−prober, ATTEST−independent,
ATTEST full — sharing one decomposition pass per example for a fair
comparison, wired to the real `EntailmentVerifier`/`AdversarialProber`/
`IndependentVerifier`/`reconcile`), `bench/metrics.py` (precision/recall/F1,
hand-rolled bootstrap 95% CI, cost/p95-latency, FRAGILE precision-risk rate,
UNVERIFIABLE abstention rate, STALE fire-rate diagnostic), `bench/report.py`
(`bench/results.md`/`results.json`), `bench/run_benchmark.py` (CLI). 58 tests
against a small **real** RAGTruth/HaluEval sample with the LLM mocked.

**Important: `bench/results.md`/`results.json` are honest PENDING templates,
not real numbers.** This environment has no `.env` and no real API key, so
no real LLM-backed run has happened — every metric cell literally says
`PENDING`/`null`, on purpose, per CLAUDE.md's "no mock data" spirit extended
to "no fabricated results either." To get real numbers: put a real
`ANTHROPIC_API_KEY` in `.env`, then
`uv run python -m bench.run_benchmark --n 250 --seed 42 --dataset ragtruth`.
This also means PLAN.md §13's "benchmark table filled with real numbers
including ablations" is genuinely not done yet — don't mark it complete
until that command has actually been run.

Full suite after integration: `uv run pytest` → 221 passed. `uv run ruff
check .` → clean. `bench/data_cache/` (real downloaded datasets, ~41MB) and
`data/search_cache/` are gitignored — never commit them.

---

## Environment

```bash
uv sync                 # installs Python 3.11.15 (uv-managed) + all deps
cp .env.example .env    # fill in SUPABASE_URL, SUPABASE_KEY, ANTHROPIC_API_KEY
uv run pytest           # 22 passed
uv run ruff check .     # All checks passed
uv run ruff format .    # do NOT run this over the whole repo — it will try
                         # to reformat PLAN.md/CLAUDE.md. Scope it:
uv run ruff format attest tests bench demo
```

Tests never touch the network or real API keys: `tests/conftest.py` sets
placeholder env vars before any import, and `test_contracts.py` mocks the
provider layer directly (`llm._PROVIDER_CHAIN`).

---

## `attest/models.py` — import paths

```python
from attest.models import (
    Verdict,              # StrEnum: GROUNDED, FRAGILE, UNSUPPORTED, CONTRADICTED, STALE, UNVERIFIABLE
    MutationType,          # StrEnum: negation, entity_swap, quantifier_shift
    RunStatus,              # Literal["pending", "running", "complete", "error", "skipped"]
    VerifierName,           # Literal["entailment", "independent", "prober"]
    Stance,                 # Literal["support", "refute"]
    RetrievedChunk,         # client-supplied chunk (no id/run_id) — used in ObserveRequest
    Chunk,                  # RetrievedChunk + id, run_id — matches retrieved_chunks row
    Claim,                  # matches claims row; verdict/confidence/disagreement/rationale
                            #   are Optional and None until the reconciler fills them in
    Evidence,               # one entry in Verification.evidence: chunk_id|url, quote_span, stance
    Verification,           # matches verifications row
    Probe,                  # matches probes row
    ClaimDetail,            # Claim + nested verifications + probes (for RunDetail)
    RunSummary,             # matches runs row — GET /runs
    RunDetail,               # RunSummary + retrieved_chunks + claims (ClaimDetail) — GET /runs/{id}
    AttestConfig,            # per-run config nested in ObserveRequest.config
    ObserveRequest,          # POST /observe body
    ObserveResponse,         # POST /observe 202 response: {run_id, status}
    VerifyContext,           # what a verifier is allowed to see: run_id, query, answer,
                            #   retrieved_chunks, config, prior_entailment — see note below
    VerifierProtocol,        # runtime_checkable Protocol every verifier implements
)
```

### The verifier interface (A1, A2, A3 — this is what you implement)

```python
class VerifierProtocol(Protocol):
    async def verify(self, claim: Claim, ctx: VerifyContext) -> Verification: ...
```

- `claim.verdict` will be `None` when you receive it — you are producing
  *one* of the three independent opinions, not the final answer. Never read
  another verifier's `Verification`; there is no channel for that in `ctx`.
- Return a `Verification` with `verifier` set to your literal name
  (`"entailment"` / `"prober"` / `"independent"`), a `verdict` from the six
  frozen values, and `evidence` as **char-span** entries into
  `ctx.retrieved_chunks[i].text` — never quoted text.
- `isinstance(your_verifier_instance, VerifierProtocol)` works — it's
  `@runtime_checkable` — but only checks the method exists, not its
  signature, so match the signature by hand.

### `VerifyContext.prior_entailment` (added post-A0, see CONTRACT_CHANGE_REQUEST.md)

`Verdict | None`, default `None`. Set by the orchestrator only on the
context it passes to `IndependentVerifier` — the entailment verdict, so
independent verification can distinguish `STALE` (entailed by context, but
independent evidence disagrees) from plain `CONTRADICTED`. Nobody else reads
it; the prober re-derives its own baseline directly.

### Claim lifecycle (why `verdict` is Optional)

Decomposer (A1) creates `Claim` objects with `verdict=None`. Those go to all
three verifiers unchanged. The reconciler (A4) is the only place that
computes the final `verdict` / `confidence` / `disagreement` / `rationale`
and copies them onto the claim (`claim.model_copy(update={...})`) before
`store.py` persists it — the DB column is `NOT NULL`, so store.py must
reject/never call insert with a `None` verdict. This isn't enforced by the
type system on purpose; decomposer and reconciler run at different points in
the pipeline and both need to construct a `Claim`.

---

## `attest/config.py` — import path

```python
from attest.config import settings   # a Settings instance, already validated
```

Do not call `Settings()` yourself (except in tests, where it isn't needed —
see below). `settings` is built once at import time and raises `RuntimeError`
immediately if a required key is missing. Required: `supabase_url`,
`supabase_key`, `anthropic_api_key`. Everything else has a default or is
`None`-able (`groq_api_key`, `gemini_api_key`, `tavily_api_key`).

Model tier → concrete model name fields (override via env, see
`.env.example`): `anthropic_fast_model`, `anthropic_judge_model`,
`groq_fast_model`, `groq_judge_model`, `gemini_fast_model`,
`gemini_judge_model`.

Other fields available: `chroma_persist_dir` (A5), `app_env`, `log_level`,
`app_version` (A4, for `/health`).

**In tests:** `attest.config` is imported the moment anything imports
`attest.llm`, `attest.store`, etc. `tests/conftest.py` sets
`SUPABASE_URL` / `SUPABASE_KEY` / `ANTHROPIC_API_KEY` env vars via
`os.environ.setdefault` before collection, so `settings` constructs cleanly
with fake values. Don't remove that — it's what keeps the whole suite
network-free.

---

## `attest/llm.py` — import path and signature

```python
from attest.llm import complete, LLMResult, LLMError, ModelTier

result: LLMResult = await complete(
    prompt,
    model_tier="fast",       # or "judge" — Literal["fast", "judge"]
    schema=SomeBaseModel,    # optional; omit for plain-text completions
)
result.text          # str — raw text (post-repair, if repair happened)
result.parsed         # SomeBaseModel | dict | list | None — populated iff schema was passed and parsing succeeded
result.provider       # "anthropic" | "groq" | "gemini" — whichever one actually served the request
result.model          # concrete model name used
result.tokens_in / result.tokens_out
result.latency_ms
result.cost_usd       # log this into verifications.cost_usd / runs.cost_usd
```

- `temperature=0` is hardcoded, no override exists — every judge call is
  deterministic by construction.
- Anthropic is tried first, then Groq, then Gemini, skipping any provider
  whose API key isn't configured. On a provider exception (rate limit, API
  error, anything) it logs a warning and falls through to the next provider.
- If `schema` is given and the response doesn't parse: (1) local repair
  (strip ```fences```, isolate outermost `{}`/`[]`) is tried first, free of
  charge; (2) if that still fails, exactly one more call is made to the
  *same* provider with a "return valid JSON only" nudge prompt, and tokens
  from that retry are added to the totals. If it still doesn't parse,
  `result.parsed` is `None` and `result.text` is whatever came back — check
  `parsed is None` and handle it (this is expected to happen sometimes;
  don't treat it as a crash).
- Raises `attest.llm.LLMError` only when every configured provider failed or
  none is configured. This is the one path where a verifier calling
  `attest.llm.complete` should catch and turn into a degraded verdict
  (e.g. skip that verifier for this claim) rather than letting it propagate
  — `asyncio.gather(..., return_exceptions=True)` in the orchestrator is the
  other safety net, but don't rely on that alone in unit tests.
- **Testing pattern:** don't monkeypatch `attest.llm._call_anthropic` etc.
  directly — `_PROVIDER_CHAIN` captures those function objects by reference
  at module load, so patching the name afterward does nothing. Patch the
  list itself: `monkeypatch.setattr(llm, "_PROVIDER_CHAIN", [("anthropic", fake_call)])`
  where `fake_call(prompt: str, model: str) -> tuple[str, int, int]` returns
  `(text, tokens_in, tokens_out)`. See `tests/test_contracts.py` for full
  examples including the schema/repair/retry paths and provider fallback.

---

## Migrations

`migrations/001_init.sql` is PLAN.md §5.1 verbatim — apply it in Supabase
exactly as written. `migrations/001_init_rollback.sql` drops everything in
reverse FK order (probes → verifications → claims → retrieved_chunks →
runs) and removes the tables from `supabase_realtime` first.

---

## Skeleton — who owns what (unchanged from PLAN.md §7)

Every file below exists and is empty except for a one-line ownership
comment. Fill in your own; don't touch anyone else's — if you need a change
in one of A0's files (`models.py`, `config.py`, `llm.py`) or in another
agent's file, write it to `HANDOFF.md` instead of editing directly.

| File | Owner |
|---|---|
| `attest/search.py` | A3 |
| `attest/store.py`, `attest/orchestrator.py`, `attest/reconciler.py` | A4 |
| `attest/verifiers/decomposer.py`, `attest/verifiers/entailment.py` | A1 |
| `attest/verifiers/prober.py`, `attest/verifiers/mutations.py` | A2 |
| `attest/verifiers/independent.py` | A3 |
| `attest/api/main.py`, `attest/api/routes.py`, `attest/api/stream.py` | A4 |
| `attest/sdk/__init__.py`, `attest/sdk/decorator.py`, `attest/sdk/wrappers.py` | A5 |
| `demo/` | A5 |
| `bench/` | A6 |

`tests/fixtures/` exists (currently just a `.gitkeep`) — per CLAUDE.md, this
is the *only* place mock/fixture data may live; nothing in the main path may
be backed by hardcoded verdicts.
