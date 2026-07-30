# HANDOFF

Cross-cutting notes and open items surfaced by A1–A5 during the parallel
build, plus the integration fixes applied to reconcile their outputs. Read
this before touching `attest/api/routes.py`, `attest/orchestrator.py`,
`attest/verifiers/prober.py`, or `attest/sdk/wrappers.py`.

## Integration fixes already applied (in `attest/api/routes.py`)

A4's routes.py was written in parallel with A1/A2/A3 and guessed at two
shapes that turned out to differ from what actually landed. Both are fixed
now, not just noted:

1. **Prober wiring.** `get_prober_verifier()` assumed a no-arg `ProberVerifier`
   class; A2 shipped `AdversarialProber(entailment_verifier: VerifierProtocol)`
   (dependency-injected, per its own design — see A2's report). Fixed to
   construct `AdversarialProber(entailment_verifier=get_entailment_verifier())`.
2. **Decomposer signature.** `orchestrator.run` calls
   `decompose(request: ObserveRequest, run_id: UUID)`; A1 shipped
   `decompose(answer: str, query: str, *, run_id: UUID)`. `get_decompose()`
   now wraps it in a small adapter closure rather than either agent's file
   being changed.
3. **Probe persistence.** `AdversarialProber` has no way to return `Probe`
   rows through the plain `VerifierProtocol.verify()` call the orchestrator
   makes (frozen return type) — it exposes them via a `last_probes`
   side-effect attribute instead. Wired `get_probes_hook()` to read
   `prober.last_probes` right after `verify()` returns. This is safe only
   because `orchestrator.run` processes claims in a strictly sequential
   `for` loop (confirmed by reading `orchestrator.py`) — if that ever
   becomes concurrent across claims, this hook needs to move to
   `AdversarialProber.verify_with_probes()` instead (also implemented,
   currently unused by the wiring — see `attest/verifiers/prober.py`).

Fixed `tests/test_api.py::test_observe_returns_503_when_verifiers_not_implemented`
(renamed to `..._when_a_dependency_is_not_ready`) — it asserted the 503 path
by relying on A1/A2/A3's modules being real stubs, which stopped being true
once they landed. Now forces `DependencyNotReady` explicitly via monkeypatch,
independent of whether the real modules exist.

Also added the missing `attest/__init__.py` re-export
(`from attest.sdk import Output, init, observe, trace, wrap`) so PLAN.md
§5.3's `import attest; attest.init(...)` usage actually works — it only
exported `__version__` before.

Added `duckduckgo-search` to `pyproject.toml` (A3's dependency; resolved to
`8.1.1` — verified `DDGS().text(keywords, max_results=...)` is still the
correct signature at that version).

## Open items — not blocking, but not forgotten

- **Prober cost accounting is incomplete.** `mutations.py`'s up-to-3
  fast-tier generation calls per claim aren't reflected in the `Verification`
  the prober returns — only the entailment baseline + re-verify calls are
  summed (A2's own report flags this). Undercounts `runs.cost_usd`/
  `verifications.cost_usd` slightly; doesn't affect correctness. Fix by
  having `generate_mutations()` return cost data alongside `MutatedClaim`,
  or by having the orchestrator log it separately.
- **`attest.wrap()` LlamaIndex support is unverified.** The extraction logic
  happens to duck-type against LlamaIndex's `Response` shape too, but A5
  never tested it against a real LlamaIndex install. Treat as
  LangChain-only until someone verifies it.
- **`/evaluate` is a stub** (empty `results`, `status="not_implemented"`) —
  correct shape, waiting on A6's `bench/` harness. Wire it up the same
  lazy-import way as the verifier getters in `routes.py` once that lands.
- **No CORS configuration yet** for the dashboard's dev origin — A4 left
  this as a deliberate TODO in `attest/api/main.py`.
- **Auth is unenforced.** A5's SDK sends `Authorization: Bearer {api_key}`
  on every `POST /observe`; A4's routes never check for it. PLAN.md §5.2
  doesn't specify an auth contract either way — fine for the hackathon demo,
  but worth an explicit decision before this runs anywhere less trusted.
- **Demo corpus generation half is untested end-to-end.** A5 smoke-tested
  retrieval (ChromaDB) directly with no API key needed, but the
  `attest.llm.complete` generation step and the full `run_demo.py` path have
  not been run against a real `ANTHROPIC_API_KEY` yet. Do this once, for
  real, before the actual demo — see `demo/SEED_NOTES.md`.

## Current state

`uv run pytest` → 163 passed. `uv run ruff check .` → clean. `POST /observe`
now runs the real pipeline end-to-end (verifiers are no longer stubs).
`import attest; attest.init(...)` works.
