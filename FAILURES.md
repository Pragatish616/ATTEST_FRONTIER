# FAILURES

What broke, what was tried, what actually fixed it — so the same dead end isn't
walked twice. Append a new entry at the top of the relevant section every time a
non-trivial debugging session ends.

Entry format: **symptom** → what was tried → what fixed it → how to avoid it.

---

## Integration (A1–A5 parallel build)

### Prober constructor mismatch

**Symptom:** `routes.py::get_prober_verifier()` blew up constructing the prober.
A4 wrote the wiring in parallel with A2 and assumed a no-arg `ProberVerifier`
class.

**Cause:** A2 shipped `AdversarialProber(entailment_verifier: VerifierProtocol)`
— dependency-injected by design (see `DECISIONS.md`).

**Fix:** `AdversarialProber(entailment_verifier=get_entailment_verifier())` in
the wiring. Neither agent's file changed.

**Avoid:** when tracks build in parallel against a written contract, the contract
covers *data shapes*, not constructor signatures. Check the actual `__init__`
before wiring.

### Decomposer signature mismatch

**Symptom:** `orchestrator.run` calls `decompose(request, run_id)`; A1 shipped
`decompose(answer: str, query: str, *, run_id: UUID)`.

**Fix:** `get_decompose()` wraps it in a small adapter closure. Neither agent's
file was edited — that would have violated the file-ownership lanes.

**Avoid:** same as above. Adapters at the wiring layer beat cross-lane edits.

### `test_observe_returns_503_when_verifiers_not_implemented` started failing

**Symptom:** the test passed for two days, then broke with no change to it.

**Cause:** it asserted the 503 path *implicitly*, by relying on A1/A2/A3's
modules still being empty stubs. Once those landed, the 503 stopped happening.

**Fix:** renamed to `..._when_a_dependency_is_not_ready` and now forces
`DependencyNotReady` explicitly via monkeypatch.

**Avoid:** never let a test's premise depend on unimplemented code. A test that
passes because something doesn't exist yet is a landmine with a date on it.

### `import attest; attest.init(...)` raised `AttributeError`

**Cause:** `attest/__init__.py` only exported `__version__`. PLAN.md §5.3's
documented usage was never actually importable.

**Fix:** added `from attest.sdk import Output, init, observe, trace, wrap`.

### `duckduckgo-search` missing from `pyproject.toml`

A3's fallback search provider was imported but never declared. Added, resolved to
`8.1.1`, and `DDGS().text(keywords, max_results=...)` verified as still the
correct signature at that version.

---

## Testing

### Monkeypatching `attest.llm._call_anthropic` does nothing

**Symptom:** tests patch the provider function by name, and the real provider
gets called anyway.

**Cause:** `_PROVIDER_CHAIN` captures those function objects **by reference at
module load**. Rebinding the module-level name afterwards doesn't touch the list.

**Fix:** patch the list itself:

```python
monkeypatch.setattr(llm, "_PROVIDER_CHAIN", [("anthropic", fake_call)])
# fake_call(prompt: str, model: str) -> tuple[str, int, int]
```

See `tests/test_contracts.py` for the full pattern, including the
schema/repair/retry paths and provider fallback.

---

## Repository / tooling

### Every tracked file showed as modified — 16,602 insertions for ~1,465 real lines

**Cause:** Windows checkout with CRLF against an LF repo. Real diffs became
unreadable and a targeted revert became impossible.

**Fix:** `.gitattributes` with `* text=auto eol=lf` plus explicit `binary` rules
for images and SQLite files, then a one-time renormalization commit.

**Avoid:** `.gitattributes` goes in on day one, before the first Windows clone.

### Git index left in a half-reset state

**Symptom:** `git status` showed the same files as both staged deletions (`D `)
and untracked (`??`) — an interrupted `git rm --cached` during the line-ending
renormalization.

**Fix:** resolved as part of the clean-history reset (see `DECISIONS.md`).

**Avoid:** don't renormalize line endings and stage new work in the same session.

---

## Open, not yet failed but known fragile

These are tracked in `HANDOFF.md` and repeated here because they are the most
likely sources of the next entry in this file.

- **Prober cost accounting undercounts.** `mutations.py`'s up-to-3 fast-tier
  generation calls per claim aren't summed into the returned `Verification`.
  Affects reported `runs.cost_usd`, not correctness.
- **`attest.wrap()`'s LlamaIndex path is unverified.** The extraction logic
  duck-types against LlamaIndex's `Response` shape but was never run against a
  real install. Treat as LangChain-only.
- **The demo's generation half has not been run end to end with a real key.**
  Retrieval was smoke-tested; `attest.llm.complete` inside `run_demo.py` was not.
  Do this once, for real, before the demo — `demo/SEED_NOTES.md`.
- **`last_probes` breaks if the claim loop is parallelized.** See `DECISIONS.md`.
