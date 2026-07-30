# CLAUDE.md — ATTEST backend

Read `PLAN.md` before doing anything. It is the single source of truth. §5 contracts are FROZEN.

## What you are building

The backend for ATTEST: an adversarial grounding layer that decomposes AI outputs into atomic claims and verifies each one three independent ways. You own **only** the Python backend, SDK, and benchmark. The dashboard is built separately by another team against the frozen API contract. The demo agent is built separately in Google Opal. Do not build UI. Do not build the demo agent.

## Hard rules

1. **Contracts are frozen.** The Supabase schema (§5.1), REST/SSE API (§5.2), SDK surface (§5.3), and verdict taxonomy (§3) may not change. Another team is building against them right now. If you believe a contract is wrong, STOP, write the proposed change to `CONTRACT_CHANGE_REQUEST.md`, and continue with the existing contract.
2. **Stay in your lane.** Each agent writes only the files listed in its row of PLAN.md §7. If you need a change in another agent's file, write the request to `HANDOFF.md` instead of editing.
3. **The SDK never raises into the host pipeline.** Every SDK path is wrapped; on failure it logs and returns control. A grounding checker that crashes the app it observes is worse than useless.
4. **Verifiers return structured objects, never free text.** Every verifier returns a `Verification` Pydantic model. Parse LLM output as JSON with a repair fallback; never regex prose.
5. **`temperature=0` for all verification calls.** Non-determinism in a judge is a bug.
6. **Every LLM call is logged** with model, tokens, latency, and cost into `verifications`. Cost visibility is a demo feature, not an afterthought.
7. **No mock data in the main path.** Fixtures live in `tests/fixtures/` only. A demo backed by hardcoded verdicts loses the hackathon on the first question.

## Repo layout

```
attest/
  config.py          # pydantic-settings, fails loud at boot
  models.py          # ALL shared Pydantic models (A0 owns)
  llm.py             # provider router: anthropic | groq | gemini
  search.py          # web search client
  store.py           # Supabase read/write
  orchestrator.py    # fan-out + budget + sampling
  reconciler.py      # verdict resolution + disagreement score
  verifiers/
    decomposer.py entailment.py prober.py mutations.py independent.py
  api/
    main.py routes.py stream.py
  sdk/
    __init__.py decorator.py wrappers.py
demo/                # seeded corpus + reference RAG pipeline
bench/               # eval harness + ablation runner
migrations/          # SQL
tests/
```

## Conventions

- Python 3.11, `uv` for deps, `ruff` for lint, full type hints.
- All I/O is `async`. Fan-out uses `asyncio.gather(..., return_exceptions=True)` — one dead verifier must not kill a run.
- Structured logging (`structlog`), JSON output, `run_id` on every line.
- Tests: `pytest`, `pytest-asyncio`. Every verifier needs a unit test against a fixture that runs without network (mock the LLM).
- Secrets in `.env`, never committed. `.env.example` must stay current.

## Prompt-engineering rules for the verifiers

- Give the model the claim and the context, and ask for a verdict from the frozen taxonomy plus a rationale plus evidence spans. Nothing else.
- Force `UNVERIFIABLE` for subjective, predictive, or opinion claims. Precision matters more than coverage.
- Ask for evidence **character spans into the provided chunk**, not quoted text. Spans are checkable; quotes hallucinate.
- Never let a verifier see another verifier's output. Independence is the whole design.

## Working style

- Commit after every green test. Small commits, present-tense messages.
- Before starting a task, restate in one line which files you will touch and confirm they are in your lane.
- When a task is done, append a 3-line summary to `PROGRESS.md`: what shipped, what's stubbed, what the next agent needs.
- If you are more than 30 minutes into a component with nothing runnable, stub it behind the interface and move on. Shipping beats completeness.
