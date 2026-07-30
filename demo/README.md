# Demo corpus + reference RAG pipeline

Owned by A5 (SDK + Demo corpus).

A ~30-document fictional consumer-tech corpus ("Northwind Devices" —
earbuds, smartwatches, a home hub, and support apps) with a small ChromaDB
+ `attest.llm` RAG pipeline over it, seeded with one stale doc and one
subtly-unsupported claim (PLAN.md §8, hour 14-18 gate).

## Files

- `build_corpus.py` — materializes `corpus/*.md` (run once, or after
  editing the `DOCS` dict in this file).
- `corpus/` — the 30 generated markdown documents (generated artifact, not
  hand-edited directly — edit `build_corpus.py` and re-run).
- `rag_pipeline.py` — retrieval (ChromaDB, bundled default embedding
  function) + generation (`attest.llm.complete`, tier `"fast"`).
- `run_demo.py` — end-to-end CLI: retrieve, generate, submit to ATTEST via
  the SDK, report back a run id/URL.
- `SEED_NOTES.md` — exactly which doc + query is seeded for which verdict,
  and why (`STALE` and `UNSUPPORTED`/`FRAGILE`).

## Quickstart

```bash
uv run python demo/build_corpus.py     # once, populates demo/corpus/
uv run python demo/run_demo.py         # runs both seeded queries end-to-end
```

Requires a real `ANTHROPIC_API_KEY` in `.env` (generation step) and the
ATTEST API running at `--api-url` (default `http://localhost:8000`) for the
submission step. `run_demo.py` fails loudly with a plain-English message
(not a stack trace) if either isn't available — see its docstring.
