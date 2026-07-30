# Demo seed notes

Two documents in `demo/corpus/` (of 30, built by `demo/build_corpus.py`) are
deliberately seeded to produce specific verdicts when run through the full
ATTEST pipeline (decomposer → entailment / prober / independent →
reconciler). This doc pins down exactly which doc, which query, and which
verdict, so the demo is reproducible under the stress of live judging
(PLAN.md §8, §9).

Both seeds concern the fictional company **Northwind Devices** (earbuds,
smartwatches, a home hub, and support apps) that the whole corpus is built
around — see `demo/build_corpus.py` for the full doc list.

---

## Seed (a): `STALE`

| | |
|---|---|
| **Doc** | `demo/corpus/19-kb-supportbot-desktop-requirements.md` (`source_id: doc-19`) |
| **Query** | `If I install SupportBot Desktop on Windows 10, will it keep getting Microsoft security updates?` |
| **Expected verdict** | `STALE` |

**The claim in the doc:** doc-19's "Windows Support Notes" section states:

> "Windows 10 continues to receive regular security patches directly from
> Microsoft, so it remains a fully supported, secure choice for running
> SupportBot Desktop... No action is needed for Windows 10 users beyond
> staying current on Windows Update."

**Why this triggers `STALE`, not `GROUNDED` or `CONTRADICTED`:**
- The retrieved chunk (doc-19) *does* support the claim "Windows 10 still
  gets Microsoft security updates" — so the **entailment** verifier should
  return `GROUNDED` (the answer is faithful to the retrieved context).
- But this is factually stale relative to current reality: **Microsoft
  ended standard support for Windows 10 on 2025-10-14** — a real,
  fixed, one-time historical event that is well documented and
  web-verifiable (not a moving target like "the current version of X"),
  so it stays valid for the demo regardless of exactly when it's run,
  as long as that's after October 2025.
- The **independent retriever** re-searches the open web, finds Windows 10's
  actual end-of-support date, and disagrees with the (internally
  consistent, entailed) retrieved context — which is exactly PLAN.md §3's
  definition of `STALE`: "entailed by context, but independent retrieval
  disagrees with the context."
- This is why `VerifyContext.prior_entailment` exists (see
  `CONTRACT_CHANGE_REQUEST.md`): the independent verifier needs to know
  entailment already said `GROUNDED` to tell `STALE` apart from plain
  `CONTRADICTED`.

**Retrieval sanity-checked:** `demo/rag_pipeline.retrieve(...)` for this
query returns `doc-19` as the top result (score ≈0.22, next-closest doc
scores 0.0) — see the retrieval smoke test run during development.

---

## Seed (b): `UNSUPPORTED` (and a good `FRAGILE` probe target)

| | |
|---|---|
| **Doc** | `demo/corpus/01-spec-aria-earbuds-gen2.md` (`source_id: doc-01`) — plus the *absence* of the fact anywhere else in the corpus |
| **Query** | `Does the Northwind Aria Earbuds charging case support Qi wireless charging?` |
| **Expected verdict** | `UNSUPPORTED` |

**What's actually in the corpus:** doc-01 (Aria Earbuds 2nd Gen) states the
case "charges via USB-C only" — wired charging, no mention of wireless.
**No document anywhere in the 30-doc corpus states that any Aria Earbuds
case supports wireless/Qi charging** — this was deliberately scrubbed from
every doc that could plausibly imply it:
- `doc-02` (Aria Earbuds Pro) charges via USB-C only (no Qi mention).
- `doc-06` (Zephyr Dock Mini) and `doc-10` (Drift Travel Charger) — both
  charging accessories that *do* support Qi wireless charging for the
  **Comet Watch** — charge Aria Earbuds cases via a plain USB-C port, not
  a Qi pad.

**Why this is a good hallucination trap (the "lazy entailment" test):** a
fluent LLM asked "does the earbuds case support Qi wireless charging?"
will very plausibly answer "Yes" from general world knowledge — wireless
charging cases are a common feature on premium true-wireless earbuds, so
this is exactly the kind of confident, plausible-sounding, *wrong* claim
ATTEST exists to catch. A **careless/shallow** entailment check that
retrieves doc-06 or doc-10 (both of which do talk about "Qi wireless
charging" in the same document, just for the *watch*, and about USB-C for
the earbuds) and pattern-matches on the phrase "wireless charging" being
merely *present* in a retrieved chunk — without checking which product it
attaches to — could be tempted to wave the claim through as grounded. A
careful entailment check must notice the retrieved chunk's "Qi wireless
charging" clause is scoped to Comet Watch, not the earbuds case, and
correctly return `UNSUPPORTED`.

**Why it's also a good `FRAGILE` probe target:** negating the claim
("the Aria Earbuds charging case does **not** support Qi wireless
charging") is actually the *true* statement here. A verifier that isn't
really reading the context carefully might return the same verdict
direction for both the original and negated claim (i.e. fail to flip when
it logically should), which is precisely what marks a claim `FRAGILE`
(PLAN.md §3) — this claim is a deliberately good stress test for the
prober's negation mutation, independent of whether the base verdict lands
as `UNSUPPORTED` or (if a verifier is too lenient) `GROUNDED`.

**Retrieval sanity-checked:** `demo/rag_pipeline.retrieve(...)` for this
query returns `doc-01` as the top result (score ≈0.19), with `doc-10` and
`doc-06` next (both USB-C-for-earbuds, Qi-for-watch — no wireless-earbuds
claim in any of them) and `doc-02` last. No retrieved chunk in the top-4
ever asserts Aria Earbuds support wireless charging.

---

## Notes for whoever wires this into the full pipeline (A1/A2/A3/A4)

- Both queries and their expected top-k retrieval were verified against
  the real `demo/rag_pipeline.py` (ChromaDB + its bundled default
  embedding function) during development — see the retrieval sanity checks
  above. The *generation* step (`attest.llm.complete`, tier `"fast"`) was
  not exercised end-to-end in this environment (no live `ANTHROPIC_API_KEY`
  configured here) — if the generator ever answers seed (a) or (b)
  differently than expected (e.g., correctly hedges on the wireless
  charging question), that's a property of the specific generator model in
  use on demo day, not of the corpus or retrieval — rerun
  `demo/run_demo.py` once with real credentials before the actual demo and
  confirm both answers still land on the intended hallucination/staleness
  before judging.
- `demo/run_demo.py` runs both seed queries (plus any ad-hoc query) through
  the pipeline and submits via the SDK; see that file for usage.
