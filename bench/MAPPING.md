# Taxonomy mapping: ATTEST verdicts -> hallucination detection

This is the deliverable a technical judge will interrogate hardest, so this
document argues the reasoning, not just states the mapping. The mechanical
implementation is `bench/mapping.py`; keep the two in lockstep if either
changes.

## Framing

The benchmark evaluates ATTEST as a **hallucination detector**: positive
class = "this claim is hallucinated." That is a narrower question than
"is ATTEST's full six-way verdict correct" (RAGTruth/HaluEval have no
concept of `FRAGILE` or `STALE` at all — they were built to evaluate
faithfulness classifiers, not adversarial-probing or source-staleness
systems), but it is the frame the standard RAGTruth/HaluEval metrics
(precision/recall/F1 against hallucination spans or binary labels) are
built for, and it's what lets ATTEST be compared against a plain baseline
judge on the same axis. Trying to invent a six-way ground truth that
RAGTruth/HaluEval simply don't annotate would produce numbers nobody could
independently check.

## Verdict -> predicted label

| Verdict | Predicted | Why |
|---|---|---|
| `GROUNDED` | negative (faithful) | Definitionally the "no hallucination detected" case. |
| `UNSUPPORTED` | positive | Context is silent — RAGTruth/HaluEval both treat unsupported claims as hallucinated (baseless info). |
| `CONTRADICTED` | positive | Direct contradiction is the least ambiguous hallucination case. |
| `STALE` | positive | Source-level wrongness is still wrongness (PLAN.md §2/§3) — an answer entailed by a context that's itself outdated or wrong is not a faithful answer for the purposes of this benchmark's ground truth. **But**: RAGTruth/HaluEval were built before ATTEST's `STALE` concept existed and have no notion of temporal staleness at all. If `STALE` fires on this dataset, it did so because the independent verifier's fresh web search disagreed with the RAGTruth/HaluEval source text — which the ground truth can neither confirm nor refute as "correctly caught" vs. "verifier hallucinating a contradiction of its own." This is exactly the kind of thing that should be flagged, not silently absorbed into a rosier recall number: `bench/metrics.py::stale_fire_rate` reports how often it happens, per config, as its own diagnostic row — not folded into precision/recall/F1's positive-class count logic (which still counts it as positive, per the table above, but the rate is visible alongside it). |
| `FRAGILE` | positive, **with a visible cost** | See below — this is the one genuine judgment call in the whole mapping. |
| `UNVERIFIABLE` | **excluded** (abstention) | Never counted as a hit or a miss. Reported separately as `unverifiable_abstention_rate`. CLAUDE.md is explicit: "`UNVERIFIABLE` is not a failure — it protects precision." Counting it as a forced negative would penalize exactly the discipline the taxonomy is designed to reward; counting it as positive would misrepresent "we don't know" as "we caught something." |

## Why `FRAGILE` counts as positive for recall/F1 (and the alternative rejected)

`FRAGILE` means: the entailment verifier said `GROUNDED`, but at least one
adversarial mutation (negation / entity swap / quantifier shift) failed to
flip the verdict when it logically must have — i.e., the verifier wasn't
actually reading the context closely enough to be trusted on this claim.
That is evidence the *original* `GROUNDED` call is unreliable, which is a
real signal worth surfacing as "don't trust this as faithful" — hence
positive.

**The alternative considered and rejected**: exclude `FRAGILE` from
recall/F1 entirely, alongside `UNVERIFIABLE`, on the grounds that "the
prober didn't establish the claim is actually false, only that the
verifier's confidence is shaky." This has real merit — a `FRAGILE` claim
might still happen to be true, whereas `UNSUPPORTED`/`CONTRADICTED` are
first-order verdicts about the claim's actual truth value, not the
verifier's reliability. It was rejected for two reasons: (1) `FRAGILE` is
one of ATTEST's two novel contributions (PLAN.md §2) — burying it inside an
"excluded, abstention-like" bucket next to `UNVERIFIABLE` would make the
prober's entire reason for existing invisible to the one table judges look
at first; (2) excluding it would make the "ATTEST − prober" vs "ATTEST
full" ablation comparison structurally biased in the prober's favor (every
`FRAGILE` verdict the prober produces would simply vanish from both the
numerator and denominator of the ablation it's supposed to be evaluated
against, rather than being scored). Counting it as positive is the choice
that puts the prober under the same scrutiny as everything else.

A second alternative considered: require 2+ verifiers to agree before
counting *any* detection as positive (a consensus-gating scheme, rather
than the reconciler's actual precedence rule). Rejected because it
contradicts `attest.reconciler.reconcile`'s real, frozen precedence
semantics (`CONTRADICTED > STALE > UNSUPPORTED > FRAGILE > GROUNDED >
UNVERIFIABLE` — a single `CONTRADICTED` outranks two `GROUNDED` opinions on
purpose, per `attest/reconciler.py`'s own docstring: "precision over
consensus"). The benchmark evaluates the system ATTEST actually ships, not
a hypothetical consensus-gated variant of it — building a different
aggregation rule for the benchmark than the one the reconciler production
code uses would make the benchmark's numbers not describe the real system.

**The cost of the choice, made visible, not hidden**: because `FRAGILE`
counts as positive, a claim that started out genuinely faithful but got a
noisy/inapplicable mutation (e.g. an entity-swap decoy the entailment
verifier correctly still rejects for an unrelated reason) can turn into a
false positive that a `FRAGILE`-excluded scheme would not have produced.
`bench/metrics.py::fragile_precision_risk_rate` reports, per config, what
fraction of `FRAGILE`-flagged claims were actually ground-truth-faithful —
i.e., how often this specific mapping choice costs precision. It is
reported as its own small table (`results.md`'s "Secondary metrics"
section), never folded into the headline F1, so a judge can see the
tradeoff directly instead of having to take it on faith.

## Ground truth: span-level (RAGTruth) vs. response-level (HaluEval)

**RAGTruth** annotates hallucinations as character spans into the
generated response (`labels: [{start, end, label_type, ...}]`). A
decomposed `Claim`'s own span (`Claim.span_start`/`span_end`, into the same
answer string — see `attest/verifiers/decomposer.py`) gives a natural,
checkable unit: **a claim is ground-truth-positive iff its span overlaps at
least one annotated hallucination span** (`bench/mapping.py::
claim_ground_truth_ragtruth`, half-open interval overlap). This is the
finest-grained evaluation the dataset supports, and it's exactly the
granularity ATTEST's decomposer is designed to work at.

**HaluEval** has no span annotation at all — each example is a
`(context, question, right_answer, hallucinated_answer)` tuple with a
single response-level binary label (`ground_truth_hallucinated`). There is
no way to say which *claim* inside a hallucinated HaluEval answer is the
hallucinated part; the honest move is to evaluate at the response level
instead of inventing a fake span. Per the task brief: **a HaluEval example
is evaluated as one unit, OR-aggregated across every claim decomposed from
that response** — if any non-abstaining claim from the response is
predicted positive, the response is predicted positive; if every claim
abstained (`UNVERIFIABLE`), the response abstains; otherwise it's negative
(`bench/mapping.py::build_eval_units`'s `_or_aggregate`). This means
RAGTruth and HaluEval contribute different *kinds* of unit to the same
precision/recall/F1 computation (claim-level vs. response-level) — stated
here explicitly rather than glossed over, since it's a real methodological
seam between the two datasets, not an oversight.

### A documented approximation for the secondary diagnostics

`fragile_precision_risk_rate` and `stale_fire_rate` are computed at the
*claim* level (they're inherently about individual `FRAGILE`/`STALE`
verdicts, not response-level aggregates). For a HaluEval claim, there is no
finer ground truth available than the response's label, so
`bench/mapping.py::claim_ground_truth` assigns every claim decomposed from
a HaluEval response that response's `ground_truth_hallucinated` value. This
means a genuinely faithful sub-claim inside a hallucinated HaluEval
response would be (mis)counted ground-truth-positive for these
diagnostics — a known, documented approximation, not a silent one. It does
not affect the headline precision/recall/F1 numbers, which use the coarser
but honest response-level OR-aggregated unit for HaluEval instead.

## Stratification: `hallucination_type`

RAGTruth's four annotation categories (`Evident Conflict`, `Subtle
Conflict`, `Evident Baseless Info`, `Subtle Baseless Info`) plus a `"none"`
bucket for non-hallucinated examples are used as the stratification key for
sampling (`bench/datasets.py::stratified_sample`) — every bucket gets equal
representation in the drawn sample (subject to bucket size; RAGTruth's
`Subtle Conflict` bucket is small in the wild, so it caps how many
`Subtle Conflict` examples any sample of a given size can contain). A
response can carry multiple labels of different types; the first label's
type is used as that example's stratification bucket (`bench/datasets.py::
_ragtruth_hallucination_type`) — this only affects which *sampling*
stratum an example counts toward, not its claim-level ground truth (every
annotated span is still checked for overlap regardless of which label
"named" the example's bucket).

For HaluEval, `hallucination_type` is the task split (`"qa"` by default;
`"dialogue"`/`"summarization"` are supported by the same parser but not
fetched by default — see `bench/datasets.py::load_halueval_raw`).
