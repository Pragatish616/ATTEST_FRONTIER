-- ATTEST — initial schema (PLAN.md §5.1, verbatim). FROZEN — do not edit.
-- Any needed change must go through CONTRACT_CHANGE_REQUEST.md.

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
