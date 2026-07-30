-- 002_rls.sql — Row Level Security for the five trace tables.
--
-- WHY: 001_init.sql creates runs / retrieved_chunks / claims / verifications /
-- probes with RLS disabled, which is Postgres's default and Supabase's most
-- common production mistake. Supabase exposes every table over PostgREST at
-- https://<project>.supabase.co/rest/v1/<table>. With RLS off and the anon key
-- in play, anyone holding that publishable key can read and write these tables
-- directly, completely bypassing the ATTEST API and every control in
-- attest/api/security.py.
--
-- Trace rows are not innocuous. Each one stores the observed system's query,
-- its answer, and the retrieved context that produced it — in any real
-- deployment that is customer data belonging to whoever installed the SDK.
--
-- POLICY MODEL: ATTEST has no end-user auth and no per-tenant column, so there
-- is no row-ownership predicate to express. The honest model is therefore
-- "backend only": enable RLS and add no permissive policy for anon or
-- authenticated. The service_role key bypasses RLS by design, so the backend
-- keeps full access while the publishable key gets nothing.
--
-- THIS MEANS: SUPABASE_KEY in the deploy environment must be the service_role
-- key, and it must never be shipped to a browser. The dashboard reads traces
-- through the ATTEST API, not through Supabase directly — if any track is
-- planning to query Supabase from client-side JS, this migration breaks it and
-- they need to hear about it before it is applied.
--
-- Additive only: no existing table, column, or constraint is altered, so the
-- frozen §5.1 schema is untouched.

alter table runs              enable row level security;
alter table retrieved_chunks  enable row level security;
alter table claims            enable row level security;
alter table verifications     enable row level security;
alter table probes            enable row level security;

-- Force RLS for table owners too. Without this, a connection that happens to
-- own the table silently bypasses the policies above and the lockdown is
-- theatre. service_role still bypasses RLS at the role level, which is the
-- intended backend path.
alter table runs              force row level security;
alter table retrieved_chunks  force row level security;
alter table claims            force row level security;
alter table verifications     force row level security;
alter table probes            force row level security;

-- Belt and braces: revoke the table grants PostgREST relies on, so the anon and
-- authenticated roles cannot reach these tables even if a permissive policy is
-- added later by accident.
revoke all on runs             from anon, authenticated;
revoke all on retrieved_chunks from anon, authenticated;
revoke all on claims           from anon, authenticated;
revoke all on verifications    from anon, authenticated;
revoke all on probes           from anon, authenticated;
