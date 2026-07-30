-- Rollback for 002_rls.sql. Restores the pre-migration state: RLS disabled and
-- the default PostgREST grants back in place.
--
-- Only run this if a track genuinely needs client-side Supabase access. It
-- reopens the trace tables to anyone holding the publishable key.

alter table runs              no force row level security;
alter table retrieved_chunks  no force row level security;
alter table claims            no force row level security;
alter table verifications     no force row level security;
alter table probes            no force row level security;

alter table runs              disable row level security;
alter table retrieved_chunks  disable row level security;
alter table claims            disable row level security;
alter table verifications     disable row level security;
alter table probes            disable row level security;

grant all on runs             to anon, authenticated;
grant all on retrieved_chunks to anon, authenticated;
grant all on claims           to anon, authenticated;
grant all on verifications    to anon, authenticated;
grant all on probes           to anon, authenticated;
