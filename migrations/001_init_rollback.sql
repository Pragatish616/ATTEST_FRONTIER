-- Rollback for 001_init.sql. Drops in reverse dependency order.

alter publication supabase_realtime drop table runs, claims, verifications, probes;

drop table if exists probes;
drop table if exists verifications;
drop table if exists claims;
drop table if exists retrieved_chunks;
drop table if exists runs;
