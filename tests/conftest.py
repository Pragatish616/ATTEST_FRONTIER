"""Test environment bootstrap.

`attest.config` validates and constructs its `settings` singleton at import
time (by design — CLAUDE.md: fail loud at boot, never at request time). That
means required env vars must exist before anything imports `attest.config`
(directly or transitively). pytest imports conftest.py before collecting any
test module, so setting them here — rather than in a fixture — is what
makes that work.
"""

import os

os.environ.setdefault("SUPABASE_URL", "https://test.supabase.co")
os.environ.setdefault("SUPABASE_KEY", "test-supabase-key")
os.environ.setdefault("ANTHROPIC_API_KEY", "test-anthropic-key")
os.environ.setdefault("LLM_TRANSIENT_RETRIES", "0")  # unit tests must never sleep
