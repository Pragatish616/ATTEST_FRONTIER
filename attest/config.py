"""Application settings.

Loaded once, at import time, via pydantic-settings. A missing required key
must fail loudly here — at boot — never mid-request during a demo
(CLAUDE.md hard rule). Import `settings` from this module; do not construct
`Settings()` yourself outside of tests.
"""

from __future__ import annotations

from pydantic import ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Supabase (required — trace store) ---
    supabase_url: str
    supabase_key: str

    # --- LLM providers ---
    # At least ONE provider key is required; the rest are fallbacks tried in
    # `attest.llm._PROVIDER_CHAIN` order (anthropic -> groq -> gemini) on
    # error or rate limit. A provider with no key is skipped entirely, so
    # setting only GEMINI_API_KEY makes Gemini the effective primary.
    anthropic_api_key: str | None = None
    groq_api_key: str | None = None
    gemini_api_key: str | None = None

    # Transient-failure retry (attest.llm). Rate limits and capacity blips are
    # recoverable — the provider is telling us to wait, not that the request
    # was wrong. Retrying the same provider before advancing the chain is what
    # keeps a demo alive when only one provider key is configured. Set
    # LLM_TRANSIENT_RETRIES=0 to disable (tests do this).
    llm_transient_retries: int = 2
    llm_retry_base_delay_s: float = 2.0

    # Model tier mapping (attest.llm.complete(..., model_tier=...)).
    anthropic_fast_model: str = "claude-haiku-4-5-20251001"
    anthropic_judge_model: str = "claude-sonnet-5"
    groq_fast_model: str = "llama-3.1-8b-instant"
    groq_judge_model: str = "llama-3.3-70b-versatile"
    gemini_fast_model: str = "gemini-2.0-flash"
    gemini_judge_model: str = "gemini-2.0-flash"

    # --- Independent web search (A3) ---
    tavily_api_key: str | None = None
    # How long a cached web-search result stays usable. The cache exists so
    # demo day never depends on a live search API, but it must expire: the
    # independent verifier's whole job is spotting STALE claims, and an
    # immortal cache means it can never see that the world moved. 6h keeps
    # rehearsal runs fast while still re-checking within a build cycle.
    search_cache_ttl_seconds: float = 6 * 60 * 60

    # --- Local corpus store (A5 demo) ---
    chroma_persist_dir: str = "./data/chroma"

    # --- App ---
    app_env: str = "dev"
    log_level: str = "INFO"
    app_version: str = "0.1.0"

    # Comma-separated allowed browser origins for the dashboard (deployed
    # separately on Vercel). Kept as a string, not list[str], because
    # pydantic-settings requires JSON syntax for complex types in env vars and
    # `CORS_ALLOW_ORIGINS=https://a.vercel.app,https://b.vercel.app` is what a
    # deploy dashboard's env editor makes easy to get right at 3am.
    #
    # Default stays "*" so local development and the other tracks are never
    # blocked by CORS. Set it explicitly in production: "*" combined with
    # allow_credentials=True is rejected by browsers, so a credentialed
    # request against a wildcard deployment fails in a way that looks like a
    # backend outage.
    cors_allow_origins: str = "*"

    @property
    def cors_origins(self) -> list[str]:
        """`cors_allow_origins` split into the list CORSMiddleware wants."""
        return [origin.strip() for origin in self.cors_allow_origins.split(",") if origin.strip()]

    # --- API hardening (attest/api/security.py) ---
    #
    # Shared bearer token for the public API. Unset (the default) disables auth
    # entirely, which is what keeps local development, the dashboard track, and
    # the Opal demo agent working unchanged against the frozen §5.2 contract.
    # Set it in production. The SDK already sends this value as
    # `Authorization: Bearer <api_key>` via attest.init(api_key=...).
    attest_api_key: str | None = None

    # Server-side ceiling on the caller-supplied AttestConfig.budget_usd.
    #
    # budget_usd arrives in the request body with only a `> 0` validator, so
    # without this an unauthenticated caller sets it to 1e9 and spends our
    # provider credits until the keys are exhausted. Enforced in
    # api/routes.py::observe, not in the model, because AttestConfig is part of
    # the frozen SDK surface and its shape may not change.
    max_budget_usd: float = 0.25

    # Edge caps. 512 KiB is ~100x the largest realistic RAG answer + chunks
    # payload, so it rejects abuse without ever tripping on legitimate traffic.
    max_request_bytes: int = 512 * 1024
    rate_limit_read_per_minute: int = 120
    rate_limit_write_per_minute: int = 20
    # /demo/query and /evaluate each cost several LLM calls — a full RAG
    # generation plus the three-verifier fan-out, or a whole benchmark sweep.
    # They get their own, much tighter bucket.
    rate_limit_expensive_per_minute: int = 5

    # Number of reverse proxies in front of the app. MUST match the deployment:
    # 0 for local/direct, 1 behind Railway or Render's edge. Rate limiting reads
    # the caller IP from X-Forwarded-For counting this many hops from the right;
    # setting it too high lets a caller forge their own bucket, too low buckets
    # every caller in the world together. See api/security.py::client_ip.
    trusted_proxy_hops: int = 0

    @property
    def is_production(self) -> bool:
        return self.app_env.lower() in {"prod", "production"}


def _load_settings() -> Settings:
    try:
        loaded = Settings()
    except ValidationError as exc:
        missing = ", ".join(
            ".".join(str(part) for part in error["loc"])
            for error in exc.errors()
            if error["type"] == "missing"
        )
        raise RuntimeError(
            "ATTEST failed to start: missing required configuration "
            f"({missing or exc}). Copy .env.example to .env and fill in the "
            "required keys (SUPABASE_URL, SUPABASE_KEY, and at least one of "
            "ANTHROPIC_API_KEY / GROQ_API_KEY / GEMINI_API_KEY)."
        ) from exc

    # Fail loud at boot, not mid-demo: with no provider key at all, every
    # llm.complete() call would raise LLMError on the first verified claim.
    if not any(
        (loaded.anthropic_api_key, loaded.groq_api_key, loaded.gemini_api_key)
    ):
        raise RuntimeError(
            "ATTEST failed to start: no LLM provider configured. Set at least "
            "one of ANTHROPIC_API_KEY, GROQ_API_KEY, or GEMINI_API_KEY in .env."
        )
    return loaded


settings = _load_settings()

__all__ = ["Settings", "settings"]
