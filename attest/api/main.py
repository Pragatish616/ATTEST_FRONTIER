"""FastAPI app entrypoint. Mounts `routes.py` and `stream.py`.

Run with `uvicorn attest.api.main:app`. All dependency wiring for
components owned by other in-progress agents (decomposer, verifiers) lives
in `routes.py`, lazily — this module just assembles the app so it stays
importable (and `/health` stays reachable) regardless of what has or hasn't
landed elsewhere in the repo yet.
"""

from __future__ import annotations

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from attest.api import routes, stream
from attest.api.security import (
    BearerAuthMiddleware,
    MaxBodySizeMiddleware,
    RateLimitMiddleware,
    SecurityHeadersMiddleware,
)
from attest.config import settings

logger = structlog.get_logger(__name__)


def create_app() -> FastAPI:
    """Build the ATTEST FastAPI app.

    DEPLOYMENT CONSTRAINT: run this single-process (`uvicorn
    attest.api.main:app` with the default one worker). The SSE event bus in
    `attest.api.stream` is in-process memory, so `--workers > 1` or multiple
    replicas silently break live streaming — see `EventBus`'s docstring.
    """
    app = FastAPI(
        title="ATTEST",
        description="Adversarial grounding layer for agentic systems.",
        version=settings.app_version,
    )

    # Scoped via CORS_ALLOW_ORIGINS (comma-separated); defaults to "*" so local
    # development and the dashboard/Opal tracks are never blocked. In production
    # set it to the dashboard's Vercel origin.
    #
    # allow_credentials is tied to the wildcard check on purpose: browsers
    # reject `Access-Control-Allow-Origin: *` together with
    # `Allow-Credentials: true`, and the resulting failure looks like the API is
    # down rather than misconfigured. Wildcard => no credentials.
    allowed_origins = settings.cors_origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials="*" not in allowed_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Hardening stack. `add_middleware` builds the chain outward-in, so the LAST
    # one added runs FIRST on the way in. Intended inbound order is therefore
    # written here in reverse: headers -> rate limit -> body cap -> auth.
    #
    # Auth deliberately runs innermost of the four so that a flood of
    # unauthenticated requests is rate-limited before it can generate log noise,
    # and an oversized body is rejected before anything reads it.
    #
    # All four are pure-ASGI (see security.py) because BaseHTTPMiddleware can
    # stall EventSourceResponse, and live SSE streaming is the demo.
    #
    # NOTE FOR THE OTHER TRACKS: with ATTEST_API_KEY unset this is a no-op and
    # the §5.2 contract is unchanged. Setting it in production means the
    # dashboard and the Opal agent must send `Authorization: Bearer <key>` on
    # every call except /v1/health. Tell both tracks before enabling it.
    app.add_middleware(BearerAuthMiddleware, api_key=settings.attest_api_key)
    app.add_middleware(MaxBodySizeMiddleware, max_bytes=settings.max_request_bytes)
    app.add_middleware(
        RateLimitMiddleware,
        read_per_minute=settings.rate_limit_read_per_minute,
        write_per_minute=settings.rate_limit_write_per_minute,
        expensive_per_minute=settings.rate_limit_expensive_per_minute,
        trusted_proxy_hops=settings.trusted_proxy_hops,
    )
    # HSTS only in production: sending it from a local http:// dev server pins
    # the browser to https for localhost and breaks the next person's setup.
    app.add_middleware(SecurityHeadersMiddleware, hsts=settings.is_production)

    if settings.is_production and settings.trusted_proxy_hops == 0:
        logger.warning(
            "trusted_proxy_hops_is_zero_in_production",
            detail=(
                "TRUSTED_PROXY_HOPS=0 with APP_ENV=prod — rate limiting will key "
                "every caller off the platform proxy's IP, collapsing them into a "
                "single bucket. Set TRUSTED_PROXY_HOPS=1 behind Railway/Render."
            ),
        )

    if settings.is_production and not settings.attest_api_key:
        # Loud, but not fatal: refusing to boot would take the demo down for a
        # missing env var, which is worse than an open API you were warned about.
        logger.warning(
            "api_key_not_set_in_production",
            detail=(
                "ATTEST_API_KEY is unset with APP_ENV=prod — /v1/observe is "
                "reachable unauthenticated and will spend LLM credits for "
                "anyone who finds the URL."
            ),
        )

    app.include_router(routes.router, prefix="/v1")
    app.include_router(stream.router, prefix="/v1")

    return app


app = create_app()

__all__ = ["app", "create_app"]
