"""Edge hardening for the public API: bearer auth, rate limiting, request-size
caps, and security headers.

WHY THIS FILE EXISTS
--------------------
Every endpoint in `routes.py` was reachable unauthenticated, and
`AttestConfig.budget_usd` is supplied by the caller. Together those two facts
mean anyone who learns the deployed URL can POST `/v1/observe` with a long
answer, `sample_rate=1.0`, and a large `budget_usd`, and spend our Groq /
Gemini / Anthropic credits without limit. That is the primary threat against
this service — not data theft. The controls here are ordered accordingly.

Note that the SDK has *always* sent `Authorization: Bearer <api_key>`
(`attest/sdk/__init__.py:237`). The server simply never read it. So this is
not a new contract — it completes one that was already half-implemented, and
`api_key` is already part of the frozen §5.3 SDK surface.

FROZEN-CONTRACT SAFETY
----------------------
Auth is OFF unless `ATTEST_API_KEY` is set. With it unset the middleware is a
no-op, so the dashboard track and the Opal demo agent keep working against
PLAN.md §5.2 exactly as before the freeze. Turning it on is a deployment
decision that requires telling both of those tracks first — see the note in
`create_app()`.

WHY PURE ASGI AND NOT BaseHTTPMiddleware
----------------------------------------
Starlette's `BaseHTTPMiddleware` (and the `@app.middleware("http")` sugar built
on it) wraps responses in a way that buffers and can stall
`EventSourceResponse`. Live SSE streaming is this project's whole demo, so
these are raw ASGI middlewares: they inspect the scope, optionally short-circuit
with their own response, and otherwise pass the untouched send/receive channels
straight through. Streaming is unaffected.
"""

from __future__ import annotations

import json
import secrets
import time
from collections import deque
from collections.abc import Awaitable, Callable

import structlog

logger = structlog.get_logger(__name__)

ASGIApp = Callable[[dict, Callable[[], Awaitable[dict]], Callable[[dict], Awaitable[None]]], Awaitable[None]]

# Open endpoints. `/v1/health` must stay reachable unauthenticated or the
# Railway/Render healthcheck marks the service unhealthy and restart-loops it;
# it returns only `{"ok": true, "version": ...}`, no run data. The docs paths
# expose the API shape only, which is already public in PLAN.md §5.2.
_PUBLIC_PATHS = frozenset(
    {"/v1/health", "/docs", "/redoc", "/openapi.json", "/docs/oauth2-redirect"}
)

# Endpoints whose per-request cost is dominated by LLM calls, not I/O.
# Matched by suffix so the router's /v1 prefix does not have to be repeated.
_EXPENSIVE_PATHS_SUFFIX = ("/demo/query", "/evaluate")


def client_ip(scope: dict, trusted_proxy_hops: int) -> str:
    """Best-available caller IP, correct behind a reverse proxy.

    `scope["client"]` is the *proxy's* address once this runs on Railway or
    Render, so keying a rate limit on it collapses every caller in the world
    into one bucket — the per-client limit then protects nothing. This was a
    real bug in the first version of `RateLimitMiddleware`.

    `X-Forwarded-For` is `client, proxy1, proxy2, ...` where each hop *appends*
    the address it saw. A caller can therefore forge leading entries — sending
    `X-Forwarded-For: 1.2.3.4` just yields `1.2.3.4, <real caller>` after the
    platform's proxy appends. The only trustworthy entries are the rightmost
    ones, written by infrastructure we control, so we index from the right by
    the number of proxies we actually sit behind.

    `trusted_proxy_hops` must therefore match reality: 0 for direct/local (use
    the socket address, ignore the header entirely), 1 behind Railway or
    Render's single edge proxy. Setting it too high lets a caller choose their
    own bucket and bypass the limit; too low buckets everyone together.
    """
    if trusted_proxy_hops <= 0:
        return (scope.get("client") or ("unknown", 0))[0]

    for name, value in scope.get("headers", []):
        if name == b"x-forwarded-for":
            hops = [part.strip() for part in value.decode("latin-1").split(",") if part.strip()]
            if len(hops) >= trusted_proxy_hops:
                return hops[-trusted_proxy_hops]
            # Fewer hops than configured means the request did not traverse the
            # expected chain. Fall back to the socket rather than trusting a
            # header we cannot account for.
            break

    return (scope.get("client") or ("unknown", 0))[0]


async def _send_json(send: Callable[[dict], Awaitable[None]], status: int, body: dict) -> None:
    payload = json.dumps(body).encode()
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(payload)).encode()),
            ],
        }
    )
    await send({"type": "http.response.body", "body": payload})


class BearerAuthMiddleware:
    """Validate `Authorization: Bearer <key>` against a single shared secret.

    A shared secret rather than per-tenant keys is a deliberate scope decision:
    ATTEST has no user model, no registration, and no sessions, so there is
    nothing for a JWT's `sub`/`aud` claims to mean and no password to hash. A
    single high-entropy deployment key is the honest control for "keep strangers
    off our LLM billing". If ATTEST ever grows tenants, this is the seam to
    replace.

    Comparison uses `secrets.compare_digest`, not `==`, so a network observer
    cannot recover the key byte-by-byte from response-time differences.
    """

    def __init__(self, app: ASGIApp, api_key: str | None) -> None:
        self.app = app
        self.api_key = api_key

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http" or not self.api_key:
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        # CORS preflights carry no Authorization header by design; rejecting
        # them would surface as an opaque browser CORS error rather than a 401.
        if path in _PUBLIC_PATHS or scope.get("method") == "OPTIONS":
            await self.app(scope, receive, send)
            return

        header = ""
        for name, value in scope.get("headers", []):
            if name == b"authorization":
                header = value.decode("latin-1")
                break

        scheme, _, token = header.partition(" ")
        if scheme.lower() != "bearer" or not secrets.compare_digest(token, self.api_key):
            # Deliberately generic: never reveal whether the key was absent,
            # malformed, or merely wrong.
            logger.warning(
                "auth_rejected",
                path=path,
                client=(scope.get("client") or ("unknown", 0))[0],
            )
            await _send_json(send, 401, {"detail": "Unauthorized."})
            return

        await self.app(scope, receive, send)


class MaxBodySizeMiddleware:
    """Reject oversized request bodies before they are read into memory.

    `ObserveRequest.answer` and `retrieved_chunks` have no length bound, and
    claim count scales with answer length — so a multi-megabyte answer becomes a
    very large number of LLM calls. Capping bytes at the edge is the cheapest
    place to stop that.
    """

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        for name, value in scope.get("headers", []):
            if name == b"content-length":
                try:
                    declared = int(value)
                except ValueError:
                    break
                if declared > self.max_bytes:
                    logger.warning(
                        "request_too_large", declared_bytes=declared, limit=self.max_bytes
                    )
                    await _send_json(
                        send,
                        413,
                        {"detail": f"Request body exceeds {self.max_bytes} bytes."},
                    )
                    return
                break

        # A chunked request sends no Content-Length. Wrap receive so the body is
        # counted as it streams and the connection is cut if it overruns —
        # otherwise the cap is trivially bypassed by omitting the header.
        received = 0

        async def counting_receive() -> dict:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise ValueError(
                        f"request body exceeded {self.max_bytes} bytes while streaming"
                    )
            return message

        await self.app(scope, counting_receive, send)


class RateLimitMiddleware:
    """Fixed-window-per-client rate limit, in process memory.

    In-memory state is correct here *only* because the service is pinned to a
    single replica for the SSE event bus (see `create_app`). If that ever
    changes, this must move to Redis at the same time — and so must the bus.

    Write endpoints get a much tighter budget than reads: every POST /observe
    costs real money, whereas GET /runs costs a Supabase query. The SSE stream
    path is exempt because the dashboard holds one long-lived connection per
    run; counting it would throttle normal use.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        read_per_minute: int,
        write_per_minute: int,
        expensive_per_minute: int,
        trusted_proxy_hops: int = 0,
    ) -> None:
        self.app = app
        self.read_per_minute = read_per_minute
        self.write_per_minute = write_per_minute
        self.expensive_per_minute = expensive_per_minute
        self.trusted_proxy_hops = trusted_proxy_hops
        self._hits: dict[tuple[str, str], deque[float]] = {}

    def _allow(self, client: str, bucket: str, limit: int) -> bool:
        now = time.monotonic()
        window = self._hits.setdefault((client, bucket), deque())
        while window and now - window[0] > 60.0:
            window.popleft()
        if len(window) >= limit:
            return False
        window.append(now)
        return True

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        method = scope.get("method", "GET")

        if path in _PUBLIC_PATHS or method == "OPTIONS" or path.endswith("/stream"):
            await self.app(scope, receive, send)
            return

        client = client_ip(scope, self.trusted_proxy_hops)

        # Three tiers, priced by what a single request actually costs us.
        #
        # /demo/query runs the whole reference RAG pipeline *and* the
        # three-verifier fan-out, and /evaluate runs a benchmark sweep — both are
        # multiples more expensive than /observe, which is itself far more
        # expensive than a Supabase read. A single write limit would price the
        # cheapest and most costly endpoints identically.
        if _EXPENSIVE_PATHS_SUFFIX and path.endswith(_EXPENSIVE_PATHS_SUFFIX):
            bucket, limit = "expensive", self.expensive_per_minute
        elif method in {"POST", "PUT", "PATCH", "DELETE"}:
            bucket, limit = "write", self.write_per_minute
        else:
            bucket, limit = "read", self.read_per_minute

        if not self._allow(client, bucket, limit):
            logger.warning("rate_limited", client=client, bucket=bucket, limit=limit)
            await _send_json(
                send,
                429,
                {"detail": f"Rate limit exceeded ({limit} {bucket} requests/minute)."},
            )
            return

        await self.app(scope, receive, send)


class SecurityHeadersMiddleware:
    """Attach baseline hardening headers to every response.

    No CSP: this app serves JSON, and the dashboard is a separate static
    deployment on Vercel whose CSP belongs in *its* host config, not here. A CSP
    on an API response protects nothing and invites cargo-culting.
    """

    def __init__(self, app: ASGIApp, *, hsts: bool) -> None:
        self.app = app
        self.hsts = hsts

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_headers(message: dict) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.extend(
                    [
                        (b"x-content-type-options", b"nosniff"),
                        (b"x-frame-options", b"DENY"),
                        (b"referrer-policy", b"no-referrer"),
                        (b"cross-origin-opener-policy", b"same-origin"),
                    ]
                )
                if self.hsts:
                    headers.append(
                        (b"strict-transport-security", b"max-age=31536000; includeSubDomains")
                    )
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, send_with_headers)


__all__ = [
    "BearerAuthMiddleware",
    "MaxBodySizeMiddleware",
    "RateLimitMiddleware",
    "SecurityHeadersMiddleware",
]
