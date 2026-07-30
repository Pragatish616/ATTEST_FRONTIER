"""Validation for attest/api/security.py.

Each control is asserted to actually fire, not merely to be installed — an
untested middleware chain is indistinguishable from no middleware. The
`create_app()` calls patch `attest.config.settings` attributes first, because
`settings` is an import-time singleton (see tests/conftest.py) and `create_app`
reads it when called.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from attest.api.main import create_app
from attest.api.security import client_ip
from attest.config import settings

API_KEY = "test-deployment-key-do-not-reuse"


def _scope(xff: str | None = None, peer: str = "10.0.0.1") -> dict:
    headers = [(b"x-forwarded-for", xff.encode())] if xff is not None else []
    return {"type": "http", "client": (peer, 1234), "headers": headers}


# --- Caller identification behind a proxy --------------------------------


def test_client_ip_ignores_forwarded_header_when_not_behind_a_proxy():
    """hops=0 means we are reachable directly, so the header is untrusted input."""
    assert client_ip(_scope(xff="1.2.3.4"), trusted_proxy_hops=0) == "10.0.0.1"


def test_client_ip_reads_real_caller_behind_one_proxy():
    assert client_ip(_scope(xff="203.0.113.9"), trusted_proxy_hops=1) == "203.0.113.9"


def test_forged_forwarded_entries_cannot_choose_a_bucket():
    """A caller sending `X-Forwarded-For: 1.2.3.4` gets `1.2.3.4, <real>` once the
    platform proxy appends. Counting from the right must yield the real caller,
    or the rate limit is bypassable with one header."""
    assert client_ip(_scope(xff="1.2.3.4, 203.0.113.9"), trusted_proxy_hops=1) == "203.0.113.9"


def test_client_ip_falls_back_when_chain_is_shorter_than_configured():
    """Fewer hops than configured means the request skipped the expected chain;
    trust the socket, not a header we cannot account for."""
    assert client_ip(_scope(xff="", peer="10.0.0.7"), trusted_proxy_hops=2) == "10.0.0.7"


@pytest.fixture
def restore_settings():
    """Snapshot and restore the mutated settings fields."""
    fields = (
        "attest_api_key",
        "cors_allow_origins",
        "max_request_bytes",
        "rate_limit_read_per_minute",
        "rate_limit_write_per_minute",
        "rate_limit_expensive_per_minute",
        "trusted_proxy_hops",
        "app_env",
        "max_budget_usd",
    )
    saved = {name: getattr(settings, name) for name in fields}
    yield
    for name, value in saved.items():
        object.__setattr__(settings, name, value)


def _client(**overrides) -> TestClient:
    for name, value in overrides.items():
        object.__setattr__(settings, name, value)
    return TestClient(create_app())


# --- Authentication -------------------------------------------------------


def test_auth_disabled_by_default_keeps_the_frozen_contract_open(restore_settings):
    """With ATTEST_API_KEY unset the middleware must be a no-op, or the
    dashboard and Opal tracks break against the frozen §5.2 contract."""
    client = _client(attest_api_key=None)
    assert client.get("/v1/health").status_code == 200


def test_missing_bearer_token_is_rejected(restore_settings):
    client = _client(attest_api_key=API_KEY)
    response = client.get("/v1/runs")
    assert response.status_code == 401


def test_wrong_bearer_token_is_rejected(restore_settings):
    client = _client(attest_api_key=API_KEY)
    response = client.get("/v1/runs", headers={"Authorization": "Bearer wrong-key"})
    assert response.status_code == 401


def test_auth_failure_message_leaks_nothing(restore_settings):
    """A 401 must not reveal whether the key was absent, malformed or wrong."""
    client = _client(attest_api_key=API_KEY)
    absent = client.get("/v1/runs")
    malformed = client.get("/v1/runs", headers={"Authorization": "Basic abc"})
    wrong = client.get("/v1/runs", headers={"Authorization": "Bearer nope"})

    bodies = {absent.json()["detail"], malformed.json()["detail"], wrong.json()["detail"]}
    assert bodies == {"Unauthorized."}
    for response in (absent, malformed, wrong):
        assert API_KEY not in response.text


def test_health_stays_public_so_the_platform_healthcheck_survives(restore_settings):
    """Railway/Render probe /v1/health with no credentials. If auth blocked it,
    the platform would mark the service unhealthy and restart-loop it."""
    client = _client(attest_api_key=API_KEY)
    assert client.get("/v1/health").status_code == 200


def test_valid_bearer_token_passes_through(restore_settings):
    client = _client(attest_api_key=API_KEY)
    # 401 is what we are ruling out; any non-401 means auth let the request
    # reach the handler (which then fails on the unreachable test Supabase).
    response = client.get("/v1/runs", headers={"Authorization": f"Bearer {API_KEY}"})
    assert response.status_code != 401


def test_preflight_is_not_blocked_by_auth(restore_settings):
    """Browsers never attach Authorization to a preflight; blocking it would
    surface as an opaque CORS error instead of a 401."""
    client = _client(attest_api_key=API_KEY, cors_allow_origins="https://dash.vercel.app")
    response = client.options(
        "/v1/runs",
        headers={
            "Origin": "https://dash.vercel.app",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.status_code == 200


# --- CORS -----------------------------------------------------------------


def test_cors_allowlist_admits_only_configured_origins(restore_settings):
    client = _client(
        attest_api_key=None,
        cors_allow_origins="https://dash.vercel.app, http://localhost:5173",
    )
    allowed = client.options(
        "/v1/runs",
        headers={
            "Origin": "https://dash.vercel.app",
            "Access-Control-Request-Method": "GET",
        },
    )
    blocked = client.options(
        "/v1/runs",
        headers={
            "Origin": "https://evil.example",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert allowed.headers["access-control-allow-origin"] == "https://dash.vercel.app"
    assert "access-control-allow-origin" not in blocked.headers


def test_wildcard_cors_does_not_also_claim_credentials(restore_settings):
    """`*` plus Allow-Credentials: true is spec-invalid; browsers reject it and
    the failure looks like a backend outage."""
    client = _client(attest_api_key=None, cors_allow_origins="*")
    response = client.options(
        "/v1/runs",
        headers={"Origin": "https://anything.example", "Access-Control-Request-Method": "GET"},
    )
    assert response.headers.get("access-control-allow-credentials") != "true"


# --- Request size cap -----------------------------------------------------


def test_oversized_body_is_rejected_with_413(restore_settings):
    client = _client(attest_api_key=None, max_request_bytes=1024)
    response = client.post("/v1/observe", content=b"x" * 4096)
    assert response.status_code == 413


def test_normal_sized_body_is_not_rejected(restore_settings):
    client = _client(attest_api_key=None, max_request_bytes=512 * 1024)
    response = client.post("/v1/observe", json={"bad": "payload"})
    # 422 from validation is fine — the point is that the cap did not fire.
    assert response.status_code != 413


# --- Rate limiting --------------------------------------------------------


def test_read_rate_limit_actually_triggers(restore_settings):
    client = _client(attest_api_key=None, rate_limit_read_per_minute=3)
    statuses = [client.get("/v1/runs").status_code for _ in range(5)]
    assert 429 in statuses
    assert statuses.count(429) == 2


def test_write_rate_limit_is_tighter_than_read(restore_settings):
    """POST /observe spends real LLM credit; GET /runs spends a DB query."""
    assert settings.rate_limit_write_per_minute < settings.rate_limit_read_per_minute


def test_expensive_endpoints_get_their_own_tighter_bucket(restore_settings):
    """/demo/query costs a full RAG generation plus a three-verifier fan-out, so
    it must not share the ordinary write allowance."""
    client = _client(
        attest_api_key=None,
        rate_limit_write_per_minute=50,
        rate_limit_expensive_per_minute=2,
    )
    statuses = [client.post("/v1/demo/query", json={"query": "x"}).status_code for _ in range(4)]
    assert statuses.count(429) == 2


def test_expensive_bucket_is_the_tightest_by_default(restore_settings):
    assert (
        settings.rate_limit_expensive_per_minute
        < settings.rate_limit_write_per_minute
        < settings.rate_limit_read_per_minute
    )


def test_health_is_never_rate_limited(restore_settings):
    client = _client(attest_api_key=None, rate_limit_read_per_minute=2)
    statuses = [client.get("/v1/health").status_code for _ in range(6)]
    assert statuses == [200] * 6


# --- Security headers -----------------------------------------------------


def test_security_headers_present_on_every_response(restore_settings):
    client = _client(attest_api_key=None, app_env="dev")
    headers = client.get("/v1/health").headers
    assert headers["x-content-type-options"] == "nosniff"
    assert headers["x-frame-options"] == "DENY"
    assert headers["referrer-policy"] == "no-referrer"


def test_hsts_only_in_production(restore_settings):
    dev = _client(attest_api_key=None, app_env="dev")
    assert "strict-transport-security" not in dev.get("/v1/health").headers

    prod = _client(attest_api_key=None, app_env="prod")
    assert "strict-transport-security" in prod.get("/v1/health").headers
