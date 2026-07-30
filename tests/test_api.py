"""API-layer tests: route shape, status codes, and the "clear startup
error" behavior when a verifier owned by another in-progress agent
(A1/A2/A3) hasn't landed with real content yet.

`/runs` and `/runs/{run_id}` are exercised against a monkeypatched
`get_store()` so nothing here touches a real Supabase project or the
network — the network-free `Store` behavior itself is covered by
tests/test_store.py.
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from attest.api import routes
from attest.api.main import app
from attest.models import RunDetail, RunSummary


@pytest.fixture
def client():
    return TestClient(app)


def test_health(client: TestClient):
    response = client.get("/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert "version" in body


def test_observe_returns_503_when_a_dependency_is_not_ready(client: TestClient, monkeypatch):
    """A1/A2/A3's verifiers are real now, so this can no longer rely on the
    modules actually being absent — it forces the same failure mode
    `_load_attr` raises when a dependency genuinely hasn't landed
    (ImportError/AttributeError -> DependencyNotReady), and asserts /observe
    turns that into a clear 503 rather than a bare 500/traceback.
    """

    def _not_ready():
        raise routes.DependencyNotReady("EntailmentVerifier is not available yet in ...")

    monkeypatch.setattr(routes, "get_entailment_verifier", _not_ready)

    response = client.post(
        "/v1/observe",
        json={"pipeline_name": "demo", "query": "q", "answer": "a"},
    )

    assert response.status_code == 503
    assert "not available yet" in response.json()["detail"]


def test_observe_returns_202_and_run_id_once_dependencies_are_wired(
    client: TestClient, monkeypatch
):
    """Simulates A1/A2/A3 having landed: monkeypatch the four getters so the
    route shape (202 + ObserveResponse) is verified independently of
    whether the real verifier modules exist yet.
    """

    class _StubVerifier:
        async def verify(self, claim, ctx):
            raise AssertionError("should not be called synchronously by the route")

    async def _decompose(request, run_id):
        return []

    class _StubStore:
        async def create_run(self, run_id, request):
            return None

        async def update_run_status(self, run_id, status, **fields):
            return None

        async def save_chunks(self, run_id, chunks):
            return []

        async def save_claim(self, claim):
            return None

        async def save_verification(self, verification):
            return None

        async def save_probe(self, probe):
            return None

        async def finalize_run(self, run_id, **fields):
            return None

    monkeypatch.setattr(routes, "get_store", lambda: _StubStore())
    monkeypatch.setattr(routes, "get_entailment_verifier", lambda: _StubVerifier())
    monkeypatch.setattr(routes, "get_prober_verifier", lambda: _StubVerifier())
    monkeypatch.setattr(routes, "get_independent_verifier", lambda: _StubVerifier())
    monkeypatch.setattr(routes, "get_decompose", lambda: _decompose)

    response = client.post(
        "/v1/observe",
        json={"pipeline_name": "demo", "query": "q", "answer": "a"},
    )

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "pending"
    assert "run_id" in body


def test_list_runs(client: TestClient, monkeypatch):
    summary = RunSummary(
        id=uuid4(),
        created_at="2026-01-01T00:00:00Z",
        pipeline_name="demo",
        query="q",
        answer="a",
    )

    class _StubStore:
        async def list_runs(self, *, limit, offset):
            return [summary]

    monkeypatch.setattr(routes, "get_store", lambda: _StubStore())

    response = client.get("/v1/runs?limit=10&offset=0")

    assert response.status_code == 200
    body = response.json()
    assert len(body["runs"]) == 1
    assert body["runs"][0]["pipeline_name"] == "demo"


def test_get_run_404_when_missing(client: TestClient, monkeypatch):
    class _StubStore:
        async def get_run_detail(self, run_id):
            return None

    monkeypatch.setattr(routes, "get_store", lambda: _StubStore())

    response = client.get(f"/v1/runs/{uuid4()}")

    assert response.status_code == 404


def test_get_run_returns_detail(client: TestClient, monkeypatch):
    run_id = uuid4()
    detail = RunDetail(
        id=run_id,
        created_at="2026-01-01T00:00:00Z",
        pipeline_name="demo",
        query="q",
        answer="a",
        retrieved_chunks=[],
        claims=[],
    )

    class _StubStore:
        async def get_run_detail(self, rid):
            assert rid == run_id
            return detail

    monkeypatch.setattr(routes, "get_store", lambda: _StubStore())

    response = client.get(f"/v1/runs/{run_id}")

    assert response.status_code == 200
    assert response.json()["id"] == str(run_id)


def test_evaluate_stub_returns_contract_shape(client: TestClient):
    response = client.post(
        "/v1/evaluate", json={"dataset": "ragtruth", "n": 50, "ablation": "no_prober"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["dataset"] == "ragtruth"
    assert body["status"] == "not_implemented"
    assert body["results"] == []


# ---------------------------------------------------------------------------
# POST /demo/query — demo-only route (additive to §5.2, see
# CONTRACT_CHANGE_REQUEST.md). ATTEST verifies but does not generate, so this
# runs the bundled RAG first and returns the answer synchronously while
# verification continues in the background.
# ---------------------------------------------------------------------------


class _FakeDoc:
    def __init__(self, i: int) -> None:
        self.chunk_index = i
        self.source_id = f"doc-{i:02d}"
        self.source_url = None
        self.text = f"chunk text {i}"
        self.score = 0.5


def _stub_dependencies(monkeypatch):
    """Make every verifier dependency resolvable without real modules."""
    monkeypatch.setattr(routes, "get_store", lambda: object())
    monkeypatch.setattr(routes, "get_entailment_verifier", lambda: object())
    monkeypatch.setattr(routes, "get_prober_verifier", lambda: object())
    monkeypatch.setattr(routes, "get_independent_verifier", lambda: object())
    monkeypatch.setattr(routes, "get_decompose", lambda: (lambda *a, **k: []))
    monkeypatch.setattr(routes, "get_probes_hook", lambda: None)
    # Never actually run the orchestrator in a route test.
    async def _noop(*args, **kwargs):
        return None
    monkeypatch.setattr(routes, "_run_in_background", _noop)


def test_demo_query_returns_answer_synchronously(client: TestClient, monkeypatch):
    _stub_dependencies(monkeypatch)

    async def fake_answer_query(query: str, k: int = 4):
        return "Windows 10 still receives security patches.", [_FakeDoc(0), _FakeDoc(1)]

    import sys
    import types

    module = types.ModuleType("demo.rag_pipeline")
    module.answer_query = fake_answer_query
    pkg = types.ModuleType("demo")
    monkeypatch.setitem(sys.modules, "demo", pkg)
    monkeypatch.setitem(sys.modules, "demo.rag_pipeline", module)

    response = client.post("/v1/demo/query", json={"query": "does it get updates?", "k": 2})

    assert response.status_code == 202
    body = response.json()
    assert body["answer"] == "Windows 10 still receives security patches."
    assert body["query"] == "does it get updates?"
    assert len(body["retrieved_chunks"]) == 2
    assert body["retrieved_chunks"][0]["source_id"] == "doc-00"
    assert body["run_id"]  # caller polls GET /runs/{run_id} for verdicts


def test_demo_query_rejects_empty_query(client: TestClient):
    assert client.post("/v1/demo/query", json={"query": ""}).status_code == 422


def test_demo_query_502_when_generation_fails(client: TestClient, monkeypatch):
    _stub_dependencies(monkeypatch)

    async def boom(query: str, k: int = 4):
        raise RuntimeError("all providers failed")

    import sys
    import types

    module = types.ModuleType("demo.rag_pipeline")
    module.answer_query = boom
    monkeypatch.setitem(sys.modules, "demo", types.ModuleType("demo"))
    monkeypatch.setitem(sys.modules, "demo.rag_pipeline", module)

    response = client.post("/v1/demo/query", json={"query": "anything"})

    assert response.status_code == 502
    assert "could not generate an answer" in response.json()["detail"]
