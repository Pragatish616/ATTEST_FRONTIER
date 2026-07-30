# ATTEST backend — portable image for Railway (primary) or Render (fallback).
#
# DEPLOYMENT CONSTRAINT (see attest/api/main.py::create_app): the SSE event bus
# in attest/api/stream.py is in-process memory. This image therefore runs a
# SINGLE uvicorn worker and MUST be deployed as a single replica with no
# autoscaling. More than one worker or replica does not error — it silently
# splits the bus, so a dashboard connected to replica B never sees events
# emitted on replica A. Live streaming is the demo; do not scale this out.

FROM python:3.11-slim

# pyproject pins requires-python >=3.11,<3.12. 3.11-slim satisfies that; do not
# bump the base image to 3.12 without changing the pin (StrEnum/datetime.UTC are
# 3.11+, and the codebase relies on TimeoutError being aliased to
# asyncio.TimeoutError, which is only true from 3.11 onward).

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CHROMA_PERSIST_DIR=/app/data/chroma \
    UV_PROJECT_ENVIRONMENT=/usr/local

WORKDIR /app

# uv, matching the project's declared toolchain.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Dependencies first, so a source-only change doesn't re-resolve the lock.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Application source.
COPY attest/ ./attest/
COPY demo/ ./demo/
COPY migrations/ ./migrations/
COPY README.md ./

# Drop root BEFORE warming Chroma. The container needs no privileged ports and
# writes only to the Chroma directory, so running as uid 10001 means a
# code-execution bug in a dependency lands on an unprivileged account instead of
# owning the container.
#
# ORDER MATTERS, AND GETTING IT WRONG ONLY FAILS IN THE CONTAINER.
# The first version of this file warmed Chroma as root and *then* created the
# user with --no-create-home. Two things broke at runtime, neither visible
# locally:
#   1. chromadb's ONNX MiniLM model caches under $HOME/.cache, so warming as
#      root put it in /root/.cache — unreadable by uid 10001.
#   2. --no-create-home left $HOME=/home/attest non-existent and uncreatable by
#      an unprivileged user, so the fallback download died with
#      `[Errno 13] Permission denied: '/home/attest'` and POST /v1/demo/query
#      returned 502 on every call.
# Creating the home directory, exporting HOME/XDG_CACHE_HOME explicitly, and
# running the warm step as the runtime user bakes the model into the image at a
# path that user owns.
RUN useradd --uid 10001 --create-home --home-dir /home/attest --shell /usr/sbin/nologin attest \
    && mkdir -p /app/data \
    && chown -R 10001:10001 /app/data /home/attest

ENV HOME=/home/attest \
    XDG_CACHE_HOME=/home/attest/.cache

USER 10001

# Warm the Chroma index at BUILD time, not first request.
#
# demo/corpus/*.md is tracked in git but data/chroma/ is gitignored, so the
# index does not exist in a fresh checkout. get_collection() builds it lazily on
# first call — which would mean the first demo query pays for both a network
# download of chromadb's bundled ONNX MiniLM embedding model and a full reindex.
# Doing it here bakes model + index into the image so the demo has no cold-start
# network dependency.
#
# The dummy env vars exist only because attest/config.py instantiates Settings()
# at import time and fails loud on missing required keys. Nothing here contacts
# Supabase or an LLM provider; these values are never used at runtime, where the
# host's real environment variables take over.
RUN SUPABASE_URL=https://build.invalid \
    SUPABASE_KEY=build-time-placeholder \
    GEMINI_API_KEY=build-time-placeholder \
    python -c "from demo.rag_pipeline import get_collection; c = get_collection(); print('chroma warmed, docs =', c.count())"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os,urllib.request,sys; port=os.environ.get('PORT','8000'); sys.exit(0 if urllib.request.urlopen(f'http://127.0.0.1:{port}/v1/health', timeout=4).status == 200 else 1)"

# Shell form so $PORT (injected by Railway/Render) expands. --workers is
# deliberately absent: uvicorn defaults to one worker. Do not add it.
#
# --proxy-headers plus --forwarded-allow-ips='*' makes uvicorn populate the
# client address from the platform's X-Forwarded-For. The wildcard is safe only
# because the container is never directly reachable — Railway and Render put
# their edge proxy in front of it — and the rate limiter does not rely on
# uvicorn's parsing anyway: it counts hops itself via TRUSTED_PROXY_HOPS
# (see attest/api/security.py::client_ip). Set TRUSTED_PROXY_HOPS=1 in the
# deploy environment.
CMD uvicorn attest.api.main:app \
    --host 0.0.0.0 --port ${PORT:-8000} \
    --timeout-keep-alive 75 \
    --proxy-headers --forwarded-allow-ips='*'
