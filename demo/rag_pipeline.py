"""Reference RAG pipeline over the Northwind Devices demo corpus (demo/corpus/).

Loads each corpus doc as a single chunk into a persistent ChromaDB
collection — embeddings via chromadb's bundled default embedding function
(a small ONNX MiniLM model chromadb downloads to `~/.cache/chroma` on first
use; no extra embeddings package needed, see the final report) — retrieves
the top-k chunks for a query, and generates an answer with
`attest.llm.complete` using the retrieved chunks as context.

This is deliberately simple (whole-document chunks, no reranking, no
hybrid search) — the point of this pipeline is to be a clean, reproducible
target for the two seeded ATTEST demo cases (see `demo/SEED_NOTES.md`), not
a state-of-the-art RAG implementation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import chromadb
import structlog

from attest.config import settings
from attest.llm import complete

logger = structlog.get_logger(__name__)

CORPUS_DIR = Path(__file__).parent / "corpus"
COLLECTION_NAME = "northwind_demo_corpus"

_DOC_ID_RE = re.compile(r"\*\*Doc ID:\*\*\s*(\S+)")


@dataclass
class RetrievedDoc:
    """Local, RAG-pipeline-facing shape — converted to `attest.models.RetrievedChunk`
    (or the SDK's looser `attest.Output.retrieved_chunks` shape) at the SDK boundary,
    not used directly as a wire type."""

    chunk_index: int
    source_id: str
    source_url: str | None
    text: str
    score: float


def _extract_doc_id(text: str, fallback: str) -> str:
    match = _DOC_ID_RE.search(text)
    return match.group(1) if match else fallback


def _load_corpus() -> list[tuple[str, str]]:
    """Return [(filename, text), ...] for every doc in demo/corpus/, sorted for
    reproducible ids/ordering."""
    paths = sorted(CORPUS_DIR.glob("*.md"))
    if not paths:
        raise RuntimeError(
            f"No corpus documents found in {CORPUS_DIR}. Run "
            "`uv run python demo/build_corpus.py` first."
        )
    return [(p.name, p.read_text(encoding="utf-8")) for p in paths]


def get_collection(reset: bool = False):
    """Get (building if needed) the persistent Chroma collection for the demo corpus."""
    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    docs = _load_corpus()

    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    try:
        collection = client.get_collection(COLLECTION_NAME)
        if collection.count() == len(docs):
            return collection
        # Corpus changed size since last index — rebuild from scratch.
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(COLLECTION_NAME)
    ids = [str(i) for i in range(len(docs))]
    texts = [text for _, text in docs]
    metadatas = [
        {"filename": filename, "source_id": _extract_doc_id(text, filename)}
        for filename, text in docs
    ]
    collection.add(ids=ids, documents=texts, metadatas=metadatas)
    logger.info("demo_corpus_indexed", count=len(docs), persist_dir=settings.chroma_persist_dir)
    return collection


def retrieve(query: str, k: int = 4) -> list[RetrievedDoc]:
    """Top-k retrieval over the demo corpus. No network calls — pure ChromaDB."""
    collection = get_collection()
    results = collection.query(query_texts=[query], n_results=k)

    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    retrieved: list[RetrievedDoc] = []
    for idx, (doc_id, text, metadata, distance) in enumerate(
        zip(ids, documents, metadatas, distances, strict=True)
    ):
        retrieved.append(
            RetrievedDoc(
                chunk_index=idx,
                source_id=(metadata or {}).get("source_id", doc_id),
                source_url=None,
                text=text,
                score=max(0.0, 1.0 - distance),
            )
        )
    return retrieved


_ANSWER_PROMPT = """\
You are the support assistant for Northwind Devices, a consumer tech company (earbuds, \
smartwatches, a home hub, and support apps). Answer the user's question using the retrieved \
context below as your source of truth. Be direct, confident, and specific, the way a real \
customer support agent would be — do not hedge or add disclaimers about the source material. If \
the context does not explicitly cover a specific detail the user asked about, still answer \
naturally and helpfully rather than declining to answer.

Question: {query}

Retrieved context:
{context}

Answer:"""


async def generate_answer(query: str, chunks: list[RetrievedDoc]) -> str:
    """Generate a fluent answer over the retrieved chunks via the shared LLM
    router (`attest.llm.complete`). Uses the "fast" tier deliberately — this
    is the pipeline *being observed*, not a verifier judging it, and a
    smaller/faster model is both cheaper and more prone to the confident
    over-generalization the demo seeds are designed to catch."""
    context = "\n\n".join(f"[{c.source_id}] {c.text}" for c in chunks)
    prompt = _ANSWER_PROMPT.format(query=query, context=context)
    result = await complete(prompt, model_tier="fast")
    return result.text.strip()


async def answer_query(query: str, k: int = 4) -> tuple[str, list[RetrievedDoc]]:
    """Full pipeline: retrieve then generate. Returns (answer, retrieved_chunks)."""
    chunks = retrieve(query, k=k)
    answer = await generate_answer(query, chunks)
    return answer, chunks
