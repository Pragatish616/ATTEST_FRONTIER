"""Unit tests for the SDK's bounded fire-and-forget queue and chunk
normalization (attest/sdk/__init__.py).

Per the task brief: the drop-oldest overflow behavior is tested directly
against the pure `_put_drop_oldest` helper (no threads, no HTTP, no
`attest.init`) — this is the exact function `_AttestClient.submit()` funnels
every enqueue through, so exercising it in isolation covers the guarantee
precisely without flakiness from thread/timing.
"""

import asyncio

from attest.models import RetrievedChunk
from attest.sdk import Output, _normalize_chunks, _put_drop_oldest


async def test_put_drop_oldest_drops_oldest_item_when_queue_is_full():
    queue: asyncio.Queue = asyncio.Queue(maxsize=2)
    await _put_drop_oldest(queue, "a")
    await _put_drop_oldest(queue, "b")
    await _put_drop_oldest(queue, "c")  # queue full at [a, b] -> drop "a", keep [b, c]

    remaining = []
    while not queue.empty():
        remaining.append(queue.get_nowait())
    assert remaining == ["b", "c"]


async def test_put_drop_oldest_drops_multiple_times_under_sustained_overflow():
    queue: asyncio.Queue = asyncio.Queue(maxsize=1)
    for item in ("a", "b", "c", "d"):
        await _put_drop_oldest(queue, item)

    remaining = []
    while not queue.empty():
        remaining.append(queue.get_nowait())
    assert remaining == ["d"]  # only the most recent survives a maxsize=1 queue


async def test_put_drop_oldest_does_not_drop_when_there_is_room():
    queue: asyncio.Queue = asyncio.Queue(maxsize=5)
    await _put_drop_oldest(queue, 1)
    await _put_drop_oldest(queue, 2)

    remaining = []
    while not queue.empty():
        remaining.append(queue.get_nowait())
    assert remaining == [1, 2]


# ---------------------------------------------------------------------------
# Output + chunk normalization
# ---------------------------------------------------------------------------


def test_output_holds_answer_and_arbitrary_chunk_shapes():
    out = Output(answer="hello", retrieved_chunks=[{"text": "x"}, "y"])
    assert out.answer == "hello"
    assert out.retrieved_chunks == [{"text": "x"}, "y"]


def test_normalize_chunks_handles_retrievedchunk_dict_langchain_dict_and_string():
    chunks = [
        RetrievedChunk(chunk_index=0, text="already a model"),
        {"text": "from a plain dict", "source_id": "doc-1"},
        {"page_content": "langchain Document-style dict", "source_url": "https://x"},
        "a bare string chunk",
    ]
    normalized = _normalize_chunks(chunks)

    assert len(normalized) == 4
    assert all(isinstance(c, RetrievedChunk) for c in normalized)
    assert normalized[0].text == "already a model"
    assert normalized[1].text == "from a plain dict"
    assert normalized[1].source_id == "doc-1"
    assert normalized[2].text == "langchain Document-style dict"
    assert normalized[2].source_url == "https://x"
    assert normalized[3].text == "a bare string chunk"


def test_normalize_chunks_handles_duck_typed_document_object():
    class FakeDocument:
        def __init__(self, page_content, metadata):
            self.page_content = page_content
            self.metadata = metadata

    doc = FakeDocument("duck-typed content", {"source_id": "doc-9", "source_url": "https://y"})
    normalized = _normalize_chunks([doc])

    assert len(normalized) == 1
    assert normalized[0].text == "duck-typed content"
    assert normalized[0].source_id == "doc-9"
    assert normalized[0].source_url == "https://y"


def test_normalize_chunks_skips_malformed_entries_without_raising():
    class NoUsefulAttributes:
        """Doesn't crash — falls back to str(obj) — but is clearly 'malformed'."""

    chunks = [
        {"this_dict_has_no_text_field": True},  # missing required 'text' -> should be skipped
        NoUsefulAttributes(),  # falls back to str(obj), not skipped
        "ok",
    ]

    normalized = _normalize_chunks(chunks)  # must not raise

    assert len(normalized) == 2
    assert normalized[-1].text == "ok"


def test_normalize_chunks_handles_none_and_empty():
    assert _normalize_chunks(None) == []
    assert _normalize_chunks([]) == []
