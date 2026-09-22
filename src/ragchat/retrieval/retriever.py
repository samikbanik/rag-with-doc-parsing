"""Retriever: query → ranked chunks.

Milestone 2 is dense-only. Rule 8's later stages (rewrite, hybrid RRF, rerank, neighbour
expansion) are M4 config flags and are added here only once the eval harness shows a gain.
What is applied now: dense top-`prefetch_k`, then at most `max_chunks_per_doc` per document,
then `top_k`.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel
from qdrant_client import models

from ragchat.core.logging import get_logger
from ragchat.core.settings import Settings
from ragchat.retrieval.vectorstore import VectorStore

log = get_logger(__name__)
EmbedFn = Callable[[list[str]], Awaitable[list[list[float]]]]


class RetrievedChunk(BaseModel):
    chunk_id: str
    doc_id: str
    chunk_index: int
    score: float
    title: str
    uri: str
    source: str
    doc_type: str
    breadcrumb: str
    text: str
    kind: str = "text"
    page_start: int | None = None
    page_end: int | None = None

    @classmethod
    def from_point(cls, point: models.ScoredPoint) -> RetrievedChunk:
        p: dict[str, Any] = point.payload or {}
        return cls(
            chunk_id=str(point.id),
            doc_id=p["doc_id"],
            chunk_index=p["chunk_index"],
            score=float(point.score),
            title=p.get("title", ""),
            uri=p.get("uri", ""),
            source=p.get("source", ""),
            doc_type=p.get("doc_type", ""),
            breadcrumb=p.get("breadcrumb", ""),
            text=p.get("text", ""),
            kind=p.get("kind", "text"),
            page_start=p.get("page_start"),
            page_end=p.get("page_end"),
        )


class Retriever:
    def __init__(self, settings: Settings, store: VectorStore, embed_fn: EmbedFn) -> None:
        self.s = settings.retrieval
        self.store = store
        self.embed_fn = embed_fn

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        query_filter: models.Filter | None = None,
    ) -> list[RetrievedChunk]:
        top_k = top_k or self.s.top_k
        vector = (await self.embed_fn([query]))[0]
        points = await self.store.search_dense(
            vector, limit=max(self.s.prefetch_k, top_k), query_filter=query_filter
        )
        hits = [RetrievedChunk.from_point(p) for p in points]
        return cap_per_doc(hits, self.s.max_chunks_per_doc)[:top_k]


def cap_per_doc(hits: list[RetrievedChunk], max_per_doc: int) -> list[RetrievedChunk]:
    """Keep ranking order but allow at most `max_per_doc` chunks from one document."""
    seen: dict[str, int] = {}
    kept: list[RetrievedChunk] = []
    for h in hits:
        n = seen.get(h.doc_id, 0)
        if n < max_per_doc:
            kept.append(h)
            seen[h.doc_id] = n + 1
    return kept
