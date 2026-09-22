"""Qdrant collection schema, index config guard, upsert and no-gap re-ingest.

Collection layout: named dense vector (`embedding.dimensions`, cosine) plus a sparse vector slot
(BM25 with server-side IDF) that stays empty until hybrid retrieval is enabled in M4. Payload
indexes on `doc_id`, `source`, `doc_type`, `last_modified`, `content_hash`, `chunk_index`.

Index config guard (PLAN.md rule 5): collection metadata records embedding model, dims and
pipeline_version; `ensure_collection` refuses to proceed on mismatch so points from different
pipelines never mix. `rag reindex --recreate` drops and rebuilds.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from qdrant_client import AsyncQdrantClient, models

from ragchat.core.logging import get_logger
from ragchat.core.settings import Settings, get_settings
from ragchat.ingest.chunking import Chunk
from ragchat.ingest.ir import ParsedDocument

log = get_logger(__name__)

UPSERT_BATCH = 128
PAYLOAD_INDEXES: dict[str, models.PayloadSchemaType] = {
    "doc_id": models.PayloadSchemaType.KEYWORD,
    "source": models.PayloadSchemaType.KEYWORD,
    "doc_type": models.PayloadSchemaType.KEYWORD,
    "content_hash": models.PayloadSchemaType.KEYWORD,
    "chunk_index": models.PayloadSchemaType.INTEGER,
    "last_modified": models.PayloadSchemaType.DATETIME,
}


class IndexConfigMismatchError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def get_qdrant() -> AsyncQdrantClient:
    s = get_settings()
    return AsyncQdrantClient(
        url=s.qdrant_url,
        api_key=s.qdrant_api_key.get_secret_value() if s.qdrant_api_key else None,
    )


async def ping() -> str:
    """Return a short description of the Qdrant server; raises on failure."""
    client = get_qdrant()
    collections = await client.get_collections()
    names = [c.name for c in collections.collections]
    return f"{len(names)} collection(s): {', '.join(names) or '-'}"


def index_config(s: Settings) -> dict[str, Any]:
    """What must match between the running config and the collection."""
    return {
        "embedding_model": s.embedding.model,
        "dims": s.embedding.dimensions,
        "pipeline_version": s.pipeline_version,
    }


class VectorStore:
    def __init__(self, settings: Settings, client: AsyncQdrantClient | None = None) -> None:
        self.s = settings
        self.client = client or get_qdrant()
        self.collection = settings.vectorstore.collection
        self.dense = settings.vectorstore.dense_vector_name
        self.sparse = settings.vectorstore.sparse_vector_name

    # -- schema ---------------------------------------------------------------------------------

    async def ensure_collection(self) -> dict[str, Any]:
        """Create the collection if missing; otherwise verify its index config matches."""
        expected = index_config(self.s)
        if not await self.client.collection_exists(self.collection):
            await self.client.create_collection(
                collection_name=self.collection,
                vectors_config={
                    self.dense: models.VectorParams(
                        size=self.s.embedding.dimensions, distance=models.Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    self.sparse: models.SparseVectorParams(modifier=models.Modifier.IDF)
                },
                metadata=expected,
            )
            for field, schema in PAYLOAD_INDEXES.items():
                await self.client.create_payload_index(
                    collection_name=self.collection, field_name=field, field_schema=schema
                )
            log.info("created collection", collection=self.collection, **expected)
            return expected
        actual = await self.current_config()
        mismatch = {k: (actual.get(k), v) for k, v in expected.items() if actual.get(k) != v}
        if mismatch:
            detail = ", ".join(
                f"{k}: collection={a!r} settings={e!r}" for k, (a, e) in mismatch.items()
            )
            raise IndexConfigMismatchError(
                f"collection '{self.collection}' was built with a different index config "
                f"({detail}). Run `rag reindex --recreate` or fix config/settings.yaml."
            )
        return actual

    async def current_config(self) -> dict[str, Any]:
        info = await self.client.get_collection(self.collection)
        meta = dict(info.config.metadata or {})
        vectors = info.config.params.vectors
        if isinstance(vectors, dict) and self.dense in vectors:
            meta.setdefault("dims", vectors[self.dense].size)
        return meta

    async def drop_collection(self) -> bool:
        return await self.client.delete_collection(self.collection)

    # -- points ---------------------------------------------------------------------------------

    async def upsert_chunks(
        self, doc: ParsedDocument, chunks: list[Chunk], vectors: list[list[float]]
    ) -> int:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors differ in length")
        points = [
            models.PointStruct(id=c.chunk_id, vector={self.dense: v}, payload=self.payload(doc, c))
            for c, v in zip(chunks, vectors, strict=True)
        ]
        for i in range(0, len(points), UPSERT_BATCH):
            await self.client.upsert(
                collection_name=self.collection, points=points[i : i + UPSERT_BATCH], wait=True
            )
        return len(points)

    async def delete_stale(self, doc_id: str, keep_ids: list[str]) -> None:
        """No-gap re-ingest: after the new points are in, drop every other point of `doc_id`.

        Filtering on ids (rather than only `content_hash != new`) also covers re-chunking that
        yields fewer chunks for the same content, and pipeline_version bumps.
        """
        must_not = [models.HasIdCondition(has_id=keep_ids)] if keep_ids else []
        await self.client.delete(
            collection_name=self.collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[_match("doc_id", doc_id)],
                    must_not=must_not,
                )
            ),
            wait=True,
        )

    async def delete_document(self, doc_id: str) -> None:
        await self.delete_stale(doc_id, keep_ids=[])

    async def count(self, doc_id: str | None = None) -> int:
        flt = models.Filter(must=[_match("doc_id", doc_id)]) if doc_id else None
        res = await self.client.count(collection_name=self.collection, count_filter=flt, exact=True)
        return res.count

    async def search_dense(
        self,
        vector: list[float],
        *,
        limit: int,
        query_filter: models.Filter | None = None,
    ) -> list[models.ScoredPoint]:
        res = await self.client.query_points(
            collection_name=self.collection,
            query=vector,
            using=self.dense,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        )
        return res.points

    async def scroll_doc(self, doc_id: str) -> list[models.Record]:
        records, _ = await self.client.scroll(
            collection_name=self.collection,
            scroll_filter=models.Filter(must=[_match("doc_id", doc_id)]),
            limit=10_000,
            with_payload=True,
            with_vectors=False,
        )
        return sorted(records, key=lambda r: (r.payload or {}).get("chunk_index", 0))

    def payload(self, doc: ParsedDocument, c: Chunk) -> dict[str, Any]:
        return {
            "doc_id": c.doc_id,
            "chunk_index": c.chunk_index,
            "section_id": c.section_id,
            "breadcrumb": c.breadcrumb,
            "text": c.text,
            "kind": c.kind,
            "table_part": list(c.table_part) if c.table_part else None,
            "page_start": c.page_start,
            "page_end": c.page_end,
            "token_count": c.token_count,
            "content_hash": c.content_hash,
            "title": doc.title,
            "uri": doc.uri,
            "source": doc.source,
            "doc_type": doc.doc_type,
            "last_modified": doc.last_modified.isoformat(),
            "pipeline_version": self.s.pipeline_version,
            "acl": list(doc.acl),
        }


def _match(field: str, value: str) -> models.FieldCondition:
    return models.FieldCondition(key=field, match=models.MatchValue(value=value))
