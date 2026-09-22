"""Ingest state in Postgres: `documents`, `chunks` (metadata only) and `embedding_cache`.

Incremental ingest keys on (`source`, `uri`) and compares `content_hash` plus the index config
(pipeline_version, embedding model/dims): a document is skipped only when all of them match.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    delete,
    func,
    select,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, REAL, insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from ragchat.core.db import Base
from ragchat.ingest.chunking import Chunk
from ragchat.ingest.ir import ParsedDocument


class DocumentStatus:
    INDEXED = "indexed"
    EMPTY = "empty"  # parsed fine but produced no chunks (e.g. image-only PDF)
    FAILED = "failed"


class Document(Base):
    __tablename__ = "documents"
    __table_args__ = (UniqueConstraint("source", "uri", name="uq_documents_source_uri"),)

    doc_id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    source: Mapped[str] = mapped_column(String(32), index=True)
    uri: Mapped[str] = mapped_column(Text)
    title: Mapped[str] = mapped_column(Text, default="")
    doc_type: Mapped[str] = mapped_column(String(16), default="")
    content_hash: Mapped[str] = mapped_column(String(64), default="")
    last_modified: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    parser: Mapped[str | None] = mapped_column(String(32))
    parser_version: Mapped[str | None] = mapped_column(String(32))
    pipeline_version: Mapped[int | None] = mapped_column(Integer)
    embedding_model: Mapped[str | None] = mapped_column(String(64))
    embedding_dims: Mapped[int | None] = mapped_column(Integer)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(16), default=DocumentStatus.INDEXED, index=True)
    error: Mapped[str | None] = mapped_column(Text)
    parsed_path: Mapped[str | None] = mapped_column(Text)
    meta: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, default=dict)
    acl: Mapped[list[str]] = mapped_column(JSONB, default=list)
    indexed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class ChunkRecord(Base):
    __tablename__ = "chunks"
    __table_args__ = (UniqueConstraint("doc_id", "chunk_index", name="uq_chunks_doc_index"),)

    chunk_id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    doc_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.doc_id", ondelete="CASCADE"), index=True
    )
    chunk_index: Mapped[int] = mapped_column(Integer)
    section_id: Mapped[uuid.UUID]
    breadcrumb: Mapped[str] = mapped_column(Text)
    kind: Mapped[str] = mapped_column(String(8))
    page_start: Mapped[int | None] = mapped_column(Integer)
    page_end: Mapped[int | None] = mapped_column(Integer)
    token_count: Mapped[int] = mapped_column(Integer)
    content_hash: Mapped[str] = mapped_column(String(64))


class EmbeddingCache(Base):
    __tablename__ = "embedding_cache"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)  # sha256(text+model+dims)
    model: Mapped[str] = mapped_column(String(64))
    dims: Mapped[int] = mapped_column(Integer)
    vector: Mapped[list[float]] = mapped_column(ARRAY(REAL))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


# -- documents ------------------------------------------------------------------------------


async def get_document(session: AsyncSession, source: str, uri: str) -> Document | None:
    return await session.scalar(
        select(Document).where(Document.source == source, Document.uri == uri)
    )


async def list_documents(
    session: AsyncSession, source: str | None = None, uri_prefix: str | None = None
) -> list[Document]:
    stmt = select(Document).order_by(Document.uri)
    if source:
        stmt = stmt.where(Document.source == source)
    if uri_prefix:
        stmt = stmt.where(Document.uri.startswith(uri_prefix))
    return list((await session.scalars(stmt)).all())


async def record_indexed(
    session: AsyncSession,
    doc: ParsedDocument,
    chunks: list[Chunk],
    *,
    pipeline_version: int,
    embedding_model: str,
    embedding_dims: int,
    parsed_path: str | None,
) -> Document:
    """Insert or update the document row and replace its chunk rows."""
    row = await session.get(Document, uuid.UUID(doc.doc_id))
    if row is None:
        row = Document(doc_id=uuid.UUID(doc.doc_id), source=doc.source, uri=doc.uri)
        session.add(row)
    row.title = doc.title
    row.doc_type = doc.doc_type
    row.content_hash = doc.content_hash
    row.last_modified = doc.last_modified
    row.parser = doc.parser
    row.parser_version = doc.parser_version
    row.pipeline_version = pipeline_version
    row.embedding_model = embedding_model
    row.embedding_dims = embedding_dims
    row.chunk_count = len(chunks)
    row.status = DocumentStatus.INDEXED if chunks else DocumentStatus.EMPTY
    row.error = None
    row.parsed_path = parsed_path
    row.meta = _jsonable(doc.metadata)
    row.acl = list(doc.acl)
    row.indexed_at = datetime.now(UTC)

    await session.execute(delete(ChunkRecord).where(ChunkRecord.doc_id == row.doc_id))
    if chunks:
        await session.execute(
            insert(ChunkRecord),
            [
                {
                    "chunk_id": uuid.UUID(c.chunk_id),
                    "doc_id": row.doc_id,
                    "chunk_index": c.chunk_index,
                    "section_id": uuid.UUID(c.section_id),
                    "breadcrumb": c.breadcrumb,
                    "kind": c.kind,
                    "page_start": c.page_start,
                    "page_end": c.page_end,
                    "token_count": c.token_count,
                    "content_hash": c.content_hash,
                }
                for c in chunks
            ],
        )
    await session.flush()
    return row


async def record_failed(
    session: AsyncSession, *, doc_id: str, source: str, uri: str, error: str, content_hash: str
) -> Document:
    row = await session.get(Document, uuid.UUID(doc_id))
    if row is None:
        row = Document(doc_id=uuid.UUID(doc_id), source=source, uri=uri)
        session.add(row)
    row.status = DocumentStatus.FAILED
    row.error = error[:4000]
    row.content_hash = content_hash
    await session.flush()
    return row


async def delete_document(session: AsyncSession, doc_id: str) -> None:
    await session.execute(delete(Document).where(Document.doc_id == uuid.UUID(doc_id)))


# -- embedding cache ------------------------------------------------------------------------


async def cache_lookup(session: AsyncSession, keys: list[str]) -> dict[str, list[float]]:
    found: dict[str, list[float]] = {}
    for i in range(0, len(keys), 1000):
        batch = keys[i : i + 1000]
        rows = await session.execute(
            select(EmbeddingCache.key, EmbeddingCache.vector).where(EmbeddingCache.key.in_(batch))
        )
        found.update({k: list(v) for k, v in rows.all()})
    return found


async def cache_store(
    session: AsyncSession, entries: dict[str, list[float]], *, model: str, dims: int
) -> None:
    if not entries:
        return
    values = [{"key": k, "model": model, "dims": dims, "vector": v} for k, v in entries.items()]
    for i in range(0, len(values), 500):
        stmt = insert(EmbeddingCache).values(values[i : i + 500])
        await session.execute(stmt.on_conflict_do_nothing(index_elements=["key"]))


def _jsonable(data: dict[str, Any]) -> dict[str, Any]:
    import json

    return json.loads(json.dumps(data, default=str))
