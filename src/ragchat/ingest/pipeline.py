"""Synchronous (no job queue) ingest pipeline: connector → parse → chunk → embed → Qdrant.

- Incremental: a document is skipped when its content hash *and* index config (pipeline
  version, embedding model/dims) match the `documents` row and it was indexed successfully.
- Parsing runs in a process pool (CPU-bound, Docling); embedding runs async in the main process.
- No-gap re-ingest: new points are upserted before stale points of the same doc are deleted.
- Documents that disappeared from the source are pruned from Qdrant and Postgres.
"""

from __future__ import annotations

import asyncio
import multiprocessing
import os
import traceback
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from ragchat.core.logging import get_logger
from ragchat.core.settings import Settings
from ragchat.ingest import state
from ragchat.ingest.chunking import Chunker, CountTokens
from ragchat.ingest.connectors.base import Connector, SourceItem
from ragchat.ingest.embedding import Embedder, EmbedStats
from ragchat.ingest.ir import ParsedDocument, file_content_hash, make_doc_id
from ragchat.ingest.parsers.router import parse_file
from ragchat.retrieval.vectorstore import VectorStore

log = get_logger(__name__)


@dataclass
class IngestReport:
    scanned: int = 0
    unsupported: int = 0
    unchanged: int = 0
    indexed: int = 0
    empty: int = 0
    failed: int = 0
    pruned: int = 0
    chunks: int = 0
    embed: EmbedStats = field(default_factory=EmbedStats)
    errors: list[tuple[str, str]] = field(default_factory=list)
    planned: list[str] = field(default_factory=list)  # dry-run: uris that would be (re)indexed


@dataclass
class _Job:
    item: SourceItem
    content_hash: str
    doc_id: str


def _parse_job(path: str, source: str, uri: str) -> ParsedDocument | tuple[str, str]:
    """Process-pool entry point. Returns the document or (error type, message)."""
    try:
        return parse_file(Path(path), source=source, uri=uri)
    except Exception as exc:  # noqa: BLE001 - surfaced per document in the report
        return type(exc).__name__, f"{exc}\n{traceback.format_exc(limit=3)}"


class IngestPipeline:
    def __init__(
        self,
        settings: Settings,
        session_factory: async_sessionmaker[AsyncSession],
        store: VectorStore,
        embedder: Embedder,
        *,
        count_tokens: CountTokens | None = None,
        workers: int | None = None,
        parsed_dir: Path | None = None,
    ) -> None:
        self.s = settings
        self.session_factory = session_factory
        self.store = store
        self.embedder = embedder
        self.chunker = Chunker(settings.chunking, count_tokens or embedder.count)
        self.workers = workers or max(1, min(4, (os.cpu_count() or 2) - 1))
        self.parsed_dir = parsed_dir or settings.parsed_dir

    # -- ingest ---------------------------------------------------------------------------------

    async def run(
        self,
        connector: Connector,
        *,
        force: bool = False,
        prune: bool = True,
        dry_run: bool = False,
    ) -> IngestReport:
        report = IngestReport()
        self.embedder.stats = EmbedStats()  # per-run numbers in the report
        if not dry_run:
            await self.store.ensure_collection()

        items = list(connector.iter_items())
        report.scanned = len(items)
        report.unsupported = connector.skipped
        async with self.session_factory() as session:
            known = {
                d.uri: d
                for d in await state.list_documents(
                    session, source=connector.name, uri_prefix=connector.uri_prefix
                )
            }

        jobs: list[_Job] = []
        for item in items:
            content_hash = file_content_hash(item.path)
            if not force and self._is_current(known.get(item.uri), content_hash):
                report.unchanged += 1
                continue
            jobs.append(_Job(item, content_hash, make_doc_id(connector.name, item.uri)))

        seen_uris = {item.uri for item in items}
        stale = [d for uri, d in known.items() if uri not in seen_uris] if prune else []

        if dry_run:
            report.planned = [j.item.uri for j in jobs]
            report.pruned = len(stale)
            return report

        await self._process_jobs(connector.name, jobs, report)

        for d in stale:
            await self._delete(str(d.doc_id))
            report.pruned += 1
            log.info("pruned", uri=d.uri)

        report.embed = self.embedder.stats
        return report

    def _is_current(self, row: state.Document | None, content_hash: str) -> bool:
        return (
            row is not None
            and row.status in (state.DocumentStatus.INDEXED, state.DocumentStatus.EMPTY)
            and row.content_hash == content_hash
            and row.pipeline_version == self.s.pipeline_version
            and row.embedding_model == self.s.embedding.model
            and row.embedding_dims == self.s.embedding.dimensions
        )

    async def _process_jobs(self, source: str, jobs: list[_Job], report: IngestReport) -> None:
        if not jobs:
            return
        loop = asyncio.get_running_loop()

        async def handle(job: _Job, result: ParsedDocument | tuple[str, str]) -> None:
            if isinstance(result, tuple):
                await self._fail(job, source, f"{result[0]}: {result[1]}", report)
                return
            try:
                n = await self.index_document(result)
            except Exception as exc:  # noqa: BLE001 - keep going, report per document
                await self._fail(job, source, f"{type(exc).__name__}: {exc}", report)
                return
            report.chunks += n
            if n:
                report.indexed += 1
            else:
                report.empty += 1
            log.info("indexed", uri=job.item.uri, chunks=n)

        if self.workers == 1 or len(jobs) == 1:
            for job in jobs:
                result = await loop.run_in_executor(
                    None, _parse_job, str(job.item.path), source, job.item.uri
                )
                await handle(job, result)
            return

        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=min(self.workers, len(jobs)), mp_context=ctx) as pool:

            async def parse(job: _Job) -> tuple[_Job, ParsedDocument | tuple[str, str]]:
                result = await loop.run_in_executor(
                    pool, _parse_job, str(job.item.path), source, job.item.uri
                )
                return job, result

            # Index documents as their parses finish while the pool keeps parsing the rest.
            for done in asyncio.as_completed([parse(j) for j in jobs]):
                job, result = await done
                await handle(job, result)

    async def _fail(self, job: _Job, source: str, error: str, report: IngestReport) -> None:
        report.failed += 1
        report.errors.append((job.item.uri, error.splitlines()[0]))
        log.error("failed", uri=job.item.uri, error=error.splitlines()[0])
        async with self.session_factory() as session:
            await state.record_failed(
                session,
                doc_id=job.doc_id,
                source=source,
                uri=job.item.uri,
                error=error,
                content_hash=job.content_hash,
            )
            await session.commit()

    # -- per-document indexing (shared by ingest and reindex) -----------------------------------

    async def index_document(self, doc: ParsedDocument, *, save_ir: bool = True) -> int:
        """Chunk, embed, upsert, drop stale points, record state. Returns the chunk count."""
        chunks = self.chunker.chunk(doc)
        parsed_path = str(doc.save(self.parsed_dir)) if save_ir else None
        if chunks:
            vectors = await self.embedder.embed([c.embed_text for c in chunks])
            await self.store.upsert_chunks(doc, chunks, vectors)
        await self.store.delete_stale(doc.doc_id, [c.chunk_id for c in chunks])
        async with self.session_factory() as session:
            await state.record_indexed(
                session,
                doc,
                chunks,
                pipeline_version=self.s.pipeline_version,
                embedding_model=self.s.embedding.model,
                embedding_dims=self.s.embedding.dimensions,
                parsed_path=parsed_path,
            )
            await session.commit()
        return len(chunks)

    async def _delete(self, doc_id: str) -> None:
        await self.store.delete_document(doc_id)
        async with self.session_factory() as session:
            await state.delete_document(session, doc_id)
            await session.commit()

    # -- reindex --------------------------------------------------------------------------------

    async def reindex(self, *, recreate: bool = False) -> IngestReport:
        """Re-chunk and re-embed every document in `data/parsed/` without re-parsing."""
        report = IngestReport()
        self.embedder.stats = EmbedStats()
        if recreate:
            await self.store.drop_collection()
        await self.store.ensure_collection()
        for path in sorted(self.parsed_dir.glob("*.json")):
            report.scanned += 1
            try:
                doc = ParsedDocument.load(path)
                n = await self.index_document(doc, save_ir=False)
            except Exception as exc:  # noqa: BLE001
                report.failed += 1
                report.errors.append((path.name, f"{type(exc).__name__}: {exc}"))
                log.error("reindex failed", path=str(path), error=str(exc))
                continue
            report.chunks += n
            if n:
                report.indexed += 1
            else:
                report.empty += 1
        report.embed = self.embedder.stats
        return report
