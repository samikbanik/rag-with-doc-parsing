from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from sqlalchemy import func, select

from ragchat.ingest import state
from ragchat.ingest.connectors.local import LocalFolderConnector
from ragchat.ingest.ir import make_doc_id
from ragchat.ingest.pipeline import IngestPipeline
from ragchat.retrieval.vectorstore import IndexConfigMismatchError, VectorStore

pytestmark = pytest.mark.integration
FIXTURES = Path(__file__).parent.parent / "fixtures"


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    root = tmp_path / "corpus"
    (root / "sub").mkdir(parents=True)
    shutil.copy(FIXTURES / "handbook.md", root / "handbook.md")
    shutil.copy(FIXTURES / "wiki.html", root / "sub" / "wiki.html")
    (root / "notes.txt").write_text("unsupported")
    (root / ".hidden.md").write_text("# hidden")
    return root


async def _doc_rows(sessions) -> dict[str, state.Document]:  # noqa: ANN001
    async with sessions() as s:
        return {Path(d.uri).name: d for d in await state.list_documents(s)}


async def test_ingest_end_to_end(env: dict, corpus: Path):
    pipeline: IngestPipeline = env["pipeline"]
    store: VectorStore = env["store"]
    connector = LocalFolderConnector(corpus)

    # 1. first run indexes everything supported
    r1 = await pipeline.run(connector)
    assert (r1.scanned, r1.unsupported, r1.indexed, r1.failed) == (2, 1, 2, 0)
    assert r1.chunks > 0 and r1.embed.api_calls > 0 and r1.embed.cache_hits == 0
    assert await store.count() == r1.chunks
    rows = await _doc_rows(env["sessions"])
    assert set(rows) == {"handbook.md", "wiki.html"}
    assert all(d.status == "indexed" for d in rows.values())
    assert sum(d.chunk_count for d in rows.values()) == r1.chunks
    assert (env["settings"].parsed_dir / f"{rows['handbook.md'].doc_id}.json").exists()

    # payload carries what citations and filters need
    doc_id = str(rows["handbook.md"].doc_id)
    records = await store.scroll_doc(doc_id)
    assert [r.payload["chunk_index"] for r in records] == list(range(len(records)))
    p = records[0].payload
    assert p["title"] == "Employee Handbook" and p["source"] == "local"
    assert p["uri"].endswith("handbook.md") and p["doc_type"] == "md"
    assert p["breadcrumb"].startswith("Employee Handbook") and p["text"]
    assert p["pipeline_version"] == env["settings"].pipeline_version

    # 2. unchanged dir → no work at all
    calls_before = len(env["embed_fn"].calls)
    r2 = await pipeline.run(connector)
    assert (r2.unchanged, r2.indexed, r2.pruned) == (2, 0, 0)
    assert len(env["embed_fn"].calls) == calls_before
    assert await store.count() == r1.chunks

    # 3. change one file → only it is re-indexed; ids rotate with the content hash, no gap
    old_ids = {r.id for r in records}
    text = (corpus / "handbook.md").read_text().replace("25 days", "30 days")
    (corpus / "handbook.md").write_text(text)
    r3 = await pipeline.run(connector)
    assert (r3.unchanged, r3.indexed) == (1, 1)
    new_records = await store.scroll_doc(doc_id)
    assert {r.id for r in new_records}.isdisjoint(old_ids)
    assert len(new_records) == len(records)  # same structure, new hash
    assert await store.count() == r1.chunks  # stale points are gone
    assert any("30 days" in r.payload["text"] for r in new_records)
    # unchanged chunks hit the embedding cache (identical embed_text) rather than the API
    assert r3.embed.cache_hits > 0
    rows = await _doc_rows(env["sessions"])
    assert rows["handbook.md"].content_hash != p["content_hash"]

    # 4. delete a file → pruned from Qdrant and Postgres
    (corpus / "sub" / "wiki.html").unlink()
    r4 = await pipeline.run(connector)
    assert r4.pruned == 1 and r4.scanned == 1
    rows = await _doc_rows(env["sessions"])
    assert set(rows) == {"handbook.md"}
    assert await store.count(make_doc_id("local", str(corpus / "sub" / "wiki.html"))) == 0
    async with env["sessions"]() as s:
        n_chunks = await s.scalar(select(func.count()).select_from(state.ChunkRecord))
    assert n_chunks == rows["handbook.md"].chunk_count == await store.count()

    # 5. --no-prune keeps rows; dry-run reports without touching anything
    (corpus / "new.md").write_text("# New\n\nfresh content")
    dry = await pipeline.run(connector, dry_run=True)
    assert dry.planned == [str(corpus / "new.md")] and dry.indexed == 0
    assert await store.count() == n_chunks


async def test_force_reindexes_unchanged(env: dict, corpus: Path):
    pipeline: IngestPipeline = env["pipeline"]
    connector = LocalFolderConnector(corpus)
    await pipeline.run(connector)
    r = await pipeline.run(connector, force=True)
    assert r.indexed == 2 and r.unchanged == 0
    assert r.embed.cache_hits == r.embed.texts  # everything came from the cache


async def test_failed_parse_is_recorded_and_retried(env: dict, tmp_path: Path):
    root = tmp_path / "bad"
    root.mkdir()
    (root / "broken.pdf").write_bytes(b"%PDF-1.4 not really a pdf")
    (root / "ok.md").write_text("# ok\n\nfine")
    pipeline: IngestPipeline = env["pipeline"]
    r = await pipeline.run(LocalFolderConnector(root))
    assert (r.indexed, r.failed) == (1, 1)
    assert r.errors and r.errors[0][0].endswith("broken.pdf")
    rows = await _doc_rows(env["sessions"])
    assert rows["broken.pdf"].status == "failed" and rows["broken.pdf"].error
    r2 = await pipeline.run(LocalFolderConnector(root))
    assert (r2.unchanged, r2.failed) == (1, 1)  # failures are retried, successes are not


async def test_index_config_guard(env: dict, corpus: Path):
    pipeline: IngestPipeline = env["pipeline"]
    await pipeline.run(LocalFolderConnector(corpus))
    settings = env["settings"].model_copy(update={"pipeline_version": 99})
    other = VectorStore(settings, env["client"])
    with pytest.raises(IndexConfigMismatchError, match="pipeline_version"):
        await other.ensure_collection()
    # a bumped pipeline_version also invalidates the incremental skip
    bumped = IngestPipeline(
        settings, env["sessions"], env["store"], env["embedder"], count_tokens=len, workers=1
    )
    dry = await bumped.run(LocalFolderConnector(corpus), dry_run=True)
    assert len(dry.planned) == 2


async def test_reindex_from_parsed_ir(env: dict, corpus: Path):
    pipeline: IngestPipeline = env["pipeline"]
    store: VectorStore = env["store"]
    r1 = await pipeline.run(LocalFolderConnector(corpus))
    await store.drop_collection()
    r = await pipeline.reindex()
    assert (r.scanned, r.indexed, r.failed) == (2, 2, 0)
    assert await store.count() == r1.chunks
    assert r.embed.api_calls == 0  # vectors all came from the embedding cache
