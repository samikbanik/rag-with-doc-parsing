"""Real Qdrant + Postgres via testcontainers; OpenAI replaced by a deterministic fake embedder."""

from __future__ import annotations

import hashlib
import math
import shutil
import subprocess
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from ragchat.core.settings import Settings, get_settings
from ragchat.ingest.embedding import Embedder
from ragchat.ingest.pipeline import IngestPipeline
from ragchat.retrieval.vectorstore import VectorStore

pytestmark = pytest.mark.integration

DIMS = 32


def _docker_available() -> bool:
    if not shutil.which("docker"):
        return False
    return subprocess.run(["docker", "info"], capture_output=True, timeout=20).returncode == 0


@pytest.fixture(scope="session")
def containers():
    if not _docker_available():
        pytest.skip("Docker not available")
    from testcontainers.community.postgres import PostgresContainer
    from testcontainers.community.qdrant import QdrantContainer

    with (
        PostgresContainer("postgres:16-alpine", driver="psycopg") as pg,
        QdrantContainer("qdrant/qdrant:v1.19.1") as qd,
    ):
        yield pg, qd


@pytest.fixture(scope="session")
def migrated_db_url(containers) -> str:
    pg, _ = containers
    sync_url = pg.get_connection_url()  # postgresql+psycopg://...
    subprocess.run(
        ["uv", "run", "alembic", "upgrade", "head"],
        check=True,
        env={
            **__import__("os").environ,
            "RAGCHAT_DATABASE_URL": sync_url.replace("+psycopg", "+asyncpg"),
        },
        capture_output=True,
    )
    return sync_url.replace("+psycopg", "+asyncpg")


@pytest.fixture
def settings(containers, migrated_db_url, tmp_path: Path) -> Settings:
    _, qd = containers
    base = get_settings()
    return base.model_copy(
        update={
            "database_url": migrated_db_url,
            "qdrant_url": f"http://{qd.get_container_host_ip()}:{qd.get_exposed_port(6333)}",
            "data_dir": tmp_path / "data",
            "embedding": base.embedding.model_copy(
                update={"model": "fake-embed", "dimensions": DIMS, "batch_size": 8}
            ),
            "vectorstore": base.vectorstore.model_copy(update={"collection": "test_chunks"}),
        }
    )


def fake_vector(text: str) -> list[float]:
    """Deterministic unit vector from a hash of the text."""
    h = hashlib.sha256(text.encode()).digest()
    raw = [(b - 128) / 128 for b in h[:DIMS]]
    norm = math.sqrt(sum(x * x for x in raw))
    return [x / norm for x in raw]


class FakeEmbedFn:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def __call__(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [fake_vector(t) for t in texts]


def words(text: str) -> int:
    return len(text.split())


@pytest.fixture
async def env(settings: Settings) -> AsyncIterator[dict]:
    from qdrant_client import AsyncQdrantClient

    engine = create_async_engine(settings.database_url)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    client = AsyncQdrantClient(url=settings.qdrant_url)
    store = VectorStore(settings, client)
    embed_fn = FakeEmbedFn()
    embedder = Embedder(settings.embedding, sessions, embed_fn=embed_fn, count_tokens=words)
    pipeline = IngestPipeline(settings, sessions, store, embedder, count_tokens=words, workers=1)
    yield {
        "settings": settings,
        "sessions": sessions,
        "client": client,
        "store": store,
        "embed_fn": embed_fn,
        "embedder": embedder,
        "pipeline": pipeline,
    }
    if await client.collection_exists(settings.vectorstore.collection):
        await client.delete_collection(settings.vectorstore.collection)
    from sqlalchemy import text

    async with engine.begin() as conn:
        await conn.execute(text("TRUNCATE documents, chunks, embedding_cache, traces CASCADE"))
    await client.close()
    await engine.dispose()
