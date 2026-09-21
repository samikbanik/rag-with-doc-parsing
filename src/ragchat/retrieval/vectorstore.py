"""Qdrant client wrapper. Collection schema/upsert/search land in Milestone 1."""

from __future__ import annotations

from functools import lru_cache

from qdrant_client import AsyncQdrantClient

from ragchat.core.settings import get_settings


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
