"""Embedding with a Postgres cache (PLAN.md rule 6).

Cache key = sha256(embed_text + model + dims). Misses are embedded in batches (≤ 2048 inputs,
`embedding.batch_size` by default) concurrently under a semaphore; `LLMClient.embed` retries.
A token/cost estimate is logged before any API call.
"""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from ragchat.core.logging import get_logger
from ragchat.core.settings import EmbeddingSettings
from ragchat.ingest import state
from ragchat.ingest.chunking import CountTokens, tiktoken_counter

log = get_logger(__name__)

EmbedFn = Callable[[list[str]], Awaitable[list[list[float]]]]
OPENAI_MAX_BATCH = 2048


def cache_key(text: str, model: str, dims: int) -> str:
    return hashlib.sha256(f"{text}\x00{model}\x00{dims}".encode()).hexdigest()


@dataclass
class EmbedStats:
    texts: int = 0
    cache_hits: int = 0
    embedded: int = 0
    api_calls: int = 0
    tokens: int = 0
    usd: float = 0.0

    def add(self, other: EmbedStats) -> None:
        for f in ("texts", "cache_hits", "embedded", "api_calls", "tokens"):
            setattr(self, f, getattr(self, f) + getattr(other, f))
        self.usd += other.usd


class Embedder:
    def __init__(
        self,
        settings: EmbeddingSettings,
        session_factory: async_sessionmaker[AsyncSession],
        embed_fn: EmbedFn | None = None,
        count_tokens: CountTokens | None = None,
    ) -> None:
        self.s = settings
        self.session_factory = session_factory
        self._embed_fn = embed_fn
        self.count = count_tokens or tiktoken_counter()
        self.batch_size = min(settings.batch_size, OPENAI_MAX_BATCH)
        self.semaphore = asyncio.Semaphore(settings.max_concurrency)
        self.stats = EmbedStats()

    @property
    def embed_fn(self) -> EmbedFn:
        if self._embed_fn is None:
            from ragchat.core.llm import get_llm

            self._embed_fn = get_llm().embed
        return self._embed_fn

    def estimate(self, texts: list[str]) -> tuple[int, float]:
        tokens = sum(self.count(t) for t in texts)
        return tokens, tokens / 1_000_000 * self.s.price_per_million_tokens

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Vectors for `texts`, in order. Duplicated texts are embedded once."""
        if not texts:
            return []
        keys = [cache_key(t, self.s.model, self.s.dimensions) for t in texts]
        async with self.session_factory() as session:
            cached = await state.cache_lookup(session, list(set(keys)))

        missing: dict[str, str] = {}  # key → text (dedup)
        for k, t in zip(keys, texts, strict=True):
            if k not in cached:
                missing[k] = t
        stats = EmbedStats(texts=len(texts), cache_hits=len(texts) - len(missing))

        if missing:
            miss_keys = list(missing)
            miss_texts = [missing[k] for k in miss_keys]
            tokens, usd = self.estimate(miss_texts)
            stats.tokens, stats.usd, stats.embedded = tokens, usd, len(miss_texts)
            log.info(
                "embedding",
                texts=len(miss_texts),
                cached=stats.cache_hits,
                tokens=tokens,
                est_usd=round(usd, 4),
                model=self.s.model,
            )
            batches = [
                miss_texts[i : i + self.batch_size]
                for i in range(0, len(miss_texts), self.batch_size)
            ]
            results = await asyncio.gather(*(self._embed_batch(b) for b in batches))
            stats.api_calls = len(batches)
            vectors = [v for batch in results for v in batch]
            fresh = dict(zip(miss_keys, vectors, strict=True))
            async with self.session_factory() as session:
                await state.cache_store(session, fresh, model=self.s.model, dims=self.s.dimensions)
                await session.commit()
            cached.update(fresh)

        self.stats.add(stats)
        return [cached[k] for k in keys]

    async def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        async with self.semaphore:
            vectors = await self.embed_fn(batch)
        if len(vectors) != len(batch):
            raise RuntimeError(f"embedding returned {len(vectors)} vectors for {len(batch)} inputs")
        for v in vectors:
            if len(v) != self.s.dimensions:
                raise RuntimeError(
                    f"embedding has {len(v)} dims, expected {self.s.dimensions}; check settings"
                )
        return vectors
