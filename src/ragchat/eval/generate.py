"""Synthetic golden candidates: sample indexed chunks, ask the small model for questions each
chunk answers, keep only those whose supporting span is verbatim in the chunk. Output goes to a
candidates file for human curation into `eval/golden.jsonl`."""

from __future__ import annotations

import asyncio
import random
from pathlib import Path

from pydantic import BaseModel, Field
from qdrant_client import models

from ragchat.core.llm import LLMClient
from ragchat.core.logging import get_logger
from ragchat.core.settings import Settings
from ragchat.eval.golden import Expected, GoldenItem
from ragchat.retrieval.retriever import RetrievedChunk
from ragchat.retrieval.vectorstore import VectorStore

log = get_logger(__name__)

SYSTEM = """\
You write evaluation questions for an internal company knowledge assistant.
Given one passage from a company document, write up to {n} distinct questions that an employee
might realistically ask and that this passage answers on its own. For each question give a
concise reference answer and a short verbatim span (5-20 words, copied exactly) from the passage
that supports it. Skip the passage (return an empty list) if it is boilerplate, a bare heading,
navigation, or has no self-contained facts. Do not invent facts not in the passage.
"""

USER = """\
Document title: {title}
Section: {breadcrumb}

<passage>
{text}
</passage>
"""


class Candidate(BaseModel):
    question: str
    answer: str
    span: str = Field(description="Verbatim phrase from the passage supporting the answer.")


class CandidateList(BaseModel):
    candidates: list[Candidate]


def _norm(text: str) -> str:
    return " ".join(text.split()).lower()


async def sample_chunks(store: VectorStore, max_chunks: int, seed: int) -> list[RetrievedChunk]:
    records, _ = await store.client.scroll(
        collection_name=store.collection, limit=10_000, with_payload=True, with_vectors=False
    )
    chunks = [
        RetrievedChunk.from_point(
            models.ScoredPoint(id=r.id, version=0, score=0.0, payload=r.payload)
        )
        for r in records
    ]
    chunks = [c for c in chunks if len(c.text.split()) >= 12]  # skip near-empty chunks
    random.Random(seed).shuffle(chunks)
    return chunks[:max_chunks]


async def generate_candidates(
    settings: Settings,
    store: VectorStore,
    llm: LLMClient,
    *,
    max_chunks: int | None = None,
    per_chunk: int | None = None,
    seed: int = 0,
    concurrency: int = 4,
) -> list[GoldenItem]:
    per_chunk = per_chunk or settings.eval.questions_per_chunk
    chunks = await sample_chunks(store, max_chunks or settings.eval.max_chunks, seed)
    sem = asyncio.Semaphore(concurrency)

    async def one(chunk: RetrievedChunk) -> list[GoldenItem]:
        async with sem:
            result = await llm.complete_structured(
                SYSTEM.format(n=per_chunk),
                USER.format(title=chunk.title, breadcrumb=chunk.breadcrumb, text=chunk.text),
                CandidateList,
            )
        items = []
        for cand in result.candidates[:per_chunk]:
            if _norm(cand.span) not in _norm(chunk.text):
                log.debug("span not verbatim; dropped", question=cand.question)
                continue
            items.append(
                GoldenItem(
                    id="",  # assigned after dedupe
                    question=cand.question.strip(),
                    answer=cand.answer.strip(),
                    expected=[
                        Expected(
                            uri=chunk.uri,
                            chunk_index=chunk.chunk_index,
                            chunk_id=chunk.chunk_id,
                            span=cand.span.strip(),
                        )
                    ],
                    origin="synthetic",
                )
            )
        return items

    nested = await asyncio.gather(*(one(c) for c in chunks))
    seen: set[str] = set()
    out: list[GoldenItem] = []
    for items in nested:
        for item in items:
            key = _norm(item.question)
            if key in seen:
                continue
            seen.add(key)
            item.id = f"syn-{len(out) + 1:04d}"
            out.append(item)
    log.info("golden candidates", chunks=len(chunks), candidates=len(out))
    return out


def write_candidates(items: list[GoldenItem], path: Path) -> None:
    from ragchat.eval.golden import save_golden

    save_golden(path, items)
