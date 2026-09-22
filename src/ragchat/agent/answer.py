"""Baseline RAG: retrieve → assemble → cited answer → trace (PLAN.md rules 8, 9, 11).

Answer contract: `{answer, citations[{chunk_id, title, uri, page, snippet}], retrieved_chunk_ids,
trace_id}`. An answer without at least one valid citation is refused.
"""

from __future__ import annotations

import re
import time
import uuid
from collections.abc import Callable

from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from ragchat.agent.prompts import ANSWER_SYSTEM, ANSWER_USER
from ragchat.agent.traces import Trace
from ragchat.core.llm import LLMClient, StructuredResult, Usage
from ragchat.core.logging import get_logger
from ragchat.core.settings import Settings
from ragchat.retrieval.assemble import AssembledContext, assemble
from ragchat.retrieval.retriever import RetrievedChunk, Retriever

log = get_logger(__name__)

REFUSAL = "I couldn't find an answer to that in the indexed documents."
SNIPPET_CHARS = 240
_CITE_RE = re.compile(r"\[(\d{1,3})\]")


class AnswerDraft(BaseModel):
    """Structured output requested from the model."""

    answer: str = Field(description="The answer with [n] citation markers, or a brief refusal.")
    citations: list[int] = Field(
        default_factory=list, description="Document numbers actually cited in the answer."
    )
    insufficient_context: bool = Field(
        default=False, description="True when the documents do not answer the question."
    )


class Citation(BaseModel):
    number: int  # the [n] marker used in the answer text
    chunk_id: str
    doc_id: str
    title: str
    uri: str
    page: int | None = None
    snippet: str
    score: float


class Answer(BaseModel):
    answer: str
    citations: list[Citation]
    retrieved_chunk_ids: list[str]
    trace_id: str
    refused: bool = False
    usage: dict[str, int] = Field(default_factory=dict)
    latency_ms: int = 0

    @property
    def cited_chunk_ids(self) -> list[str]:
        return [c.chunk_id for c in self.citations]


def validate_citations(draft: AnswerDraft, ctx: AssembledContext) -> tuple[str, list[Citation]]:
    """Map [n] markers (and the model's citation list) to chunks, dropping invalid numbers.

    Returns the answer text and citations in order of first appearance; numbers that do not
    resolve to a context document are stripped from the text so the UI never shows a dangling
    marker.
    """
    numbers: list[int] = []
    for n in [int(m) for m in _CITE_RE.findall(draft.answer)] + list(draft.citations):
        if n not in numbers and ctx.chunk_for(n) is not None:
            numbers.append(n)
    text = _CITE_RE.sub(lambda m: m.group(0) if int(m.group(1)) in numbers else "", draft.answer)
    text = re.sub(r"[ \t]+([.,;:])", r"\1", re.sub(r"[ \t]{2,}", " ", text)).strip()
    citations = [_citation(n, ctx.chunk_for(n)) for n in numbers]  # type: ignore[arg-type]
    return text, citations


def _citation(number: int, c: RetrievedChunk) -> Citation:
    snippet = c.text if len(c.text) <= SNIPPET_CHARS else c.text[: SNIPPET_CHARS - 1] + "…"
    return Citation(
        number=number,
        chunk_id=c.chunk_id,
        doc_id=c.doc_id,
        title=c.title,
        uri=c.uri,
        page=c.page_start,
        snippet=snippet.replace("\n", " "),
        score=c.score,
    )


class AnswerService:
    def __init__(
        self,
        settings: Settings,
        retriever: Retriever,
        llm: LLMClient,
        session_factory: async_sessionmaker[AsyncSession] | None,
        count_tokens: Callable[[str], int],
    ) -> None:
        self.s = settings
        self.retriever = retriever
        self.llm = llm
        self.session_factory = session_factory  # None → traces are not persisted
        self.count = count_tokens

    async def answer(self, question: str, *, top_k: int | None = None) -> Answer:
        t0 = time.perf_counter()
        trace = Trace(trace_id=uuid.uuid4(), query=question)
        try:
            hits = await self.retriever.retrieve(question, top_k=top_k)
            trace.retrieval_ms = int((time.perf_counter() - t0) * 1000)
            trace.retrieved = [
                {"chunk_id": h.chunk_id, "doc_id": h.doc_id, "score": round(h.score, 4)}
                for h in hits
            ]
            ctx = assemble(
                hits, max_tokens=self.s.retrieval.max_context_tokens, count_tokens=self.count
            )
            trace.meta = {"context_tokens": ctx.tokens, "context_dropped": ctx.dropped}

            if not ctx.chunks:
                text, citations, usage, model = REFUSAL, [], Usage(), None
            else:
                t1 = time.perf_counter()
                result = await self.generate(question, ctx)
                trace.llm_ms = int((time.perf_counter() - t1) * 1000)
                usage, model = result.usage, result.model
                text, citations = validate_citations(result.parsed, ctx)
                if result.parsed.insufficient_context or not citations:
                    # Rule 9: never answer without citations.
                    text, citations = (
                        (result.parsed.answer if result.parsed.insufficient_context else REFUSAL),
                        [],
                    )
            refused = not citations
            trace.answer, trace.refused, trace.model = text, refused, model
            trace.cited_chunk_ids = [c.chunk_id for c in citations]
            trace.input_tokens, trace.output_tokens = usage.input_tokens, usage.output_tokens
        except Exception as exc:
            trace.error = f"{type(exc).__name__}: {exc}"
            trace.latency_ms = int((time.perf_counter() - t0) * 1000)
            await self._save(trace)
            raise
        trace.latency_ms = int((time.perf_counter() - t0) * 1000)
        await self._save(trace)
        return Answer(
            answer=text,
            citations=citations,
            retrieved_chunk_ids=[h.chunk_id for h in hits],
            trace_id=str(trace.trace_id),
            refused=refused,
            usage={"input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens},
            latency_ms=trace.latency_ms,
        )

    async def generate(self, question: str, ctx: AssembledContext) -> StructuredResult[AnswerDraft]:
        return await self.llm.generate_structured(
            ANSWER_SYSTEM,
            ANSWER_USER.format(context=ctx.text, question=question),
            AnswerDraft,
            model=self.s.llm.model,
        )

    async def _save(self, trace: Trace) -> None:
        if self.session_factory is None:
            return
        try:
            async with self.session_factory() as session:
                session.add(trace)
                await session.commit()
        except Exception as exc:  # noqa: BLE001 - a trace failure must not break the answer
            log.warning("trace not saved", error=str(exc), trace_id=str(trace.trace_id))
