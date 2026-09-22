from __future__ import annotations

import pytest

from ragchat.agent.answer import REFUSAL, AnswerDraft, AnswerService, validate_citations
from ragchat.core.llm import StructuredResult, Usage
from ragchat.core.settings import Settings
from ragchat.retrieval.assemble import assemble, render_document
from ragchat.retrieval.retriever import RetrievedChunk, cap_per_doc


def words(text: str) -> int:
    return len(text.split())


def chunk(
    i: int, doc: str = "d1", text: str | None = None, page: int | None = None
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"c{i}",
        doc_id=doc,
        chunk_index=i,
        score=1.0 - i / 100,
        title=f"Doc {doc}",
        uri=f"/docs/{doc}.md",
        source="local",
        doc_type="md",
        breadcrumb=f"Doc {doc} > Section {i}",
        text=text or f"Sentence number {i} with some words.",
        page_start=page,
        page_end=page,
    )


# -- retriever ----------------------------------------------------------------------------------


def test_cap_per_doc_keeps_order():
    docs = ["a", "a", "b", "a", "b", "c"]
    hits = [chunk(i, d) for i, d in enumerate(docs, start=1)]
    kept = cap_per_doc(hits, 2)
    assert [h.chunk_id for h in kept] == ["c1", "c2", "c3", "c5", "c6"]


# -- assembly -----------------------------------------------------------------------------------


def test_render_document_attributes_and_pages():
    c = chunk(1, page=3)
    out = render_document(1, c)
    assert out.startswith('<document id="1" title="Doc d1" uri="/docs/d1.md" page="3">')
    assert "Doc d1 > Section 1\n\nSentence number 1" in out and out.endswith("</document>")
    c2 = c.model_copy(update={"page_end": 5, "title": 'He said "hi"'})
    out2 = render_document(2, c2)
    assert 'page="3-5"' in out2 and "title='He said \"hi\"'" in out2
    assert "page=" not in render_document(1, chunk(2))


def test_assemble_respects_budget_and_numbers_in_order():
    hits = [chunk(i) for i in range(1, 6)]
    per_block = words(render_document(1, hits[0]))
    ctx = assemble(hits, max_tokens=per_block * 3 + 1, count_tokens=words)
    assert len(ctx.chunks) == 3 and ctx.dropped == 2
    assert ctx.text.count("<document ") == 3
    assert ctx.chunk_for(1) is hits[0] and ctx.chunk_for(3) is hits[2]
    assert ctx.chunk_for(4) is None and ctx.chunk_for(0) is None
    # the first chunk is always included even if it alone exceeds the budget
    assert len(assemble(hits, max_tokens=1, count_tokens=words).chunks) == 1
    assert assemble([], max_tokens=100, count_tokens=words).text == ""


# -- citation validation ------------------------------------------------------------------------


def test_validate_citations_maps_markers_and_strips_invalid():
    ctx = assemble([chunk(1), chunk(2)], max_tokens=10_000, count_tokens=words)
    draft = AnswerDraft(answer="Leave is 25 days [1]. Also see [7] and [2][1].", citations=[1, 9])
    text, cites = validate_citations(draft, ctx)
    assert text == "Leave is 25 days [1]. Also see and [2][1]."
    assert [c.number for c in cites] == [1, 2]
    assert cites[0].chunk_id == "c1" and cites[1].chunk_id == "c2"
    assert cites[0].snippet.startswith("Sentence number 1")


def test_validate_citations_uses_list_when_text_has_no_markers():
    ctx = assemble([chunk(1)], max_tokens=10_000, count_tokens=words)
    text, cites = validate_citations(AnswerDraft(answer="Plain answer.", citations=[1]), ctx)
    assert text == "Plain answer." and [c.chunk_id for c in cites] == ["c1"]


def test_snippet_is_truncated():
    ctx = assemble([chunk(1, text="x" * 1000)], max_tokens=10_000, count_tokens=words)
    _, cites = validate_citations(AnswerDraft(answer="[1]"), ctx)
    assert len(cites[0].snippet) == 240 and cites[0].snippet.endswith("…")


# -- service (fake retriever + fake llm, no network, no DB) ---------------------------------------


class FakeRetriever:
    def __init__(self, hits: list[RetrievedChunk]) -> None:
        self.hits = hits

    async def retrieve(self, query: str, *, top_k=None):  # noqa: ANN001
        return self.hits[:top_k] if top_k else self.hits


class FakeLLM:
    def __init__(self, draft: AnswerDraft) -> None:
        self.draft = draft
        self.calls: list[tuple[str, str]] = []

    async def generate_structured(self, system, user, schema, *, model=None):  # noqa: ANN001
        self.calls.append((system, user))
        return StructuredResult(parsed=self.draft, usage=Usage(100, 20), model=model or "fake")


def service(hits: list[RetrievedChunk], draft: AnswerDraft) -> tuple[AnswerService, FakeLLM]:
    llm = FakeLLM(draft)
    svc = AnswerService(Settings(), FakeRetriever(hits), llm, None, words)  # type: ignore[arg-type]
    return svc, llm


async def test_answer_happy_path():
    draft = AnswerDraft(answer="It is 25 days [2].", citations=[2])
    svc, llm = service([chunk(1), chunk(2)], draft)
    a = await svc.answer("how many days?")
    assert a.answer == "It is 25 days [2]." and not a.refused
    assert [c.chunk_id for c in a.citations] == ["c2"]
    assert a.retrieved_chunk_ids == ["c1", "c2"]
    assert a.usage == {"input_tokens": 100, "output_tokens": 20}
    system, user = llm.calls[0]
    assert '<document id="1"' in user and "Question: how many days?" in user
    assert "only" in system.lower()


async def test_answer_refuses_without_valid_citations():
    svc, _ = service([chunk(1)], AnswerDraft(answer="Made up fact [5].", citations=[5]))
    a = await svc.answer("q")
    assert a.refused and a.answer == REFUSAL and a.citations == []


async def test_answer_passes_through_insufficient_context_message():
    draft = AnswerDraft(answer="The documents do not mention parking.", insufficient_context=True)
    svc, _ = service([chunk(1)], draft)
    a = await svc.answer("parking?")
    assert a.refused and a.answer == "The documents do not mention parking."


async def test_answer_refuses_when_nothing_retrieved_without_calling_llm():
    svc, llm = service([], AnswerDraft(answer="should not be used", citations=[1]))
    a = await svc.answer("q")
    assert a.refused and a.answer == REFUSAL and llm.calls == []


async def test_answer_error_is_raised_after_tracing():
    class BoomLLM(FakeLLM):
        async def generate_structured(self, *a, **k):  # noqa: ANN002, ANN003
            raise RuntimeError("boom")

    llm = BoomLLM(AnswerDraft(answer=""))
    svc = AnswerService(Settings(), FakeRetriever([chunk(1)]), llm, None, words)  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="boom"):
        await svc.answer("q")
