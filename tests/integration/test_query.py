"""Retrieval + answer service over a real index (fake embedder, fake LLM)."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import pytest
from sqlalchemy import select

from ragchat.agent.answer import AnswerDraft, AnswerService
from ragchat.agent.traces import Trace
from ragchat.core.llm import StructuredResult, Usage
from ragchat.ingest.connectors.local import LocalFolderConnector
from ragchat.ingest.ir import make_doc_id
from ragchat.retrieval.retriever import Retriever

pytestmark = pytest.mark.integration
FIXTURES = Path(__file__).parent.parent / "fixtures"


def words(text: str) -> int:
    return len(text.split())


@pytest.fixture
async def indexed(env: dict, tmp_path: Path) -> dict:
    root = tmp_path / "corpus"
    root.mkdir()
    for name in ("handbook.md", "wiki.html"):
        shutil.copy(FIXTURES / name, root / name)
    report = await env["pipeline"].run(LocalFolderConnector(root))
    assert report.indexed == 2
    records = await env["store"].scroll_doc(make_doc_id("local", str(root / "handbook.md")))
    return {**env, "records": records}


async def test_retrieve_returns_exact_chunk_first(indexed: dict):
    settings = indexed["settings"]
    retriever = Retriever(settings, indexed["store"], indexed["embed_fn"])
    target = indexed["records"][2].payload
    # With the hash embedder, a query equal to the embed text is an exact vector match.
    hits = await retriever.retrieve(f"{target['breadcrumb']}\n\n{target['text']}")
    assert hits[0].chunk_id == str(indexed["records"][2].id)
    assert hits[0].score == pytest.approx(1.0, abs=1e-5)
    assert hits[0].breadcrumb == target["breadcrumb"] and hits[0].title == "Employee Handbook"
    assert len(hits) <= settings.retrieval.top_k
    # per-document cap holds
    per_doc: dict[str, int] = {}
    for h in hits:
        per_doc[h.doc_id] = per_doc.get(h.doc_id, 0) + 1
    assert max(per_doc.values()) <= settings.retrieval.max_chunks_per_doc


async def test_answer_persists_trace(indexed: dict):
    settings = indexed["settings"]
    retriever = Retriever(settings, indexed["store"], indexed["embed_fn"])

    class FakeLLM:
        async def generate_structured(self, system, user, schema, *, model=None):  # noqa: ANN001
            return StructuredResult(
                parsed=AnswerDraft(answer="Answer [1].", citations=[1]),
                usage=Usage(50, 7),
                model="fake-model",
            )

    svc = AnswerService(settings, retriever, FakeLLM(), indexed["sessions"], words)  # type: ignore[arg-type]
    target = indexed["records"][0].payload
    answer = await svc.answer(f"{target['breadcrumb']}\n\n{target['text']}")
    assert not answer.refused and answer.citations[0].chunk_id == answer.retrieved_chunk_ids[0]

    async with indexed["sessions"]() as s:
        trace = await s.scalar(select(Trace).where(Trace.trace_id == uuid.UUID(answer.trace_id)))
    assert trace is not None
    assert trace.query.startswith("Employee Handbook") and trace.answer == "Answer [1]."
    assert trace.cited_chunk_ids == [answer.citations[0].chunk_id]
    assert [r["chunk_id"] for r in trace.retrieved] == answer.retrieved_chunk_ids
    assert trace.retrieved[0]["score"] == pytest.approx(1.0, abs=1e-4)
    assert (trace.input_tokens, trace.output_tokens, trace.model) == (50, 7, "fake-model")
    assert trace.latency_ms >= 0 and trace.error is None and not trace.refused
    assert trace.meta["context_tokens"] > 0
