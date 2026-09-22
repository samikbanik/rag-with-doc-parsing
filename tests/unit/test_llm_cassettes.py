"""OpenAI calls replayed from vcrpy cassettes (tests/cassettes/). Offline by default.

Re-record after changing prompts/schemas:  uv run pytest tests/unit/test_llm_cassettes.py \
    --record-mode=once   (delete the cassette first; needs OPENAI_API_KEY in .env)
"""

from __future__ import annotations

import pytest
from openai import AsyncOpenAI

from ragchat.agent.answer import AnswerDraft, AnswerService, validate_citations
from ragchat.core.llm import LLMClient
from ragchat.core.settings import get_settings
from ragchat.retrieval.assemble import assemble
from ragchat.retrieval.retriever import RetrievedChunk

pytestmark = pytest.mark.vcr


_DROP_RESPONSE_HEADERS = {"set-cookie", "openai-organization", "openai-project", "x-request-id"}


def _scrub_response(response):  # noqa: ANN001, ANN202
    headers = response.get("headers", {})
    for name in list(headers):
        if name.lower() in _DROP_RESPONSE_HEADERS:
            del headers[name]
    return response


@pytest.fixture(scope="module")
def vcr_config():
    return {
        "filter_headers": ["authorization", "openai-organization", "openai-project", "cookie"],
        "before_record_response": _scrub_response,
        "cassette_library_dir": "tests/cassettes",
    }


@pytest.fixture
def llm() -> LLMClient:
    key = get_settings().openai_api_key.get_secret_value() or "sk-test"
    return LLMClient(client=AsyncOpenAI(api_key=key))


def _chunk(i: int, text: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"chunk-{i}",
        doc_id="handbook",
        chunk_index=i,
        score=0.6,
        title="Employee Handbook",
        uri="/docs/handbook.md",
        source="local",
        doc_type="md",
        breadcrumb="Employee Handbook > Welcome > Time off",
        text=text,
    )


HITS = [
    _chunk(0, "Everyone gets 25 days of paid leave per year. Unused days carry over."),
    _chunk(1, "Reimbursement takes 5 business days."),
]


async def test_embed_query_returns_configured_dims(llm: LLMClient):
    vec = (await llm.embed(["How many days of paid leave do I get?"]))[0]
    assert len(vec) == get_settings().embedding.dimensions
    assert abs(sum(x * x for x in vec) - 1.0) < 1e-3  # OpenAI embeddings are unit-normalised


async def test_generate_cited_answer(llm: LLMClient):
    s = get_settings()
    svc = AnswerService(s, None, llm, None, len)  # type: ignore[arg-type]
    ctx = assemble(HITS, max_tokens=10_000, count_tokens=len)
    result = await svc.generate("How many days of paid leave do employees get?", ctx)
    assert isinstance(result.parsed, AnswerDraft)
    assert result.usage.input_tokens > 0 and result.usage.output_tokens > 0
    text, cites = validate_citations(result.parsed, ctx)
    assert "25" in text
    assert [c.chunk_id for c in cites] == ["chunk-0"]
    assert not result.parsed.insufficient_context


async def test_generate_refuses_when_context_is_irrelevant(llm: LLMClient):
    svc = AnswerService(get_settings(), None, llm, None, len)  # type: ignore[arg-type]
    ctx = assemble(HITS, max_tokens=10_000, count_tokens=len)
    result = await svc.generate("What is the office parking policy?", ctx)
    text, cites = validate_citations(result.parsed, ctx)
    assert result.parsed.insufficient_context or not cites
