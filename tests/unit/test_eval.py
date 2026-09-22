from __future__ import annotations

import json
from pathlib import Path

import pytest

from ragchat.agent.answer import Answer, Citation
from ragchat.core.settings import Settings
from ragchat.eval.golden import Expected, GoldenItem, load_golden, save_golden
from ragchat.eval.metrics import score_answer, score_retrieval, summarize
from ragchat.eval.runner import EvalRun, EvalRunner, compare, load_baseline, save_baseline
from ragchat.retrieval.retriever import RetrievedChunk


def chunk(i: int, uri: str = "/data/handbook.md", text: str = "some text") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"c{i}",
        doc_id="d",
        chunk_index=i,
        score=1.0,
        title="t",
        uri=uri,
        source="local",
        doc_type="md",
        breadcrumb="b",
        text=text,
    )


# -- golden matching ------------------------------------------------------------------------


def test_expected_matching_rules():
    c = chunk(3, "/other/checkout/handbook.md", "Everyone gets 25 days of  paid leave.")
    assert Expected(uri="x", chunk_id="c3").matches(c)  # id wins regardless of uri
    assert Expected(uri="tests/fixtures/handbook.md", chunk_index=3).matches(c)  # by file name
    assert not Expected(uri="tests/fixtures/wiki.html", chunk_index=3).matches(c)
    assert Expected(uri="handbook.md", span="25 DAYS of paid").matches(c)  # normalised span
    assert not Expected(uri="handbook.md", span="30 days").matches(c)
    assert not Expected(uri="handbook.md", chunk_index=4).matches(c)


def test_golden_round_trip_and_duplicate_ids(tmp_path: Path):
    items = [
        GoldenItem(id="a", question="q?", expected=[Expected(uri="f.md", span="s")]),
        GoldenItem(id="b", question="parking?", kind="refusal"),
    ]
    path = tmp_path / "golden.jsonl"
    save_golden(path, items)
    loaded = load_golden(path)
    assert loaded == items
    assert loaded[1].expects_refusal and not loaded[0].expects_refusal
    assert "chunk_id" not in path.read_text()  # None fields are not written
    path.write_text(path.read_text() * 2)
    with pytest.raises(ValueError, match="duplicate"):
        load_golden(path)


# -- metrics --------------------------------------------------------------------------------


def test_score_retrieval_ranks_and_recall():
    expected = [Expected(uri="f.md", chunk_index=2), Expected(uri="f.md", chunk_index=9)]
    hits = [chunk(5, "f.md"), chunk(2, "f.md"), chunk(7, "f.md"), chunk(9, "f.md")]
    s = score_retrieval(expected, hits, ks=(1, 3, 5))
    assert s.first_hit_rank == 2 and s.reciprocal_rank == 0.5
    assert s.recall_at == {1: 0.0, 3: 0.5, 5: 1.0}
    assert s.hit_at == {1: 0.0, 3: 1.0, 5: 1.0}
    miss = score_retrieval(expected, [chunk(1, "g.md")], ks=(1,))
    assert miss.first_hit_rank is None and miss.reciprocal_rank == 0.0


def test_score_answer_citation_precision_and_refusals():
    item = GoldenItem(id="a", question="q", expected=[Expected(uri="f.md", chunk_index=1)])
    s = score_answer(item, False, [chunk(1, "f.md"), chunk(2, "f.md")])
    assert s.citation_precision == 0.5 and s.citation_hit and s.refusal_correct
    s = score_answer(item, True, [])
    assert s.refused and not s.refusal_correct and s.citation_precision is None
    refusal = GoldenItem(id="b", question="q", kind="refusal")
    assert score_answer(refusal, True, []).refusal_correct
    assert not score_answer(refusal, False, [chunk(1)]).refusal_correct


def test_summarize_aggregates():
    exp = [Expected(uri="f.md", chunk_index=1)]
    r1 = score_retrieval(exp, [chunk(1, "f.md")], ks=(1, 3))
    r2 = score_retrieval(exp, [chunk(2), chunk(1, "f.md")], ks=(1, 3))
    item = GoldenItem(id="a", question="q", expected=[Expected(uri="f.md", chunk_index=1)])
    a1 = score_answer(item, False, [chunk(1, "f.md")])
    a2 = score_answer(item, True, [])
    ref = score_answer(GoldenItem(id="r", question="q", kind="refusal"), True, [])
    s = summarize([r1, r2], [a1, a2], [ref], {"faithfulness": [1.0, 0.5], "context_recall": []})
    assert s.n == 2 and s.mrr == 0.75
    assert s.recall_at == {1: 0.5, 3: 1.0}
    assert s.false_refusal_rate == 0.5 and s.citation_precision == 1.0
    assert s.citation_hit_rate == 0.5 and s.refusal_accuracy == 1.0
    assert s.ragas == {"faithfulness": 0.75}
    flat = s.flat(3)
    assert flat["recall@3"] == 1.0 and flat["ragas_faithfulness"] == 0.75


# -- runner with fakes ------------------------------------------------------------------------


class FakeRetriever:
    """Returns chunks by looking the question up in a table; unknown → unrelated chunk."""

    def __init__(self, table: dict[str, list[RetrievedChunk]]) -> None:
        self.table = table

    async def retrieve(self, query: str, *, top_k=None):  # noqa: ANN001
        return self.table.get(query, [chunk(99, "unrelated.md")])[:top_k]


class FakeAnswers:
    def __init__(self, retriever: FakeRetriever) -> None:
        self.retriever = retriever

    async def answer(self, question: str, *, top_k=None) -> Answer:  # noqa: ANN001
        hits = await self.retriever.retrieve(question)
        refused = hits[0].uri == "unrelated.md"
        cites = (
            []
            if refused
            else [
                Citation(
                    number=1,
                    chunk_id=hits[0].chunk_id,
                    doc_id="d",
                    title="t",
                    uri=hits[0].uri,
                    snippet="s",
                    score=1.0,
                )
            ]
        )
        return Answer(
            answer="no" if refused else "yes",
            citations=cites,
            retrieved_chunk_ids=[h.chunk_id for h in hits],
            trace_id="t",
            refused=refused,
        )


GOLDEN = [
    GoldenItem(id="g1", question="leave?", expected=[Expected(uri="h.md", chunk_index=1)]),
    GoldenItem(id="g2", question="meals?", expected=[Expected(uri="h.md", chunk_index=3)]),
    GoldenItem(id="g3", question="parking?", kind="refusal"),
]
TABLE = {
    "leave?": [chunk(1, "h.md"), chunk(2, "h.md")],
    "meals?": [chunk(2, "h.md"), chunk(3, "h.md")],  # expected chunk at rank 2
}


async def test_runner_end_to_end(tmp_path: Path):
    golden_path = tmp_path / "golden.jsonl"
    save_golden(golden_path, GOLDEN)
    retriever = FakeRetriever(TABLE)
    runner = EvalRunner(Settings(), retriever, FakeAnswers(retriever))  # type: ignore[arg-type]
    run = await runner.run(GOLDEN, golden_path)
    s = run.summary
    assert (s.n, s.n_refusal) == (2, 1)
    assert s.mrr == 0.75 and s.recall_at[1] == 0.5 and s.recall_at[3] == 1.0
    assert s.refusal_accuracy == 1.0 and s.false_refusal_rate == 0.0
    assert s.citation_precision == 0.5 and s.citation_hit_rate == 0.5  # g2 cited rank-1 chunk
    assert s.ragas == {} and run.config["answers"] and not run.config["ragas"]
    rows = [r.row() for r in run.results]
    assert rows[1]["first_hit_rank"] == 2 and rows[2]["refused"] is True
    json.dumps(rows)  # serialisable for --out


async def test_baseline_save_compare_and_regression_gate(tmp_path: Path):
    golden_path = tmp_path / "golden.jsonl"
    save_golden(golden_path, GOLDEN)
    settings = Settings()
    good = await EvalRunner(settings, FakeRetriever(TABLE)).run(GOLDEN, golden_path)  # type: ignore[arg-type]
    path = tmp_path / "current.json"
    save_baseline(good, path, "current")
    base = load_baseline(path)
    assert base["name"] == "current" and base["golden"]["n"] == 3
    assert base["metrics"]["recall@8"] == 1.0 and base["config"]["k"] == settings.eval.k

    # same run → no regression, deltas are zero
    cmp = compare(good, base, threshold=0.05)
    assert not cmp.regressed and cmp.gate_metric == f"recall@{settings.eval.k}"
    assert all(d == 0 for _, _, _, d in cmp.rows if d is not None) and cmp.warnings == []

    # a retriever that loses one question → recall@8 drops 0.5 → regression
    bad_table = {"leave?": TABLE["leave?"]}
    bad = await EvalRunner(settings, FakeRetriever(bad_table)).run(GOLDEN, golden_path)  # type: ignore[arg-type]
    cmp = compare(bad, base, threshold=0.05)
    assert cmp.regressed
    delta = dict((name, d) for name, _, _, d in cmp.rows)[f"recall@{settings.eval.k}"]
    assert delta == pytest.approx(-0.5)

    # changed golden set is flagged
    save_golden(golden_path, GOLDEN[:2])
    run2 = await EvalRunner(settings, FakeRetriever(TABLE)).run(GOLDEN[:2], golden_path)  # type: ignore[arg-type]
    assert any("golden set changed" in w for w in compare(run2, base, 0.05).warnings)


def test_eval_run_baseline_shape():
    run = EvalRun(summarize([], None, None), [], {"k": 8}, "g", "abc", "now")
    b = run.to_baseline("x")
    assert set(b) == {"name", "created_at", "git_commit", "golden", "config", "metrics"}
