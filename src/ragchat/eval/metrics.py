"""Pure metric functions over one question's retrieval / answer result."""

from __future__ import annotations

from dataclasses import dataclass, field

from ragchat.eval.golden import Expected, GoldenItem
from ragchat.retrieval.retriever import RetrievedChunk

KS = (1, 3, 5, 8, 10)


@dataclass
class RetrievalScore:
    first_hit_rank: int | None  # 1-based rank of the first chunk matching any expected entry
    recall_at: dict[int, float]  # fraction of expected entries matched within top-k
    hit_at: dict[int, float]  # 1.0 if any expected matched within top-k

    @property
    def reciprocal_rank(self) -> float:
        return 1.0 / self.first_hit_rank if self.first_hit_rank else 0.0


def score_retrieval(
    expected: list[Expected], hits: list[RetrievedChunk], ks: tuple[int, ...] = KS
) -> RetrievalScore:
    if not expected:
        return RetrievalScore(None, dict.fromkeys(ks, 0.0), dict.fromkeys(ks, 0.0))
    # rank (1-based) at which each expected entry is first matched
    ranks: list[int | None] = []
    for e in expected:
        rank = next((i for i, h in enumerate(hits, start=1) if e.matches(h)), None)
        ranks.append(rank)
    found = [r for r in ranks if r is not None]
    first = min(found) if found else None
    recall = {k: sum(1 for r in found if r <= k) / len(expected) for k in ks}
    hit = {k: 1.0 if first is not None and first <= k else 0.0 for k in ks}
    return RetrievalScore(first, recall, hit)


@dataclass
class AnswerScore:
    refused: bool
    refusal_correct: bool  # refused ⇔ refusal expected
    citation_precision: float | None  # cited chunks that match expected / cited (None if none)
    citation_hit: bool  # at least one cited chunk matches an expected entry


def score_answer(
    item: GoldenItem, refused: bool, cited: list[RetrievedChunk] | list
) -> AnswerScore:
    if item.expects_refusal:
        return AnswerScore(refused, refused, None, False)
    matched = [c for c in cited if any(e.matches(c) for e in item.expected)]
    precision = len(matched) / len(cited) if cited else None
    return AnswerScore(refused, not refused, precision, bool(matched))


@dataclass
class Summary:
    """Aggregate over a run. `n` counts factual questions for retrieval/citation metrics."""

    n: int = 0
    n_refusal: int = 0
    mrr: float = 0.0
    recall_at: dict[int, float] = field(default_factory=dict)
    hit_at: dict[int, float] = field(default_factory=dict)
    refusal_accuracy: float | None = None  # over refusal-kind questions
    false_refusal_rate: float | None = None  # factual questions that were refused
    citation_precision: float | None = None
    citation_hit_rate: float | None = None
    ragas: dict[str, float] = field(default_factory=dict)

    def flat(self, k: int) -> dict[str, float | None]:
        out: dict[str, float | None] = {"mrr": self.mrr}
        for kk in sorted(self.recall_at):
            out[f"recall@{kk}"] = self.recall_at[kk]
            out[f"hit@{kk}"] = self.hit_at[kk]
        out.update(
            {
                "false_refusal_rate": self.false_refusal_rate,
                "refusal_accuracy": self.refusal_accuracy,
                "citation_precision": self.citation_precision,
                "citation_hit_rate": self.citation_hit_rate,
            }
        )
        out.update({f"ragas_{name}": v for name, v in self.ragas.items()})
        return out


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize(
    retrieval: list[RetrievalScore],
    answers: list[AnswerScore] | None,
    refusal_answers: list[AnswerScore] | None,
    ragas: dict[str, list[float]] | None = None,
) -> Summary:
    s = Summary(n=len(retrieval), n_refusal=len(refusal_answers or []))
    if retrieval:
        s.mrr = sum(r.reciprocal_rank for r in retrieval) / len(retrieval)
        ks = retrieval[0].recall_at.keys()
        s.recall_at = {k: sum(r.recall_at[k] for r in retrieval) / len(retrieval) for k in ks}
        s.hit_at = {k: sum(r.hit_at[k] for r in retrieval) / len(retrieval) for k in ks}
    if answers:
        s.false_refusal_rate = sum(a.refused for a in answers) / len(answers)
        s.citation_precision = _mean(
            [a.citation_precision for a in answers if a.citation_precision is not None]
        )
        s.citation_hit_rate = sum(a.citation_hit for a in answers) / len(answers)
    if refusal_answers:
        s.refusal_accuracy = sum(a.refusal_correct for a in refusal_answers) / len(refusal_answers)
    if ragas:
        s.ragas = {name: m for name, vals in ragas.items() if (m := _mean(vals)) is not None}
    return s
