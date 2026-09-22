"""Run the golden set through retrieval (and optionally answering + ragas), summarise, and
compare with a committed baseline (PLAN.md rule 12: measure before adding retrieval features)."""

from __future__ import annotations

import asyncio
import json
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ragchat.agent.answer import Answer, AnswerService
from ragchat.core.logging import get_logger
from ragchat.core.settings import Settings
from ragchat.eval.golden import GoldenItem, golden_digest
from ragchat.eval.metrics import (
    AnswerScore,
    RetrievalScore,
    Summary,
    score_answer,
    score_retrieval,
    summarize,
)
from ragchat.retrieval.retriever import RetrievedChunk, Retriever

log = get_logger(__name__)

RAGAS_METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")


@dataclass
class ItemResult:
    item: GoldenItem
    hits: list[RetrievedChunk]
    retrieval: RetrievalScore | None  # None for refusal-kind questions
    answer: Answer | None = None
    answer_score: AnswerScore | None = None
    ragas: dict[str, float] = field(default_factory=dict)

    def row(self) -> dict[str, Any]:
        return {
            "id": self.item.id,
            "question": self.item.question,
            "kind": self.item.kind,
            "first_hit_rank": self.retrieval.first_hit_rank if self.retrieval else None,
            "retrieved": [h.chunk_id for h in self.hits],
            "answer": self.answer.answer if self.answer else None,
            "refused": self.answer.refused if self.answer else None,
            "cited": [c.chunk_id for c in self.answer.citations] if self.answer else None,
            "citation_precision": (
                self.answer_score.citation_precision if self.answer_score else None
            ),
            "ragas": self.ragas or None,
        }


@dataclass
class EvalRun:
    summary: Summary
    results: list[ItemResult]
    config: dict[str, Any]
    golden_path: str
    golden_digest: str
    started_at: str

    def to_baseline(self, name: str) -> dict[str, Any]:
        return {
            "name": name,
            "created_at": self.started_at,
            "git_commit": _git_commit(),
            "golden": {
                "path": self.golden_path,
                "digest": self.golden_digest,
                "n": len(self.results),
            },
            "config": self.config,
            "metrics": self.summary.flat(self.config["k"]),
        }


class RagasJudge:
    """Thin wrapper over ragas' metric classes, using the project's OpenAI client."""

    def __init__(self, settings: Settings, client: Any) -> None:
        from ragas.embeddings.base import embedding_factory
        from ragas.llms import llm_factory
        from ragas.metrics.collections import (
            AnswerRelevancy,
            ContextPrecision,
            ContextRecall,
            Faithfulness,
        )

        llm = llm_factory(settings.llm.small_model, client=client)
        emb = embedding_factory(
            "openai", model=settings.embedding.model, client=client, interface="modern"
        )
        self.faithfulness = Faithfulness(llm=llm)
        self.answer_relevancy = AnswerRelevancy(llm=llm, embeddings=emb)
        self.context_precision = ContextPrecision(llm=llm)
        self.context_recall = ContextRecall(llm=llm)

    async def score(self, item: GoldenItem, contexts: list[str], answer: str) -> dict[str, float]:
        q = item.question
        out: dict[str, float] = {}
        f = await self.faithfulness.ascore(
            user_input=q, response=answer, retrieved_contexts=contexts
        )
        out["faithfulness"] = float(f.value)
        r = await self.answer_relevancy.ascore(user_input=q, response=answer)
        out["answer_relevancy"] = float(r.value)
        if item.answer:
            p = await self.context_precision.ascore(
                user_input=q, reference=item.answer, retrieved_contexts=contexts
            )
            out["context_precision"] = float(p.value)
            c = await self.context_recall.ascore(
                user_input=q, retrieved_contexts=contexts, reference=item.answer
            )
            out["context_recall"] = float(c.value)
        return out


class EvalRunner:
    def __init__(
        self,
        settings: Settings,
        retriever: Retriever,
        answer_service: AnswerService | None = None,
        judge: RagasJudge | None = None,
        concurrency: int = 4,
    ) -> None:
        self.s = settings
        self.retriever = retriever
        self.answer_service = answer_service
        self.judge = judge
        self.sem = asyncio.Semaphore(concurrency)

    def config(self) -> dict[str, Any]:
        r = self.s.retrieval
        return {
            "k": self.s.eval.k,
            "embedding_model": self.s.embedding.model,
            "dims": self.s.embedding.dimensions,
            "llm_model": self.s.llm.model,
            "pipeline_version": self.s.pipeline_version,
            "chunking": self.s.chunking.model_dump(),
            "retrieval": {
                "prefetch_k": r.prefetch_k,
                "top_k": r.top_k,
                "max_chunks_per_doc": r.max_chunks_per_doc,
                "hybrid": r.hybrid,
                "rerank": r.rerank,
                "query_rewrite": r.query_rewrite,
            },
            "answers": self.answer_service is not None,
            "ragas": self.judge is not None,
        }

    async def run(self, items: list[GoldenItem], golden_path: Path) -> EvalRun:
        started = datetime.now(UTC).isoformat(timespec="seconds")
        results = await asyncio.gather(*(self._one(item) for item in items))
        factual = [r for r in results if not r.item.expects_refusal]
        refusal = [r for r in results if r.item.expects_refusal]
        ragas: dict[str, list[float]] = {m: [] for m in RAGAS_METRICS}
        for r in factual:
            for m, v in r.ragas.items():
                ragas[m].append(v)
        summary = summarize(
            [r.retrieval for r in factual if r.retrieval],
            [r.answer_score for r in factual if r.answer_score] or None,
            [r.answer_score for r in refusal if r.answer_score] or None,
            ragas if self.judge else None,
        )
        summary.n_refusal = len(refusal)
        return EvalRun(
            summary=summary,
            results=list(results),
            config=self.config(),
            golden_path=str(golden_path),
            golden_digest=golden_digest(golden_path),
            started_at=started,
        )

    async def _one(self, item: GoldenItem) -> ItemResult:
        async with self.sem:
            # Retrieve deeper than top_k so recall@k for k > top_k is measurable.
            depth = max(self.s.eval.k, max(10, self.s.retrieval.top_k))
            hits = await self.retriever.retrieve(item.question, top_k=depth)
            retrieval = None if item.expects_refusal else score_retrieval(item.expected, hits)
            result = ItemResult(item=item, hits=hits, retrieval=retrieval)
            if self.answer_service is None:
                return result
            answer = await self.answer_service.answer(item.question)
            cited_ids = {c.chunk_id for c in answer.citations}
            cited = [h for h in hits if h.chunk_id in cited_ids]
            result.answer = answer
            result.answer_score = score_answer(item, answer.refused, cited)
            if self.judge and not item.expects_refusal and not answer.refused:
                # Judge against what the model saw: breadcrumb + text (as in render_document).
                contexts = [
                    f"{h.breadcrumb}\n\n{h.text}" if h.breadcrumb else h.text
                    for h in hits[: self.s.retrieval.top_k]
                ]
                try:
                    result.ragas = await self.judge.score(item, contexts, answer.answer)
                except Exception as exc:  # noqa: BLE001 - judge failures should not kill the run
                    log.warning("ragas failed", id=item.id, error=str(exc))
            return result


# -- baselines ------------------------------------------------------------------------------


def save_baseline(run: EvalRun, path: Path, name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(run.to_baseline(name), indent=2) + "\n", encoding="utf-8")


def load_baseline(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass
class Comparison:
    rows: list[tuple[str, float | None, float | None, float | None]]  # metric, base, cur, delta
    regressed: bool
    gate_metric: str
    warnings: list[str] = field(default_factory=list)


def compare(run: EvalRun, baseline: dict[str, Any], threshold: float) -> Comparison:
    k = run.config["k"]
    current = run.summary.flat(k)
    base_metrics: dict[str, float | None] = baseline.get("metrics", {})
    rows = []
    for name in sorted(set(current) | set(base_metrics), key=_metric_order):
        b, c = base_metrics.get(name), current.get(name)
        delta = (c - b) if (b is not None and c is not None) else None
        rows.append((name, b, c, delta))
    gate = f"recall@{k}"
    b, c = base_metrics.get(gate), current.get(gate)
    regressed = b is not None and c is not None and (b - c) > threshold
    warnings = []
    if baseline.get("golden", {}).get("digest") != run.golden_digest:
        warnings.append("golden set changed since the baseline was recorded")
    if baseline.get("config", {}).get("k") != k:
        warnings.append(f"baseline k={baseline.get('config', {}).get('k')} differs from k={k}")
    return Comparison(rows, regressed, gate, warnings)


def _metric_order(name: str) -> tuple[int, int, str]:
    if name == "mrr":
        return (0, 0, name)
    if "@" in name:
        prefix, k = name.split("@")
        return (1 if prefix == "recall" else 2, int(k), name)
    return (3 if not name.startswith("ragas_") else 4, 0, name)


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except Exception:  # noqa: BLE001
        return None
