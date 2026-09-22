"""`rag` command-line interface.

Milestone 0: `status`, `config`. Milestone 1: `ingest`, `reindex`. Milestone 2: `query`.
Milestone 3: `eval`. Later milestones add worker.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from ragchat.core.logging import configure_logging
from ragchat.core.settings import get_settings

app = typer.Typer(help="Enterprise agentic RAG chat", no_args_is_help=True)
console = Console()


@app.callback()
def _main(verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging")) -> None:
    configure_logging("DEBUG" if verbose else "INFO")


async def _check(name: str, coro) -> tuple[str, bool, str]:  # noqa: ANN001
    try:
        detail = await coro
        return name, True, str(detail)
    except Exception as exc:  # noqa: BLE001 - we want to report any failure
        return name, False, f"{type(exc).__name__}: {exc}"


@app.command()
def status() -> None:
    """Check connectivity to Qdrant, Postgres and OpenAI."""
    from ragchat.core import db
    from ragchat.core.llm import get_llm
    from ragchat.retrieval import vectorstore

    s = get_settings()

    async def run() -> list[tuple[str, bool, str]]:
        return await asyncio.gather(
            _check(f"Qdrant ({s.qdrant_url})", vectorstore.ping()),
            _check("Postgres", db.ping()),
            _check(f"OpenAI (model={s.llm.model})", get_llm().ping()),
        )

    results = asyncio.run(run())

    table = Table(title="ragchat status")
    table.add_column("Service")
    table.add_column("OK")
    table.add_column("Detail", overflow="fold")
    for name, ok, detail in results:
        table.add_row(name, "[green]yes[/]" if ok else "[red]no[/]", detail)
    console.print(table)

    console.print(
        f"embedding={s.embedding.model}@{s.embedding.dimensions}d  "
        f"collection={s.vectorstore.collection}  pipeline_version={s.pipeline_version}"
    )
    if not all(ok for _, ok, _ in results):
        raise typer.Exit(code=1)


@app.command()
def config() -> None:
    """Print effective configuration (secrets redacted)."""
    console.print_json(get_settings().model_dump_json(indent=2))


def _pipeline(workers: int | None = None):  # noqa: ANN202
    from ragchat.core.db import get_sessionmaker
    from ragchat.ingest.embedding import Embedder
    from ragchat.ingest.pipeline import IngestPipeline
    from ragchat.retrieval.vectorstore import VectorStore

    s = get_settings()
    sessions = get_sessionmaker()
    return IngestPipeline(
        s, sessions, VectorStore(s), Embedder(s.embedding, sessions), workers=workers
    )


def _print_report(report, title: str) -> None:  # noqa: ANN001
    table = Table(title=title)
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    rows = [
        ("scanned", report.scanned),
        ("unsupported (skipped)", report.unsupported),
        ("unchanged", report.unchanged),
        ("indexed", report.indexed),
        ("empty (no chunks)", report.empty),
        ("failed", report.failed),
        ("pruned", report.pruned),
        ("chunks written", report.chunks),
        ("embeddings: cache hits", report.embed.cache_hits),
        ("embeddings: API calls", report.embed.api_calls),
        ("embeddings: tokens", report.embed.tokens),
        ("embeddings: est. USD", f"{report.embed.usd:.4f}"),
    ]
    for k, v in rows:
        table.add_row(k, str(v))
    console.print(table)
    if report.planned:
        console.print("[bold]Would (re)index:[/]")
        for uri in report.planned:
            console.print(f"  {uri}")
    for uri, err in report.errors:
        console.print(f"[red]failed[/] {uri}: {err}")


def _run_reporting(coro, title: str) -> None:  # noqa: ANN001
    from sqlalchemy.exc import ProgrammingError

    from ragchat.retrieval.vectorstore import IndexConfigMismatchError

    try:
        report = asyncio.run(coro)
    except IndexConfigMismatchError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(code=2) from None
    except ProgrammingError as exc:
        if "does not exist" in str(exc):
            console.print("[red]database schema missing; run `make migrate` first[/]")
            raise typer.Exit(code=2) from None
        raise
    _print_report(report, title)
    if report.failed:
        raise typer.Exit(code=1)


@app.command()
def ingest(
    path: Annotated[Path, typer.Option("--path", exists=True, file_okay=False, resolve_path=True)],
    workers: int | None = typer.Option(None, help="Parser processes (default: min(4, cpus-1))"),
    force: bool = typer.Option(False, help="Re-index even if unchanged"),
    prune: bool = typer.Option(True, help="Remove documents no longer present in the directory"),
    dry_run: bool = typer.Option(False, help="Only report what would be (re)indexed"),
) -> None:
    """Ingest every supported file under a directory (incremental by content hash)."""
    from ragchat.ingest.connectors.local import LocalFolderConnector

    connector = LocalFolderConnector(path)
    pipeline = _pipeline(workers)
    _run_reporting(
        pipeline.run(connector, force=force, prune=prune, dry_run=dry_run),
        f"rag ingest {path}{' (dry run)' if dry_run else ''}",
    )


def _answer_service():  # noqa: ANN202
    from ragchat.agent.answer import AnswerService
    from ragchat.core.db import get_sessionmaker
    from ragchat.core.llm import get_llm
    from ragchat.ingest.chunking import tiktoken_counter
    from ragchat.retrieval.retriever import Retriever
    from ragchat.retrieval.vectorstore import VectorStore

    s = get_settings()
    llm = get_llm()
    retriever = Retriever(s, VectorStore(s), llm.embed)
    return AnswerService(s, retriever, llm, get_sessionmaker(), tiktoken_counter())


@app.command()
def query(
    question: str = typer.Argument(..., help="Question to answer from the indexed documents"),
    top_k: int | None = typer.Option(None, help="Chunks to retrieve (default: settings)"),
    show_context: bool = typer.Option(False, help="Print the retrieved chunks too"),
    as_json: bool = typer.Option(False, "--json", help="Print the answer contract as JSON"),
) -> None:
    """Answer a question with citations (baseline dense RAG)."""
    from rich.markdown import Markdown
    from rich.panel import Panel

    service = _answer_service()
    answer = asyncio.run(service.answer(question, top_k=top_k))

    if as_json:
        console.print_json(answer.model_dump_json())
        return

    style = "yellow" if answer.refused else "green"
    console.print(Panel(Markdown(answer.answer), title="Answer", border_style=style))

    if answer.citations:
        table = Table(title="Citations")
        table.add_column("#", justify="right")
        table.add_column("Title")
        table.add_column("Page", justify="right")
        table.add_column("Score", justify="right")
        table.add_column("Snippet", overflow="fold")
        for c in answer.citations:
            table.add_row(
                str(c.number),
                f"{c.title}\n[dim]{c.uri}[/]",
                str(c.page) if c.page is not None else "-",
                f"{c.score:.3f}",
                c.snippet,
            )
        console.print(table)

    if show_context:
        console.print(f"[bold]Retrieved {len(answer.retrieved_chunk_ids)} chunk(s):[/]")
        for cid in answer.retrieved_chunk_ids:
            console.print(f"  {cid}")

    console.print(
        f"[dim]trace={answer.trace_id}  tokens={answer.usage.get('input_tokens', 0)}+"
        f"{answer.usage.get('output_tokens', 0)}  latency={answer.latency_ms}ms[/]"
    )


@app.command()
def eval(  # noqa: A001 - CLI verb
    golden: Annotated[
        Path | None, typer.Option(help="Golden set (default: settings.eval.golden_path)")
    ] = None,
    answers: bool = typer.Option(False, help="Also generate answers: citation/refusal metrics"),
    ragas: bool = typer.Option(False, help="Also run ragas judges (implies --answers; costs)"),
    compare: str | None = typer.Option(
        "current", help="Baseline name to diff against ('' to skip)", show_default=True
    ),
    save_baseline: str | None = typer.Option(None, help="Save run as eval/baselines/<name>.json"),
    out: Annotated[Path | None, typer.Option(help="Write per-question results as JSONL")] = None,
    limit: int | None = typer.Option(None, help="Only the first N golden items"),
) -> None:
    """Score retrieval (recall@k, MRR) and optionally answers against the golden set."""
    from ragchat.eval.golden import load_golden
    from ragchat.eval.runner import EvalRunner, RagasJudge, load_baseline
    from ragchat.eval.runner import compare as compare_runs
    from ragchat.eval.runner import save_baseline as save_baseline_file

    s = get_settings()
    golden_path = golden or s.eval.golden_path
    if not golden_path.exists():
        console.print(f"[red]golden set not found: {golden_path}[/] (see eval/generate_golden.py)")
        raise typer.Exit(code=2)
    items = load_golden(golden_path)[:limit]

    service = _answer_service() if (answers or ragas) else None
    if service is None:
        from ragchat.core.llm import get_llm
        from ragchat.retrieval.retriever import Retriever
        from ragchat.retrieval.vectorstore import VectorStore

        retriever = Retriever(s, VectorStore(s), get_llm().embed)
    else:
        retriever = service.retriever
    judge = RagasJudge(s, service.llm.client) if (ragas and service) else None
    runner = EvalRunner(s, retriever, service, judge)
    run = asyncio.run(runner.run(items, golden_path))

    k = s.eval.k
    table = Table(title=f"rag eval  ({run.summary.n} factual, {run.summary.n_refusal} refusal)")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    for name, value in run.summary.flat(k).items():
        if value is not None:
            table.add_row(name, f"{value:.3f}")
    console.print(table)

    misses = [r for r in run.results if r.retrieval and r.retrieval.first_hit_rank is None]
    if misses:
        depth = len(misses[0].hits)
        console.print(f"[yellow]{len(misses)} question(s) with no hit in top {depth}:[/]")
        for r in misses:
            console.print(f"  {r.item.id}: {r.item.question}")

    if out:
        import json

        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("".join(json.dumps(r.row()) + "\n" for r in run.results))
        console.print(f"[dim]per-question results → {out}[/]")

    exit_code = 0
    if compare:
        base_path = s.eval.baselines_dir / f"{compare}.json"
        if base_path.exists():
            cmp = compare_runs(run, load_baseline(base_path), s.eval.regression_threshold)
            diff = Table(title=f"vs baseline '{compare}' ({base_path.name})")
            for col in ("Metric", "Baseline", "Current", "Δ"):
                diff.add_column(col, justify="right" if col != "Metric" else "left")
            for name, b, c, d in cmp.rows:
                fmt = lambda v: "-" if v is None else f"{v:.3f}"  # noqa: E731
                colour = "" if d is None or abs(d) < 1e-9 else ("green" if d > 0 else "red")
                diff.add_row(name, fmt(b), fmt(c), f"[{colour}]{fmt(d)}[/]" if colour else fmt(d))
            console.print(diff)
            for w in cmp.warnings:
                console.print(f"[yellow]warning: {w}[/]")
            if cmp.regressed:
                console.print(
                    f"[red]REGRESSION: {cmp.gate_metric} dropped by more than "
                    f"{s.eval.regression_threshold:.2f}[/]"
                )
                exit_code = 1
        else:
            console.print(f"[dim]no baseline '{compare}' to compare with[/]")

    if save_baseline:
        path = s.eval.baselines_dir / f"{save_baseline}.json"
        save_baseline_file(run, path, save_baseline)
        console.print(f"baseline saved → {path}")
    raise typer.Exit(code=exit_code)


@app.command()
def reindex(
    recreate: bool = typer.Option(
        False, help="Drop the Qdrant collection first (needed after an index config change)"
    ),
) -> None:
    """Re-chunk and re-embed from data/parsed/ without re-parsing."""
    if recreate and not typer.confirm(
        f"Drop collection '{get_settings().vectorstore.collection}' and rebuild?"
    ):
        raise typer.Abort()
    _run_reporting(_pipeline().reindex(recreate=recreate), "rag reindex")


if __name__ == "__main__":
    app()
