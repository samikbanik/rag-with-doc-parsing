"""`rag` command-line interface.

Milestone 0: `status`, `config`. Milestone 1: `ingest`, `reindex`.
Later milestones add query, eval, worker.
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
