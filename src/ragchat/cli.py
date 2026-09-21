"""`rag` command-line interface.

Milestone 0 provides `status`. Later milestones add ingest, query, eval, worker, reindex.
"""

from __future__ import annotations

import asyncio

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


if __name__ == "__main__":
    app()
