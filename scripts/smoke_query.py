"""Run the smoke questions through `AnswerService` and check citations by hand-readable output.

    uv run python scripts/smoke_query.py [eval/smoke_questions.jsonl]

Each line: {"question", "expect_file" (a cited uri must end with it), "expect_text" (answer must
contain it, case-insensitive)} or {"question", "expect_refusal": true}. This is the Milestone 2
acceptance check; Milestone 3 replaces it with the real eval harness.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from ragchat.agent.answer import AnswerService
from ragchat.core.db import get_sessionmaker
from ragchat.core.llm import get_llm
from ragchat.core.settings import get_settings
from ragchat.ingest.chunking import tiktoken_counter
from ragchat.retrieval.retriever import Retriever
from ragchat.retrieval.vectorstore import VectorStore


async def main(path: Path) -> int:
    s = get_settings()
    llm = get_llm()
    svc = AnswerService(
        s, Retriever(s, VectorStore(s), llm.embed), llm, get_sessionmaker(), tiktoken_counter()
    )
    cases = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    console = Console()
    table = Table(title=f"smoke: {path}", show_lines=True)
    for col in ("ok", "question", "answer", "cited files", "top score", "ms"):
        table.add_column(col, overflow="fold")
    failures = 0
    for case in cases:
        a = await svc.answer(case["question"])
        files = sorted({Path(c.uri).name for c in a.citations})
        if case.get("expect_refusal"):
            ok = a.refused
        else:
            ok = any(f.endswith(case["expect_file"]) for f in files) and (
                case["expect_text"].lower() in a.answer.lower()
            )
        failures += not ok
        table.add_row(
            "[green]yes[/]" if ok else "[red]NO[/]",
            case["question"],
            a.answer,
            ", ".join(files) or ("(refused)" if a.refused else "-"),
            f"{a.citations[0].score:.3f}" if a.citations else "-",
            str(a.latency_ms),
        )
    console.print(table)
    console.print(f"{len(cases) - failures}/{len(cases)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("eval/smoke_questions.jsonl")
    raise SystemExit(asyncio.run(main(target)))
