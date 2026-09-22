"""Benchmark Docling on one or more PDFs: cold (model load) vs warm conversion time.

    uv run python scripts/bench_docling.py tests/fixtures/report.pdf [more.pdf ...] [--runs 3]

Numbers feed the "parsing runs in a process pool" sizing and the pymupdf4llm-fast-path
decision in PLAN.md.
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

from ragchat.ingest.parsers.docling_parser import DoclingParser


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()

    parser = DoclingParser()
    t0 = time.perf_counter()
    parser.converter  # noqa: B018 - force model load
    print(f"converter init: {time.perf_counter() - t0:.2f}s")

    for path in args.files:
        times: list[float] = []
        pages = blocks = 0
        for _ in range(args.runs):
            t0 = time.perf_counter()
            res = parser.parse(path)
            times.append(time.perf_counter() - t0)
            pages = res.metadata.get("page_count", 0)
            blocks = len(res.blocks)
        cold, warm = times[0], times[1:] or times
        wmed = statistics.median(warm)
        print(
            f"{path.name}: pages={pages} blocks={blocks} size={path.stat().st_size / 1024:.0f}KB "
            f"cold={cold:.2f}s warm(median of {len(warm)})={wmed:.2f}s "
            f"-> {pages / wmed if pages else 0:.2f} pages/s"
        )


if __name__ == "__main__":
    main()
