"""Generate synthetic golden candidates from the indexed chunks.

    uv run python eval/generate_golden.py [--max-chunks 60] [--per-chunk 2] [--seed 0]
                                          [--out eval/golden.candidates.jsonl]

Review the candidates by hand (fix questions, drop weak ones, add manual questions and
refusal cases) and save the curated set as eval/golden.jsonl. Candidates are never used directly.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from ragchat.core.llm import get_llm
from ragchat.core.settings import get_settings
from ragchat.eval.generate import generate_candidates, write_candidates
from ragchat.retrieval.vectorstore import VectorStore


async def main(args: argparse.Namespace) -> None:
    s = get_settings()
    items = await generate_candidates(
        s,
        VectorStore(s),
        get_llm(),
        max_chunks=args.max_chunks,
        per_chunk=args.per_chunk,
        seed=args.seed,
    )
    write_candidates(items, args.out)
    print(f"{len(items)} candidates → {args.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-chunks", type=int, default=None)
    ap.add_argument("--per-chunk", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("eval/golden.candidates.jsonl"))
    asyncio.run(main(ap.parse_args()))
