"""Golden set: questions with the chunks that should be retrieved.

`expected` entries are matched against retrieved chunks by `chunk_id`, or, so the set survives
re-ingests (ids rotate with content hashes) and different checkouts (uris are absolute paths),
by document file name plus either `chunk_index` or a verbatim `span` of the chunk text.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from ragchat.retrieval.retriever import RetrievedChunk


class Expected(BaseModel):
    uri: str
    chunk_index: int | None = None
    chunk_id: str | None = None
    span: str | None = None  # short verbatim phrase from the chunk text

    def matches(self, chunk: RetrievedChunk) -> bool:
        if self.chunk_id and chunk.chunk_id == self.chunk_id:
            return True
        if Path(chunk.uri).name != Path(self.uri).name:
            return False
        if self.chunk_index is not None and chunk.chunk_index == self.chunk_index:
            return True
        return bool(self.span) and _norm(self.span) in _norm(chunk.text)


class GoldenItem(BaseModel):
    id: str
    question: str
    answer: str | None = None  # reference answer (ragas context_recall / precision)
    expected: list[Expected] = Field(default_factory=list)  # empty ⇒ a refusal is expected
    kind: Literal["factual", "refusal"] = "factual"
    origin: Literal["synthetic", "manual"] = "manual"
    notes: str = ""

    @property
    def expects_refusal(self) -> bool:
        return self.kind == "refusal" or not self.expected


def _norm(text: str | None) -> str:
    return " ".join((text or "").split()).lower()


def load_golden(path: Path) -> list[GoldenItem]:
    items = [
        GoldenItem.model_validate(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ids = [i.id for i in items]
    if len(ids) != len(set(ids)):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"duplicate golden ids: {dupes}")
    return items


def save_golden(path: Path, items: list[GoldenItem]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(item.model_dump_json(exclude_none=True, exclude_defaults=True) + "\n")


def golden_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]
