"""Intermediate representation produced by every parser and consumed by the chunker.

The IR is *typed*, not Markdown: each `Block` keeps its kind, heading level, page number,
bounding box and (for tables) the HTML form, so citations can point back at the source and the
chunker can make structural decisions (never split code, keep small tables whole, ...).

`ParsedDocument` is serialised to `data/parsed/<doc_id>.json`; `rag reindex` re-chunks from those
files without re-parsing.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

# Namespace for every uuid5 in the project (doc_id, chunk_id, section_id).
NAMESPACE = uuid.UUID("6f7a0d5c-2b0e-4b6f-9a1f-3a2c8e5d1b47")

_WS_RE = re.compile(r"[ \t\r\f\v]+")
_BLANK_LINES_RE = re.compile(r"\n{3,}")


class BlockType(StrEnum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    CODE = "code"
    TABLE = "table"
    CAPTION = "caption"
    OTHER = "other"


class Block(BaseModel):
    type: BlockType
    text: str
    level: int | None = None  # heading level (1..6); None for non-headings
    page: int | None = None  # 1-based page number when the source is paginated
    bbox: tuple[float, float, float, float] | None = None  # (l, t, r, b) in source page units
    table_html: str | None = None  # canonical table form; `text` holds a Markdown rendering
    language: str | None = None  # code block language hint

    @field_validator("level")
    @classmethod
    def _level_range(cls, v: int | None) -> int | None:
        if v is not None and not 1 <= v <= 6:
            raise ValueError("heading level must be in 1..6")
        return v

    @property
    def is_heading(self) -> bool:
        return self.type is BlockType.HEADING


class ParsedDocument(BaseModel):
    doc_id: str  # uuid5(NAMESPACE, f"{source}:{uri}") — stable across re-ingests of one item
    source: str  # connector name: local | upload | notion
    uri: str  # absolute path, URL, or connector-specific locator
    title: str
    doc_type: str  # file extension without the dot, lower-case (pdf, docx, md, html, ...)
    content_hash: str  # sha256 of the raw bytes; drives incremental ingest and no-gap re-ingest
    last_modified: datetime
    parser: str
    parser_version: str
    blocks: list[Block]
    metadata: dict[str, Any] = Field(default_factory=dict)
    acl: list[str] = Field(default_factory=list)  # unused for now; kept for the schema

    # -- helpers ------------------------------------------------------------------------------

    def normalized(self) -> ParsedDocument:
        """Return a copy with whitespace collapsed and empty blocks dropped.

        Code and table blocks keep their internal whitespace (indentation matters).
        """
        blocks: list[Block] = []
        for b in self.blocks:
            text = b.text if b.type in (BlockType.CODE, BlockType.TABLE) else normalize_text(b.text)
            text = text.strip("\n") if b.type is BlockType.CODE else text.strip()
            if not text and not b.table_html:
                continue
            blocks.append(b.model_copy(update={"text": text}))
        return self.model_copy(update={"blocks": blocks, "title": normalize_text(self.title)})

    def plain_text(self) -> str:
        return "\n\n".join(b.text for b in self.blocks)

    def page_count(self) -> int | None:
        pages = [b.page for b in self.blocks if b.page is not None]
        return max(pages) if pages else None

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.doc_id}.json"
        path.write_text(self.to_json(), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> ParsedDocument:
        return cls.model_validate(json.loads(path.read_text(encoding="utf-8")))


# -- id / hash helpers ------------------------------------------------------------------------


def make_doc_id(source: str, uri: str) -> str:
    return str(uuid.uuid5(NAMESPACE, f"{source}:{uri}"))


def content_hash_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_content_hash(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def file_last_modified(path: Path) -> datetime:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)


def normalize_text(text: str) -> str:
    """Collapse runs of spaces/tabs, trim line ends, cap blank lines at one."""
    lines = [_WS_RE.sub(" ", line).strip() for line in text.replace("\xa0", " ").splitlines()]
    return _BLANK_LINES_RE.sub("\n\n", "\n".join(lines)).strip()
