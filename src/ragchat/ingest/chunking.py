"""Structure-aware chunker: ParsedDocument → list[Chunk].

Rules (PLAN.md "Key design rules" 2 and 3):
- Split on heading hierarchy first; every chunk carries `section_id` and a `breadcrumb`
  ("Title > H1 > H2"). Sections are hard boundaries.
- Within a section, pack blocks into ~`target_tokens` chunks (never above `max_tokens`) with
  ~10 % sentence-level overlap between consecutive text chunks.
- Code blocks are never split. Tables up to `table_max_tokens` are one chunk of their own;
  larger tables are split by rows with the header row repeated.
- `chunk_id = uuid5(NS, f"{doc_id}:{chunk_index}:{content_hash}")`.
- Embedded text = breadcrumb + text; displayed text = text.

Token counting is injected (`count_tokens`) so unit tests stay offline; production uses tiktoken.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Literal

from pydantic import BaseModel

from ragchat.core.logging import get_logger
from ragchat.core.settings import ChunkingSettings
from ragchat.ingest.ir import NAMESPACE, Block, BlockType, ParsedDocument
from ragchat.ingest.parsers.tables import html_to_rows, rows_to_markdown

log = get_logger(__name__)

CountTokens = Callable[[str], int]
ChunkKind = Literal["text", "table", "code"]

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[^\s])")
BREADCRUMB_SEP = " > "


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    chunk_index: int
    section_id: str
    breadcrumb: str
    text: str
    embed_text: str
    token_count: int  # tokens of embed_text
    kind: ChunkKind
    page_start: int | None = None
    page_end: int | None = None
    content_hash: str
    table_part: tuple[int, int] | None = None  # (part, total) when a table was split by rows


# -- tokenizer --------------------------------------------------------------------------------


@lru_cache(maxsize=1)
def tiktoken_counter(encoding: str = "cl100k_base") -> CountTokens:
    """Token counter matching the OpenAI embedding models (cl100k_base)."""
    import tiktoken

    enc = tiktoken.get_encoding(encoding)
    return lambda text: len(enc.encode(text, disallowed_special=()))


# -- internal units ---------------------------------------------------------------------------


@dataclass
class _Unit:
    text: str
    kind: ChunkKind
    tokens: int
    page: int | None = None
    table_part: tuple[int, int] | None = None


@dataclass
class _Section:
    breadcrumb: list[str]
    blocks: list[Block] = field(default_factory=list)


def make_chunk_id(doc_id: str, chunk_index: int, content_hash: str) -> str:
    return str(uuid.uuid5(NAMESPACE, f"{doc_id}:{chunk_index}:{content_hash}"))


def make_section_id(doc_id: str, breadcrumb: str, occurrence: int) -> str:
    return str(uuid.uuid5(NAMESPACE, f"{doc_id}:section:{occurrence}:{breadcrumb}"))


class Chunker:
    def __init__(self, settings: ChunkingSettings, count_tokens: CountTokens | None = None) -> None:
        self.s = settings
        self.count = count_tokens or tiktoken_counter()

    # -- public ---------------------------------------------------------------------------------

    def chunk(self, doc: ParsedDocument) -> list[Chunk]:
        chunks: list[Chunk] = []
        seen: dict[str, int] = {}
        for section in self._sections(doc):
            crumb = BREADCRUMB_SEP.join(section.breadcrumb)
            seen[crumb] = seen.get(crumb, 0) + 1
            section_id = make_section_id(doc.doc_id, crumb, seen[crumb])
            for group in self._pack(self._units(section.blocks)):
                text = "\n\n".join(u.text for u in group)
                embed_text = f"{crumb}\n\n{text}"
                pages = [u.page for u in group if u.page is not None]
                kind: ChunkKind = group[0].kind if len(group) == 1 else "text"
                index = len(chunks)
                chunks.append(
                    Chunk(
                        chunk_id=make_chunk_id(doc.doc_id, index, doc.content_hash),
                        doc_id=doc.doc_id,
                        chunk_index=index,
                        section_id=section_id,
                        breadcrumb=crumb,
                        text=text,
                        embed_text=embed_text,
                        token_count=self.count(embed_text),
                        kind=kind,
                        page_start=min(pages) if pages else None,
                        page_end=max(pages) if pages else None,
                        content_hash=doc.content_hash,
                        table_part=group[0].table_part if len(group) == 1 else None,
                    )
                )
        return chunks

    # -- sections -------------------------------------------------------------------------------

    @staticmethod
    def _sections(doc: ParsedDocument) -> list[_Section]:
        stack: list[tuple[int, str]] = []  # (level, heading text)
        title = doc.title.strip()
        sections: list[_Section] = []
        current = _Section(breadcrumb=[title])
        for block in doc.blocks:
            if block.type is BlockType.HEADING:
                level = block.level or 1
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, block.text))
                crumbs = [title] + [h for _, h in stack]
                if len(crumbs) > 1 and crumbs[1] == title:
                    crumbs.pop(1)  # first heading repeats the title
                if current.blocks:
                    sections.append(current)
                current = _Section(breadcrumb=crumbs)
            else:
                current.blocks.append(block)
        if current.blocks:
            sections.append(current)
        return sections

    # -- blocks → units -------------------------------------------------------------------------

    def _units(self, blocks: list[Block]) -> list[_Unit]:
        units: list[_Unit] = []
        for b in blocks:
            if b.type is BlockType.TABLE:
                units.extend(self._table_units(b))
            elif b.type is BlockType.CODE:
                fence = f"```{b.language or ''}\n{b.text}\n```"
                tokens = self.count(fence)
                if tokens > self.s.max_tokens:
                    log.warning("code block exceeds max_tokens; kept whole", tokens=tokens)
                units.append(_Unit(fence, "code", tokens, b.page))
            else:
                text = b.text
                if b.type is BlockType.LIST_ITEM:
                    text = "  " * max((b.level or 1) - 1, 0) + "- " + text
                units.extend(self._text_units(text, b.page))
        return units

    def _text_units(self, text: str, page: int | None) -> list[_Unit]:
        tokens = self.count(text)
        if tokens <= self.s.max_tokens:
            return [_Unit(text, "text", tokens, page)]
        # Long paragraph: split by sentences, then words, into ≤ target-sized pieces.
        pieces = self._split_long(text)
        return [_Unit(p, "text", self.count(p), page) for p in pieces]

    def _split_long(self, text: str) -> list[str]:
        parts: list[str] = []
        buf: list[str] = []
        buf_tokens = 0
        for sent in self._sentences(text):
            st = self.count(sent)
            if st > self.s.max_tokens:  # pathological sentence: split on words
                if buf:
                    parts.append(" ".join(buf))
                    buf, buf_tokens = [], 0
                parts.extend(self._split_words(sent))
                continue
            if buf and buf_tokens + st > self.s.target_tokens:
                parts.append(" ".join(buf))
                buf, buf_tokens = [], 0
            buf.append(sent)
            buf_tokens += st
        if buf:
            parts.append(" ".join(buf))
        return parts

    def _split_words(self, text: str) -> list[str]:
        words = text.split()
        parts: list[str] = []
        buf: list[str] = []
        for w in words:
            buf.append(w)
            if self.count(" ".join(buf)) > self.s.target_tokens and len(buf) > 1:
                buf.pop()
                parts.append(" ".join(buf))
                buf = [w]
        if buf:
            parts.append(" ".join(buf))
        return parts

    @staticmethod
    def _sentences(text: str) -> list[str]:
        return [s for s in _SENTENCE_RE.split(text) if s.strip()]

    def _table_units(self, block: Block) -> list[_Unit]:
        tokens = self.count(block.text)
        if tokens <= self.s.table_max_tokens:
            return [_Unit(block.text, "table", tokens, block.page)]
        rows, header = html_to_rows(block.table_html or "")
        if not rows:
            return [_Unit(block.text, "table", tokens, block.page)]
        # Split by rows; each part repeats the header row.
        parts: list[list[list[str]]] = []
        buf: list[list[str]] = []
        for row in rows:
            candidate = rows_to_markdown(buf + [row], header)
            if buf and self.count(candidate) > self.s.table_max_tokens:
                parts.append(buf)
                buf = []
            buf.append(row)
        if buf:
            parts.append(buf)
        total = len(parts)
        return [
            _Unit(
                rows_to_markdown(part, header),
                "table",
                self.count(rows_to_markdown(part, header)),
                block.page,
                table_part=(i, total),
            )
            for i, part in enumerate(parts, start=1)
        ]

    # -- units → chunks -------------------------------------------------------------------------

    def _pack(self, units: list[_Unit]) -> list[list[_Unit]]:
        groups: list[list[_Unit]] = []
        cur: list[_Unit] = []
        cur_tokens = 0
        prev_text_tail: str | None = None  # overlap carried into the next text chunk

        def flush() -> None:
            nonlocal cur, cur_tokens, prev_text_tail
            if cur:
                groups.append(cur)
                prev_text_tail = self._tail(cur) if all(u.kind == "text" for u in cur) else None
            cur, cur_tokens = [], 0

        for u in units:
            if u.kind == "table":
                flush()
                groups.append([u])
                prev_text_tail = None
                continue
            if cur and cur_tokens + u.tokens > self.s.target_tokens:
                flush()
            if not cur and prev_text_tail and u.kind == "text":
                tail = _Unit(prev_text_tail, "text", self.count(prev_text_tail), u.page)
                if tail.tokens + u.tokens <= self.s.max_tokens:
                    cur.append(tail)
                    cur_tokens += tail.tokens
            cur.append(u)
            cur_tokens += u.tokens
        flush()
        return groups

    def _tail(self, units: list[_Unit]) -> str | None:
        """Last sentences of a text chunk, up to `overlap_tokens`."""
        if self.s.overlap_tokens <= 0:
            return None
        sentences = self._sentences("\n\n".join(u.text for u in units))
        tail: list[str] = []
        total = 0
        for s in reversed(sentences):
            t = self.count(s)
            if tail and total + t > self.s.overlap_tokens:
                break
            tail.insert(0, s)
            total += t
            if total >= self.s.overlap_tokens:
                break
        text = " ".join(tail).strip()
        # Never overlap the whole previous chunk.
        return text if text and len(tail) < len(sentences) else None


def chunk_document(
    doc: ParsedDocument, settings: ChunkingSettings, count_tokens: CountTokens | None = None
) -> list[Chunk]:
    return Chunker(settings, count_tokens).chunk(doc)
