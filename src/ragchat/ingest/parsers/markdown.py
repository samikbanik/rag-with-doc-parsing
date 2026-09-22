"""Markdown → IR via markdown-it-py (GFM-like: tables, strikethrough)."""

from __future__ import annotations

import importlib.metadata
import re
from pathlib import Path
from typing import Any

import yaml
from markdown_it import MarkdownIt
from markdown_it.token import Token

from ragchat.ingest.ir import Block, BlockType
from ragchat.ingest.parsers.base import BaseParser, ParseResult
from ragchat.ingest.parsers.tables import rows_to_html, rows_to_markdown

_FRONT_MATTER_RE = re.compile(r"\A---[ \t]*\n(.*?)\n---[ \t]*\n", re.DOTALL)


def _inline_text(token: Token) -> str:
    """Plain text of an inline token: keeps link/emphasis text, drops the markup."""
    if not token.children:
        return token.content
    out: list[str] = []
    for c in token.children:
        if c.type in ("text", "code_inline"):
            out.append(c.content)
        elif c.type in ("softbreak", "hardbreak"):
            out.append(" ")
        elif c.type == "image":
            out.append(c.attrGet("alt") or "")
        elif c.type == "html_inline":
            continue
        elif c.children:
            out.append(_inline_text(c))
    return "".join(out)


def split_front_matter(text: str) -> tuple[dict[str, Any], str]:
    m = _FRONT_MATTER_RE.match(text)
    if not m:
        return {}, text
    try:
        data = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError:
        return {}, text
    return (data if isinstance(data, dict) else {}), text[m.end() :]


class MarkdownParser(BaseParser):
    name = "markdown"
    version = importlib.metadata.version("markdown-it-py")
    extensions = frozenset({"md", "markdown", "mdx"})

    def __init__(self) -> None:
        self.md = MarkdownIt("gfm-like", {"linkify": False})

    def parse(self, path: Path) -> ParseResult:
        front, body = split_front_matter(path.read_text(encoding="utf-8", errors="replace"))
        blocks = self.blocks_from_text(body)
        title = str(front["title"]) if front.get("title") else None
        if title is None:
            title = next((b.text for b in blocks if b.type is BlockType.HEADING), None)
        return ParseResult(blocks=blocks, title=title, metadata={"front_matter": front})

    def blocks_from_text(self, text: str) -> list[Block]:
        tokens = self.md.parse(text)
        blocks: list[Block] = []
        list_depth = 0
        i = 0
        while i < len(tokens):
            t = tokens[i]
            if t.type == "heading_open":
                text_ = _inline_text(tokens[i + 1])
                blocks.append(Block(type=BlockType.HEADING, text=text_, level=int(t.tag[1])))
                i += 3
            elif t.type in ("bullet_list_open", "ordered_list_open"):
                list_depth += 1
                i += 1
            elif t.type in ("bullet_list_close", "ordered_list_close"):
                list_depth -= 1
                i += 1
            elif t.type == "paragraph_open":
                text_ = _inline_text(tokens[i + 1])
                if list_depth:
                    blocks.append(
                        Block(type=BlockType.LIST_ITEM, text=text_, level=min(list_depth, 6))
                    )
                else:
                    blocks.append(Block(type=BlockType.PARAGRAPH, text=text_))
                i += 3
            elif t.type in ("fence", "code_block"):
                lang = (t.info or "").strip().split()[0] if t.info and t.info.strip() else None
                blocks.append(Block(type=BlockType.CODE, text=t.content, language=lang))
                i += 1
            elif t.type == "table_open":
                i = self._table(tokens, i, blocks)
            elif t.type == "html_block":
                from bs4 import BeautifulSoup

                text_ = BeautifulSoup(t.content, "lxml").get_text(" ", strip=True)
                if text_:
                    blocks.append(Block(type=BlockType.PARAGRAPH, text=text_))
                i += 1
            else:
                i += 1
        return blocks

    @staticmethod
    def _table(tokens: list[Token], i: int, blocks: list[Block]) -> int:
        header: list[str] | None = None
        rows: list[list[str]] = []
        current: list[str] | None = None
        in_head = False
        while tokens[i].type != "table_close":
            t = tokens[i]
            if t.type == "thead_open":
                in_head = True
            elif t.type == "thead_close":
                in_head = False
            elif t.type == "tr_open":
                current = []
            elif t.type == "tr_close":
                if in_head:
                    header = current
                elif current is not None:
                    rows.append(current)
                current = None
            elif t.type in ("th_open", "td_open") and current is not None:
                current.append(_inline_text(tokens[i + 1]))
            i += 1
        blocks.append(
            Block(
                type=BlockType.TABLE,
                text=rows_to_markdown(rows, header),
                table_html=rows_to_html(rows, header),
            )
        )
        return i + 1
