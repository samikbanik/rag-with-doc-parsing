"""HTML → IR via BeautifulSoup (lxml). Walks the body in document order."""

from __future__ import annotations

import importlib.metadata
import re
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString, Tag

from ragchat.ingest.ir import Block, BlockType
from ragchat.ingest.parsers.base import BaseParser, ParseResult
from ragchat.ingest.parsers.tables import rows_to_html, rows_to_markdown

_DROP = {"script", "style", "noscript", "template", "svg", "head", "iframe", "nav", "footer"}
_WS_RE = re.compile(r"\s+")
_HEADINGS = {f"h{n}": n for n in range(1, 7)}
_BLOCK_TAGS = [*_HEADINGS, "p", "table", "ul", "ol", "pre"]
_CONTAINERS = {
    "body",
    "html",
    "div",
    "section",
    "article",
    "main",
    "aside",
    "header",
    "blockquote",
    "details",
    "summary",
    "figure",
    "form",
    "fieldset",
    "center",
    "span",
    "dl",
    "dd",
    "dt",
}


def _text(node: Tag | NavigableString) -> str:
    """Inline text with the source's own whitespace, collapsed; no separator is injected so
    `the <em>team</em>!` stays `the team!`. `<br>` tags were replaced by spaces up front."""
    raw = node.get_text("") if isinstance(node, Tag) else str(node)
    return _WS_RE.sub(" ", raw).strip()


class HTMLParser(BaseParser):
    name = "html"
    version = importlib.metadata.version("beautifulsoup4")
    extensions = frozenset({"html", "htm", "xhtml"})

    def parse(self, path: Path) -> ParseResult:
        soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="replace"), "lxml")
        title = soup.title.get_text(strip=True) if soup.title else None
        for tag in soup.find_all(_DROP):
            tag.decompose()
        for br in soup.find_all("br"):
            br.replace_with(" ")
        blocks: list[Block] = []
        root = soup.body or soup
        self._walk(root, blocks, list_depth=0)
        if not title:
            title = next((b.text for b in blocks if b.type is BlockType.HEADING), None)
        return ParseResult(blocks=blocks, title=title)

    def _walk(self, node: Tag, blocks: list[Block], list_depth: int) -> None:
        pending: list[str] = []  # raw loose inline text between block elements

        def flush() -> None:
            text = _WS_RE.sub(" ", "".join(pending)).strip()
            if text:
                kind = BlockType.LIST_ITEM if list_depth else BlockType.PARAGRAPH
                blocks.append(Block(type=kind, text=text, level=list_depth or None))
            pending.clear()

        for child in node.children:
            if isinstance(child, NavigableString):
                if child.parent is node:
                    pending.append(str(child))
                continue
            if not isinstance(child, Tag):
                continue
            name = child.name.lower()
            if name in _HEADINGS:
                flush()
                level = _HEADINGS[name]
                blocks.append(Block(type=BlockType.HEADING, text=_text(child), level=level))
            elif name == "p":
                flush()
                text = _text(child)
                if text:
                    kind = BlockType.LIST_ITEM if list_depth else BlockType.PARAGRAPH
                    blocks.append(Block(type=kind, text=text, level=list_depth or None))
            elif name == "pre":
                flush()
                code = child.find("code")
                lang = None
                if code is not None:
                    for cls in code.get("class") or []:
                        if cls.startswith("language-") or cls.startswith("lang-"):
                            lang = cls.split("-", 1)[1]
                blocks.append(Block(type=BlockType.CODE, text=child.get_text(), language=lang))
            elif name in ("ul", "ol"):
                flush()
                for li in child.find_all("li", recursive=False):
                    self._list_item(li, blocks, list_depth + 1)
            elif name == "table":
                flush()
                blocks.append(self._table(child))
            elif name == "figcaption":
                flush()
                blocks.append(Block(type=BlockType.CAPTION, text=_text(child)))
            elif name in ("img", "hr", "br", "picture", "video", "audio", "canvas"):
                continue
            elif name in _CONTAINERS or child.find(_BLOCK_TAGS):
                flush()
                self._walk(child, blocks, list_depth)
            else:  # inline element: b, a, em, code, ...
                pending.append(child.get_text(""))
        flush()

    def _list_item(self, li: Tag, blocks: list[Block], depth: int) -> None:
        own: list[str] = []
        nested: list[Tag] = []
        for c in li.children:
            if isinstance(c, Tag) and c.name in ("ul", "ol"):
                nested.append(c)
            elif isinstance(c, Tag) and c.name in ("table", "pre"):
                nested.append(c)
            else:
                t = _text(c)
                if t:
                    own.append(t)
        text = " ".join(own).strip()
        if text:
            blocks.append(Block(type=BlockType.LIST_ITEM, text=text, level=min(depth, 6)))
        for n in nested:
            if n.name in ("ul", "ol"):
                for sub in n.find_all("li", recursive=False):
                    self._list_item(sub, blocks, depth + 1)
            elif n.name == "table":
                blocks.append(self._table(n))
            else:
                blocks.append(Block(type=BlockType.CODE, text=n.get_text()))

    @staticmethod
    def _table(table: Tag) -> Block:
        header: list[str] | None = None
        rows: list[list[str]] = []
        for tr in table.find_all("tr"):
            if tr.find_parent("table") is not table:
                continue  # nested table row
            cells = tr.find_all(["th", "td"], recursive=False)
            if not cells:
                continue
            texts = [_text(c) for c in cells]
            in_thead = tr.find_parent("thead") is not None
            if header is None and not rows and (in_thead or all(c.name == "th" for c in cells)):
                header = texts
            else:
                rows.append(texts)
        return Block(
            type=BlockType.TABLE,
            text=rows_to_markdown(rows, header),
            table_html=rows_to_html(rows, header),
        )
