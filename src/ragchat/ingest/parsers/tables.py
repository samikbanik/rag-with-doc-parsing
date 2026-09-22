"""Table rendering shared by parsers and the chunker.

Every parser reduces a table to `rows: list[list[str]]` (+ optional header) and calls these to
produce the IR's `table_html` (canonical) and `text` (Markdown, what gets embedded/displayed).
The chunker parses `table_html` back into rows when a table must be split.
"""

from __future__ import annotations

import html
import re

from bs4 import BeautifulSoup

_PIPE_RE = re.compile(r"\|")
_NL_RE = re.compile(r"\s*\n\s*")


def _cell(text: str) -> str:
    return _PIPE_RE.sub(r"\\|", _NL_RE.sub(" ", text.strip()))


def rows_to_markdown(rows: list[list[str]], header: list[str] | None = None) -> str:
    """GitHub-flavoured Markdown table. Without a header, an empty header row is emitted so
    the table still renders and the first data row is not mistaken for a header."""
    if not rows and not header:
        return ""
    width = max([len(header or [])] + [len(r) for r in rows])
    if width == 0:
        return ""

    def line(cells: list[str]) -> str:
        cells = [_cell(c) for c in cells] + [""] * (width - len(cells))
        return "| " + " | ".join(cells) + " |"

    out = [line(header or [""] * width), "| " + " | ".join(["---"] * width) + " |"]
    out.extend(line(r) for r in rows)
    return "\n".join(out)


def rows_to_html(rows: list[list[str]], header: list[str] | None = None) -> str:
    def tr(cells: list[str], tag: str) -> str:
        return "<tr>" + "".join(f"<{tag}>{html.escape(c.strip())}</{tag}>" for c in cells) + "</tr>"

    parts = ["<table>"]
    if header:
        parts.append("<thead>" + tr(header, "th") + "</thead>")
    parts.append("<tbody>" + "".join(tr(r, "td") for r in rows) + "</tbody>")
    parts.append("</table>")
    return "".join(parts)


def html_to_rows(table_html: str) -> tuple[list[list[str]], list[str] | None]:
    """Inverse of `rows_to_html` (also tolerant of arbitrary HTML tables).

    A header is recognised from `<thead>` or from a first row made only of `<th>` cells.
    """
    soup = BeautifulSoup(table_html, "lxml")
    table = soup.find("table") or soup
    header: list[str] | None = None
    rows: list[list[str]] = []
    for tr in table.find_all("tr"):
        cells = tr.find_all(["th", "td"], recursive=False)
        if not cells:
            continue
        texts = [c.get_text(" ", strip=True) for c in cells]
        in_thead = tr.find_parent("thead") is not None
        if header is None and not rows and (in_thead or all(c.name == "th" for c in cells)):
            header = texts
        else:
            rows.append(texts)
    return rows, header
