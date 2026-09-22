"""Parser snapshot tests for Markdown/HTML plus targeted assertions on structure."""

from __future__ import annotations

from pathlib import Path

import pytest

from ragchat.ingest.ir import BlockType, ParsedDocument
from ragchat.ingest.parsers import router
from ragchat.ingest.parsers.base import UnsupportedFileTypeError
from ragchat.ingest.parsers.tables import html_to_rows, rows_to_html, rows_to_markdown


def _snapshot_view(doc: ParsedDocument) -> dict:
    """Everything except machine/time-dependent fields."""
    return {
        "title": doc.title,
        "doc_type": doc.doc_type,
        "parser": doc.parser,
        "metadata": {k: v for k, v in doc.metadata.items() if k not in ("size_bytes",)},
        "blocks": [b.model_dump(exclude_none=True) for b in doc.blocks],
    }


def test_markdown_snapshot(fixtures_dir: Path, snapshot):
    doc = router.parse_file(fixtures_dir / "handbook.md", source="local", uri="fixture:handbook.md")
    snapshot("handbook_md", _snapshot_view(doc))


def test_markdown_structure(fixtures_dir: Path):
    doc = router.parse_file(fixtures_dir / "handbook.md", source="local", uri="fixture:handbook.md")
    assert doc.title == "Employee Handbook"  # from front matter
    assert doc.metadata["front_matter"]["owner"] == "people-ops"
    types = [b.type for b in doc.blocks]
    assert types[0] is BlockType.HEADING and doc.blocks[0].level == 1
    # inline markup is stripped, link text is kept
    assert doc.blocks[1].text.startswith("This handbook covers policies and benefits")
    items = [b for b in doc.blocks if b.type is BlockType.LIST_ITEM]
    assert [i.level for i in items] == [1, 1, 2, 1, 1, 1]
    table = next(b for b in doc.blocks if b.type is BlockType.TABLE)
    assert table.text.splitlines()[0] == "| Category | Limit | Receipt required |"
    assert html_to_rows(table.table_html) == (
        [["Meals", "$50/day", "yes"], ["Hotel", "$200/night", "yes"], ["Taxi", "$30", "no"]],
        ["Category", "Limit", "Receipt required"],
    )
    code = next(b for b in doc.blocks if b.type is BlockType.CODE)
    assert code.language == "python"
    assert code.text == "def reimburse(amount: float) -> bool:\n    return amount <= LIMIT"
    assert "Raw HTML block." in [b.text for b in doc.blocks]
    assert "Quoted note about reimbursements." in [b.text for b in doc.blocks]


def test_html_snapshot(fixtures_dir: Path, snapshot):
    doc = router.parse_file(fixtures_dir / "wiki.html", source="local", uri="fixture:wiki.html")
    snapshot("wiki_html", _snapshot_view(doc))


def test_html_structure(fixtures_dir: Path):
    doc = router.parse_file(fixtures_dir / "wiki.html", source="local", uri="fixture:wiki.html")
    assert doc.title == "Onboarding Wiki"
    texts = [b.text for b in doc.blocks]
    assert "alert('x')" not in doc.plain_text()
    assert "color: red" not in doc.plain_text()
    assert "Home | Wiki" not in doc.plain_text()  # <nav> dropped
    assert "Example Corp" not in doc.plain_text()  # <footer> dropped
    assert "Welcome to the team! Read this handbook first." in texts
    assert "Loose text inside a div with bold." in texts
    items = [(b.text, b.level) for b in doc.blocks if b.type is BlockType.LIST_ITEM]
    assert items == [
        ("Get a laptop", 1),
        ("Set up accounts", 1),
        ("Email", 2),
        ("Slack", 2),
        ("Meet your buddy", 1),
    ]
    tables = [b for b in doc.blocks if b.type is BlockType.TABLE]
    assert len(tables) == 2
    assert html_to_rows(tables[0].table_html) == (
        [["Jira", "Tickets"], ["Confluence", "Docs & wiki"]],
        ["Tool", "Purpose"],
    )
    assert html_to_rows(tables[1].table_html) == ([["no", "header"], ["second", "row"]], None)
    code = next(b for b in doc.blocks if b.type is BlockType.CODE)
    assert code.language == "bash" and code.text == "git clone repo\ncd repo"
    assert next(b for b in doc.blocks if b.type is BlockType.CAPTION).text.startswith("Figure 1")


def test_router_rejects_unknown_extension(tmp_path: Path):
    p = tmp_path / "x.xyz"
    p.write_text("hi")
    assert not router.is_supported(p)
    with pytest.raises(UnsupportedFileTypeError):
        router.parse_file(p, source="local")


def test_router_document_identity(tmp_path: Path):
    p = tmp_path / "note.md"
    p.write_text("# T\n\nbody")
    a = router.parse_file(p, source="local")
    b = router.parse_file(p, source="local")
    assert a.doc_id == b.doc_id and a.content_hash == b.content_hash
    assert a.uri == str(p.resolve()) and a.title == "T"
    p.write_text("# T\n\nbody changed")
    c = router.parse_file(p, source="local")
    assert c.doc_id == a.doc_id and c.content_hash != a.content_hash


def test_router_title_falls_back_to_stem(tmp_path: Path):
    p = tmp_path / "untitled-note.md"
    p.write_text("just a paragraph")
    assert router.parse_file(p, source="local").title == "untitled-note"


def test_table_helpers_round_trip():
    rows = [["a|b", "line\nbreak"], ["c", ""]]
    header = ["H1", "H2"]
    md = rows_to_markdown(rows, header)
    assert md.splitlines()[0] == "| H1 | H2 |"
    assert md.splitlines()[2] == "| a\\|b | line break |"
    assert html_to_rows(rows_to_html(rows, header)) == (rows, header)
    assert html_to_rows(rows_to_html(rows)) == (rows, None)
    assert rows_to_markdown([], None) == ""
