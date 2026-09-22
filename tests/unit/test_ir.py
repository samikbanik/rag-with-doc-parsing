from datetime import UTC, datetime
from pathlib import Path

import pytest

from ragchat.ingest.ir import (
    Block,
    BlockType,
    ParsedDocument,
    content_hash_of,
    file_content_hash,
    make_doc_id,
    normalize_text,
)


def _doc(blocks: list[Block]) -> ParsedDocument:
    return ParsedDocument(
        doc_id=make_doc_id("local", "/tmp/a.md"),
        source="local",
        uri="/tmp/a.md",
        title="  A   title ",
        doc_type="md",
        content_hash=content_hash_of(b"x"),
        last_modified=datetime(2026, 1, 1, tzinfo=UTC),
        parser="markdown",
        parser_version="1",
        blocks=blocks,
    )


def test_doc_id_is_stable_and_source_scoped():
    assert make_doc_id("local", "/a") == make_doc_id("local", "/a")
    assert make_doc_id("local", "/a") != make_doc_id("upload", "/a")
    assert len(make_doc_id("local", "/a")) == 36


def test_content_hash_matches_file_hash(tmp_path: Path):
    p = tmp_path / "f.bin"
    p.write_bytes(b"hello" * 1000)
    assert file_content_hash(p) == content_hash_of(b"hello" * 1000)


def test_heading_level_validated():
    Block(type=BlockType.HEADING, text="h", level=6)
    with pytest.raises(ValueError):
        Block(type=BlockType.HEADING, text="h", level=7)
    with pytest.raises(ValueError):
        Block(type=BlockType.HEADING, text="h", level=0)


def test_normalize_text_collapses_whitespace():
    assert normalize_text("a \t b\r\n\n\n\n c\xa0d  ") == "a b\n\nc d"


def test_normalized_drops_empty_and_keeps_code_whitespace():
    doc = _doc(
        [
            Block(type=BlockType.PARAGRAPH, text="  hello   world \n"),
            Block(type=BlockType.PARAGRAPH, text="   "),
            Block(type=BlockType.CODE, text="\ndef f():\n    return 1\n", language="python"),
            Block(type=BlockType.TABLE, text="", table_html="<table><tr><td>1</td></tr></table>"),
        ]
    )
    n = doc.normalized()
    assert n.title == "A title"
    assert [b.type for b in n.blocks] == [BlockType.PARAGRAPH, BlockType.CODE, BlockType.TABLE]
    assert n.blocks[0].text == "hello world"
    assert n.blocks[1].text == "def f():\n    return 1"
    # original is untouched
    assert len(doc.blocks) == 4


def test_json_round_trip(tmp_path: Path):
    doc = _doc(
        [
            Block(type=BlockType.HEADING, text="H", level=1, page=1, bbox=(0, 0, 10.5, 20)),
            Block(type=BlockType.PARAGRAPH, text="p", page=1),
        ]
    ).normalized()
    path = doc.save(tmp_path)
    assert path == tmp_path / f"{doc.doc_id}.json"
    loaded = ParsedDocument.load(path)
    assert loaded == doc
    assert loaded.blocks[0].bbox == (0, 0, 10.5, 20)
    assert loaded.last_modified.tzinfo is not None


def test_page_count_and_plain_text():
    doc = _doc(
        [
            Block(type=BlockType.PARAGRAPH, text="a", page=2),
            Block(type=BlockType.PARAGRAPH, text="b", page=5),
        ]
    )
    assert doc.page_count() == 5
    assert doc.plain_text() == "a\n\nb"
    assert _doc([Block(type=BlockType.PARAGRAPH, text="a")]).page_count() is None
