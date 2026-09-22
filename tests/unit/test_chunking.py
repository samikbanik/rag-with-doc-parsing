from __future__ import annotations

import os
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest

from ragchat.core.settings import ChunkingSettings
from ragchat.ingest.chunking import Chunker, chunk_document, make_chunk_id
from ragchat.ingest.ir import Block, BlockType, ParsedDocument
from ragchat.ingest.parsers import router
from ragchat.ingest.parsers.tables import rows_to_html, rows_to_markdown


def words(text: str) -> int:
    """Offline stand-in for tiktoken: one token per whitespace-separated word."""
    return len(text.split())


SETTINGS = ChunkingSettings(target_tokens=40, max_tokens=60, overlap_tokens=6, table_max_tokens=30)


def _doc(blocks: list[Block], title: str = "Doc", content_hash: str = "h1") -> ParsedDocument:
    return ParsedDocument(
        doc_id=str(uuid.uuid5(uuid.NAMESPACE_URL, "doc")),
        source="local",
        uri="/x",
        title=title,
        doc_type="md",
        content_hash=content_hash,
        last_modified=datetime(2026, 1, 1, tzinfo=UTC),
        parser="test",
        parser_version="0",
        blocks=blocks,
    )


def _para(n: int, prefix: str = "w") -> Block:
    return Block(type=BlockType.PARAGRAPH, text=" ".join(f"{prefix}{i}." for i in range(n)))


def h(text: str, level: int) -> Block:
    return Block(type=BlockType.HEADING, text=text, level=level)


def test_breadcrumbs_follow_heading_hierarchy():
    doc = _doc(
        [
            h("Doc", 1),  # repeats the title → not duplicated in the breadcrumb
            _para(3, "intro"),
            h("A", 2),
            _para(3, "a"),
            h("A.1", 3),
            _para(3, "a1"),
            h("B", 2),
            _para(3, "b"),
            h("Empty", 2),
            h("C", 2),
            _para(3, "c"),
        ]
    )
    chunks = chunk_document(doc, SETTINGS, words)
    assert [c.breadcrumb for c in chunks] == [
        "Doc",
        "Doc > A",
        "Doc > A > A.1",
        "Doc > B",
        "Doc > C",
    ]
    assert [c.chunk_index for c in chunks] == list(range(5))
    assert chunks[1].embed_text == "Doc > A\n\n" + chunks[1].text
    assert len({c.section_id for c in chunks}) == 5


def test_ids_are_deterministic_and_change_with_content_hash():
    blocks = [h("A", 1), _para(5)]
    a = chunk_document(_doc(blocks, content_hash="h1"), SETTINGS, words)
    b = chunk_document(_doc(blocks, content_hash="h1"), SETTINGS, words)
    c = chunk_document(_doc(blocks, content_hash="h2"), SETTINGS, words)
    assert [x.chunk_id for x in a] == [x.chunk_id for x in b]
    assert a[0].chunk_id != c[0].chunk_id
    assert a[0].section_id == c[0].section_id  # sections are content-hash independent
    assert a[0].chunk_id == make_chunk_id(a[0].doc_id, 0, "h1")
    uuid.UUID(a[0].chunk_id)  # valid uuid for Qdrant


def test_packing_respects_target_and_overlaps():
    doc = _doc([h("S", 1)] + [_para(15, f"p{i}") for i in range(6)])  # 6 × 15 words
    chunks = chunk_document(doc, SETTINGS, words)
    # target=40 → two paragraphs (30) + overlap fit, a third would not
    assert len(chunks) >= 3
    for c in chunks:
        assert c.token_count <= SETTINGS.max_tokens + words(c.breadcrumb)
    # each later chunk starts with the tail of the previous one
    for prev, cur in zip(chunks, chunks[1:], strict=False):
        overlap = cur.text.split("\n\n")[0]
        assert overlap in prev.text  # whole trailing sentences of the previous chunk
        assert prev.text.endswith(overlap)
        assert len(overlap.split()) <= SETTINGS.overlap_tokens


def test_no_overlap_when_disabled():
    settings = SETTINGS.model_copy(update={"overlap_tokens": 0})
    doc = _doc([_para(15, f"p{i}") for i in range(4)])
    chunks = chunk_document(doc, settings, words)
    assert len(chunks) == 2
    assert chunks[0].text.split() != chunks[1].text.split()
    assert not set(chunks[0].text.split()) & set(chunks[1].text.split())


def test_long_paragraph_is_split_by_sentences():
    doc = _doc([_para(200)])  # 200 one-word sentences
    chunks = chunk_document(doc, SETTINGS, words)
    assert len(chunks) > 3
    assert all(words(c.text) <= SETTINGS.max_tokens for c in chunks)
    # every sentence survives (order preserved, ignoring overlap duplicates)
    seen: list[str] = []
    for c in chunks:
        for w in c.text.split():
            if not seen or w != seen[-1] and w not in seen[-SETTINGS.overlap_tokens :]:
                seen.append(w)
    assert seen == [f"w{i}." for i in range(200)]


def test_giant_sentence_is_split_by_words():
    text = " ".join(f"w{i}" for i in range(150))  # no sentence punctuation at all
    doc = _doc([Block(type=BlockType.PARAGRAPH, text=text)])
    chunks = chunk_document(doc, SETTINGS, words)
    assert all(words(c.text) <= SETTINGS.max_tokens for c in chunks)
    assert sum(c.text.split().count("w149") for c in chunks) == 1


def test_code_block_is_never_split_and_is_fenced():
    code = "\n".join(f"line{i} = {i}" for i in range(100))
    doc = _doc([_para(3), Block(type=BlockType.CODE, text=code, language="python"), _para(3)])
    chunks = chunk_document(doc, SETTINGS, words)
    code_chunks = [c for c in chunks if "```python" in c.text]
    assert len(code_chunks) == 1
    assert code_chunks[0].text.endswith("line99 = 99\n```")
    assert code_chunks[0].kind == "code"


def test_small_table_is_one_chunk():
    rows = [["a", "1"], ["b", "2"]]
    tbl = Block(
        type=BlockType.TABLE,
        text=rows_to_markdown(rows, ["k", "v"]),
        table_html=rows_to_html(rows, ["k", "v"]),
        page=3,
    )
    doc = _doc([_para(5), tbl, _para(5)])
    chunks = chunk_document(doc, SETTINGS, words)
    assert [c.kind for c in chunks] == ["text", "table", "text"]
    assert chunks[1].text == tbl.text
    assert chunks[1].page_start == chunks[1].page_end == 3
    assert chunks[1].table_part is None
    assert chunks[2].text.startswith("w0.")  # no overlap carried across a table


def test_large_table_split_by_rows_repeats_header():
    rows = [[f"row{i}", f"value {i}"] for i in range(20)]  # ≈ 4 words/row + header
    tbl = Block(
        type=BlockType.TABLE,
        text=rows_to_markdown(rows, ["name", "value"]),
        table_html=rows_to_html(rows, ["name", "value"]),
    )
    chunks = chunk_document(_doc([tbl]), SETTINGS, words)
    assert len(chunks) > 1
    parts = [c.table_part for c in chunks]
    assert parts == [(i, len(chunks)) for i in range(1, len(chunks) + 1)]
    for c in chunks:
        assert c.kind == "table"
        assert c.text.splitlines()[0] == "| name | value |"
        assert words(c.text) <= SETTINGS.table_max_tokens
    all_rows = [ln for c in chunks for ln in c.text.splitlines()[2:]]
    assert len(all_rows) == 20 and all_rows[0].startswith("| row0 ") and "| row19 " in all_rows[-1]


def test_list_items_render_with_markers_and_indent():
    doc = _doc(
        [
            Block(type=BlockType.LIST_ITEM, text="one", level=1),
            Block(type=BlockType.LIST_ITEM, text="two", level=2),
        ]
    )
    (chunk,) = chunk_document(doc, SETTINGS, words)
    assert chunk.text == "- one\n\n  - two"


def test_pages_span_units():
    doc = _doc(
        [
            Block(type=BlockType.PARAGRAPH, text="a b c.", page=2),
            Block(type=BlockType.PARAGRAPH, text="d e f.", page=4),
        ]
    )
    (chunk,) = chunk_document(doc, SETTINGS, words)
    assert (chunk.page_start, chunk.page_end) == (2, 4)


def test_heading_only_document_yields_no_chunks():
    assert chunk_document(_doc([h("A", 1), h("B", 2)]), SETTINGS, words) == []


def test_repeated_breadcrumbs_get_distinct_section_ids():
    doc = _doc([h("A", 1), _para(3, "x"), h("A", 1), _para(3, "y")])
    a, b = chunk_document(doc, SETTINGS, words)
    assert a.breadcrumb == b.breadcrumb == "Doc > A"
    assert a.section_id != b.section_id


def test_chunk_real_fixture_with_default_settings(fixtures_dir: Path):
    doc = router.parse_file(fixtures_dir / "handbook.md", source="local", uri="fixture:hb")
    chunks = Chunker(ChunkingSettings(), words).chunk(doc)
    assert [c.breadcrumb for c in chunks] == [
        "Employee Handbook > Welcome",
        "Employee Handbook > Welcome > Time off",
        "Employee Handbook > Welcome > Time off > Parental leave",
        "Employee Handbook > Welcome > Expenses",  # table stands alone …
        "Employee Handbook > Welcome > Expenses",  # … then the prose + code
        "Employee Handbook > Welcome > Contact",
    ]
    assert chunks[3].kind == "table"
    assert "```python" in chunks[4].text


def _tiktoken_cached() -> bool:
    cache = os.environ.get("TIKTOKEN_CACHE_DIR") or str(
        Path(tempfile.gettempdir()) / "data-gym-cache"
    )
    return Path(cache).exists() and any(Path(cache).iterdir())


@pytest.mark.skipif(not _tiktoken_cached(), reason="tiktoken encoding not cached (would download)")
def test_tiktoken_counter_is_used_by_default():
    chunker = Chunker(ChunkingSettings())
    assert chunker.count("hello world") == 2
