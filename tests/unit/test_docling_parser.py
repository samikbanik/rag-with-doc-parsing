"""Docling snapshot tests. Skipped unless the Docling models are already cached locally, so
CI without network/model access is not blocked (unit tests must not hit the network)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from ragchat.ingest.ir import BlockType, ParsedDocument
from ragchat.ingest.parsers import router

_HF_HUB = Path.home() / ".cache" / "huggingface" / "hub"
_MODEL_CACHE_DIRS = [
    Path.home() / ".cache" / "docling" / "models",
    _HF_HUB / "models--docling-project--docling-layout-heron",
    _HF_HUB / "models--ds4sd--docling-models",
]


def _models_available() -> bool:
    if os.getenv("RAGCHAT_RUN_DOCLING_TESTS") == "1":
        return True
    return any(p.exists() for p in _MODEL_CACHE_DIRS)


pytestmark = [
    pytest.mark.docling,
    pytest.mark.skipif(not _models_available(), reason="Docling models not cached locally"),
]


def _view(doc: ParsedDocument) -> dict:
    return {
        "title": doc.title,
        "doc_type": doc.doc_type,
        "parser": doc.parser,
        "metadata": {k: v for k, v in doc.metadata.items() if k not in ("size_bytes",)},
        "blocks": [b.model_dump(exclude_none=True) for b in doc.blocks],
    }


@pytest.mark.parametrize("name", ["policy.docx", "roadmap.pptx", "report.pdf"])
def test_docling_snapshot(fixtures_dir: Path, snapshot, name: str):
    doc = router.parse_file(fixtures_dir / name, source="local", uri=f"fixture:{name}")
    assert doc.parser == "docling"
    snapshot(f"{Path(name).stem}_{doc.doc_type}", _view(doc))


def test_docx_structure(fixtures_dir: Path):
    doc = router.parse_file(fixtures_dir / "policy.docx", source="local", uri="fixture:policy")
    assert doc.title == "Security Policy"
    headings = [(b.text, b.level) for b in doc.blocks if b.type is BlockType.HEADING]
    assert headings == [
        ("Security Policy", 1),
        ("Passwords", 2),
        ("Devices", 2),
        ("Incident response", 3),
    ]
    assert [b.level for b in doc.blocks if b.type is BlockType.LIST_ITEM] == [1, 1]
    table = next(b for b in doc.blocks if b.type is BlockType.TABLE)
    assert "| Laptop | Employee |" in table.text


def test_pdf_structure(fixtures_dir: Path):
    doc = router.parse_file(fixtures_dir / "report.pdf", source="local", uri="fixture:report")
    assert doc.metadata["page_count"] == 2
    assert "image_only_pages" not in doc.metadata
    pages = {b.page for b in doc.blocks}
    assert pages == {1, 2}
    assert all(b.bbox is not None for b in doc.blocks)
    table = next(b for b in doc.blocks if b.type is BlockType.TABLE)
    assert table.page == 1
    assert "| Revenue | 120 | 142 |" in table.text
    outlook = next(b for b in doc.blocks if b.text == "Outlook")
    assert outlook.level == 2  # smaller font than the level-1 headings


def test_pptx_pages_are_slides(fixtures_dir: Path):
    doc = router.parse_file(fixtures_dir / "roadmap.pptx", source="local", uri="fixture:roadmap")
    assert doc.metadata["page_count"] == 3
    assert [b.page for b in doc.blocks if b.type is BlockType.HEADING] == [1, 2, 3]
