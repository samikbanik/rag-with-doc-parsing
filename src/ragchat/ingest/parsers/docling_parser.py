"""PDF / DOCX / PPTX / XLSX → IR via Docling (OCR disabled).

The converter is built once per process (model loading takes seconds), which suits the
process-pool parsing in `ingest.pipeline`. Image-only PDF pages produce no blocks: they are
logged and listed in `metadata["image_only_pages"]`, per the no-OCR decision in PLAN.md.
"""

from __future__ import annotations

import importlib.metadata
from pathlib import Path
from typing import Any

from ragchat.core.logging import get_logger
from ragchat.ingest.ir import Block, BlockType
from ragchat.ingest.parsers.base import BaseParser, ParseError, ParseResult
from ragchat.ingest.parsers.tables import rows_to_html, rows_to_markdown

log = get_logger(__name__)

# Docling labels that are layout furniture rather than content.
_SKIP_LABELS = {"page_header", "page_footer"}


class DoclingParser(BaseParser):
    name = "docling"
    version = importlib.metadata.version("docling")
    extensions = frozenset({"pdf", "docx", "pptx", "xlsx"})

    def __init__(self) -> None:
        self._converter: Any = None

    @property
    def converter(self):  # noqa: ANN201 - docling types are heavy to import at module level
        if self._converter is None:
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.document_converter import DocumentConverter, PdfFormatOption

            pdf_opts = PdfPipelineOptions(do_ocr=False, do_table_structure=True)
            pdf_opts.table_structure_options.do_cell_matching = True
            # Infer heading levels from bookmarks / numbering / font size; the layout model
            # alone labels every heading as level 1. The style pass needs parsed pages.
            pdf_opts.heading_hierarchy_options.enabled = True
            pdf_opts.generate_parsed_pages = True
            self._converter = DocumentConverter(
                allowed_formats=[
                    InputFormat.PDF,
                    InputFormat.DOCX,
                    InputFormat.PPTX,
                    InputFormat.XLSX,
                ],
                format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_opts)},
            )
        return self._converter

    def parse(self, path: Path) -> ParseResult:
        from docling.datamodel.base_models import ConversionStatus

        result = self.converter.convert(path, raises_on_error=False)
        if result.status not in (ConversionStatus.SUCCESS, ConversionStatus.PARTIAL_SUCCESS):
            errors = "; ".join(e.error_message for e in result.errors) or str(result.status)
            raise ParseError(errors)
        doc = result.document
        blocks, meta = self._blocks(doc)
        meta["docling_status"] = str(result.status.value)
        if doc.pages:
            meta["page_count"] = len(doc.pages)
            with_text = {b.page for b in blocks if b.page is not None}
            image_only = sorted(p for p in doc.pages if p not in with_text)
            if image_only:
                log.warning(
                    "image-only pages skipped (OCR disabled)", path=str(path), pages=image_only
                )
                meta["image_only_pages"] = image_only
        title = doc.name if doc.name and doc.name != path.stem else None
        title = title or next((b.text for b in blocks if b.type is BlockType.HEADING), None)
        return ParseResult(blocks=blocks, title=title, metadata=meta)

    # -- docling item → Block -------------------------------------------------------------------

    def _blocks(self, doc) -> tuple[list[Block], dict[str, Any]]:  # noqa: ANN001
        from docling_core.types.doc import (
            CodeItem,
            DocItemLabel,
            GroupItem,
            GroupLabel,
            ListItem,
            PictureItem,
            SectionHeaderItem,
            TableItem,
            TextItem,
            TitleItem,
        )

        # Docling numbers section headers from 1; if the document also has a Title, shift
        # them down so the title is the only level-1 heading.
        has_title = any(isinstance(t, TitleItem) for t in doc.texts)
        offset = 1 if has_title else 0
        blocks: list[Block] = []
        pictures = 0

        list_labels = {GroupLabel.LIST, GroupLabel.ORDERED_LIST}

        def list_depth(item) -> int:  # noqa: ANN001 - nesting = number of enclosing list groups
            depth, node = 0, item
            while node.parent is not None:
                node = node.parent.resolve(doc)
                if isinstance(node, GroupItem) and node.label in list_labels:
                    depth += 1
            return depth

        for item, _level in doc.iterate_items(with_groups=False):
            page, bbox = _prov(item)
            if isinstance(item, PictureItem):
                pictures += 1
                continue
            if isinstance(item, TableItem):
                rows, header = _table_rows(item)
                if not rows and not header:
                    continue
                blocks.append(
                    Block(
                        type=BlockType.TABLE,
                        text=rows_to_markdown(rows, header),
                        table_html=rows_to_html(rows, header),
                        page=page,
                        bbox=bbox,
                    )
                )
                continue
            if not isinstance(item, TextItem):
                continue
            text = item.text or ""
            if isinstance(item, TitleItem):
                blocks.append(
                    Block(type=BlockType.HEADING, text=text, level=1, page=page, bbox=bbox)
                )
            elif isinstance(item, SectionHeaderItem):
                lvl = min(max(item.level + offset, 1), 6)
                blocks.append(
                    Block(type=BlockType.HEADING, text=text, level=lvl, page=page, bbox=bbox)
                )
            elif isinstance(item, ListItem):
                blocks.append(
                    Block(
                        type=BlockType.LIST_ITEM,
                        text=text,
                        level=min(max(list_depth(item), 1), 6),
                        page=page,
                        bbox=bbox,
                    )
                )
            elif isinstance(item, CodeItem):
                lang = getattr(item.code_language, "value", None)
                if lang in (None, "unknown"):
                    lang = None
                blocks.append(
                    Block(type=BlockType.CODE, text=text, language=lang, page=page, bbox=bbox)
                )
            elif item.label == DocItemLabel.CAPTION:
                blocks.append(Block(type=BlockType.CAPTION, text=text, page=page, bbox=bbox))
            elif item.label in _SKIP_LABELS:
                continue
            else:
                blocks.append(Block(type=BlockType.PARAGRAPH, text=text, page=page, bbox=bbox))

        meta: dict[str, Any] = {}
        if pictures:
            meta["pictures_skipped"] = pictures
        return blocks, meta


def _prov(item) -> tuple[int | None, tuple[float, float, float, float] | None]:  # noqa: ANN001
    prov = getattr(item, "prov", None)
    if not prov:
        return None, None
    p = prov[0]
    bbox = p.bbox
    return p.page_no, (round(bbox.l, 2), round(bbox.t, 2), round(bbox.r, 2), round(bbox.b, 2))


def _table_rows(item) -> tuple[list[list[str]], list[str] | None]:  # noqa: ANN001
    grid = item.data.grid if item.data else []
    if not grid:
        return [], None
    rows = [[(c.text or "").strip() for c in row] for row in grid]
    header: list[str] | None = None
    if any(c.column_header for c in grid[0]):
        header = rows.pop(0)
    return rows, header
