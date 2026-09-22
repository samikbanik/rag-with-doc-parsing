"""Pick a parser by file extension and wrap its output in a `ParsedDocument`."""

from __future__ import annotations

from functools import cache, lru_cache
from pathlib import Path

from ragchat.core.logging import get_logger
from ragchat.ingest.ir import (
    ParsedDocument,
    file_content_hash,
    file_last_modified,
    make_doc_id,
)
from ragchat.ingest.parsers.base import BaseParser, ParseError, UnsupportedFileTypeError

log = get_logger(__name__)

DOCLING_EXTENSIONS = frozenset({"pdf", "docx", "pptx", "xlsx"})


@cache
def _parsers() -> dict[str, BaseParser]:
    from ragchat.ingest.parsers.html import HTMLParser
    from ragchat.ingest.parsers.markdown import MarkdownParser

    registry: dict[str, BaseParser] = {}
    for parser in (MarkdownParser(), HTMLParser()):
        for ext in parser.extensions:
            registry[ext] = parser
    return registry


@lru_cache(maxsize=1)
def _docling() -> BaseParser:
    # Imported lazily: docling pulls in torch and takes seconds to import.
    from ragchat.ingest.parsers.docling_parser import DoclingParser

    return DoclingParser()


def supported_extensions() -> frozenset[str]:
    return frozenset(_parsers()) | DOCLING_EXTENSIONS


def is_supported(path: Path) -> bool:
    return path.suffix.lower().lstrip(".") in supported_extensions()


def parser_for(path: Path) -> BaseParser:
    ext = path.suffix.lower().lstrip(".")
    if ext in DOCLING_EXTENSIONS:
        return _docling()
    try:
        return _parsers()[ext]
    except KeyError:
        raise UnsupportedFileTypeError(f"no parser for '.{ext}' ({path})") from None


def parse_file(path: Path, *, source: str, uri: str | None = None) -> ParsedDocument:
    """Parse `path` into a normalised `ParsedDocument`.

    `uri` defaults to the absolute path; connectors pass their own locator so `doc_id` stays
    stable even if the local staging path changes (uploads, Notion exports).
    """
    path = Path(path)
    uri = uri or str(path.resolve())
    parser = parser_for(path)
    try:
        result = parser.parse(path)
    except UnsupportedFileTypeError:
        raise
    except Exception as exc:  # noqa: BLE001 - normalise parser failures
        raise ParseError(f"{parser.name} failed on {path}: {exc}") from exc
    doc = ParsedDocument(
        doc_id=make_doc_id(source, uri),
        source=source,
        uri=uri,
        title=(result.title or path.stem).strip() or path.stem,
        doc_type=path.suffix.lower().lstrip("."),
        content_hash=file_content_hash(path),
        last_modified=file_last_modified(path),
        parser=parser.name,
        parser_version=parser.version,
        blocks=result.blocks,
        metadata={"filename": path.name, "size_bytes": path.stat().st_size, **result.metadata},
    ).normalized()
    log.debug("parsed", path=str(path), parser=parser.name, blocks=len(doc.blocks))
    return doc
