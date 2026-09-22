"""Parser interface. Parsers turn one file into blocks; the router wraps them in a
`ParsedDocument` with ids, hashes and timestamps."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ragchat.ingest.ir import Block


class UnsupportedFileTypeError(ValueError):
    pass


class ParseError(RuntimeError):
    pass


@dataclass
class ParseResult:
    blocks: list[Block]
    title: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseParser(ABC):
    name: str
    version: str
    extensions: frozenset[str]  # lower-case, without the dot

    @abstractmethod
    def parse(self, path: Path) -> ParseResult: ...
