"""Connector interface: a connector enumerates source items as local files to parse."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SourceItem:
    uri: str  # stable locator; with `source` it determines doc_id
    path: Path  # local file to hand to the parser


class Connector(ABC):
    name: str  # value of ParsedDocument.source

    @property
    @abstractmethod
    def uri_prefix(self) -> str:
        """Every item's uri starts with this; used to find documents to prune."""

    @abstractmethod
    def iter_items(self) -> Iterator[SourceItem]: ...

    @property
    def skipped(self) -> int:
        """Files seen but not yielded (unsupported types); for the ingest report."""
        return 0
