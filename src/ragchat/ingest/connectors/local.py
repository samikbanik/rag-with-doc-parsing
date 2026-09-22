"""Local directory connector: every supported file under a root, recursively."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

from ragchat.core.logging import get_logger
from ragchat.ingest.connectors.base import Connector, SourceItem
from ragchat.ingest.parsers.router import is_supported

log = get_logger(__name__)


class LocalFolderConnector(Connector):
    name = "local"

    def __init__(self, root: Path, include_hidden: bool = False) -> None:
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise NotADirectoryError(self.root)
        self.include_hidden = include_hidden
        self._skipped = 0

    @property
    def uri_prefix(self) -> str:
        return str(self.root) + os.sep

    @property
    def skipped(self) -> int:
        return self._skipped

    def iter_items(self) -> Iterator[SourceItem]:
        self._skipped = 0
        for dirpath, dirnames, filenames in os.walk(self.root):
            if not self.include_hidden:
                dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            dirnames.sort()
            for name in sorted(filenames):
                if name.startswith(".") and not self.include_hidden:
                    continue
                path = Path(dirpath) / name
                if not path.is_file():
                    continue
                if not is_supported(path):
                    self._skipped += 1
                    log.debug("unsupported file skipped", path=str(path))
                    continue
                yield SourceItem(uri=str(path), path=path)
