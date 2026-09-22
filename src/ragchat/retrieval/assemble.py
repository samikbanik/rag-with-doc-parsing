"""Context assembly: retrieved chunks → numbered `<document>` blocks within a token budget.

Chunks are numbered 1..n in ranking order; the model cites them as [n] and the numbers map
back to chunk ids for the answer contract.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from xml.sax.saxutils import quoteattr

from ragchat.retrieval.retriever import RetrievedChunk


@dataclass
class AssembledContext:
    text: str
    chunks: list[RetrievedChunk] = field(default_factory=list)  # in citation-number order
    tokens: int = 0
    dropped: int = 0  # retrieved but over budget

    def chunk_for(self, number: int) -> RetrievedChunk | None:
        return self.chunks[number - 1] if 1 <= number <= len(self.chunks) else None


def render_document(number: int, c: RetrievedChunk) -> str:
    attrs = [f'id="{number}"', f"title={quoteattr(c.title)}", f"uri={quoteattr(c.uri)}"]
    if c.page_start is not None:
        pages = (
            str(c.page_start)
            if c.page_end in (None, c.page_start)
            else f"{c.page_start}-{c.page_end}"
        )
        attrs.append(f'page="{pages}"')
    body = f"{c.breadcrumb}\n\n{c.text}" if c.breadcrumb else c.text
    return f"<document {' '.join(attrs)}>\n{body}\n</document>"


def assemble(
    chunks: list[RetrievedChunk], *, max_tokens: int, count_tokens: Callable[[str], int]
) -> AssembledContext:
    parts: list[str] = []
    kept: list[RetrievedChunk] = []
    total = 0
    for c in chunks:
        block = render_document(len(kept) + 1, c)
        n = count_tokens(block)
        if kept and total + n > max_tokens:
            break
        parts.append(block)
        kept.append(c)
        total += n
    return AssembledContext(
        text="\n\n".join(parts), chunks=kept, tokens=total, dropped=len(chunks) - len(kept)
    )
