"""Trace of one question → answer run (PLAN.md rule 11). Feeds the UI trace panel and eval."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from ragchat.core.db import Base


class Trace(Base):
    __tablename__ = "traces"

    trace_id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
    conversation_id: Mapped[uuid.UUID | None] = mapped_column(index=True)  # set from M6 on
    query: Mapped[str] = mapped_column(Text)
    rewritten_query: Mapped[str | None] = mapped_column(Text)  # M4 query rewrite
    retrieved: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, default=list)
    cited_chunk_ids: Mapped[list[str]] = mapped_column(JSONB, default=list)
    answer: Mapped[str | None] = mapped_column(Text)
    refused: Mapped[bool] = mapped_column(default=False)
    tool_calls: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, default=list)  # M6
    model: Mapped[str | None] = mapped_column(String(64))
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    retrieval_ms: Mapped[int] = mapped_column(Integer, default=0)
    llm_ms: Mapped[int] = mapped_column(Integer, default=0)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text)
    meta: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, default=dict)
