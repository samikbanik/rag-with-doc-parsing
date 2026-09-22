"""Thin wrapper over the OpenAI SDK.

Everything that talks to OpenAI goes through here so that model IDs, retries,
usage accounting and (later) request/response tracing live in one place.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import TypeVar

from openai import (
    AsyncOpenAI,
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    PermissionDeniedError,
)
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from ragchat.core.logging import get_logger
from ragchat.core.settings import get_settings

log = get_logger(__name__)
T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True)
class StructuredResult[S: BaseModel]:
    parsed: S
    usage: Usage
    model: str


# Retry transient failures (rate limits, 5xx, network); never retry bad requests/credentials.
_NO_RETRY = (AuthenticationError, PermissionDeniedError, BadRequestError, NotFoundError)
_retry = retry(
    retry=retry_if_not_exception_type(_NO_RETRY),
    stop=stop_after_attempt(5),
    wait=wait_exponential_jitter(initial=1, max=20),
    reraise=True,
)


class LLMClient:
    def __init__(self, client: AsyncOpenAI | None = None) -> None:
        self.settings = get_settings()
        self.client = client or AsyncOpenAI(api_key=self.settings.openai_api_key.get_secret_value())

    # -- embeddings -------------------------------------------------------------------------

    @_retry
    async def embed(self, texts: list[str]) -> list[list[float]]:
        cfg = self.settings.embedding
        resp = await self.client.embeddings.create(
            model=cfg.model, input=texts, dimensions=cfg.dimensions
        )
        # API returns in input order, but sort defensively by index.
        return [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]

    # -- text generation --------------------------------------------------------------------

    @_retry
    async def complete(self, system: str, user: str, *, model: str | None = None) -> str:
        cfg = self.settings.llm
        resp = await self.client.responses.create(
            model=model or cfg.model,
            instructions=system,
            input=user,
            temperature=cfg.temperature,
            max_output_tokens=cfg.max_output_tokens,
        )
        return resp.output_text

    async def complete_structured(
        self, system: str, user: str, schema: type[T], *, model: str | None = None
    ) -> T:
        """Structured output: the model must return JSON matching `schema`."""
        return (await self.generate_structured(system, user, schema, model=model)).parsed

    @_retry
    async def generate_structured(
        self, system: str, user: str, schema: type[T], *, model: str | None = None
    ) -> StructuredResult[T]:
        """Like `complete_structured` but also returns token usage (for traces)."""
        cfg = self.settings.llm
        model = model or cfg.small_model
        resp = await self.client.responses.parse(
            model=model,
            instructions=system,
            input=user,
            temperature=cfg.temperature,
            max_output_tokens=cfg.max_output_tokens,
            text_format=schema,
        )
        parsed = resp.output_parsed
        if parsed is None:
            raise ValueError("model returned no parsable structured output")
        usage = Usage(
            input_tokens=resp.usage.input_tokens if resp.usage else 0,
            output_tokens=resp.usage.output_tokens if resp.usage else 0,
        )
        return StructuredResult(parsed=parsed, usage=usage, model=model)

    # -- health -----------------------------------------------------------------------------

    async def ping(self) -> str:
        """Cheapest possible authenticated call: confirms the key works and the model exists."""
        model = await self.client.models.retrieve(self.settings.llm.model)
        return model.id


@lru_cache(maxsize=1)
def get_llm() -> LLMClient:
    return LLMClient()
