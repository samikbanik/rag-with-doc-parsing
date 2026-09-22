"""Application settings.

Layering (lowest to highest precedence):
  1. config/settings.yaml   -- non-secret defaults, committed
  2. .env                   -- local secrets / URLs, gitignored
  3. environment variables  -- RAGCHAT_<SECTION>__<KEY>, plus OPENAI_API_KEY
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "settings.yaml"


class LLMSettings(BaseModel):
    model: str = "gpt-4.1"
    small_model: str = "gpt-4.1-mini"
    temperature: float = 0.0
    max_output_tokens: int = 2048


class EmbeddingSettings(BaseModel):
    model: str = "text-embedding-3-large"
    dimensions: int = 1024
    batch_size: int = 256
    max_concurrency: int = 4
    price_per_million_tokens: float = 0.13  # USD, for the pre-embedding cost estimate


class VectorStoreSettings(BaseModel):
    collection: str = "ragchat_chunks"
    dense_vector_name: str = "dense"
    sparse_vector_name: str = "sparse"


class ChunkingSettings(BaseModel):
    target_tokens: int = 600
    max_tokens: int = 800
    overlap_tokens: int = 60
    table_max_tokens: int = 1500


class RetrievalSettings(BaseModel):
    prefetch_k: int = 40
    top_k: int = 8
    max_chunks_per_doc: int = 3
    max_context_tokens: int = 8000  # budget for assembled <document> context
    hybrid: bool = False
    rerank: Literal["none", "cross_encoder", "llm"] = "none"
    query_rewrite: bool = False


class EvalSettings(BaseModel):
    k: int = 8  # recall@k that gates regressions
    regression_threshold: float = 0.05  # fail `rag eval` if recall@k drops by more than this
    golden_path: Path = PROJECT_ROOT / "eval" / "golden.jsonl"
    baselines_dir: Path = PROJECT_ROOT / "eval" / "baselines"
    questions_per_chunk: int = 2  # synthetic golden generation
    max_chunks: int = 60  # chunks sampled for synthetic generation


class AgentSettings(BaseModel):
    max_tool_iterations: int = 6
    max_context_tokens: int = 24000


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAGCHAT_",
        env_nested_delimiter="__",
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Secrets / environment-specific
    openai_api_key: SecretStr = Field(default=SecretStr(""), validation_alias="OPENAI_API_KEY")
    database_url: str = "postgresql+asyncpg://ragchat:ragchat@localhost:5432/ragchat"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: SecretStr | None = None

    # Paths
    data_dir: Path = PROJECT_ROOT / "data"

    # Sections (defaults come from settings.yaml)
    llm: LLMSettings = LLMSettings()
    embedding: EmbeddingSettings = EmbeddingSettings()
    vectorstore: VectorStoreSettings = VectorStoreSettings()
    chunking: ChunkingSettings = ChunkingSettings()
    retrieval: RetrievalSettings = RetrievalSettings()
    agent: AgentSettings = AgentSettings()
    eval: EvalSettings = EvalSettings()
    pipeline_version: int = 1

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Init kwargs carry the YAML defaults, so they must have the *lowest* precedence:
        # env > .env > yaml. (pydantic-settings default puts init kwargs first.)
        return env_settings, dotenv_settings, init_settings, file_secret_settings

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def parsed_dir(self) -> Path:
        return self.data_dir / "parsed"

    @property
    def sync_database_url(self) -> str:
        """Driver-swapped URL for Alembic and other sync consumers."""
        return self.database_url.replace("+asyncpg", "+psycopg")


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as f:
        return yaml.safe_load(f) or {}


@lru_cache(maxsize=1)
def get_settings(config_path: Path | None = None) -> Settings:
    """Build settings: YAML defaults first, then env/.env override via pydantic-settings."""
    yaml_values = _load_yaml(config_path or DEFAULT_CONFIG_PATH)
    return Settings(**yaml_values)
