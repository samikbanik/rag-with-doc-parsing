from pathlib import Path

from ragchat.core.settings import DEFAULT_CONFIG_PATH, Settings, get_settings


def test_yaml_defaults_are_loaded():
    s = get_settings()
    assert s.embedding.dimensions == 1024
    assert s.vectorstore.collection == "ragchat_chunks"
    assert s.pipeline_version >= 1


def test_env_overrides_yaml(monkeypatch):
    monkeypatch.setenv("RAGCHAT_EMBEDDING__DIMENSIONS", "256")
    s = Settings(embedding={"dimensions": 1024})
    assert s.embedding.dimensions == 256


def test_sync_url_swaps_driver(monkeypatch):
    # env has the highest precedence, so it wins over any .env on the developer machine
    monkeypatch.setenv("RAGCHAT_DATABASE_URL", "postgresql+asyncpg://u:p@h/db")
    s = Settings()
    assert s.sync_database_url == "postgresql+psycopg://u:p@h/db"


def test_config_file_exists():
    assert Path(DEFAULT_CONFIG_PATH).exists()
