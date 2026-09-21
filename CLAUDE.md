# ragchat — working notes for Claude Code

Read `PLAN.md` first: it holds the reviewed architecture, design rules, milestone table, and a
**Status** section saying what is done and what is next. Update Status when a milestone lands.

## Commands
- `make install` / `make up` / `make status` / `make test` / `make lint` / `make fmt` / `make migrate`
- CLI: `uv run rag <command>` (defined in `src/ragchat/cli.py`, Typer)
- Frontend: `cd frontend && npm run dev` (Vite, proxies `/api` → `localhost:8000`)

## Conventions
- Python 3.12, `uv`, ruff (line length 100). Async by default (SQLAlchemy async, `AsyncOpenAI`, `AsyncQdrantClient`).
- All OpenAI calls go through `ragchat.core.llm.LLMClient`; all config through `ragchat.core.settings.get_settings()`.
  Tunables live in `config/settings.yaml`; secrets/URLs in `.env` (never commit).
- SQLAlchemy models inherit `ragchat.core.db.Base` and are imported in `alembic/env.py`; schema changes = Alembic migration.
- Follow the "Key design rules" in `PLAN.md` (typed IR, uuid5 chunk IDs, no-gap re-ingest, index config guard, measure before adding retrieval features).
- Tests: unit tests must not hit the network; OpenAI via vcrpy cassettes, Qdrant/Postgres via testcontainers.
- Work milestone by milestone; do not start the next milestone without being asked.
