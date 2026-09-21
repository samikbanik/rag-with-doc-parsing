# ragchat

Enterprise agentic RAG chat over parsed documents and wikis. See `PLAN.md` for the design and milestones.

## Quick start

```bash
cp .env.example .env         # add your OPENAI_API_KEY
make install                 # uv sync
make up                      # Qdrant + Postgres via docker compose
make status                  # rag status: checks Qdrant, Postgres, OpenAI
```

Frontend (Vite + React + TS):

```bash
make frontend-install
make frontend-dev
```
