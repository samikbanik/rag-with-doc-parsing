.PHONY: install up down status lint test fmt frontend-install frontend-dev

install:            ## Install backend deps into .venv
	uv sync --extra dev

up:                 ## Start Qdrant + Postgres
	docker compose up -d

down:               ## Stop infra
	docker compose down

status:             ## Check connectivity to Qdrant, Postgres, OpenAI
	uv run rag status

migrate:            ## Apply DB migrations
	uv run alembic upgrade head

lint:
	uv run ruff check . && uv run ruff format --check .

fmt:
	uv run ruff check --fix . && uv run ruff format .

test:
	uv run pytest -q

frontend-install:
	cd frontend && npm install

frontend-dev:
	cd frontend && npm run dev
