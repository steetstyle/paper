.PHONY: help install install-dev up down logs db-up db-down migrate revision serve test lint fmt typecheck doctor clean ingest search shell

SHELL := /bin/bash
PY    := .venv/bin/python
PIP   := uv pip

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Create the venv and install the package with dev extras
	uv venv --python 3.12
	uv pip install -e ".[dev,html]"

install-all: ## Install every optional extra (heavy: mineru, torch, qdrant)
	uv pip install -e ".[dev,postgres,openai,html,qdrant,numpy]"

up: ## Start Postgres + pgvector
	docker compose up -d db

down: ## Stop everything and drop volumes
	docker compose --profile app --profile qdrant down -v

logs: ## Tail API logs
	docker compose --profile app logs -f api

db-up: ## Start only Postgres
	docker compose up -d db

db-down: ## Stop Postgres
	docker compose stop db

migrate: ## Apply Alembic migrations
	.venv/bin/paper db upgrade

revision: ## Autogenerate a migration: make revision m="add x"
	.venv/bin/paper db revision -m "$(m)"

serve: ## Run the API with reload
	.venv/bin/paper serve --reload

test: ## Run the test suite
	.venv/bin/pytest -q

test-cov: ## Run tests with a coverage report
	.venv/bin/pytest --cov=paper_app --cov-report=term-missing

lint: ## Lint
	.venv/bin/ruff check paper_app tests

fmt: ## Format and autofix
	.venv/bin/ruff format paper_app tests
	.venv/bin/ruff check --fix paper_app tests

typecheck: ## Static types
	.venv/bin/mypy paper_app

doctor: ## Check DB, vector store, MinerU and embedding config
	.venv/bin/paper doctor

ingest: ## Ingest a paper: make ingest id=1706.03762
	.venv/bin/paper ingest $(id)

search: ## Search ArXiv: make search q="cat:cs.CL"
	.venv/bin/paper search "$(q)" --max 10

clean: ## Remove caches and build artifacts
	rm -rf .pytest_cache .ruff_cache .mypy_cache **/__pycache__ .data/dbg.db

.PHONY: check
check: lint typecheck test ## Lint + types + tests