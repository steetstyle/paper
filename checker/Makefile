.PHONY: help bootstrap bootstrap-cpu install install-link install-all test lint fmt typecheck check \
        scan similarity compliance calibrate doctor models segment diff clean requirements

# This project has its OWN virtualenv (checker/.venv). If it does not exist yet,
# the paper-app venv is used as a fallback so the repo stays usable without a
# 5 GB install; `make bootstrap` creates the real one.
PAPER_VENV := ../paper/.venv
PY         := $(if $(wildcard .venv/bin/python),.venv/bin/python,$(if $(wildcard $(PAPER_VENV)/bin/python),$(PAPER_VENV)/bin/python,python3))
PIP        := $(PY) -m pip
CHECKER    := .venv/bin/checker

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

bootstrap: ## Create checker/.venv from requirements-dev.txt (CUDA torch, ~5 GB)
	bash scripts/bootstrap.sh --dev

bootstrap-cpu: ## Same, but CPU-only torch (~1 GB)
	bash scripts/bootstrap.sh --cpu --dev

install: ## Create checker/.venv from requirements-dev.txt (alias for bootstrap)
	bash scripts/bootstrap.sh --dev

install-link: ## Install this package into the active venv (editable)
	$(PIP) install -e .

install-all: ## Alias for bootstrap
	bash scripts/bootstrap.sh --dev

test: ## Run the test suite (no model downloads)
	$(PY) -m pytest -q

test-cov: ## Tests with a coverage report
	$(PY) -m pytest --cov=checker_app --cov-report=term-missing

mcp-test: ## End-to-end MCP test: launches `checker mcp` and speaks the protocol
	$(PY) scripts/test_mcp_stdio.py $(f)

lint: ## Lint
	$(PY) -m ruff check checker_app tests scripts

fmt: ## Format and autofix
	$(PY) -m ruff format checker_app tests scripts
	$(PY) -m ruff check --fix checker_app tests scripts

typecheck: ## Static types
	$(PY) -m mypy checker_app

check: lint typecheck test ## Lint + types + tests

scan: ## Full thesis scan: make scan f=tez.md ref=./kaynaklar
	$(CHECKER) scan $(f) --profile thesis --ref-dir $(ref) --json report.json --md report.md --annotate tez.annotated.md

similarity: ## Turnitin-style similarity with institutional filters
	$(CHECKER) similarity $(f) --ref-dir $(ref) --json similarity.json

compliance: ## YOK-style compliance report (disclosure, citation status, AI sections)
	$(CHECKER) compliance $(f) --ref-dir $(ref) --json compliance.json

diff: ## Sentence-level diff of two revisions
	$(CHECKER) compare $(f) $(g)

segment: ## Show only the sentence segmentation: make segment f=tez.md
	$(CHECKER) segment $(f)

doctor: ## Check dependencies, models and settings
	$(CHECKER) doctor

models: ## Show the models that will be used
	$(CHECKER) models

calibrate: ## Re-measure perplexity calibration on the bundled samples
	$(PY) scripts/calibrate.py

calibrate-local: ## Calibrate the FPR operating point on YOUR human corpus
	$(CHECKER) calibrate --human-dir $(h) --out .checker_cache/operating_point.json

mcp: ## Start the MCP server on stdio (Claude Desktop / Claude Code)
	$(CHECKER) mcp

mcp-http: ## Start the MCP server on streamable HTTP
	$(CHECKER) mcp --transport http --port 8090

clean: ## Remove caches and artifacts
	rm -rf .pytest_cache .ruff_cache .mypy_cache .checker_cache **/__pycache__
requirements: ## Print the installed versions of the direct dependencies
	@$(PY) -c "import importlib.metadata as m; \
	[print(f'{p}=={m.version(p)}') for p in ['pydantic','pydantic-settings','httpx','typer','rich','beautifulsoup4','pypdf','torch','transformers','huggingface-hub','safetensors','numpy','sentencepiece','pytest','ruff','mypy'] if True]"
