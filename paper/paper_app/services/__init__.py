"""Application services: use-case orchestration on top of the pipeline.

Imports are lazy on purpose: ``app.pipeline.steps`` depends on
``paper_app.services.chunker``, so eagerly re-exporting ``ingestion`` here would close
an import cycle.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "ArxivSearchService",
    "ChunkingService",
    "IngestionService",
    "IngestionResult",
    "SemanticSearchService",
    "SemanticSearchHit",
]

_EXPORTS = {
    "ArxivSearchService": "paper_app.services.arxiv_search",
    "ChunkingService": "paper_app.services.chunker",
    "IngestionService": "paper_app.services.ingestion",
    "IngestionResult": "paper_app.services.ingestion",
    "build_ingestion_service": "paper_app.services.ingestion",
    "SemanticSearchService": "paper_app.services.semantic_search",
    "SemanticSearchHit": "paper_app.services.semantic_search",
}


def __getattr__(name: str) -> Any:
    module_path = _EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module_path), name)


def __dir__() -> list[str]:
    return sorted(__all__)