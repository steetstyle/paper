"""Application services: use-case orchestration on top of the pipeline.

Imports are lazy on purpose: ``app.pipeline.steps`` depends on
``app.services.chunker``, so eagerly re-exporting ``ingestion`` here would close
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
    "ArxivSearchService": "app.services.arxiv_search",
    "ChunkingService": "app.services.chunker",
    "IngestionService": "app.services.ingestion",
    "IngestionResult": "app.services.ingestion",
    "build_ingestion_service": "app.services.ingestion",
    "SemanticSearchService": "app.services.semantic_search",
    "SemanticSearchHit": "app.services.semantic_search",
}


def __getattr__(name: str) -> Any:
    module_path = _EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module_path), name)


def __dir__() -> list[str]:
    return sorted(__all__)