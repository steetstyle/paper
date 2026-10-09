"""Vector store factory."""

from __future__ import annotations

from paper_app.config import AppSettings, get_settings
from paper_app.db.spaces import EmbeddingSpace
from paper_app.db.vector_store.base import VectorFilter, VectorStore
from paper_app.db.vector_store.memory_store import InMemoryVectorStore
from paper_app.db.vector_store.pgvector_store import PgVectorStore
from paper_app.domain.enums import VectorBackend
from paper_app.logging import get_logger

logger = get_logger(__name__)


def build_vector_store(
    settings: AppSettings | None = None,
    space: EmbeddingSpace | None = None,
) -> VectorStore:
    """Construct the store for ``space`` (default space when omitted)."""
    settings = settings or get_settings()
    space = space or EmbeddingSpace.default_for(settings)
    backend = settings.vector.backend

    if backend == VectorBackend.MEMORY:
        return InMemoryVectorStore(space)

    if backend == VectorBackend.QDRANT:
        from paper_app.db.vector_store.qdrant_store import QdrantVectorStore  # noqa: PLC0415

        return QdrantVectorStore(
            url=settings.vector.qdrant_url,
            space=space,
            api_key=settings.vector.qdrant_api_key,
            collection_prefix=settings.vector.qdrant_collection,
        )

    if not settings.database.is_postgres:
        logger.warning(
            "pgvector_requires_postgres",
            extra={
                "url_scheme": settings.database.url.split(":", 1)[0],
                "space": space.name,
            },
        )
        return InMemoryVectorStore(space)

    from paper_app.db.session import get_session_factory  # noqa: PLC0415

    return PgVectorStore(get_session_factory(settings.database), space)


__all__ = ["build_vector_store", "VectorFilter", "VectorStore"]