"""Vector store factory."""

from __future__ import annotations

from app.config import AppSettings, get_settings
from app.db.spaces import EmbeddingSpace
from app.db.vector_store.base import VectorFilter, VectorStore
from app.db.vector_store.memory_store import InMemoryVectorStore
from app.db.vector_store.pgvector_store import PgVectorStore
from app.domain.enums import VectorBackend
from app.logging import get_logger

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
        from app.db.vector_store.qdrant_store import QdrantVectorStore  # noqa: PLC0415

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

    from app.db.session import get_session_factory  # noqa: PLC0415

    return PgVectorStore(get_session_factory(settings.database), space)


__all__ = ["build_vector_store", "VectorFilter", "VectorStore"]