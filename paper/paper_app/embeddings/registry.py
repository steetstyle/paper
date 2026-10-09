"""Embedding provider registry.

``EMBEDDING_PROVIDER`` selects the implementation; unknown names raise at
startup rather than silently degrading. All providers share the constructor
signature declared on :class:`~paper_app.embeddings.base.EmbeddingProvider`, so any
entry in the registry — including plugins — is built the same way.
"""

from __future__ import annotations

from paper_app.config import EmbeddingSettings, get_settings
from paper_app.embeddings.base import EmbeddingError, EmbeddingProvider, resolve_dimensions
from paper_app.embeddings.bge_provider import BgeEmbeddingProvider
from paper_app.embeddings.hashing_provider import HashingEmbeddingProvider
from paper_app.embeddings.openai_provider import OpenAIEmbeddingProvider
from paper_app.embeddings.sentence_transformers_provider import SentenceTransformerProvider
from paper_app.logging import get_logger

logger = get_logger(__name__)

_REGISTRY: dict[str, type[EmbeddingProvider]] = {
    "openai": OpenAIEmbeddingProvider,
    "sentence-transformers": SentenceTransformerProvider,
    "local": SentenceTransformerProvider,
    "bge": BgeEmbeddingProvider,
    "bge-large": BgeEmbeddingProvider,
    "hashing": HashingEmbeddingProvider,
    "hash": HashingEmbeddingProvider,
}


def register_provider(name: str, provider_cls: type[EmbeddingProvider]) -> None:
    """Register a custom provider (in-house models, plugins)."""
    _REGISTRY[name.strip().lower()] = provider_cls


def available_providers() -> list[str]:
    return sorted(_REGISTRY)


def build_provider(settings: EmbeddingSettings | None = None) -> EmbeddingProvider:
    settings = settings or get_settings().embedding
    provider_cls = _REGISTRY.get(settings.provider)
    if provider_cls is None:
        raise EmbeddingError(
            f"unknown embedding provider {settings.provider!r}; "
            f"available: {', '.join(available_providers())}"
        )

    if provider_cls is OpenAIEmbeddingProvider:
        return OpenAIEmbeddingProvider(
            model=settings.model,
            dimensions=settings.dimensions,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            batch_size=settings.batch_size,
            normalize=settings.normalize_embeddings,
        )
    if provider_cls is BgeEmbeddingProvider:
        return BgeEmbeddingProvider(
            model=settings.model,
            dimensions=settings.dimensions,
            batch_size=settings.batch_size,
            device=settings.device,
            cache_dir=str(settings.cache_dir),
            normalize=settings.normalize_embeddings,
        )
    if provider_cls is SentenceTransformerProvider:
        return SentenceTransformerProvider(
            model=settings.model,
            dimensions=settings.dimensions,
            batch_size=settings.batch_size,
            device=settings.device,
            cache_dir=str(settings.cache_dir),
            normalize=settings.normalize_embeddings,
        )
    return provider_cls(
        model=settings.model,
        dimensions=settings.dimensions,
        batch_size=settings.batch_size,
        normalize=settings.normalize_embeddings,
    )


def build_default_provider() -> EmbeddingProvider:
    return build_provider(get_settings().embedding)


def validate_configuration(settings: EmbeddingSettings | None = None) -> int:
    """Return the effective dimension count, warning on misconfiguration."""
    settings = settings or get_settings().embedding
    return resolve_dimensions(build_provider(settings), settings.dimensions)