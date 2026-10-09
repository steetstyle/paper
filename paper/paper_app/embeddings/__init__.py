"""Pluggable embedding layer."""

from paper_app.embeddings.base import EmbeddingError, EmbeddingProvider
from paper_app.embeddings.hashing_provider import HashingEmbeddingProvider
from paper_app.embeddings.openai_provider import OpenAIEmbeddingProvider
from paper_app.embeddings.registry import (
    available_providers,
    build_default_provider,
    build_provider,
    register_provider,
)
from paper_app.embeddings.sentence_transformers_provider import SentenceTransformerProvider

__all__ = [
    "EmbeddingError",
    "EmbeddingProvider",
    "HashingEmbeddingProvider",
    "OpenAIEmbeddingProvider",
    "SentenceTransformerProvider",
    "available_providers",
    "build_default_provider",
    "build_provider",
    "register_provider",
]