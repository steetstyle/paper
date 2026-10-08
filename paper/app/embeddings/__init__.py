"""Pluggable embedding layer."""

from app.embeddings.base import EmbeddingError, EmbeddingProvider
from app.embeddings.hashing_provider import HashingEmbeddingProvider
from app.embeddings.openai_provider import OpenAIEmbeddingProvider
from app.embeddings.registry import (
    available_providers,
    build_default_provider,
    build_provider,
    register_provider,
)
from app.embeddings.sentence_transformers_provider import SentenceTransformerProvider

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