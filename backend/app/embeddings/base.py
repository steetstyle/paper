"""Embedding provider contract.

All providers share one constructor signature so the registry can build any of
them uniformly::

    provider = SomeProvider(model=..., dimensions=..., batch_size=...)

Implementations must be async-friendly and stateless across calls so the same
instance can be shared by the pipeline and the search API.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import ClassVar

from app.logging import get_logger

logger = get_logger(__name__)


class EmbeddingError(RuntimeError):
    """Raised for provider misconfiguration or a failed upstream request."""


class EmbeddingProvider(ABC):
    """Base class for embedding backends."""

    #: stable identifier persisted alongside every vector
    name: ClassVar[str] = "base"

    def __init__(
        self,
        model: str,
        dimensions: int,
        *,
        batch_size: int = 64,
        normalize: bool = True,
    ) -> None:
        self._model = model
        self._dimensions = dimensions
        self._batch_size = max(1, batch_size)
        self._normalize = normalize

    @property
    def model(self) -> str:
        """Model identifier recorded with each vector."""
        return self._model

    @property
    def dimensions(self) -> int:
        """Vector length. Must be stable across restarts."""
        return self._dimensions

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @abstractmethod
    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of documents."""

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single search query. Defaults to document embedding."""
        vectors = await self.embed_documents([text])
        return vectors[0]

    def count_tokens(self, text: str) -> int:
        """Best-effort token count; providers with real tokenizers override."""
        from app.infra.text import approx_tokens

        return approx_tokens(text)

    async def warmup(self) -> None:
        """Preload weights / validate credentials before serving traffic."""
        return None

    def _finalise(self, vectors: list[list[float]]) -> list[list[float]]:
        return self._l2_normalise(vectors) if self._normalize else vectors

    @staticmethod
    def _l2_normalise(matrix: list[list[float]]) -> list[list[float]]:
        """Row-wise L2 normalisation; numpy is used when installed."""
        try:
            import numpy as np
        except ImportError:
            normalised: list[list[float]] = []
            for row in matrix:
                norm = math.sqrt(sum(value * value for value in row)) or 1.0
                normalised.append([value / norm for value in row])
            return normalised
        array = np.asarray(matrix, dtype=np.float32)
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (array / norms).tolist()

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<{type(self).__name__} model={self._model!r} dims={self._dimensions}>"


def resolve_dimensions(provider: EmbeddingProvider, configured: int | None) -> int:
    """Reconcile configured dimensions with what the provider actually produces."""
    if configured and configured != provider.dimensions:
        logger.warning(
            "embedding_dimension_mismatch",
            extra={
                "model": provider.model,
                "configured": configured,
                "actual": provider.dimensions,
            },
        )
    return provider.dimensions