"""Deterministic hashing embeddings.

Not semantically meaningful, but stable, dependency-free and dimension-exact —
used for local development, tests and CI so the full pipeline is runnable
without an API key or a GPU.
"""

from __future__ import annotations

import hashlib
import math
import struct
from typing import ClassVar

from paper_app.embeddings.base import EmbeddingProvider


class HashingEmbeddingProvider(EmbeddingProvider):
    name: ClassVar[str] = "hashing"

    def __init__(
        self,
        model: str = "hashing-1024",
        dimensions: int = 1536,
        *,
        batch_size: int = 64,
        normalize: bool = True,
    ) -> None:
        super().__init__(model=model, dimensions=dimensions, batch_size=batch_size, normalize=normalize)

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self._dimensions
        tokens = text.lower().split()
        if not tokens:
            return vector
        for position, token in enumerate(tokens):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = struct.unpack("<Q", digest)[0] % self._dimensions
            sign = 1.0 if digest[0] & 1 else -1.0
            # Slight positional decay keeps phrase order faintly represented.
            vector[index] += sign * (1.0 / (1.0 + position * 0.01))
        # Mix a whole-document hash in so empty-token edge cases stay non-zero.
        doc_digest = hashlib.blake2b(text.lower().encode("utf-8"), digest_size=8).digest()
        vector[int.from_bytes(doc_digest[:4], "little") % self._dimensions] += 0.5
        if self._normalize:
            norm = math.sqrt(sum(value * value for value in vector))
            if norm > 0:
                vector = [value / norm for value in vector]
        return vector

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]