"""In-process vector store, one instance per embedding space.

Used by the test-suite and by ``VECTOR_BACKEND=memory`` for local experiments.
Uses numpy when available and falls back to pure Python otherwise.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from paper_app.db.spaces import EmbeddingSpace
from paper_app.db.vector_store.base import VectorFilter, VectorStore
from paper_app.domain.models import VectorRecord
from paper_app.logging import get_logger

logger = get_logger(__name__)

try:  # pragma: no cover - optional speedup
    import numpy as _np
except ImportError:  # pragma: no cover
    _np = None  # type: ignore[assignment]


class InMemoryVectorStore(VectorStore):
    name = "memory"

    def __init__(self, space: EmbeddingSpace) -> None:
        super().__init__(space)
        self._records: dict[str, tuple[list[float], dict]] = {}

    @property
    def dimensions(self) -> int:
        return self.space.dimensions

    async def ensure_ready(self) -> None:
        return None

    async def upsert(self, records: Sequence[VectorRecord]) -> int:
        for record in records:
            if len(record.vector) != self.space.dimensions:
                raise ValueError(
                    f"space {self.space.name!r} expects {self.space.dimensions}-dim vectors, "
                    f"got {len(record.vector)}"
                )
            self._records[record.chunk_id] = (list(record.vector), dict(record.payload))
        return len(records)

    async def search(
        self,
        vector: list[float],
        *,
        top_k: int = 10,
        filters: VectorFilter | None = None,
        min_score: float | None = None,
    ) -> list[tuple[str, float]]:
        self._check_dimensions(vector)
        filters = filters or VectorFilter()
        if filters.matches_nothing:
            return []
        candidates = [
            (chunk_id, payload, _cosine(vector, stored))
            for chunk_id, (stored, payload) in self._records.items()
            if _matches(payload, filters)
        ]
        if min_score is not None:
            candidates = [c for c in candidates if c[2] >= min_score]
        candidates.sort(key=lambda item: item[2], reverse=True)
        return [(chunk_id, score) for chunk_id, _, score in candidates[:top_k]]

    async def delete_for_papers(self, paper_ids: list[str]) -> int:
        targets = set(paper_ids)
        doomed = [
            chunk_id
            for chunk_id, (_, payload) in self._records.items()
            if payload.get("paper_id") in targets
        ]
        for chunk_id in doomed:
            del self._records[chunk_id]
        return len(doomed)

    async def count(self) -> int:
        return len(self._records)

    def vectors(self) -> dict[str, list[float]]:
        """Raw vectors by chunk id — used by dimension-sweep tooling."""
        return {chunk_id: list(vector) for chunk_id, (vector, _) in self._records.items()}


def _matches(payload: dict, filters: VectorFilter) -> bool:
    # `is not None`, never truthiness: an empty list is an explicit "match
    # nothing". See VectorFilter.paper_ids.
    if filters.paper_ids is not None and payload.get("paper_id") not in filters.paper_ids:
        return False
    if filters.categories is not None:
        available = set(payload.get("categories") or [])
        if not available.intersection(filters.categories):
            return False
    if filters.content_kinds is not None and payload.get("content_kind") not in filters.content_kinds:
        return False
    if filters.sources is not None and payload.get("source") not in filters.sources:
        return False
    if filters.sections is not None:
        pair = (payload.get("paper_id"), payload.get("section_ordinal"))
        if pair not in set(filters.sections):
            return False
    return all(payload.get(key) == expected for key, expected in filters.extra.items())


def _cosine(a: list[float], b: list[float]) -> float:
    if _np is not None:
        va = _np.asarray(a, dtype=_np.float32)
        vb = _np.asarray(b, dtype=_np.float32)
        denominator = float(_np.linalg.norm(va) * _np.linalg.norm(vb))
        return 0.0 if denominator == 0 else float(_np.dot(va, vb) / denominator)
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return 0.0 if norm_a * norm_b == 0 else dot / (norm_a * norm_b)