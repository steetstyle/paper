"""Vector store contract.

The pipeline writes vectors through this interface only, so swapping pgvector
for Qdrant (or an in-memory store in tests) requires no pipeline changes.

Each store is bound to exactly one :class:`~paper_app.db.spaces.EmbeddingSpace`:
vectors from different models are never mixed in one index, because that would
be arithmetically meaningless and because pgvector's HNSW index is width-locked.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from paper_app.db.spaces import EmbeddingSpace
from paper_app.domain.models import VectorRecord


@dataclass(slots=True)
class VectorFilter:
    """Structured metadata filter applied before ranking.

    All three scoping axes are independent and compose: a query can be limited to
    one project *and* one category *and* only equations. Projects are resolved to
    their paper ids by the caller, so no store needs to know the project schema —
    the same trick keeps pgvector, Qdrant and the in-memory store agreeing.
    """

    paper_ids: list[str] | None = None
    """Already resolved. A project scope arrives here as its paper ids.

    ``None`` means "do not filter"; an **empty list means "match nothing"**.
    The distinction is load-bearing: intersecting a project with a caller-supplied
    list can legitimately produce ``[]``, and treating that as "no filter" would
    silently widen the query to the whole corpus — the opposite of what the
    caller asked for. Every store therefore tests ``is not None``, never
    truthiness.
    """

    categories: list[str] | None = None
    content_kinds: list[str] | None = None
    """Chunk kinds: body, abstract, figure, table, equation, reference, code."""

    sources: list[str] | None = None

    sections: list[tuple[str, int]] | None = None
    """``(paper_id, section_ordinal)`` pairs — the sections of a document to search.

    Ordinals are per-document, so a bare list of ints would mean different things
    in two books and could silently intersect them. Pairs are unambiguous, and the
    resolution of a *name* to an ordinal happens before this point, in SQL, where
    the section table lives — a vector store has no business knowing what a
    heading is.
    """

    extra: dict[str, Any] = field(default_factory=dict)

    def is_empty(self) -> bool:
        """True when nothing is being filtered on.

        Distinct from :attr:`matches_nothing`: ``VectorFilter(paper_ids=[])`` is
        a filter that matches nothing, so it is *not* empty. Kept separate so the
        two cannot be confused — asking for no results is not the same as asking
        for no filter.
        """
        return not self.matches_nothing and not self.extra

    @property
    def matches_nothing(self) -> bool:
        """True when a filter is present but its value set is empty.

        Checked *before* building a query on purpose. Relying on the backend to
        treat an empty ``IN ()`` as false works on PostgreSQL and not
        everywhere — Qdrant rejects an empty ``MatchAny`` outright — so each
        store short-circuits here instead of guessing.
        """
        return any(
            value is not None and not value
            for value in (
                self.paper_ids,
                self.categories,
                self.content_kinds,
                self.sources,
                self.sections,
            )
        )

    def describe(self) -> dict[str, Any]:
        """What was actually applied, for echoing back to the caller."""
        return {
            "paper_ids": len(self.paper_ids or []),
            # Counted, not listed: a book has hundreds of sections and the point
            # of this echo is "how narrow was it", not "which ones".
            "sections": len(self.sections or []),
            "categories": self.categories or [],
            "content_kinds": self.content_kinds or [],
            "sources": self.sources or [],
        }


class VectorStore(ABC):
    """Nearest-neighbour storage for one embedding space."""

    name: str = "abstract"

    #: True when the store *is* the relational record (pgvector). The pipeline
    #: uses this to skip the second write that external backends need.
    persists_relationally: bool = False

    def __init__(self, space: EmbeddingSpace) -> None:
        self.space = space

    @abstractmethod
    async def ensure_ready(self) -> None:
        """Create collections/indexes if missing. Safe to call repeatedly."""

    @abstractmethod
    async def upsert(self, records: Sequence[VectorRecord]) -> int:
        """Insert or replace vectors. Returns the number written."""

    @abstractmethod
    async def search(
        self,
        vector: list[float],
        *,
        top_k: int = 10,
        filters: VectorFilter | None = None,
        min_score: float | None = None,
    ) -> list[tuple[str, float]]:
        """Return ``(chunk_id, similarity)`` pairs, best first.

        Raises ``ValueError`` if ``vector`` has the wrong width: querying an
        index with a mismatched embedding is always a caller bug, never a
        degraded result.
        """

    @abstractmethod
    async def delete_for_papers(self, paper_ids: list[str]) -> int: ...

    @abstractmethod
    async def count(self) -> int: ...

    async def health(self) -> bool:  # pragma: no cover - overridden where useful
        try:
            await self.count()
            return True
        except Exception:  # noqa: BLE001
            return False

    async def aclose(self) -> None:
        return None

    # ------------------------------------------------------------------ helpers
    def _check_dimensions(self, vector: Sequence[float]) -> None:
        if len(vector) != self.space.dimensions:
            raise ValueError(
                f"space {self.space.name!r} expects {self.space.dimensions}-dim vectors, "
                f"got {len(vector)}. Embed the query with the space's own provider/model."
            )

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return (
            f"<{type(self).__name__} space={self.space.name!r} "
            f"dim={self.space.dimensions} backend={self.name}>"
        )