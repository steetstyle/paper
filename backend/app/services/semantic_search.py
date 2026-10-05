"""Semantic retrieval over the ingested corpus.

Embeds the query with the same provider that produced the stored vectors,
ranks candidates in the vector store, then hydrates the hits from the relational
DB so results carry full paper metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.db.repositories import SemanticSearchRepository
from app.db.spaces import EmbeddingSpace
from app.db.vector_store.base import VectorFilter, VectorStore
from app.domain.models import SearchHitWithScore, SearchResultPage
from app.embeddings.base import EmbeddingProvider
from app.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class SemanticSearchHit:
    chunk_id: str
    paper_id: str
    score: float
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "paper_id": self.paper_id,
            "score": round(self.score, 6),
            "text": self.text,
            "metadata": self.metadata,
        }


class SemanticSearchService:
    def __init__(
        self,
        *,
        vector_store: VectorStore,
        embeddings: EmbeddingProvider,
        session_factory: Any,
        space: EmbeddingSpace | None = None,
    ) -> None:
        self._store = vector_store
        self._embeddings = embeddings
        self._session_factory = session_factory
        # Recorded so callers can see which model answered the query.
        self.space = space or vector_store.space
        # What the last search was scoped to, for echoing back in a response.
        self.last_scope: dict[str, Any] = {}

    async def paper_ids_for_project(self, project: str) -> list[str]:
        """Paper ids held by a project. Raises on an unknown project name."""
        from app.db.project_repository import PaperNotFoundError, ProjectRepository

        async with self._session_factory() as session:
            repo = ProjectRepository(session)
            found = await repo.get(project)
            if found is None:
                raise PaperNotFoundError(f"no project matching {project!r}")
            return await repo.paper_ids(found)

    async def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        category: str | None = None,
        paper_ids: list[str] | None = None,
        min_score: float | None = None,
        sources: list[str] | None = None,
        content_kinds: list[str] | None = None,
        project: str | None = None,
    ) -> list[SemanticSearchHit]:
        """Search the corpus by meaning.

        Three independent scoping axes, and they compose:

        - ``project`` restricts to the papers a project holds, resolved to their
          ids here so that no vector store needs to know the project schema.
        - ``paper_ids`` restricts to specific papers.
        - ``content_kinds`` restricts to what the chunks *are* — equations,
          figures, tables, references, abstracts, code or body.

        ``project`` and ``paper_ids`` together intersect, which is the useful
        reading: "the equations in the papers of this project".
        """
        if not query.strip():
            return []
        if project:
            scoped = await self.paper_ids_for_project(project)
            if not scoped:
                # An empty project is not an error, but it must not silently
                # widen into a corpus-wide search.
                self.last_scope = {"project": project, "project_papers": 0}
                return []
            paper_ids = sorted(set(paper_ids or []) & set(scoped)) if paper_ids else scoped
        self.last_scope = {
            "project": project,
            "project_papers": len(paper_ids or []) if project else None,
            "content_kinds": content_kinds or [],
        }
        vector = await self._embeddings.embed_query(query)
        # Guard before the store: a wrong-width query vector is a caller bug and
        # the message is far clearer here than inside a similarity scan.
        self._check_query_width(vector)
        filters = VectorFilter(
            paper_ids=paper_ids,
            categories=[category] if category else None,
            content_kinds=content_kinds,
            sources=sources,
        )
        ranked = await self._store.search(
            vector, top_k=top_k, filters=filters, min_score=min_score
        )
        if not ranked:
            return []

        hits = [
            SemanticSearchHit(chunk_id=chunk_id, paper_id="", score=score, text="")
            for chunk_id, score in ranked
        ]
        async with self._session_factory() as session:
            rows = await SemanticSearchRepository(session).hydrate(
                [
                    SearchHitWithScore(
                        chunk_id=hit.chunk_id,
                        paper_id=hit.paper_id,
                        score=hit.score,
                        text=hit.text,
                    )
                    for hit in hits
                ]
            )
        hydrated = {row["chunk_id"]: row for row in rows}
        out: list[SemanticSearchHit] = []
        for hit in hits:
            row = hydrated.get(hit.chunk_id)
            if row is None:
                continue
            out.append(
                SemanticSearchHit(
                    chunk_id=row["chunk_id"],
                    paper_id=row["paper_id"],
                    score=row["score"],
                    text=row["text"],
                    metadata=row["metadata"],
                )
            )
        logger.info(
            "semantic_search",
            extra={
                "query": query[:80],
                "hits": len(out),
                "top_k": top_k,
                "store": self._store.name,
                "space": self.space.name,
                "dimensions": self.space.dimensions,
            },
        )
        return out

    def _check_query_width(self, vector: list[float]) -> None:
        expected = self.space.dimensions
        if len(vector) != expected:
            raise ValueError(
                f"space {self.space.name!r} expects {expected}-dim query vectors, "
                f"got {len(vector)}. Embed the query with that space's own provider."
            )

    async def similar_to_paper(self, arxiv_id: str, *, top_k: int = 5) -> list[SemanticSearchHit]:
        """'More like this' — uses the paper's first chunk as the query vector."""
        from app.db.repositories import ChunkRepository, PaperRepository  # noqa: PLC0415

        async with self._session_factory() as session:
            paper = await PaperRepository(session).get_by_arxiv_id(arxiv_id)
            if paper is None:
                return []
            chunks = await ChunkRepository(session).list_for_paper(paper.id, limit=1)
            if not chunks:
                return []
            seed = chunks[0]
        return await self.search(
            seed.text[:2000],
            top_k=top_k + 1,
            paper_ids=[seed.paper_id],
        )


def build_search_page(hits: list[SemanticSearchHit], total: int) -> SearchResultPage:  # pragma: no cover
    from app.domain.models import SearchHit

    return SearchResultPage(
        hits=tuple(SearchHit(metadata=h.metadata) for h in hits),  # type: ignore[arg-type]
        total_results=total,
        start=0,
        items_per_page=len(hits),
    )