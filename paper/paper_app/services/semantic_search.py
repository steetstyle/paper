"""Semantic retrieval over the ingested corpus.

Embeds the query with the same provider that produced the stored vectors,
ranks candidates in the vector store, then hydrates the hits from the relational
DB so results carry full paper metadata.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from paper_app.db.repositories import SemanticSearchRepository
from paper_app.db.spaces import EmbeddingSpace
from paper_app.db.vector_store.base import VectorFilter, VectorStore
from paper_app.domain.enums import ContentSource
from paper_app.domain.models import SearchHitWithScore, SearchResultPage
from paper_app.embeddings.base import EmbeddingProvider
from paper_app.logging import get_logger
from paper_app.services.chunk_kinds import parse_kinds

logger = get_logger(__name__)

#: Where chunk text can have come from, as stored in ``chunks.source``.
_SOURCES = frozenset(source.value for source in ContentSource)


def parse_sources(values: Sequence[str] | str | None) -> list[str] | None:
    """Validate ``sources`` against :class:`~paper_app.domain.enums.ContentSource`.

    Same contract as :func:`paper_app.services.chunk_kinds.parse_kinds`: a typo is an
    error, never a filter that silently matches nothing.

    Raises:
        ValueError: on an unknown source, naming the accepted set.
    """
    if values is None:
        return None
    raw = [values] if isinstance(values, str) else list(values)
    wanted = [item.strip().lower() for item in raw if item and item.strip()]
    if not wanted:
        return None
    resolved: list[str] = []
    unknown: list[str] = []
    for word in wanted:
        if word in _SOURCES:
            if word not in resolved:
                resolved.append(word)
        else:
            unknown.append(word)
    if unknown:
        raise ValueError(
            f"unknown source(s): {', '.join(unknown)}; "
            f"choose from: {', '.join(sorted(_SOURCES))}"
        )
    return resolved


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

    def _scope(
        self,
        *,
        project: str | None,
        project_papers: int | None,
        content_kinds: list[str] | None,
        filters: VectorFilter | None = None,
        min_score: float | None = None,
    ) -> dict[str, Any]:
        """What the last search was actually scoped to.

        Built from the resolved filter rather than from the arguments, so it
        reports what actually reached the store: a caller must never be told
        "scoped to 3 papers" when the filter turned out to be unrestricted.
        """
        scope: dict[str, Any] = {
            "project": project,
            "project_papers": project_papers,
            "content_kinds": content_kinds or [],
        }
        if filters is not None:
            described = filters.describe()
            described["content_kinds"] = filters.content_kinds or []
            scope.update(described)
        if min_score is not None:
            scope["min_score"] = min_score
        return scope

    async def paper_ids_for_project(self, project: str) -> list[str]:
        """Paper ids held by a project. Raises on an unknown project name."""
        from paper_app.db.project_repository import ProjectRepository

        async with self._session_factory() as session:
            return await ProjectRepository(session).paper_ids_for(project)

    async def resolve_paper_ids(
        self, identifiers: Sequence[str], *, strict: bool = False
    ) -> list[str]:
        """Map user input onto stored paper ids.

        Accepts arXiv ids, versioned ids and abs URLs, because that is what every
        surface receives; an internal id passes through unchanged.

        Args:
            strict: raise on anything unresolved. Off by default so a project
                intersection can discard papers it does not hold without
                failing the whole query.
        """
        from paper_app.db.project_repository import ProjectRepository

        async with self._session_factory() as session:
            resolved, unresolved = await ProjectRepository(session).resolve_paper_ids(identifiers)
        if strict and unresolved:
            raise LookupError(f"no ingested paper matching: {', '.join(unresolved)}")
        return resolved

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
        sections: list[tuple[str, int]] | None = None,
    ) -> list[SemanticSearchHit]:
        """Search the corpus by meaning.

        Three independent scoping axes, and they compose:

        - ``project`` restricts to the papers a project holds, resolved to their
          ids here so that no vector store needs to know the project schema.
        - ``paper_ids`` restricts to specific papers. Accepts arXiv ids and URLs,
          not just internal ids — see :meth:`resolve_paper_ids`.
        - ``content_kinds`` restricts to what the chunks *are* — equations,
          figures, tables, references, abstracts, code or body.
        - ``sections`` restricts to named parts of documents — "only section 2.2
          of this book". Already resolved to ``(paper_id, ordinal)`` pairs by the
          caller, because turning "Debye" into an ordinal needs the section table
          and no vector store knows it exists.

        ``project`` and ``paper_ids`` together intersect, which is the useful
        reading: "the equations in the papers of this project".

        Raises:
            ValueError: a ``content_kinds`` or ``sources`` value is unknown.
            LookupError: the project does not exist, or no requested paper does.
        """
        if not query.strip():
            return []
        # Parse before touching the vector store so a typo is a clean 422/2
        # instead of a search that quietly matched nothing.
        if content_kinds is not None:
            content_kinds = parse_kinds(content_kinds)
        if sources is not None:
            sources = parse_sources(sources)

        if project:
            scoped = set(await self.paper_ids_for_project(project))
            if not scoped:
                # An empty project is not an error, but it must not silently
                # widen into a corpus-wide search.
                self.last_scope = self._scope(
                    project=project, project_papers=0, content_kinds=content_kinds
                )
                return []
            # Intersect with any caller-supplied papers *after* they are
            # resolved, so both spellings refer to the same internal ids. A
            # project narrows the paper set and never widens it.
            if paper_ids is None:
                paper_ids = sorted(scoped)
            else:
                asked = set(await self.resolve_paper_ids(paper_ids))
                paper_ids = sorted(scoped & asked)
            if not paper_ids:
                self.last_scope = self._scope(
                    project=project, project_papers=0, content_kinds=content_kinds
                )
                return []
        elif paper_ids is not None:
            # Strict: without a project to intersect against, an unresolvable id
            # can only be a mistake, and a search that silently returns nothing
            # is indistinguishable from one that found no matching text.
            paper_ids = await self.resolve_paper_ids(paper_ids, strict=True)
            if not paper_ids:
                # Only reachable for an explicitly empty list, which means
                # "match nothing" — the opposite of "no filter".
                self.last_scope = self._scope(
                    project=project, project_papers=0, content_kinds=content_kinds
                )
                return []

        filters = VectorFilter(
            paper_ids=paper_ids,
            categories=[category] if category else None,
            content_kinds=content_kinds,
            sources=sources,
            # Intersected with the paper scope rather than replacing it: naming a
            # section of one book must not silently drop the `--paper` the caller
            # also asked for, and must not widen past it either.
            sections=sections,
        )
        self.last_scope = self._scope(
            project=project,
            project_papers=(
                len(filters.paper_ids or []) if project and filters.paper_ids is not None else None
            ),
            content_kinds=content_kinds,
            filters=filters,
            min_score=min_score,
        )

        vector = await self._embeddings.embed_query(query)
        # Guard before the store: a wrong-width query vector is a caller bug and
        # the message is far clearer here than inside a similarity scan.
        self._check_query_width(vector)
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
        from paper_app.db.repositories import ChunkRepository, PaperRepository  # noqa: PLC0415

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
    from paper_app.domain.models import SearchHit

    return SearchResultPage(
        hits=tuple(SearchHit(metadata=h.metadata) for h in hits),  # type: ignore[arg-type]
        total_results=total,
        start=0,
        items_per_page=len(hits),
    )