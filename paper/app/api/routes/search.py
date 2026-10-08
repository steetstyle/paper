"""Semantic search over the ingested corpus.

Embeds the query with the *same* model that produced the stored vectors, ranks
candidates in that space's index, then hydrates the hits from the relational DB
so results carry full paper metadata.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import ContainerDep, SessionDep, resolve_space
from app.api.schemas import (
    SectionOut,
    SemanticSearchRequest,
    SemanticSearchResponse,
)
from app.api.schemas import (
    SemanticSearchHit as SemanticSearchHitSchema,
)
from app.db.repositories import PaperRepository
from app.services.chunk_kinds import parse_kinds
from app.services.sections_query import SectionMatch, find_sections
from app.services.semantic_search import SemanticSearchService

router = APIRouter(prefix="/search", tags=["search"])


@router.post("/semantic", response_model=SemanticSearchResponse, summary="Vector search")
async def semantic_search(
    payload: SemanticSearchRequest,
    container: ContainerDep,
    session: SessionDep,
) -> SemanticSearchResponse:
    """Query one embedding space.

    The space decides both the index and the query embedding. Querying a space
    with another model's embedding would compare incomparable vectors, so the
    provider is always taken from the space.

    ``device`` is this call's alone: it selects a provider from the (space,
    device) cache, so asking for cuda moves a second copy of the weights onto the
    GPU for this query without disturbing the ones every other query uses.

    ``section`` is resolved here rather than in the service, because turning a
    name a reader uses into a row of the section table is a database question and
    no vector store knows that table exists — see :func:`_resolve_sections`.
    """
    # Before the space and the provider: a name that matches nothing must not
    # cost a model load, and must be refused whatever else the request asked for.
    section_keys: list[tuple[str, int]] | None = None
    matched: list[SectionMatch] = []
    if payload.section is not None or payload.doc_key is not None:
        matched = await _resolve_sections(session, payload.section, payload.doc_key)
        section_keys = [item.key for item in matched]

    space = await resolve_space(container, session, payload.space)
    provider = container.provider_for(space, payload.device)
    service = SemanticSearchService(
        vector_store=container.vector_store_for(space),
        embeddings=provider,
        session_factory=container.session_factory,
        space=space,
    )

    try:
        # Parsed here rather than in the service so an unknown content kind is a
        # 422 naming the accepted set, instead of a search that quietly matched
        # nothing.
        kinds = parse_kinds(payload.content_kinds)
        hits = await service.search(
            payload.query,
            top_k=payload.top_k,
            category=payload.category,
            paper_ids=payload.paper_ids,
            min_score=payload.min_score,
            content_kinds=kinds,
            project=payload.project,
            sources=payload.sources,
            sections=section_keys,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return SemanticSearchResponse(
        query=payload.query,
        provider=provider.name,
        model=provider.model,
        space=space.name,
        dimensions=space.dimensions,
        scope=service.last_scope,
        sections=[SectionOut(**item.as_dict()) for item in matched],
        hits=[
            SemanticSearchHitSchema(
                chunk_id=hit.chunk_id,
                paper_id=hit.paper_id,
                score=round(hit.score, 6),
                text=hit.text,
                metadata=hit.metadata,
            )
            for hit in hits
        ],
    )


async def _resolve_sections(
    session: AsyncSession,
    section: str | None,
    doc_key: str | None,
) -> list[SectionMatch]:
    """The sections a caller means by ``section``, narrowed to ``doc_key``.

    Every way this could come back empty is an error rather than an empty answer,
    because the two failures look identical from outside: a section that does not
    discuss the question, and a name that names nothing. A search answering zero
    hits to both would be a confident response to a question nobody asked. So an
    empty ``section`` is a 422 (omitting the field is how a caller asks for no
    section filter), a blank ``doc_key`` is a 422 rather than the whole corpus,
    a ``doc_key`` naming no document is a 404, and a name that resolves to
    nothing is a 422 that says what was searched for.

    ``doc_key`` off means the whole corpus, which is right for a rare name and
    wrong for "Introduction" — hence the field rather than a required one.
    """
    if section is None:
        # Only reachable when ``doc_key`` came alone: there is no section name to
        # resolve, and guessing one would search something the caller did not ask
        # for.
        raise HTTPException(
            status_code=422,
            detail=(
                "doc_key only says which document a section name belongs to, so "
                "it needs section as well; to search one document outright, use "
                "paper_ids"
            ),
        )
    if not section.strip():
        raise HTTPException(
            status_code=422,
            detail=(
                "section is empty; omit it to search every section, or name one "
                "the way its book does — its number (2.2) or a word from its "
                "title (Debye)"
            ),
        )

    if doc_key is None:
        where = "in the corpus"
        hint = "use read_sections(<doc_key>) to list one document's sections"
    elif not doc_key.strip():
        # Blank is not "every document": a caller who sent a doc_key has said
        # which book they meant, and quietly reading the whole library is how a
        # wrong section name becomes a confident wrong answer.
        raise HTTPException(
            status_code=422,
            detail="doc_key is empty; omit it to read every document, or name one by its doc_key",
        )
    else:
        # Any handle a paper row answers to: doc_key, arXiv id or internal id.
        # 404 rather than 422, because a corpus that also holds books cannot tell
        # a malformed id from a document that simply is not here.
        paper = await PaperRepository(session).resolve(doc_key)
        if paper is None:
            raise HTTPException(status_code=404, detail=f"no document matching {doc_key!r}")
        doc_key = paper.doc_key
        where = f"in {paper.doc_key!r}"
        hint = f"use read_sections({paper.doc_key!r}) to list them"

    found = await find_sections(
        session, section, doc_keys=[doc_key] if doc_key is not None else None
    )
    if not found:
        raise HTTPException(
            status_code=422,
            detail=(
                f"no section matching {section!r} {where}; try a word from its "
                f"title or its number (2.2), or {hint}"
            ),
        )
    return found
