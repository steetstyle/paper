"""Semantic search over the ingested corpus.

Embeds the query with the *same* model that produced the stored vectors, ranks
candidates in that space's index, then hydrates the hits from the relational DB
so results carry full paper metadata.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.api.deps import ContainerDep, SessionDep, resolve_space
from app.api.schemas import (
    SemanticSearchHit as SemanticSearchHitSchema,
)
from app.api.schemas import (
    SemanticSearchRequest,
    SemanticSearchResponse,
)
from app.services.chunk_kinds import parse_kinds
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
    """
    space = await resolve_space(container, session, payload.space)
    provider = container.provider_for(space)
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