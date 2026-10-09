"""Paper library endpoints: stored papers, chunks and extracted text."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Response
from sqlalchemy import func, select

from paper_app.api.deps import ContainerDep, SessionDep, normalize_arxiv_id_or_400
from paper_app.api.schemas import (
    ChunkOut,
    ContentOut,
    PaperDetail,
    PaperOut,
    SearchResponse,
)
from paper_app.db.models import Chunk, RawDocument
from paper_app.db.repositories import ChunkRepository, PaperRepository
from paper_app.services.chunk_kinds import parse_kinds

router = APIRouter(prefix="/papers", tags=["papers"])


@router.get("", response_model=SearchResponse, summary="List stored papers")
async def list_papers(
    session: SessionDep,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    category: str | None = Query(default=None),
    ingested_only: bool = Query(default=False),
) -> SearchResponse:
    repo = PaperRepository(session)
    papers, total = await repo.list_papers(
        limit=limit, offset=offset, category=category, ingested_only=ingested_only
    )
    return SearchResponse(
        total_results=total,
        start=offset,
        items_per_page=len(papers),
        papers=[await _to_schema(session, paper) for paper in papers],
    )


@router.get("/categories", summary="Categories and their paper counts")
async def list_categories(session: SessionDep) -> dict[str, Any]:
    """Every registered category, with the number of papers filed under it."""
    return {"categories": await PaperRepository(session).list_categories()}


@router.get("/authors", summary="Authors and their paper counts")
async def list_authors(
    session: SessionDep,
    limit: int = Query(default=50, ge=1, le=200),
) -> dict[str, Any]:
    """Prolific authors. One row per person, shared across all their papers."""
    return {"authors": await PaperRepository(session).list_authors(limit=limit)}


@router.get("/authors/{name}", summary="Every paper by one author")
async def papers_by_author(
    session: SessionDep,
    name: str,
    include_subcategories: bool = Query(
        default=False, summary="Reserved for category lookups; ignored here."
    ),
) -> dict[str, Any]:
    """Look an author up by name and list every paper they wrote.

    Matching follows the same normalization ingestion uses, so a surname, a
    full name, or any consistent capitalization all reach the same author.
    """
    repo = PaperRepository(session)
    match = await repo.find_author(name)
    if match is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"no papers found for author {name!r}. "
                "Try a full name, or GET /api/v1/papers/authors to list known authors."
            ),
        )
    papers = await repo.papers_by_author(name)
    return {
        "author": match["name"],
        "author_id": match["id"],
        "paper_count": match["paper_count"],
        "count": len(papers),
        "papers": [await _to_schema(session, paper) for paper in papers],
    }


@router.get("/{arxiv_id}", response_model=PaperDetail, summary="Paper detail")
async def get_paper(session: SessionDep, arxiv_id: str) -> PaperDetail:
    repo = PaperRepository(session)
    paper = await repo.resolve(normalize_arxiv_id_or_400(arxiv_id))
    if paper is None:
        raise HTTPException(status_code=404, detail=f"paper {arxiv_id} not ingested")

    schema = await _to_schema(session, paper)
    contents = (
        await session.execute(
            select(RawDocument)
            .where(RawDocument.paper_id == paper.id)
            .order_by(RawDocument.created_at)
        )
    ).scalars().all()
    return PaperDetail(
        **schema.model_dump(),
        contents=[
            ContentOut(
                kind=doc.kind,
                uri=doc.uri,
                content_type=doc.content_type,
                size_bytes=doc.size_bytes,
                sha256=doc.sha256,
                source_url=doc.source_url,
            )
            for doc in contents
        ],
    )


@router.get("/{arxiv_id}/chunks", summary="Chunk listing")
async def get_chunks(
    session: SessionDep,
    arxiv_id: str,
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    content_kinds: list[str] | None = Query(
        default=None,
        alias="content",
        description="Only these chunk kinds: body, abstract, figure, table, "
        "equation, reference, code. Repeatable; plurals accepted.",
    ),
) -> dict:
    try:
        # Validated here so a typo is a 422 naming the accepted set, rather
        # than a listing that silently omits everything.
        kinds = parse_kinds(content_kinds)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    paper = await PaperRepository(session).resolve(normalize_arxiv_id_or_400(arxiv_id))
    if paper is None:
        raise HTTPException(status_code=404, detail=f"paper {arxiv_id} not ingested")

    chunks_repo = ChunkRepository(session)
    # `total` counts what this request can page through, not the whole paper:
    # reporting the unfiltered total next to a filtered page is how a client
    # ends up paging forever.
    total = await chunks_repo.count(paper.id, content_kinds=kinds)
    chunks = await chunks_repo.list_for_paper(
        paper.id, limit=limit, offset=offset, content_kinds=kinds
    )
    return {
        "arxiv_id": paper.arxiv_id,
        "total": total,
        "paper_total": await chunks_repo.count(paper.id),
        "content_kinds": kinds or [],
        "chunks": [ChunkOut.model_validate(chunk) for chunk in chunks],
    }


@router.get("/{arxiv_id}/markdown", summary="Extracted markdown")
async def get_markdown(
    container: ContainerDep, session: SessionDep, arxiv_id: str
) -> Response:
    """Return the MinerU/HTML-derived markdown used for embedding."""
    paper = await PaperRepository(session).resolve(normalize_arxiv_id_or_400(arxiv_id))
    if paper is None:
        raise HTTPException(status_code=404, detail=f"paper {arxiv_id} not ingested")

    document = (
        await session.execute(
            select(RawDocument)
            .where(
                RawDocument.paper_id == paper.id,
                RawDocument.kind.in_(["mineru_markdown", "text"]),
            )
            .order_by(RawDocument.created_at.desc())
            .limit(1)
        )
    ).scalar_one_or_none()
    if document is None:
        raise HTTPException(status_code=404, detail="no extracted markdown for this paper")

    path = _path_from_uri(document.uri)
    if path is None or not path.exists():
        raise HTTPException(status_code=410, detail="extracted markdown is no longer on disk")
    return Response(content=path.read_text(errors="replace"), media_type="text/markdown")


async def _to_schema(session: SessionDep, paper) -> PaperOut:  # noqa: ANN001
    chunk_count = int(
        await session.scalar(
            select(func.count()).select_from(Chunk).where(Chunk.paper_id == paper.id)
        )
        or 0
    )
    return PaperOut(
        arxiv_id=paper.arxiv_id,
        versioned_id=paper.versioned_id,
        title=paper.title,
        abstract=paper.abstract,
        authors=paper.authors(),
        categories=paper.categories or [],
        primary_category=paper.primary_category,
        published_at=paper.published_at,
        updated_at=paper.updated_at_arxiv,
        doi=paper.doi,
        comment=paper.comment,
        journal_ref=paper.journal_ref,
        abs_url=paper.abs_url,
        pdf_url=paper.pdf_url,
        html_url=paper.html_url,
        chunk_count=chunk_count,
        ingested_at=paper.ingested_at,
    )


def _path_from_uri(uri: str) -> Path | None:
    if uri.startswith("file://"):
        return Path(uri[len("file://") :])
    path = Path(uri)
    return path if path.exists() else None