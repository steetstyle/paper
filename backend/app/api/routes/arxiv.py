"""ArXiv search endpoints.

The filter surface mirrors ArXiv's own query language one-for-one, so a query
that works against `export.arxiv.org` works here unchanged.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Query

from app.api.deps import ContainerDep, SessionDep
from app.api.schemas import AuthorOut, PaperOut, SearchResponse
from app.clients.arxiv.filters import FIELD_ALIASES, search_query_from
from app.domain.filters import FilterProblem
from app.domain.models import PaperMetadata
from app.services.arxiv_search import ArxivSearchService

router = APIRouter(prefix="/arxiv", tags=["arxiv"])


def to_schema(metadata: PaperMetadata) -> PaperOut:
    return PaperOut(
        # `display_id` / the doc_key fallback: this route renders anything that
        # can be ingested, and a local file has no arXiv id to print.
        arxiv_id=metadata.display_id,
        versioned_id=metadata.versioned_id or metadata.display_id,
        title=metadata.title,
        abstract=metadata.abstract,
        authors=[
            AuthorOut(name=a.name, affiliation=a.affiliation) for a in metadata.authors
        ],
        categories=list(metadata.categories),
        primary_category=metadata.primary_category,
        published_at=metadata.published_at,
        updated_at=metadata.updated_at,
        doi=metadata.doi,
        comment=metadata.comment,
        journal_ref=metadata.journal_ref,
        abs_url=metadata.abs_url,
        pdf_url=metadata.pdf_url,
        html_url=metadata.html_url,
    )


@router.get("/search", response_model=SearchResponse, summary="Search ArXiv")
async def search_arxiv(
    container: ContainerDep,
    session: SessionDep,
    # ArXiv's own query language, passed through verbatim.
    filter: Annotated[
        str | None,
        Query(
            description=(
                "Raw ArXiv `search_query`, e.g. 'cat:cs.CL AND ti:\"attention\"'. "
                "Supports ti/au/abs/co/jr/cat/rn/all, AND/OR/ANDNOT, parentheses "
                "and quoted phrases. Lowercase operators are reported, not applied."
            )
        ),
    ] = None,
    q: Annotated[
        str | None, Query(description="Alias for `filter`, for convenience.")
    ] = None,
    # Structured equivalents of the field prefixes.
    title: Annotated[list[str] | None, Query()] = None,
    author: Annotated[list[str] | None, Query()] = None,
    abstract: Annotated[list[str] | None, Query()] = None,
    comment: Annotated[list[str] | None, Query()] = None,
    journal: Annotated[list[str] | None, Query(description="Journal reference (jr).")] = None,
    category: Annotated[list[str] | None, Query(description="Subject category (cat).")] = None,
    report_number: Annotated[list[str] | None, Query(description="Report number (rn).")] = None,
    all_fields: Annotated[
        list[str] | None, Query(alias="all", description="Search every field.")
    ] = None,
    operator: Annotated[
        str,
        Query(description="How to combine values within one field: AND, OR or ANDNOT."),
    ] = "AND",
    arxiv_id: Annotated[
        list[str] | None,
        Query(description="Look up specific papers. Preferred over `id:`."),
    ] = None,
    submitted_from: Annotated[
        str | None,
        Query(description="Earliest submission (YYYY-MM-DD, YYYYMMDD or ISO-8601)."),
    ] = None,
    submitted_to: Annotated[str | None, Query(description="Latest submission.")] = None,
    # ArXiv request parameters.
    max_results: Annotated[int, Query(ge=1, le=100)] = 10,
    start: Annotated[int, Query(ge=0)] = 0,
    sort_by: Annotated[
        str, Query(description="relevance | lastUpdatedDate | submittedDate")
    ] = "relevance",
    sort_order: Annotated[str, Query(description="ascending | descending")] = "descending",
    # Client-side filters ArXiv cannot express.
    has_pdf: Annotated[bool | None, Query()] = None,
    has_html: Annotated[bool | None, Query()] = None,
    has_doi: Annotated[bool | None, Query()] = None,
    has_journal_ref: Annotated[bool | None, Query()] = None,
    ingested: Annotated[
        bool | None, Query(description="Only papers already in the local corpus.")
    ] = None,
    also_categories: Annotated[
        list[str] | None,
        Query(description="Keep entries carrying all of these categories."),
    ] = None,
    exclude_categories: Annotated[list[str] | None, Query()] = None,
) -> SearchResponse:
    """Search ArXiv using ArXiv's own filter language.

    Field prefixes: ``ti`` title, ``au`` author, ``abs`` abstract, ``co`` comment,
    ``jr`` journal reference, ``cat`` subject category, ``rn`` report number,
    ``all`` everything. Boolean operators are uppercase: ``AND``, ``OR``,
    ``ANDNOT``. Phrases go in double quotes; parentheses group.

    Structured parameters are equivalent to writing the prefix yourself, so
    ``?title=transformer`` compiles to the same query as
    ``?filter=ti:transformer``.
    """
    try:
        request = search_query_from(
            raw=filter or q,
            fields={
                "title": title,
                "author": author,
                "abstract": abstract,
                "comment": comment,
                "journal": journal,
                "category": category,
                "report_number": report_number,
                "all": all_fields,
            },
            operator=operator,
            id_list=arxiv_id,
            max_results=max_results,
            start=start,
            sort_by=sort_by,
            sort_order=sort_order,
            submitted_from=submitted_from,
            submitted_to=submitted_to,
            post={
                "has_pdf": has_pdf,
                "has_html": has_html,
                "has_doi": has_doi,
                "has_journal_ref": has_journal_ref,
                "ingested": ingested,
                "also_categories": also_categories,
                "exclude_categories": exclude_categories,
            },
        )
    except FilterProblem as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if request.filter.is_empty and not request.id_list:
        raise HTTPException(
            status_code=422,
            detail=(
                "nothing to search for: provide `filter`/`q`, one of the field "
                "parameters, or `arxiv_id`"
            ),
        )

    service = ArxivSearchService(container.arxiv, session_factory=container.session_factory)
    outcome = await service.search(request)
    if not outcome.ok:
        raise HTTPException(status_code=502, detail=outcome.error)

    response = SearchResponse(
        total_results=outcome.total_results,
        start=request.start,
        items_per_page=outcome.returned,
        papers=[to_schema(paper) for paper in outcome.papers],
        filtered_out=outcome.filtered_out,
        next_start=outcome.next_start,
        warnings=list(outcome.warnings),
        hint=outcome.hint,
        search_query=request.filter.compile() or None,
        post_filter=request.post_filter.to_dict(),
    )
    return response


@router.get("/paper/{arxiv_id}", response_model=PaperOut, summary="Fetch one paper")
async def get_paper(container: ContainerDep, arxiv_id: str) -> PaperOut:
    from app.clients.arxiv.exceptions import ArxivError  # noqa: PLC0415

    try:
        metadata = await container.arxiv.get_paper(arxiv_id)
    except ArxivError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return to_schema(metadata)


@router.get("/filter-help", summary="Supported filter fields")
async def filter_help() -> dict[str, Any]:
    """The field prefixes and operators this endpoint understands."""
    return {
        "fields": FIELD_ALIASES,
        "operators": ["AND", "OR", "ANDNOT"],
        "notes": (
            "Operators must be uppercase. Phrases go in double quotes. Parentheses "
            "group. Dates use submittedDate:[YYYYMMDDTTTT TO YYYYMMDDTTTT] and are "
            "evaluated by ArXiv, not locally."
        ),
    }