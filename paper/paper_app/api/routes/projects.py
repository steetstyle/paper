"""Projects and the citation graph.

Papers are global: one row in ``papers``, referenced by any number of projects
through ``project_papers``. Nothing here ever copies a paper, so two project
views of the same paper cannot drift apart.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query

from paper_app.api.deps import DeviceDep, SessionDep, normalize_arxiv_id_or_400
from paper_app.api.schemas import (
    AssetCounts,
    AssetsResponse,
    EquationOut,
    FigureOut,
    ImportResult,
    ProjectCreate,
    ProjectOut,
    ProjectPaperOut,
    ReferenceOut,
    ReferencesResponse,
    TableAssetOut,
)
from paper_app.db.asset_repository import AssetRepository
from paper_app.db.project_repository import (
    PaperNotFoundError,
    ProjectConflictError,
    ProjectRepository,
)
from paper_app.db.reference_repository import ReferenceRepository
from paper_app.db.repositories import PaperRepository

router = APIRouter(tags=["projects"])


async def _project_or_404(session: SessionDep, slug: str):  # noqa: ANN202
    projects = ProjectRepository(session)
    try:
        return projects, await projects.require(slug)
    except PaperNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/projects", response_model=list[ProjectOut], summary="List projects")
async def list_projects(
    session: SessionDep,
    include_archived: bool = Query(default=False),
) -> list[ProjectOut]:
    rows = await ProjectRepository(session).list_projects(include_archived=include_archived)
    return [
        ProjectOut(
            id=row.id,
            slug=row.slug,
            name=row.name,
            description=row.description,
            is_archived=row.is_archived,
            paper_count=row.paper_count,
            read_count=row.read_count,
        )
        for row in rows
    ]


@router.post("/projects", response_model=ProjectOut, status_code=201, summary="Create a project")
async def create_project(payload: ProjectCreate, session: SessionDep) -> ProjectOut:
    projects = ProjectRepository(session)
    try:
        project = await projects.create(payload.name, description=payload.description)
    except ProjectConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    await session.commit()
    return ProjectOut(id=project.id, slug=project.slug, name=project.name, description=project.description)


@router.get(
    "/projects/{slug}", response_model=ProjectOut, summary="Project detail"
)
async def get_project(slug: str, session: SessionDep) -> ProjectOut:
    _projects, project = await _project_or_404(session, slug)
    listed = next(
        (
            row
            for row in await _projects.list_projects(include_archived=True)
            if row.id == project.id
        ),
        None,
    )
    return ProjectOut(
        id=project.id,
        slug=project.slug,
        name=project.name,
        description=project.description,
        is_archived=project.is_archived,
        paper_count=listed.paper_count if listed else 0,
        read_count=listed.read_count if listed else 0,
    )


@router.delete("/projects/{slug}", summary="Delete a project (its papers survive)")
async def delete_project(slug: str, session: SessionDep) -> dict[str, Any]:
    _projects, project = await _project_or_404(session, slug)
    name = project.name
    await _projects.delete(project)
    await session.commit()
    return {"deleted": name, "papers_kept": True}


@router.get(
    "/projects/{slug}/papers",
    response_model=list[ProjectPaperOut],
    summary="Papers in a project",
)
async def list_project_papers(
    slug: str,
    session: SessionDep,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    unread_only: bool = Query(default=False),
    category: str | None = Query(default=None),
) -> list[ProjectPaperOut]:
    projects, project = await _project_or_404(session, slug)
    rows = await projects.list_papers(
        project, limit=limit, offset=offset, unread_only=unread_only, category=category
    )
    return [
        ProjectPaperOut(
            arxiv_id=paper.arxiv_id,
            versioned_id=paper.versioned_id,
            title=paper.title,
            categories=list(paper.categories),
            primary_category=paper.primary_category,
            authors=list(paper.author_names),
            is_read=link.is_read,
            note=link.note,
            projects=await projects.projects_for_paper(paper.id),
        )
        for paper, link in rows
    ]


@router.post(
    "/projects/{slug}/papers",
    response_model=ImportResult,
    summary="Import papers into a project",
)
async def import_papers(
    slug: str,
    arxiv_ids: list[str],
    session: SessionDep,
    note: str | None = Query(default=None),
    ingest: bool = Query(
        default=False, description="Ingest anything not already in the corpus."
    ),
    device: DeviceDep = None,
) -> ImportResult:
    """Link papers into a project. Idempotent: re-importing is a no-op.

    With ``ingest=true`` anything missing is fetched and ingested first. That is a
    write against the network and minutes of CPU per paper, so it is opt-in.

    ``?device=`` applies to that ingest only — the one case here that embeds text.
    """
    projects, project = await _project_or_404(session, slug)
    # Read out of the ORM object before anything can expire it. The ingest branch
    # below rolls the session back, and a rollback expires every loaded instance;
    # touching `project.slug` afterwards is then an implicit refresh — IO on a
    # synchronised attribute access, outside async context, which raises
    # MissingGreenlet. It failed on every single call of this route, not
    # intermittently, which is why it went unnoticed behind a green test suite.
    project_slug = project.slug
    resolved, missing = await projects.resolve_paper_ids(arxiv_ids)

    if missing and ingest:
        from paper_app.container import get_container  # noqa: PLC0415
        from paper_app.services.ingestion import build_ingestion_service  # noqa: PLC0415

        container = get_container()
        service = build_ingestion_service(container, container.active_space, device=device)
        for arxiv_id in missing:
            await service.ingest_paper(
                normalize_arxiv_id_or_400(arxiv_id),
                prefer_html=True,
                requested_by="api",
                trigger="api",
            )
        await session.rollback()
        # The rollback above expired every loaded instance, including `project`,
        # which `add_papers` needs. Reload it here, in async context — touching an
        # expired attribute would try to refresh it from a synchronous access
        # point and raise MissingGreenlet, which is what this route did on every
        # call.
        await session.refresh(project)
        projects = ProjectRepository(session)
        resolved, missing = await projects.resolve_paper_ids(arxiv_ids)

    added = await projects.add_papers(project, resolved, added_by="api", note=note)
    await session.commit()
    stored = PaperRepository(session)
    titles = []
    for paper_id in resolved:
        row = await stored.get(paper_id)
        if row is not None:
            titles.append(row.display_id)
    return ImportResult(
        project=project_slug,
        requested=len(arxiv_ids),
        resolved=len(resolved),
        added=added,
        missing=missing,
        papers=titles,
    )


@router.delete(
    "/projects/{slug}/papers/{arxiv_id}", summary="Unlink a paper from a project"
)
async def unlink_paper(slug: str, arxiv_id: str, session: SessionDep) -> dict[str, Any]:
    projects, project = await _project_or_404(session, slug)
    resolved, _missing = await projects.resolve_paper_ids([arxiv_id])
    if not resolved:
        raise HTTPException(status_code=404, detail=f"{arxiv_id} is not in the corpus")
    removed = await projects.remove_paper(project, resolved[0])
    await session.commit()
    if not removed:
        raise HTTPException(status_code=404, detail=f"{arxiv_id} is not in {project.slug}")
    return {"unlinked": arxiv_id, "project": project.slug, "paper_kept": True}


@router.patch(
    "/projects/{slug}/papers/{arxiv_id}", summary="Set read state or note"
)
async def update_project_paper(
    slug: str,
    arxiv_id: str,
    session: SessionDep,
    is_read: bool | None = Query(default=None),
    note: str | None = Query(default=None),
) -> dict[str, Any]:
    projects, project = await _project_or_404(session, slug)
    resolved, _missing = await projects.resolve_paper_ids([arxiv_id])
    if not resolved:
        raise HTTPException(status_code=404, detail=f"{arxiv_id} is not in the corpus")
    paper_id = resolved[0]
    changed = False
    if is_read is not None:
        changed |= await projects.set_read(project, paper_id, is_read=is_read)
    if note is not None:
        changed |= await projects.set_note(project, paper_id, note)
    await session.commit()
    if not changed:
        raise HTTPException(status_code=404, detail=f"{arxiv_id} is not in {project.slug}")
    return {"updated": arxiv_id, "project": project.slug}


@router.get(
    "/papers/{arxiv_id}/references",
    response_model=ReferencesResponse,
    summary="A paper's citations",
)
async def paper_references(
    arxiv_id: str,
    session: SessionDep,
    direction: str = Query(
        default="references", description="references (out) | cited-by (in)"
    ),
    limit: int = Query(default=100, ge=1, le=500),
    resolved_only: bool = Query(default=False, description="Only in-corpus targets."),
) -> ReferencesResponse:
    """Bibliography edges of one paper.

    ``direction=references`` lists what it cites; ``cited-by`` lists which stored
    papers cite it. Reference extraction happens during ingestion, so a paper
    ingested from a PDF-only source may legitimately have none.
    """
    papers = PaperRepository(session)
    normalized = normalize_arxiv_id_or_400(arxiv_id)
    paper = await papers.resolve(normalized)
    if paper is None:
        raise HTTPException(status_code=404, detail=f"paper {arxiv_id} not ingested")

    refs = ReferenceRepository(session)
    incoming = direction in {"in", "cited-by", "cited_by", "citations"}

    if incoming:
        citing = await refs.citing_papers(paper.id, limit=limit)
        return ReferencesResponse(
            # `display_id`, not `arxiv_id`: the citation graph is only defined
            # for papers, but this route is reachable for any document and must
            # answer with something rather than None.
            arxiv_id=paper.display_id,
            direction="cited-by",
            count=len(citing),
            cited_by=[paper.display_id for paper in citing],
        )

    rows = await refs.list_references(paper.id, limit=limit, resolved_only=resolved_only)
    return ReferencesResponse(
        arxiv_id=paper.display_id,
        direction="references",
        count=len(rows),
        references=[
            ReferenceOut(
                ordinal=row.ordinal,
                cited_arxiv_id=row.cited_arxiv_id,
                cited_paper_id=row.cited_paper_id,
                resolved=row.is_resolved,
                authors=row.authors,
                title=row.title,
                year=row.year,
                venue=row.venue,
                label=row.label(),
                raw_text=row.raw_text,
            )
            for row in rows
        ],
    )


@router.get("/references/most-cited", summary="Corpus papers by incoming citations")
async def most_cited(
    session: SessionDep,
    limit: int = Query(default=20, ge=1, le=200),
) -> dict[str, Any]:
    """Ranked by how many papers *in this corpus* cite them.

    An internal-cohort ranking, not a real citation count — it only sees papers
    we happen to hold.
    """
    rows = await ReferenceRepository(session).most_cited(limit=limit)
    return {"count": len(rows), "papers": rows}

# ---------------------------------------------------------------------- assets
async def _paper_or_404(session: SessionDep, arxiv_id: str):  # noqa: ANN202
    papers = PaperRepository(session)
    normalized = normalize_arxiv_id_or_400(arxiv_id)
    # get_by_arxiv_id normalises, so the versioned and bare forms both resolve.
    paper = await papers.resolve(normalized)
    if paper is None:
        raise HTTPException(status_code=404, detail=f"paper {arxiv_id} not ingested")
    return paper


@router.get("/papers/{arxiv_id}/assets", response_model=AssetsResponse, summary="Figures, tables, equations")
async def paper_assets(
    arxiv_id: str,
    session: SessionDep,
    limit: int = Query(default=50, ge=1, le=500),
    include_inline_equations: bool = Query(
        default=False,
        description="Include inline math. Off by default: a paper has ~3 real "
        "equations and ~140 inline fragments.",
    ),
) -> AssetsResponse:
    """Extracted figures, tables and equations of one paper.

    Figures record where their image is — a ``image_sha256`` in the blob store
    from the PDF path, or an ``image_url`` we have not fetched from the HTML
    path — rather than always embedding bytes.
    """
    paper = await _paper_or_404(session, arxiv_id)
    repo = AssetRepository(session)
    return AssetsResponse(
        arxiv_id=paper.arxiv_id,
        title=paper.title,
        counts=AssetCounts(**await repo.count(paper.id)),
        figures=[
            FigureOut(
                ordinal=row.ordinal,
                label=row.label,
                caption=row.caption,
                image_url=row.image_url,
                image_sha256=row.image_sha256,
                page_idx=row.page_idx,
                width=row.width,
                height=row.height,
                source=row.source,
            )
            for row in await repo.figures(paper.id, limit=limit)
        ],
        tables=[
            TableAssetOut(
                ordinal=row.ordinal,
                label=row.label,
                caption=row.caption,
                body_html=row.body_html,
                row_count=row.row_count,
                column_count=row.column_count,
                image_url=row.image_url,
                image_sha256=row.image_sha256,
                page_idx=row.page_idx,
                source=row.source,
            )
            for row in await repo.tables(paper.id, limit=limit)
        ],
        equations=[
            EquationOut(
                ordinal=row.ordinal,
                latex=row.latex,
                is_display=row.is_display,
                page_idx=row.page_idx,
                image_sha256=row.image_sha256,
                source=row.source,
            )
            for row in await repo.equations(
                paper.id, limit=limit, display_only=not include_inline_equations
            )
        ],
    )


@router.get("/papers/{arxiv_id}/tables/{ordinal}/html", summary="One table's markup")
async def table_html(arxiv_id: str, ordinal: int, session: SessionDep) -> dict[str, Any]:
    """The raw ``<table>`` subtree, for rendering or parsing downstream.

    404 when the row has no markup: a table extracted from a PDF is only ever
    available as an image.
    """
    paper = await _paper_or_404(session, arxiv_id)
    row = next(
        (t for t in await AssetRepository(session).tables(paper.id) if t.ordinal == ordinal),
        None,
    )
    if row is None:
        raise HTTPException(status_code=404, detail=f"no table #{ordinal} in {arxiv_id}")
    if not row.body_html:
        raise HTTPException(
            status_code=404,
            detail=(
                f"table #{ordinal} has no markup — it was extracted from a PDF and "
                "exists only as an image"
                + (f" (image_sha256={row.image_sha256})" if row.image_sha256 else "")
            ),
        )
    return {
        "arxiv_id": paper.arxiv_id,
        "ordinal": row.ordinal,
        "label": row.label,
        "caption": row.caption,
        "row_count": row.row_count,
        "column_count": row.column_count,
        "body_html": row.body_html,
    }


@router.get("/equations", summary="Search formulas corpus-wide")
async def search_equations(
    session: SessionDep,
    q: str = Query(min_length=1, description="Substring of the LaTeX."),
    limit: int = Query(default=20, ge=1, le=100),
    display_only: bool = Query(default=False),
    project: str | None = Query(default=None, description="Only this project's papers."),
    arxiv_id: list[str] | None = Query(default=None, description="Only these papers. Repeatable."),
) -> dict[str, Any]:
    """Find formulas whose LaTeX contains a substring.

    Scoped like semantic search: by ``project``, by ``arxiv_id``, or across the
    whole corpus. This is a *substring* match on the stored LaTeX, not a
    semantic one — ``POST /search/semantic`` with ``content_kinds=["equation"]``
    is the meaning-based equivalent.
    """
    paper_ids: list[str] | None = None
    if project:
        try:
            paper_ids = await ProjectRepository(session).paper_ids_for(project)
        except PaperNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
    if arxiv_id:
        papers = PaperRepository(session)
        wanted = [await papers.resolve(one) for one in arxiv_id]
        resolved = [paper.id for paper in wanted if paper is not None]
        missing = [one for one, paper in zip(arxiv_id, wanted, strict=True) if paper is None]
        if missing:
            raise HTTPException(
                status_code=404, detail=f"paper not ingested: {', '.join(missing)}"
            )
        paper_ids = resolved if paper_ids is None else sorted(set(paper_ids) & set(resolved))

    rows = await AssetRepository(session).search_equations(
        q, limit=limit, display_only=display_only, paper_ids=paper_ids
    )
    return {
        "query": q,
        "project": project,
        "scope_papers": len(paper_ids) if paper_ids is not None else None,
        "count": len(rows),
        "equations": [
            {
                "paper_id": row.paper_id,
                "ordinal": row.ordinal,
                "latex": row.latex,
                "is_display": row.is_display,
                "source": row.source,
            }
            for row in rows
        ],
    }


# Under /assets/ rather than /papers/missing-figures: the papers router is
# registered first, so a two-segment path here would be swallowed by
# `/papers/{arxiv_id}`.
@router.get("/assets/missing-figures", summary="Ingested papers with no figures")
async def papers_missing_figures(
    session: SessionDep,
    limit: int = Query(default=50, ge=1, le=200),
) -> dict[str, Any]:
    """The set worth re-extracting: ingested, but no figures recorded.

    Either the paper predates asset extraction, or its rendering had no figures.
    """
    rows = await AssetRepository(session).papers_missing_figures(limit=limit)
    return {
        "count": len(rows),
        "papers": [{"arxiv_id": row.arxiv_id, "title": row.title} for row in rows],
    }
