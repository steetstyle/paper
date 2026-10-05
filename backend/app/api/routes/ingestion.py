"""Ingestion endpoints."""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, status

from app.api.deps import ContainerDep, SessionDep, resolve_space
from app.api.schemas import IngestAccepted, IngestRequest, RunOut
from app.db.repositories import RunRepository
from app.domain.enums import RunStatus
from app.domain.models import SearchQuery
from app.services.ingestion import IngestionService

router = APIRouter(prefix="/ingest", tags=["ingestion"])


@router.post(
    "",
    response_model=IngestAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Ingest one paper or the top hits of a query",
)
async def ingest(
    payload: IngestRequest,
    background: BackgroundTasks,
    container: ContainerDep,
    session: SessionDep,
    wait: bool = Query(default=False, description="Run inline and return the final status."),
) -> IngestAccepted:
    """Fire-and-forget by default; set ``?wait=true`` for synchronous ingestion.

    Either ``arxiv_id`` or ``query`` must be provided. ``query`` uses the same
    syntax as the ArXiv API (e.g. ``cat:cs.CL AND ti:transformer``).

    ``space`` selects which embedding model writes the vectors. Omit it to use
    the active space; naming a different one adds vectors without disturbing the
    space you already serve from.
    """
    if not payload.arxiv_id and not payload.query:
        raise HTTPException(status_code=422, detail="provide either arxiv_id or query")

    space = await resolve_space(container, session, payload.space)
    service = build_service(container, space)

    if payload.arxiv_id:
        if wait:
            result = await service.ingest_paper(
                payload.arxiv_id,
                prefer_html=payload.prefer_html,
                force=payload.force,
                requested_by="api",
                trigger="api",
            )
            return IngestAccepted(
                run_ids=result.run_ids,
                paper_ids=result.paper_ids,
                status=result.status,
                space=space.name,
            )
        run_id = await _precreate_run(
            service,
            arxiv_id=payload.arxiv_id,
            prefer_html=payload.prefer_html,
            force=payload.force,
        )
        background.add_task(
            service.ingest_paper,
            payload.arxiv_id,
            prefer_html=payload.prefer_html,
            force=payload.force,
            requested_by="api",
            trigger="api",
            run_id=run_id,
        )
        return IngestAccepted(
            run_ids=[run_id], paper_ids=[], status=RunStatus.PENDING, space=space.name
        )

    query = SearchQuery(
        raw=payload.query,
        max_results=payload.limit,
        sort_by="relevance",
        sort_order="descending",
    )
    if wait:
        result = await service.ingest_query(
            query,
            limit=payload.limit,
            prefer_html=payload.prefer_html,
            force=payload.force,
            requested_by="api",
            trigger="api",
        )
        return IngestAccepted(
            run_ids=result.run_ids,
            paper_ids=result.paper_ids,
            status=result.status,
            space=space.name,
        )

    background.add_task(
        service.ingest_query,
        query,
        limit=payload.limit,
        prefer_html=payload.prefer_html,
        force=payload.force,
        requested_by="api",
        trigger="api",
    )
    return IngestAccepted(run_ids=[], paper_ids=[], status=RunStatus.PENDING, space=space.name)


def build_service(container: ContainerDep, space) -> IngestionService:  # noqa: ANN001
    from app.services.ingestion import build_ingestion_service  # noqa: PLC0415

    return build_ingestion_service(container, space)


async def _precreate_run(
    service: IngestionService,
    *,
    arxiv_id: str,
    prefer_html: bool,
    force: bool,
) -> str:
    """Create the run row up front so the caller can poll it immediately."""
    async with service.session_factory() as session:
        run = await RunRepository(session).create(
            arxiv_id=arxiv_id,
            requested_by="api",
            trigger="api",
            prefer_html=prefer_html,
            force=force,
        )
        await session.commit()
        return run.id


@router.get("/runs", response_model=list[RunOut], summary="Recent ingestion runs")
async def list_runs(session: SessionDep, limit: int = Query(default=20, ge=1, le=200)) -> list[RunOut]:
    runs = await RunRepository(session).recent_runs(limit=limit)
    return [RunOut.model_validate(run) for run in runs]


@router.get("/runs/{run_id}", response_model=RunOut, summary="Run status")
async def get_run(session: SessionDep, run_id: str) -> RunOut:
    run = await RunRepository(session).get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"run {run_id} not found")
    return RunOut.model_validate(run)