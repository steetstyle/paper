"""Ingestion use case.

Responsibilities:

* create the run record so callers get a trackable id immediately
* build the pipeline context and execute the canonical step list
* fan out over several papers with bounded concurrency
* degrade gracefully when the network, MinerU or the vector store misbehave
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from app.clients.arxiv.client import ArxivClient
from app.clients.arxiv.exceptions import ArxivError
from app.config import IngestionSettings
from app.db.repositories import RunRepository
from app.db.spaces import EmbeddingSpace
from app.db.vector_store.base import VectorStore
from app.domain.enums import RunStatus
from app.domain.ids import normalize_arxiv_id
from app.domain.models import PaperMetadata, SearchQuery
from app.logging import get_logger
from app.pipeline.context import PipelineContext
from app.pipeline.runner import PipelineRunner
from app.pipeline.steps import Step

logger = get_logger(__name__)


@dataclass(slots=True)
class IngestionResult:
    run_ids: list[str] = field(default_factory=list)
    paper_ids: list[str] = field(default_factory=list)
    statuses: dict[str, RunStatus] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    status: RunStatus = RunStatus.PENDING

    @property
    def succeeded(self) -> bool:
        return self.status in {RunStatus.SUCCEEDED, RunStatus.PARTIAL, RunStatus.SKIPPED}


class IngestionService:
    def __init__(
        self,
        *,
        arxiv_client: ArxivClient,
        steps: list[Step],
        session_factory: Any,
        vector_store: VectorStore,
        settings: IngestionSettings | None = None,
        pipeline_factory: Any | None = None,
        space: EmbeddingSpace | None = None,
        blob_store: Any | None = None,  # noqa: ANN401 - BlobStore, optional
    ) -> None:
        self._arxiv = arxiv_client
        self._steps = steps
        self._session_factory = session_factory
        self._blob_store = blob_store
        self._vector_store = vector_store
        self._settings = settings or IngestionSettings()
        self._pipeline_factory = pipeline_factory
        # Recorded for reporting; the steps/store were already built per space.
        self.space = space or vector_store.space

    def _runner(self) -> PipelineRunner:
        if self._pipeline_factory is not None:
            return self._pipeline_factory(self._steps)
        return PipelineRunner(self._steps)

    @property
    def session_factory(self) -> Any:
        return self._session_factory

    # ------------------------------------------------------------------ single
    async def ingest_paper(
        self,
        arxiv_id: str | None = None,
        *,
        metadata: PaperMetadata | None = None,
        prefer_html: bool = True,
        force: bool = False,
        requested_by: str = "api",
        trigger: str = "manual",
        run_id: str | None = None,
    ) -> IngestionResult:
        """Run the full pipeline for one paper."""
        target = normalize_arxiv_id(arxiv_id) if arxiv_id else (
            metadata.arxiv_id if metadata else None
        )
        run_id = run_id or await self._create_run(target, prefer_html, force, requested_by, trigger)

        try:
            await self._vector_store.ensure_ready()
        except Exception as exc:  # noqa: BLE001 - vector store may be down
            logger.error("vector_store_unavailable", extra={"error": str(exc)})
            await self._finish_run(run_id, RunStatus.FAILED, error=f"vector store: {exc}")
            return IngestionResult(
                run_ids=[run_id], status=RunStatus.FAILED, errors={run_id: str(exc)}
            )

        async with self._session_factory() as session:
            ctx = PipelineContext(
                session=session,
                run_id=run_id,
                arxiv_id=target,
                metadata=metadata,
                prefer_html=prefer_html,
                force=force,
                # The asset step hashes cropped images into the blob store.
                blob_store=self._blob_store,
            )
            try:
                await self._runner().run(ctx)
            except ArxivError as exc:
                await RunRepository(session).finish(run_id, RunStatus.FAILED, error=str(exc))
                await session.commit()
                logger.error("ingestion_arxiv_error", extra={"run_id": run_id, "error": str(exc)})
                return IngestionResult(
                    run_ids=[run_id],
                    paper_ids=[ctx.paper_id] if ctx.paper_id else [],
                    status=RunStatus.FAILED,
                    errors={run_id: str(exc)},
                )
            except Exception as exc:  # noqa: BLE001 - surfaced to the caller
                logger.exception("ingestion_failed", extra={"run_id": run_id})
                await RunRepository(session).finish(run_id, RunStatus.FAILED, error=str(exc))
                await session.commit()
                return IngestionResult(
                    run_ids=[run_id],
                    paper_ids=[ctx.paper_id] if ctx.paper_id else [],
                    status=RunStatus.FAILED,
                    errors={run_id: str(exc)},
                )

            run = await RunRepository(session).get(run_id)
            status = RunStatus(run.status) if run else RunStatus.FAILED
            return IngestionResult(
                run_ids=[run_id],
                paper_ids=[ctx.paper_id] if ctx.paper_id else [],
                statuses={run_id: status},
                errors={run_id: str(run.error)} if run and run.error else {},
                status=status,
            )

    # ------------------------------------------------------------------- batch
    async def ingest_query(
        self,
        query: SearchQuery,
        *,
        limit: int = 5,
        prefer_html: bool = True,
        force: bool = False,
        requested_by: str = "api",
        trigger: str = "query",
    ) -> IngestionResult:
        """Resolve a search query to papers, then ingest them concurrently."""
        papers = await self._arxiv.search_all(query, limit=limit)
        if not papers:
            return IngestionResult(status=RunStatus.SKIPPED)
        return await self.ingest_many(
            papers, prefer_html=prefer_html, force=force, requested_by=requested_by, trigger=trigger
        )

    async def ingest_many(
        self,
        papers: list[PaperMetadata],
        *,
        prefer_html: bool = True,
        force: bool = False,
        requested_by: str = "api",
        trigger: str = "batch",
    ) -> IngestionResult:
        semaphore = asyncio.Semaphore(self._settings.concurrency)
        run_ids: list[str] = []
        paper_ids: list[str] = []
        statuses: dict[str, RunStatus] = {}
        errors: dict[str, str] = {}

        async def worker(paper: PaperMetadata) -> None:
            async with semaphore:
                result = await self.ingest_paper(
                    paper.arxiv_id,
                    metadata=paper,
                    prefer_html=prefer_html,
                    force=force,
                    requested_by=requested_by,
                    trigger=trigger,
                )
                run_ids.extend(result.run_ids)
                paper_ids.extend(paper_id for paper_id in result.paper_ids if paper_id)
                statuses.update(result.statuses)
                errors.update(result.errors)

        await asyncio.gather(*(worker(paper) for paper in papers))

        overall = _aggregate_status(list(statuses.values()))
        return IngestionResult(
            run_ids=run_ids, paper_ids=paper_ids, statuses=statuses, errors=errors, status=overall
        )

    # ------------------------------------------------------------------ helpers
    async def _create_run(
        self,
        arxiv_id: str | None,
        prefer_html: bool,
        force: bool,
        requested_by: str,
        trigger: str,
    ) -> str:
        async with self._session_factory() as session:
            run = await RunRepository(session).create(
                arxiv_id=arxiv_id,
                requested_by=requested_by,
                trigger=trigger,
                prefer_html=prefer_html,
                force=force,
            )
            await session.commit()
            return run.id

    async def _finish_run(self, run_id: str, status: RunStatus, *, error: str) -> None:
        async with self._session_factory() as session:
            await RunRepository(session).finish(run_id, status, error=error)
            await session.commit()


def _aggregate_status(statuses: list[RunStatus]) -> RunStatus:
    if not statuses:
        return RunStatus.FAILED
    if all(status == RunStatus.SUCCEEDED for status in statuses):
        return RunStatus.SUCCEEDED
    if all(status == RunStatus.FAILED for status in statuses):
        return RunStatus.FAILED
    if any(status in {RunStatus.SUCCEEDED, RunStatus.PARTIAL} for status in statuses):
        return RunStatus.PARTIAL
    return RunStatus.FAILED


def build_ingestion_service(
    container: Any, space: EmbeddingSpace | None = None  # noqa: ANN401
) -> IngestionService:
    """Build the service from a :class:`~app.container.Container` for one space."""
    space = space or container.active_space
    return IngestionService(
        arxiv_client=container.arxiv,
        steps=container.build_steps(space),
        session_factory=container.session_factory,
        vector_store=container.vector_store_for(space),
        settings=container.settings.ingestion,
        space=space,
        blob_store=container.blob_store,
    )


__all__ = ["IngestionService", "IngestionResult", "build_ingestion_service"]