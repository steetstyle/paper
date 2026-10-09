"""Ingestion use case.

Responsibilities:

* create the run record so callers get a trackable id immediately
* build the pipeline context and execute the canonical step list
* fan out over several papers with bounded concurrency
* degrade gracefully when the network, MinerU or the vector store misbehave
"""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from paper_app.clients.arxiv.client import ArxivClient
from paper_app.clients.arxiv.exceptions import ArxivError
from paper_app.clients.content.pdf_info import PdfInfo, read_pdf_info
from paper_app.config import IngestionSettings
from paper_app.db.repositories import PaperRepository, RunRepository
from paper_app.db.spaces import EmbeddingSpace
from paper_app.db.vector_store.base import VectorStore
from paper_app.domain.enums import RunStatus
from paper_app.domain.ids import normalize_arxiv_id
from paper_app.domain.models import (
    Author,
    MineruOptions,
    PageRange,
    PaperMetadata,
    SearchQuery,
    doc_slug,
)
from paper_app.logging import get_logger
from paper_app.pipeline.context import PipelineContext
from paper_app.pipeline.runner import PipelineRunner
from paper_app.pipeline.steps import Step
from paper_app.services.runs import resume_instead_of_skip

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
        device: str | None = None,
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
        #: Which device this run will embed on. Reported rather than inferred: a
        #: caller who asked for `cuda` should be able to see that it got it.
        self.embedding_device = device or "cpu"

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
        run_id = run_id or await self._create_run(
            target, prefer_html, force, requested_by, trigger, doc_key=target
        )

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

    # ------------------------------------------------------------------- file
    async def ingest_file(
        self,
        path: Path,
        *,
        kind: str = "book",
        title: str | None = None,
        doc_key: str | None = None,
        authors: tuple[Author, ...] = (),
        page_range: PageRange | None = None,
        options: MineruOptions | None = None,
        force: bool = False,
        requested_by: str = "api",
        trigger: str = "file",
    ) -> IngestionResult:
        """Ingest a PDF from the local filesystem. No network is touched.

        The same pipeline as :meth:`ingest_paper`, from the same step list, with
        the bytes arriving from a path instead of a URL. What differs is only
        where the content comes from and what identifies it — which is the point
        of ``doc_key`` existing next to ``arxiv_id``.

        Dedupe is decided here rather than by a database constraint on
        ``file_sha256``: the same book may legitimately be ingested twice (after
        a failed run, or as a second copy under another name), so a constraint
        would make ``--force`` impossible. A file already in the corpus is
        reported and skipped unless ``force``.
        """
        # `expanduser`/`resolve` touch the filesystem, so they happen on the
        # worker thread together with the read and the hash below.
        try:
            resolved, info, fingerprint = await asyncio.to_thread(
                _probe_local_file, path
            )
        except FileNotFoundError as exc:
            # The user's own argument is wrong, not the pipeline. Reported as a
            # result rather than a traceback so the CLI can print one line.
            return IngestionResult(status=RunStatus.FAILED, errors={"": str(exc)})

        if not force:
            async with self._session_factory() as session:
                duplicate = await PaperRepository(session).get_by_file_sha256(fingerprint)
                previous = (
                    await RunRepository(session).latest_for_target(duplicate.doc_key)
                    if duplicate is not None
                    else None
                )
            if duplicate is not None and not resume_instead_of_skip(
                previous.status if previous else None
            ):
                logger.info(
                    "ingest_duplicate_file",
                    extra={"path": str(resolved), "doc_key": duplicate.doc_key},
                )
                return IngestionResult(
                    status=RunStatus.SKIPPED,
                    paper_ids=[duplicate.id],
                    errors={
                        "": (
                            f"{resolved.name} is already ingested as "
                            f"{duplicate.display_id!r} ({duplicate.title}); "
                            "pass --force to re-extract it"
                        )
                    },
                )

        handle = doc_key or doc_slug(resolved.stem)
        async with self._session_factory() as session:
            clash = await PaperRepository(session).resolve(handle)
        if clash is not None and clash.file_sha256 != fingerprint and not force:
            return IngestionResult(
                status=RunStatus.FAILED,
                errors={
                    "": (
                        f"{handle!r} is already {clash.display_id!r} "
                        f"({clash.title!r}); pass --doc-key to pick another handle"
                    )
                },
            )

        metadata = PaperMetadata.from_file(
            path=resolved,
            title=title,
            doc_key=handle,
            kind=kind,
            page_count=info.page_count,
            file_sha256=fingerprint,
            authors=authors,
            raw=info.meta,
        )
        run_id = await self._create_run(
            None, prefer_html=False, force=force, requested_by=requested_by,
            trigger=trigger, doc_key=handle,
        )

        try:
            await self._vector_store.ensure_ready()
        except Exception as exc:  # noqa: BLE001 - vector store may be down
            logger.error("vector_store_unavailable", extra={"error": str(exc)})
            await self._finish_run(run_id, RunStatus.FAILED, error=f"vector store: {exc}")
            return IngestionResult(
                run_ids=[run_id], status=RunStatus.FAILED, errors={run_id: str(exc)}
            )

        pages = page_range.clamp(info.page_count) if page_range else None
        if page_range is not None and pages is None:
            message = (
                f"pages {page_range} are past the end of a "
                f"{info.page_count}-page document"
            )
            await self._finish_run(run_id, RunStatus.FAILED, error=message)
            return IngestionResult(
                run_ids=[run_id], status=RunStatus.FAILED, errors={run_id: message}
            )

        async with self._session_factory() as session:
            ctx = PipelineContext(
                session=session,
                run_id=run_id,
                arxiv_id=None,
                metadata=metadata,
                prefer_html=False,
                force=force,
                local_path=resolved,
                page_range=pages,
                mineru_options=options or MineruOptions(),
                blob_store=self._blob_store,
            )
            try:
                await self._runner().run(ctx)
            except Exception as exc:  # noqa: BLE001 - surfaced to the caller
                logger.exception("ingestion_file_failed", extra={"run_id": run_id})
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
        doc_key: str | None = None,
    ) -> str:
        async with self._session_factory() as session:
            run = await RunRepository(session).create(
                arxiv_id=arxiv_id,
                doc_key=doc_key,
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


def _probe_local_file(path: Path) -> tuple[Path, PdfInfo, str]:
    """Resolve a path, read its structure, hash it. Blocking; via ``to_thread``.

    One round trip rather than three because each of these opens the same file
    and the answers must agree: a size checked here and a copy made there from a
    path resolved twice is how a file gets ingested from two different versions
    of itself.

    Raises :class:`FileNotFoundError` for a missing path so the caller reports
    the user's own argument rather than an internal type.
    """
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"no such file: {path}")
    return resolved, read_pdf_info(resolved), _sha256_of(resolved)


def _sha256_of(path: Path) -> str:
    """Content hash, read in chunks so a 700-page book is not held in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    container: Any,  # noqa: ANN401
    space: EmbeddingSpace | None = None,
    *,
    device: str | None = None,
) -> IngestionService:
    """Build the service from a :class:`~paper_app.container.Container` for one space.

    ``device`` selects the device for *this run's* embedding steps. It reaches the
    provider cache rather than a global setting, because a loaded model belongs to
    the device it was loaded on and one process may legitimately serve a bulk
    ingest on cuda and queries on cpu at the same time.
    """
    space = space or container.active_space
    resolved = device or container.settings.embedding.device or "cpu"
    return IngestionService(
        arxiv_client=container.arxiv,
        steps=container.build_steps(space, device=resolved),
        session_factory=container.session_factory,
        vector_store=container.vector_store_for(space),
        settings=container.settings.ingestion,
        space=space,
        blob_store=container.blob_store,
    )


__all__ = ["IngestionService", "IngestionResult", "build_ingestion_service"]