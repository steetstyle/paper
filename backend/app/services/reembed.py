"""Re-embed chunks that are already stored into another embedding space.

Switching embedding models is not a re-ingest. Chunks live in ``chunks`` and are
touched by the chunker; vectors live in one table **per space**. So a model
change only means "embed the text that is already there, into this space's
table" — no download, no PDF parse, no MinerU. That is the difference between
five minutes and five hours for a few thousand chunks.

Deliberately reuses :class:`~app.pipeline.steps.EmbedChunksStep` and
:class:`~app.pipeline.steps.IndexVectorsStep` rather than duplicating their
logic, so a re-embedded vector is byte-for-byte what a full ingest would have
written.

Each space is independent: writing into ``bge-large`` leaves the sibling space's
rows untouched, which is what lets two models be compared side by side.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from app.db.repositories import ChunkRepository, EmbeddingRepository
from app.db.spaces import EmbeddingSpace
from app.db.vector_store.schema import embedding_table, ensure_space_table
from app.domain.models import PaperMetadata
from app.embeddings.base import EmbeddingProvider
from app.logging import get_logger
from app.pipeline.context import PipelineContext
from app.pipeline.steps import EmbedChunksStep, IndexVectorsStep
from app.services.kind_backfill import backfill_chunk_kinds

logger = get_logger(__name__)


@dataclass(slots=True)
class ReembedReport:
    """What one :meth:`ReembedService.reembed` run did."""

    space: str
    model: str
    dimensions: int
    papers_seen: int = 0
    papers_embedded: int = 0
    papers_skipped: int = 0
    chunks_embedded: int = 0
    failures: list[str] = field(default_factory=list)
    seconds: float = 0.0

    def summary(self) -> str:
        return (
            f"{self.space} ({self.model}, {self.dimensions}d): "
            f"{self.chunks_embedded} chunks from {self.papers_embedded} papers "
            f"in {self.seconds:.1f}s"
            + (f", {self.papers_skipped} already present" if self.papers_skipped else "")
            + (f", {len(self.failures)} failed" if self.failures else "")
        )


class ReembedService:
    """Embed stored chunks into one space, without re-parsing any paper."""

    def __init__(
        self,
        *,
        provider: EmbeddingProvider,
        space: EmbeddingSpace,
        session_factory: Any,
        store: Any | None = None,
        batch_size: int = 64,
        concurrency: int = 1,
        ensure_table: bool = True,
        on_progress: Any | None = None,
    ) -> None:
        self._provider = provider
        self._space = space
        self._session_factory = session_factory
        self._store = store
        self._batch_size = batch_size
        # Embedding is CPU-bound in-process, so >1 worker on a local model just
        # fights the model for the same cores. Kept configurable for hosted
        # providers that are IO-bound.
        self._concurrency = max(1, concurrency)
        self._ensure_table = ensure_table
        self._on_progress = on_progress
        self._table = embedding_table(space)

    @property
    def space(self) -> EmbeddingSpace:
        return self._space

    async def reembed(
        self,
        *,
        paper_ids: list[str] | None = None,
        arxiv_ids: list[str] | None = None,
        project: str | None = None,
        force: bool = False,
        backfill_kinds: bool = True,
        limit: int | None = None,
    ) -> ReembedReport:
        """Embed into :attr:`space`.

        Args:
            paper_ids: restrict to these internal ids.
            arxiv_ids: restrict to these arXiv ids.
            project: restrict to the papers a project holds.
            force: re-embed papers that already have rows here.
            backfill_kinds: also (re)classify ``chunks.content_kind`` from the
                stored text and heading. Cheap, and it is the only way an older
                corpus picks up a classifier added after it was ingested.
            limit: stop after this many papers (handy for a smoke test).
        """
        started = time.monotonic()
        report = ReembedReport(
            space=self._space.name,
            model=self._space.model,
            dimensions=self._space.dimensions,
        )
        targets = await self._resolve_targets(paper_ids, arxiv_ids, project, limit)
        report.papers_seen = len(targets)

        # The table has to exist before the first paper is asked "are you already
        # embedded here?". A brand-new space has no table, and the skip-check
        # would fail with `no such table` on the very first paper — which is
        # exactly the case re-embedding exists for. One catalogue query, once.
        await self._make_table_ready()

        embed_step = EmbedChunksStep(
            self._provider,
            self._space,
            batch_size=self._batch_size,
            ensure_table=self._ensure_table,
        )
        index_step = IndexVectorsStep(self._store) if self._store is not None else None

        for index, (paper_id, arxiv_id) in enumerate(targets, start=1):
            try:
                embedded = await self._one_paper(
                    paper_id,
                    arxiv_id,
                    embed_step=embed_step,
                    index_step=index_step,
                    force=force,
                    backfill_kinds=backfill_kinds,
                    report=report,
                )
            except Exception as exc:  # noqa: BLE001 - one bad paper must not stop the rest
                # The message is kept verbatim in the report: "failed" with no
                # reason is a bug report nobody can act on.
                report.failures.append(f"{arxiv_id}: {type(exc).__name__}: {exc}")
                logger.warning(
                    "reembed_paper_failed",
                    extra={"arxiv_id": arxiv_id, "error": str(exc)},
                )
                embedded = 0
            if embedded:
                report.papers_embedded += 1
                report.chunks_embedded += embedded
            else:
                report.papers_skipped += 1
            if self._on_progress is not None:
                self._on_progress(index, len(targets), arxiv_id, embedded)

        report.seconds = time.monotonic() - started
        logger.info(
            "reembed_done",
            extra={
                "space": report.space,
                "papers": report.papers_embedded,
                "chunks": report.chunks_embedded,
                "seconds": round(report.seconds, 1),
            },
        )
        return report

    async def backfill_kinds(
        self, *, only_body: bool = True, paper_id: str | None = None
    ) -> dict[str, int]:
        """Label stored chunks, without embedding anything.

        Separate from :meth:`reembed` because it costs milliseconds while
        embedding costs minutes: a corpus can be made filterable before
        committing to a full backfill.
        """
        async with self._session_factory() as session:
            counts = await backfill_chunk_kinds(session, only_body=only_body, paper_id=paper_id)
            await session.commit()
        return counts

    async def preview(
        self,
        *,
        project: str | None = None,
        arxiv_ids: list[str] | None = None,
        limit: int | None = None,
    ) -> list[str]:
        """ArXiv ids that would be embedded. Embeds nothing."""
        targets = await self._resolve_targets(None, arxiv_ids, project, limit)
        return [arxiv_id for _paper_id, arxiv_id in targets]

    # -------------------------------------------------------------- internals
    async def _make_table_ready(self) -> None:
        if not self._ensure_table:
            return
        async with self._session_factory() as session:
            connection = await session.connection()
            created = await connection.run_sync(
                lambda conn: ensure_space_table(conn, self._space)
            )
            await session.commit()
        if created:
            logger.info(
                "reembed_space_table_created",
                extra={"space": self._space.name, "table": self._space.resolved_table},
            )

    async def _one_paper(
        self,
        paper_id: str,
        arxiv_id: str,
        *,
        embed_step: EmbedChunksStep,
        index_step: Any | None,
        force: bool,
        backfill_kinds: bool,
        report: ReembedReport,
    ) -> int:
        async with self._session_factory() as session:
            embeddings = EmbeddingRepository(session, self._table)
            if not force and await embeddings.count_for_paper(paper_id):
                return 0
            chunks_repo = ChunkRepository(session)
            rows = list(await chunks_repo.list_for_paper(paper_id))
            if not rows:
                return 0
            if backfill_kinds:
                await backfill_chunk_kinds(session, only_body=False, paper_id=paper_id)

            ctx = PipelineContext(
                session=session,
                run_id=f"reembed-{self._space.name}",
                paper_id=paper_id,
                arxiv_id=arxiv_id,
                force=force,
                chunk_rows=rows,
                # Steps read categories off metadata. Re-embedding does not
                # need them for pgvector, and an empty tuple is honest rather
                # than a second lookup of metadata that has not changed.
                metadata=PaperMetadata(
                    arxiv_id=arxiv_id,
                    versioned_id=arxiv_id,
                    version=None,
                    title=arxiv_id,
                    abstract="",
                ),
            )
            result = await embed_step.run(ctx)
            if not result.ok:
                detail = (result.data or {}).get("error") or "embed step failed"
                raise RuntimeError(str(detail))
            if index_step is not None:
                await index_step.run(ctx)
            await session.commit()
            return int(ctx.embedded_count)

    async def _resolve_targets(
        self,
        paper_ids: list[str] | None,
        arxiv_ids: list[str] | None,
        project: str | None,
        limit: int | None,
    ) -> list[tuple[str, str]]:
        from sqlalchemy import select  # noqa: PLC0415

        from app.db.models import Paper  # noqa: PLC0415
        from app.db.project_repository import ProjectRepository  # noqa: PLC0415

        async with self._session_factory() as session:
            if project:
                repo = ProjectRepository(session)
                found = await repo.get(project)
                if found is None:
                    raise LookupError(f"no project matching {project!r}")
                paper_ids = await repo.paper_ids(found)
            stmt = select(Paper.id, Paper.arxiv_id)
            if paper_ids:
                stmt = stmt.where(Paper.id.in_(paper_ids))
            if arxiv_ids:
                stmt = stmt.where(Paper.arxiv_id.in_(arxiv_ids))
            # Deterministic order so a --limit smoke test is reproducible.
            stmt = stmt.order_by(Paper.arxiv_id)
            if limit:
                stmt = stmt.limit(limit)
            rows = (await session.execute(stmt)).all()
        return [(row[0], row[1]) for row in rows]


__all__ = ["ReembedReport", "ReembedService"]
