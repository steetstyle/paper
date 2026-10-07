"""Interrupted ingestion: what happens after the process dies.

Two failures, both measured on the live corpus before they were fixed:

* 38 runs were stuck in ``running``, the oldest for over a day, with nothing to
  distinguish them from work in progress — 40 unfinished rows in total, of which
  2 were duplicate attempts at the same paper.
* Re-running one of them redid the extraction from scratch. MinerU is the most
  expensive step by a wide margin (measured ~1.7 pages/s), and the extracted
  markdown was already on disk.

These tests pin both: that a dead run becomes terminal and says so, and that a
retry reuses what the interrupted run already produced instead of paying for it
again.
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import select

from app.db.models import IngestionRun, Paper, PipelineStepRun
from app.db.repositories import RawDocumentRepository, RunRepository
from app.db.session import get_session_factory
from app.domain.enums import ContentKind, RunStatus
from app.pipeline.steps import ExtractTextStep
from app.services.runs import find_stale_runs, reap_stale_runs, resume_instead_of_skip


def _hours_ago(hours: float) -> datetime:
    return datetime.now(UTC) - timedelta(hours=hours)


async def _make_run(
    arxiv_id: str = "1706.03762",
    *,
    status: str = RunStatus.RUNNING.value,
    created_hours_ago: float = 1.0,
    doc_key: str | None = None,
) -> str:
    async with get_session_factory()() as session:
        run = await RunRepository(session).create(
            arxiv_id=arxiv_id, doc_key=doc_key or arxiv_id, trigger="test"
        )
        run.status = status
        run.created_at = _hours_ago(created_hours_ago)
        await session.commit()
        return run.id


async def _add_step(run_id: str, name: str, *, hours_ago: float = 1.0) -> None:
    async with get_session_factory()() as session:
        run = await session.get(IngestionRun, run_id)
        assert run is not None
        row = PipelineStepRun(
            run_id=run_id,
            name=name,
            attempt=1,
            status="succeeded",
            duration_ms=1.0,
            meta={},
            started_at=_hours_ago(hours_ago),
            finished_at=_hours_ago(hours_ago),
        )
        run.steps.append(row)
        await session.commit()


class TestFindingStaleRuns:
    async def test_a_run_nothing_has_touched_is_stale(self) -> None:
        run_id = await _make_run(created_hours_ago=30.0)
        async with get_session_factory()() as session:
            stale = await find_stale_runs(session, older_than_hours=6.0)
        assert [item.id for item in stale] == [run_id]

    async def test_a_run_in_progress_is_not_stale(self) -> None:
        """The whole point of the bound: work happening right now is untouched."""
        await _make_run(created_hours_ago=0.2)
        async with get_session_factory()() as session:
            assert await find_stale_runs(session, older_than_hours=6.0) == []

    async def test_recent_progress_outweighs_an_old_creation_time(self) -> None:
        """Liveness is the newest step, not when the run was created. A run that
        has been going for two days but stepped a minute ago is working."""
        run_id = await _make_run(created_hours_ago=48.0)
        await _add_step(run_id, "embed_chunks", hours_ago=0.02)
        async with get_session_factory()() as session:
            assert await find_stale_runs(session, older_than_hours=6.0) == []

    async def test_progress_that_has_itself_stopped_is_stale(self) -> None:
        run_id = await _make_run(created_hours_ago=48.0)
        await _add_step(run_id, "extract_text", hours_ago=20.0)
        async with get_session_factory()() as session:
            stale = await find_stale_runs(session, older_than_hours=6.0)
        assert [(item.id, item.steps_completed) for item in stale] == [(run_id, 1)]

    async def test_a_pending_run_that_never_started_is_stale_too(self) -> None:
        """It has no step record at all, so its creation time is the only evidence.
        One such run sat for 16 hours before this existed."""
        run_id = await _make_run(status=RunStatus.PENDING.value, created_hours_ago=16.0)
        async with get_session_factory()() as session:
            stale = await find_stale_runs(session, older_than_hours=6.0)
        assert [item.id for item in stale] == [run_id]

    @pytest.mark.parametrize(
        "status", [RunStatus.SUCCEEDED.value, RunStatus.FAILED.value, RunStatus.ABANDONED.value]
    )
    async def test_a_finished_run_is_never_stale(self, status: str) -> None:
        await _make_run(status=status, created_hours_ago=100.0)
        async with get_session_factory()() as session:
            assert await find_stale_runs(session, older_than_hours=6.0) == []

    async def test_the_age_and_target_are_reported(self) -> None:
        await _make_run(arxiv_id="2302.07175", doc_key="2302.07175", created_hours_ago=36.0)
        async with get_session_factory()() as session:
            stale = await find_stale_runs(session, older_than_hours=6.0)
        assert stale[0].target == "2302.07175"
        assert 35 < (stale[0].age_hours or 0) < 37

    async def test_a_document_is_named_by_its_doc_key(self) -> None:
        await _make_run(arxiv_id=None, doc_key="solid-state-basics", created_hours_ago=36.0)
        async with get_session_factory()() as session:
            stale = await find_stale_runs(session, older_than_hours=6.0)
        assert stale[0].target == "solid-state-basics"
        assert stale[0].arxiv_id is None


class TestReaping:
    async def test_the_default_is_a_report_and_nothing_more(self) -> None:
        """Deciding the fate of rows that record what happened is not something a
        read should do by surprise."""
        run_id = await _make_run(created_hours_ago=36.0)
        async with get_session_factory()() as session:
            out = await reap_stale_runs(session, older_than_hours=6.0)
        assert out["found"] == 1 and out["reaped"] == 0
        async with get_session_factory()() as session:
            run = await session.get(IngestionRun, run_id)
            assert run is not None
            assert run.status == RunStatus.RUNNING.value

    async def test_reaping_marks_the_run_terminal_and_says_why(self) -> None:
        run_id = await _make_run(arxiv_id="2302.07175", created_hours_ago=36.0)
        async with get_session_factory()() as session:
            out = await reap_stale_runs(session, older_than_hours=6.0, dry_run=False)
        assert out["reaped"] == 1
        async with get_session_factory()() as session:
            run = await session.get(IngestionRun, run_id)
            assert run is not None
            assert run.status == RunStatus.ABANDONED.value
            assert RunStatus(run.status).is_terminal
            # The error has to say what to do about it, not just that it happened.
            assert "2302.07175" in (run.error or "")
            assert "resume" in (run.error or "")

    async def test_abandoned_is_distinct_from_failed(self) -> None:
        """Nothing went wrong with the paper; the work stopped. A failed run is
        worth reading the error for, an abandoned one only needs re-running."""
        await _make_run(created_hours_ago=36.0)
        async with get_session_factory()() as session:
            await reap_stale_runs(session, older_than_hours=6.0, dry_run=False)
        async with get_session_factory()() as session:
            rows = (await session.execute(select(IngestionRun))).scalars().all()
        assert {r.status for r in rows} == {RunStatus.ABANDONED.value}

    async def test_a_reaped_run_is_not_stale_a_second_time(self) -> None:
        await _make_run(created_hours_ago=36.0)
        async with get_session_factory()() as session:
            await reap_stale_runs(session, older_than_hours=6.0, dry_run=False)
        async with get_session_factory()() as session:
            assert await find_stale_runs(session, older_than_hours=6.0) == []

    async def test_duplicate_attempts_are_grouped_by_what_to_rerun(self) -> None:
        """40 stuck rows can be 3 papers: a retry creates a new run and leaves the
        old one where it was. 'Re-run this' is a question about targets."""
        await _make_run(arxiv_id="2302.07175", doc_key="2302.07175", created_hours_ago=37.0)
        await _make_run(arxiv_id="2302.07175", doc_key="2302.07175", created_hours_ago=36.0)
        await _make_run(arxiv_id="1006.5291", doc_key="1006.5291", created_hours_ago=16.0)
        async with get_session_factory()() as session:
            out = await reap_stale_runs(session, older_than_hours=6.0)
        assert out["found"] == 3
        assert [(row["target"], row["stuck_runs"]) for row in out["targets"]] == [
            ("2302.07175", 2),
            ("1006.5291", 1),
        ]

    async def test_a_step_left_open_is_closed(self) -> None:
        """A row with no `finished_at` is what makes a run look like it is still
        working, so the reap has to close it or the report still lies."""
        run_id = await _make_run(created_hours_ago=36.0)
        async with get_session_factory()() as session:
            run = await session.get(IngestionRun, run_id)
            assert run is not None
            run.steps.append(
                PipelineStepRun(
                    run_id=run_id,
                    name="extract_text",
                    attempt=1,
                    status="running",
                    meta={},
                    started_at=_hours_ago(36.0),
                    finished_at=None,
                )
            )
            await session.commit()
        async with get_session_factory()() as session:
            await reap_stale_runs(session, older_than_hours=6.0, dry_run=False)
        async with get_session_factory()() as session:
            rows = (
                await session.execute(
                    select(PipelineStepRun).where(PipelineStepRun.run_id == run_id)
                )
            ).scalars().all()
        assert rows[0].finished_at is not None

    async def test_reaping_nothing_is_not_an_error(self) -> None:
        async with get_session_factory()() as session:
            out = await reap_stale_runs(session, older_than_hours=6.0)
        assert out == {"found": 0, "reaped": 0, "targets": [], "runs": [], "dry_run": True}


class TestSkipOrResume:
    """An already-ingested document is refused as a duplicate — unless the last
    attempt at it did not finish, in which case the retry is the way forward.

    Without this, a book whose run was interrupted could only be re-ingested with
    ``--force``, which throws away the stored markdown and pays MinerU again:
    ~96s to extract against ~18ms to reload, on the one operation nobody wants to
    repeat by hand.
    """

    @pytest.mark.parametrize(
        "status", [RunStatus.RUNNING.value, RunStatus.PENDING.value, RunStatus.PARTIAL.value]
    )
    def test_an_unfinished_previous_run_resumes(self, status: str) -> None:
        assert resume_instead_of_skip(status) is True

    @pytest.mark.parametrize(
        "status", [RunStatus.SUCCEEDED.value, RunStatus.SKIPPED.value]
    )
    def test_a_finished_previous_run_is_skipped(self, status: str) -> None:
        assert resume_instead_of_skip(status) is False

    def test_a_failed_run_resumes_rather_than_skipping(self) -> None:
        """A failed run means the work is not done. Treating it as finished would
        leave a permanently half-ingested document that can only be retried the
        expensive way."""
        assert resume_instead_of_skip(RunStatus.FAILED.value) is True

    def test_an_abandoned_run_resumes(self) -> None:
        assert resume_instead_of_skip(RunStatus.ABANDONED.value) is True

    def test_no_previous_run_resumes(self) -> None:
        """Nothing to collide with, so the only question is which path to take."""
        assert resume_instead_of_skip(None) is True

    async def test_the_latest_run_for_a_target_is_found(self) -> None:
        first = await _make_run(arxiv_id="2302.07175", doc_key="2302.07175",
                                created_hours_ago=10.0)
        second = await _make_run(arxiv_id="2302.07175", doc_key="2302.07175",
                                 created_hours_ago=1.0)
        async with get_session_factory()() as session:
            found = await RunRepository(session).latest_for_target("2302.07175")
        assert found is not None and found.id == second and found.id != first

    async def test_a_document_is_found_by_its_doc_key(self) -> None:
        await _make_run(arxiv_id=None, doc_key="solid-state-basics", created_hours_ago=1.0)
        async with get_session_factory()() as session:
            found = await RunRepository(session).latest_for_target("solid-state-basics")
        assert found is not None and found.arxiv_id is None


class TestRecordingTheSameContentTwice:
    """A content-addressed row must still learn new metadata.

    The derived block list was written on the first run and silently dropped on
    every later one: the markdown hash matched, the existing row was returned,
    and the new `meta` went with it. That made a document's pages and sections
    unrecoverable on resume even though the extraction had produced them.
    """

    async def test_new_metadata_survives_a_repeat_record(self, session, blob_store) -> None:
        from app.db.repositories import RawDocumentRepository  # noqa: PLC0415

        paper = Paper(arxiv_id="1706.03762", doc_key="1706.03762", title="T", abstract="")
        session.add(paper)
        await session.commit()
        paper_id = paper.id

        blob = blob_store.put_bytes(b"# same markdown", prefix="derived")
        repo = RawDocumentRepository(session)
        first = await repo.record(
            paper_id=paper_id,
            kind=ContentKind.MINERU_MARKDOWN,
            uri=blob.uri,
            content_type="text/markdown",
            size_bytes=blob.size_bytes,
            sha256=blob.sha256,
            meta={"source": "pdf_mineru"},
        )
        await session.commit()
        second = await repo.record(
            paper_id=paper_id,
            kind=ContentKind.MINERU_MARKDOWN,
            uri=blob.uri,
            content_type="text/markdown",
            size_bytes=blob.size_bytes,
            sha256=blob.sha256,
            meta={"source": "pdf_mineru", "blocks_sha256": "b" * 64, "blocks_uri": "x"},
        )
        await session.commit()

        assert first.id == second.id
        assert second.meta is not None
        assert second.meta["blocks_sha256"] == "b" * 64

    async def test_an_older_fact_is_not_erased_by_a_later_record(self, session, blob_store) -> None:
        from app.db.repositories import RawDocumentRepository  # noqa: PLC0415

        paper = Paper(arxiv_id="1706.03762", doc_key="1706.03762", title="T", abstract="")
        session.add(paper)
        await session.commit()
        paper_id = paper.id
        blob = blob_store.put_bytes(b"# same markdown", prefix="derived")
        repo = RawDocumentRepository(session)
        await repo.record(
            paper_id=paper_id,
            kind=ContentKind.MINERU_MARKDOWN,
            uri=blob.uri,
            content_type="text/markdown",
            size_bytes=blob.size_bytes,
            sha256=blob.sha256,
            meta={"backend": "mineru:cli:v2", "chars": 10},
        )
        await session.commit()
        await repo.record(
            paper_id=paper_id,
            kind=ContentKind.MINERU_MARKDOWN,
            uri=blob.uri,
            content_type="text/markdown",
            size_bytes=blob.size_bytes,
            sha256=blob.sha256,
            meta={"blocks_sha256": "c" * 64},
        )
        await session.commit()
        rows = await repo.list_for_paper(paper_id)
        assert rows[0].meta is not None
        assert rows[0].meta["backend"] == "mineru:cli:v2"
        assert rows[0].meta["blocks_sha256"] == "c" * 64


class TestResumingAnExtraction:
    """A retry must reuse the interrupted run's work, not redo it."""

    @pytest.fixture
    def blobs(self, tmp_path: Path):
        """A local blob store, not the in-memory one.

        The extraction path locates its source blob by filesystem path, so the
        in-memory store cannot reach MinerU at all. Using it here would make every
        test pass for the wrong reason: nothing would be extracted and nothing
        would be resumed.
        """
        from app.infra.storage import LocalBlobStore  # noqa: PLC0415

        return LocalBlobStore(root=tmp_path / "blobs")


    class _CountingExtractor:
        """Records whether MinerU was reached, and returns usable markdown."""

        def __init__(self) -> None:
            self.calls = 0

        async def extract_pdf(self, *args, **kwargs):  # noqa: ANN002, ANN003
            from app.domain.enums import ContentSource  # noqa: PLC0415
            from app.domain.models import ExtractedDocument  # noqa: PLC0415

            self.calls += 1
            text = "## MinerU ran"
            return ExtractedDocument(
                markdown=text,
                source=ContentSource.PDF_MINERU,
                backend="stub",
                text=text,
            )

    @classmethod
    def _ctx(cls, session, blobs, *, force: bool = False, paper_id: str | None = None):  # noqa: ANN205
        """A context as a resumed run has it.

        The content step has already reused the stored source blob, so `content`
        is set *and* its bytes are in the store — otherwise the extraction path
        stops at "blob missing" and the test proves nothing about resumption.
        """
        from app.domain.models import ContentPayload  # noqa: PLC0415
        from app.pipeline.context import PipelineContext  # noqa: PLC0415

        source = blobs.put_bytes(b"%PDF-1.7 source", prefix="raw")
        ctx = PipelineContext(
            session=session, run_id="r", arxiv_id="1706.03762", force=force
        )
        ctx.paper_id = paper_id
        ctx.blob_store = blobs
        ctx.content = ContentPayload(
            kind=ContentKind.PDF,
            uri=source.uri,
            content_type="application/pdf",
            size_bytes=source.size_bytes,
            sha256=source.sha256,
            source_url="https://arxiv.org/pdf/1706.03762",
        )
        return ctx

    async def _store_extraction(
        self, paper_id: str, blobs, *, blocks: bool, kind: object = ContentKind.MINERU_MARKDOWN
    ) -> str:
        """What `_persist` writes, reproduced so the test does not need MinerU."""
        markdown = "# 2.3.1 Semi-classical\n\nbody text worth keeping."
        blob = blobs.put_bytes(markdown.encode("utf-8"), prefix="derived")
        meta: dict[str, object] = {"source": "pdf_mineru", "backend": "mineru:cli:v2"}
        if blocks:
            block_blob = blobs.put_bytes(
                json.dumps([{"type": "text", "page_idx": 0, "text": "body text"}]).encode(),
                prefix="derived",
            )
            meta["blocks_sha256"] = block_blob.sha256
            meta["blocks_uri"] = block_blob.uri
        async with get_session_factory()() as session:
            await RawDocumentRepository(session).record(
                paper_id=paper_id,
                kind=kind,  # type: ignore[arg-type]
                uri=blob.uri,
                content_type="text/markdown",
                size_bytes=blob.size_bytes,
                sha256=blob.sha256,
                meta=meta,
            )
            await session.commit()
        return blob.sha256

    async def test_stored_markdown_is_reloaded_instead_of_extracted(self, blobs) -> None:
        async with get_session_factory()() as session:
            paper = Paper(arxiv_id="1706.03762", doc_key="1706.03762", title="T", abstract="")
            session.add(paper)
            await session.commit()
            paper_id = paper.id

        await self._store_extraction(paper_id, blobs, blocks=True)
        extractor = self._CountingExtractor()

        async with get_session_factory()() as session:
            ctx = self._ctx(session, blobs, paper_id=paper_id)
            result = await ExtractTextStep(extractor, blobs).run(ctx)

        assert extractor.calls == 0, "MinerU must not run on a resumed extraction"
        assert result.skipped is True
        assert result.data["resumed"] is True
        assert "Semi-classical" in (ctx.document.markdown if ctx.document else "")
        # The blocks come back too, which is what keeps pages and sections working
        # after a resume. Without them the run would be text with no provenance.
        assert ctx.document is not None and ctx.document.blocks
        assert ctx.document.meta.get("resumed") is True

    async def test_force_re_extracts_everything(self, blobs) -> None:
        """`--force` is how a user asks for the work to be done again."""
        async with get_session_factory()() as session:
            paper = Paper(arxiv_id="1706.03762", doc_key="1706.03762", title="T", abstract="")
            session.add(paper)
            await session.commit()
            paper_id = paper.id
        await self._store_extraction(paper_id, blobs, blocks=True)

        extractor = self._CountingExtractor()
        async with get_session_factory()() as session:
            ctx = self._ctx(session, blobs, force=True, paper_id=paper_id)
            result = await ExtractTextStep(extractor, blobs).run(ctx)
        assert extractor.calls == 1
        assert not result.skipped

    async def test_a_lost_blob_falls_back_to_extracting(self, blobs) -> None:
        """Continuing with nothing would be worse than extracting again, so a
        missing blob returns None and lets the normal path run."""
        async with get_session_factory()() as session:
            paper = Paper(arxiv_id="1706.03762", doc_key="1706.03762", title="T", abstract="")
            session.add(paper)
            await session.commit()
            paper_id = paper.id
        # A raw document whose blob is not in the store at all.
        async with get_session_factory()() as session:
            await RawDocumentRepository(session).record(
                paper_id=paper_id,
                kind=ContentKind.MINERU_MARKDOWN,
                uri="memory://gone",
                content_type="text/markdown",
                size_bytes=10,
                sha256="0" * 64,
                meta={},
            )
            await session.commit()

        extractor = self._CountingExtractor()
        async with get_session_factory()() as session:
            ctx = self._ctx(session, blobs, paper_id=paper_id)
            result = await ExtractTextStep(extractor, blobs).run(ctx)
        assert extractor.calls == 1, "a lost blob must fall back to extracting"
        assert not result.skipped

    async def test_a_paged_document_without_stored_blocks_is_not_resumed(self, blobs) -> None:
        """Resuming markdown-only from a PDF would quietly drop every page number
        and every section — the text would look identical, so nothing would signal
        the loss. A full extraction costs minutes once and self-heals."""
        async with get_session_factory()() as session:
            paper = Paper(arxiv_id="1706.03762", doc_key="1706.03762", title="T", abstract="")
            session.add(paper)
            await session.commit()
            paper_id = paper.id
        await self._store_extraction(paper_id, blobs, blocks=False)
        extractor = self._CountingExtractor()
        async with get_session_factory()() as session:
            ctx = self._ctx(session, blobs, paper_id=paper_id)
            result = await ExtractTextStep(extractor, blobs).run(ctx)
        assert extractor.calls == 1
        assert not result.skipped

    async def test_html_is_resumed_from_markdown_alone(self, blobs) -> None:
        """HTML has no page numbers at all, so markdown-only loses nothing."""
        async with get_session_factory()() as session:
            paper = Paper(arxiv_id="1706.03762", doc_key="1706.03762", title="T", abstract="")
            session.add(paper)
            await session.commit()
            paper_id = paper.id
        await self._store_extraction(paper_id, blobs, blocks=False, kind=ContentKind.TEXT)
        extractor = self._CountingExtractor()
        async with get_session_factory()() as session:
            ctx = self._ctx(session, blobs, paper_id=paper_id)
            # An HTML source, which is the point of the comparison.
            ctx.content = replace(ctx.content, kind=ContentKind.HTML)  # type: ignore[union-attr]
            result = await ExtractTextStep(extractor, blobs).run(ctx)
        assert extractor.calls == 0
        assert result.skipped

    async def test_a_document_with_no_stored_extraction_is_not_resumed(self, blobs) -> None:
        async with get_session_factory()() as session:
            paper = Paper(arxiv_id="1706.03762", doc_key="1706.03762", title="T", abstract="")
            session.add(paper)
            await session.commit()
            paper_id = paper.id
        extractor = self._CountingExtractor()
        async with get_session_factory()() as session:
            ctx = self._ctx(session, blobs, paper_id=paper_id)
            result = await ExtractTextStep(extractor, blobs).run(ctx)
        assert extractor.calls == 1
        assert not result.skipped
