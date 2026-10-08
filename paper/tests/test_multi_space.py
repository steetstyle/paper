"""Two models, two tables, one corpus.

This is the behaviour the space abstraction exists for: adding a model must not
disturb the incumbent, and the two indexes must stay strictly separate.
"""

from __future__ import annotations

import pytest
from sqlalchemy import func, select
from test_pipeline_e2e import FEED, PAPER_HTML, FakeContentFetcher

from app.clients.arxiv.client import ArxivClient
from app.clients.content.html_extractor import html_to_markdown
from app.config import ChunkingSettings, get_settings
from app.db.models import Chunk
from app.db.repositories import RunRepository
from app.db.session import get_session_factory
from app.db.spaces import EmbeddingSpace
from app.db.vector_store import schema
from app.db.vector_store.memory_store import InMemoryVectorStore
from app.db.vector_store.schema import embedding_table, ensure_space_table
from app.domain.enums import ContentSource
from app.domain.models import ExtractedDocument
from app.embeddings.hashing_provider import HashingEmbeddingProvider
from app.infra.storage import LocalBlobStore
from app.pipeline.context import PipelineContext
from app.pipeline.runner import PipelineRunner
from app.pipeline.steps import (
    ChunkTextStep,
    EmbedChunksStep,
    FetchContentStep,
    FetchMetadataStep,
    FinalizeStep,
    IndexVectorsStep,
    PersistMetadataStep,
    Step,
    StepResult,
)
from app.services.chunker import ChunkingService


@pytest.fixture(autouse=True)
def _clear_schema_cache():
    schema.clear_cache()
    yield
    schema.clear_cache()


class FakeHttp:
    async def get_text(self, url: str, params=None, **kwargs):  # noqa: ANN001, ANN003
        class Response:
            status_code = 200
            content = FEED.encode()
            headers = {"content-type": "application/atom+xml"}

        return Response()


class HtmlMarkdownStep(Step):
    """Stands in for `extract_text` without needing MinerU or a content fetcher."""

    name = "extract_text"

    async def run(self, ctx: PipelineContext) -> StepResult:
        markdown = html_to_markdown(PAPER_HTML)
        ctx.document = ExtractedDocument(
            markdown=markdown,
            source=ContentSource.ARXIV_HTML,
            backend="html_to_markdown",
            text=markdown,
            meta={},
        )
        return StepResult(data={"chars": len(markdown)})


@pytest.fixture
def content_fetcher(tmp_path) -> FakeContentFetcher:  # noqa: ANN001
    return FakeContentFetcher(blob_store=LocalBlobStore(root=tmp_path / "blobs"), html=PAPER_HTML)


@pytest.fixture
def chunker() -> ChunkingService:
    return ChunkingService(
        ChunkingSettings(max_tokens=80, overlap_tokens=15, min_tokens=5),
        token_counter=HashingEmbeddingProvider(dimensions=64).count_tokens,
    )


@pytest.fixture
def small() -> EmbeddingSpace:
    return EmbeddingSpace(name="small-32", provider="hashing", model="hashing-32", dimensions=32)


@pytest.fixture
def large() -> EmbeddingSpace:
    return EmbeddingSpace(name="large-128", provider="hashing", model="hashing-128", dimensions=128)


def _provider(space: EmbeddingSpace) -> HashingEmbeddingProvider:
    return HashingEmbeddingProvider(
        model=f"hashing-{space.dimensions}", dimensions=space.dimensions
    )


async def _ingest(
    settings,  # noqa: ANN001
    space: EmbeddingSpace,
    content_fetcher,
    chunker,
    *,
    steps_override: list[Step] | None = None,
) -> PipelineContext:
    """Run the pipeline into one space and return the context."""
    store = InMemoryVectorStore(space)
    steps = steps_override or [
        FetchMetadataStep(ArxivClient(get_settings().arxiv, FakeHttp())),
        PersistMetadataStep(),
        FetchContentStep(content_fetcher),
        HtmlMarkdownStep(),
        ChunkTextStep(chunker),
        EmbedChunksStep(_provider(space), space, batch_size=8),
        IndexVectorsStep(store),
        FinalizeStep(),
    ]
    factory = get_session_factory(settings.database)
    async with factory() as session:
        run = await RunRepository(session).create(arxiv_id="1706.03762", trigger="test")
        await session.commit()
        run_id = run.id
    async with factory() as session:
        ctx = PipelineContext(session=session, run_id=run_id, arxiv_id="1706.03762")
        await PipelineRunner(steps).run(ctx)
        return ctx


async def _count(session, space: EmbeddingSpace, paper_id: str) -> int:  # noqa: ANN001
    table = embedding_table(space)
    return int(
        await session.scalar(
            select(func.count()).select_from(table).where(table.c.paper_id == paper_id)
        )
        or 0
    )


class TestTwoModelsCoexist:
    async def test_each_space_writes_its_own_table(
        self, settings, small, large, content_fetcher, chunker
    ) -> None:
        ctx_small = await _ingest(settings, small, content_fetcher, chunker)
        ctx_large = await _ingest(settings, large, content_fetcher, chunker)

        assert ctx_small.embedded_count > 0
        assert ctx_large.embedded_count > 0
        assert small.resolved_table != large.resolved_table

        factory = get_session_factory(settings.database)
        async with factory() as session:
            assert await _count(session, small, ctx_small.paper_id) == ctx_small.embedded_count
            assert await _count(session, large, ctx_small.paper_id) == ctx_large.embedded_count

    async def test_widths_differ_in_the_stored_vectors(
        self, settings, small, large, content_fetcher, chunker
    ) -> None:
        await _ingest(settings, small, content_fetcher, chunker)
        ctx_large = await _ingest(settings, large, content_fetcher, chunker)

        factory = get_session_factory(settings.database)
        async with factory() as session:
            table = embedding_table(large)
            rows = (
                await session.execute(
                    select(table.c.vector, table.c.dimensions).where(
                        table.c.paper_id == ctx_large.paper_id
                    )
                )
            ).all()
        assert rows
        for vector, dimensions in rows:
            assert dimensions == 128
            assert len(vector) == 128

    async def test_reingesting_one_space_leaves_the_other_intact(
        self, settings, small, large, content_fetcher, chunker
    ) -> None:
        """The core promise: no cross-space interference.

        Chunk ids are regenerated on re-ingest, so a naive implementation would
        orphan one space's vectors or overwrite the other's.
        """
        ctx_small = await _ingest(settings, small, content_fetcher, chunker)
        await _ingest(settings, large, content_fetcher, chunker)

        factory = get_session_factory(settings.database)
        async with factory() as session:
            before = await _count(session, large, ctx_small.paper_id)
        assert before > 0

        # Re-ingest only the small space.
        await _ingest(settings, small, content_fetcher, chunker)

        async with factory() as session:
            after_large = await _count(session, large, ctx_small.paper_id)
            after_small = await _count(session, small, ctx_small.paper_id)
        assert after_large == before, "the other space's vectors were disturbed"
        assert after_small == ctx_small.embedded_count

    async def test_each_space_records_its_own_fingerprint(
        self, settings, small, large, content_fetcher, chunker
    ) -> None:
        ctx_small = await _ingest(settings, small, content_fetcher, chunker)
        ctx_large = await _ingest(settings, large, content_fetcher, chunker)

        factory = get_session_factory(settings.database)
        seen: dict[str, set[tuple[str, str]]] = {}
        async with factory() as session:
            for space, ctx in ((small, ctx_small), (large, ctx_large)):
                table = embedding_table(space)
                rows = (
                    await session.execute(
                        select(table.c.fingerprint, table.c.model).where(
                            table.c.paper_id == ctx.paper_id
                        )
                    )
                ).all()
                seen[space.name] = set(rows)

        assert seen[small.name] == {(small.fingerprint, "hashing-32")}
        assert seen[large.name] == {(large.fingerprint, "hashing-128")}

    async def test_chunks_are_not_duplicated_across_spaces(
        self, settings, small, large, content_fetcher, chunker
    ) -> None:
        """Chunk text is model-independent, so it exists exactly once."""
        ctx_small = await _ingest(settings, small, content_fetcher, chunker)
        ctx_large = await _ingest(settings, large, content_fetcher, chunker)
        assert ctx_small.paper_id == ctx_large.paper_id

        factory = get_session_factory(settings.database)
        async with factory() as session:
            chunk_rows = int(
                await session.scalar(
                    select(func.count())
                    .select_from(Chunk)
                    .where(Chunk.paper_id == ctx_small.paper_id)
                )
                or 0
            )
        assert chunk_rows == ctx_large.embedded_count


class TestDimensionMismatch:
    async def test_provider_width_must_match_the_space(
        self, settings, small, content_fetcher, chunker
    ) -> None:
        """A space is width-locked; a mismatched provider must fail loudly."""
        wrong = HashingEmbeddingProvider(model="hashing-128", dimensions=128)
        steps = [
            FetchMetadataStep(ArxivClient(get_settings().arxiv, FakeHttp())),
            PersistMetadataStep(),
            FetchContentStep(content_fetcher),
            HtmlMarkdownStep(),
            ChunkTextStep(chunker),
            EmbedChunksStep(wrong, small, batch_size=8),
            IndexVectorsStep(InMemoryVectorStore(small)),
            FinalizeStep(),
        ]
        ctx = await _ingest(settings, small, content_fetcher, chunker, steps_override=steps)
        assert ctx.embedded_count == 0

        factory = get_session_factory(settings.database)
        async with factory() as session:
            run = await RunRepository(session).get(ctx.run_id)
            embed_step = next(s for s in run.steps if s.name == "embed_chunks")
        assert embed_step.status == "failed"
        assert "is 32d but provider" in embed_step.error


class TestSpaceCreationOnDemand:
    async def test_ensure_table_is_idempotent(self, settings, small) -> None:
        factory = get_session_factory(settings.database)
        async with factory() as session:
            connection = await session.connection()
            first = await connection.run_sync(lambda c: ensure_space_table(c, small))
            await session.commit()
        async with factory() as session:
            connection = await session.connection()
            second = await connection.run_sync(lambda c: ensure_space_table(c, small))
        assert first is True
        assert second is False

    async def test_two_spaces_get_two_tables(self, settings, small, large) -> None:
        factory = get_session_factory(settings.database)
        for space in (small, large):
            async with factory() as session:
                connection = await session.connection()
                created = await connection.run_sync(lambda c, s=space: ensure_space_table(c, s))
                await session.commit()
            assert created is True

        from sqlalchemy import inspect as sa_inspect  # noqa: PLC0415

        def _tables(conn):  # noqa: ANN001, ANN202
            # Inspector is lazy, so the IO has to happen inside this sync call.
            names = set(sa_inspect(conn).get_table_names())
            return {small.resolved_table, large.resolved_table} <= names

        async with factory() as session:
            connection = await session.connection()
            assert await connection.run_sync(_tables)