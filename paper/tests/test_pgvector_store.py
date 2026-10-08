"""PgVectorStore against a real PostgreSQL with pgvector.

The suite elsewhere fakes the vector store, so nothing else would notice two
bugs that only appear once a real `vector` column is scanned:

- `1.0 - distance` inferred the whole subtraction as VECTOR and handed pgvector
  a bare float ("expected list or ndarray").
- the category filter needs a real join, and the junction tables replaced the
  jsonb containment it used to do.

Skipped unless `TEST_PG_URL` points at a database with the vector extension.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL, make_url
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.db.base import Base
from app.db.models import Chunk, PaperAuthor, PaperCategory
from app.db.repositories import PaperRepository
from app.db.spaces import EmbeddingSpace
from app.db.vector_store.base import VectorFilter
from app.db.vector_store.pgvector_store import PgVectorStore
from app.domain.models import Author as AuthorSpec
from app.domain.models import PaperMetadata

# Empty string rather than None, so the type is str and mypy stays quiet; the
# skipif below is what actually gates the run.
TEST_PG_URL = os.environ.get("TEST_PG_URL", "")
pytestmark = pytest.mark.skipif(
    not TEST_PG_URL,
    reason="set TEST_PG_URL to a postgresql+asyncpg:// url with pgvector",
)

SPACE = EmbeddingSpace(
    name="test-space",
    provider="hashing",
    model="hashing-test",
    dimensions=8,
    distance="cosine",
)


def unit(seed: int) -> list[float]:
    """A distinct 8-dimension unit-ish vector per seed."""
    raw = [float((seed * (i + 3)) % 11) for i in range(8)]
    norm = sum(value * value for value in raw) ** 0.5 or 1.0
    return [value / norm for value in raw]


def _recreate_database(url: URL, name: str) -> None:
    """(Re)create ``name`` from scratch.

    CREATE/DROP DATABASE cannot run inside a transaction block, so autocommit is
    required — ``engine.begin()`` fails with ActiveSqlTransaction.
    """
    engine = create_engine(url, isolation_level="AUTOCOMMIT")
    try:
        with engine.connect() as conn:
            conn.execute(text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
            conn.execute(text(f'CREATE DATABASE "{name}"'))
    finally:
        engine.dispose()


@pytest.fixture
def pg_url() -> Iterator[str]:
    """A dedicated throwaway database.

    Must not be the developer's own: the fixture drops and recreates the public
    schema, which would silently delete a real corpus.
    """
    if not TEST_PG_URL:
        pytest.skip("set TEST_PG_URL to run pgvector tests")
    url = make_url(TEST_PG_URL).set(drivername="postgresql+psycopg")
    name = "paper_pgvector_test"
    _recreate_database(url, name)
    yield url.set(database=name).render_as_string(hide_password=False)
    _recreate_database(url, name)


@pytest.fixture
async def factory(pg_url: str) -> Any:  # noqa: ANN201
    engine = create_async_engine(pg_url)
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    # Metadata.create_all is sync-only, and engine.sync_engine would reuse the
    # async pool's greenlet context (MissingGreenlet on close), so a separate
    # sync engine is used against the same database.
    sync_engine = create_engine(make_url(pg_url).set(drivername="postgresql+psycopg"))
    try:
        Base.metadata.create_all(sync_engine)
    finally:
        sync_engine.dispose()
    maker = async_sessionmaker(engine, expire_on_commit=False)
    await PgVectorStore(maker, SPACE).ensure_ready()
    try:
        yield maker
    finally:
        await engine.dispose()


def paper_metadata(arxiv_id: str, categories: tuple[str, ...], primary: str) -> PaperMetadata:
    return PaperMetadata(
        arxiv_id=arxiv_id,
        versioned_id=f"{arxiv_id}v1",
        version=1,
        title=f"Paper {arxiv_id}",
        abstract="An abstract.",
        authors=(AuthorSpec(name="Ashish Vaswani"),),
        categories=categories,
        primary_category=primary,
        published_at=datetime(2020, 1, 1, tzinfo=UTC),
        updated_at=datetime(2023, 1, 1, tzinfo=UTC),
        abs_url=f"https://arxiv.org/abs/{arxiv_id}",
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
        html_url=None,
    )


async def seed_paper(  # noqa: ANN202
    factory: Any,
    arxiv_id: str,
    categories: tuple[str, ...],
    primary: str,
    seed: int = 1,
) -> tuple[str, str]:
    async with factory() as session:
        repo = PaperRepository(session)
        paper = await repo.upsert(paper_metadata(arxiv_id, categories, primary))
        chunk = Chunk(
            paper_id=paper.id,
            ordinal=0,
            text=f"chunk for {arxiv_id}",
            token_count=3,
            content_hash=f"hash-{arxiv_id}",
        )
        session.add(chunk)
        # Commit before writing the vector: the store opens its own session, so
        # the chunk row must be visible outside this transaction or the FK fails.
        await session.commit()

        from app.domain.models import VectorRecord

        store = PgVectorStore(factory, SPACE)
        await store.upsert(
            [
                VectorRecord(
                    chunk_id=chunk.id,
                    paper_id=paper.id,
                    vector=unit(seed),
                    payload={},
                )
            ]
        )
        return paper.id, chunk.id


class TestSearch:
    async def test_scan_returns_ranked_chunks(self, factory) -> None:  # noqa: ANN001
        _, first = await seed_paper(factory, "2401.00001", ("cs.CL",), "cs.CL", 1)
        _, second = await seed_paper(factory, "2401.00002", ("cs.LG",), "cs.LG", 7)

        store = PgVectorStore(factory, SPACE)
        results = await store.search(unit(1), top_k=2)

        assert [chunk_id for chunk_id, _ in results] == [first, second]

    async def test_scores_are_floats_between_zero_and_one(self, factory) -> None:  # noqa: ANN001
        await seed_paper(factory, "2401.00001", ("cs.CL",), "cs.CL", 1)
        results = await PgVectorStore(factory, SPACE).search(unit(1), top_k=1)
        assert isinstance(results[0][1], float)
        assert 0.0 <= results[0][1] <= 1.0001

    async def test_min_score_filters(self, factory) -> None:  # noqa: ANN001
        await seed_paper(factory, "2401.00001", ("cs.CL",), "cs.CL", 1)
        await seed_paper(factory, "2401.00002", ("cs.LG",), "cs.LG", 9)
        store = PgVectorStore(factory, SPACE)
        near = await store.search(unit(1), top_k=5, min_score=0.99)
        assert len(near) == 1


class TestCategoryFilter:
    """The filter moved from a jsonb operator to a join on the junction table."""

    async def test_matches_only_the_requested_category(self, factory) -> None:  # noqa: ANN001
        await seed_paper(factory, "2401.00001", ("cs.CL", "cs.LG"), "cs.CL")
        await seed_paper(factory, "2401.00002", ("cs.AI",), "cs.AI")
        results = await PgVectorStore(factory, SPACE).search(
            unit(1), top_k=5, filters=VectorFilter(categories=["cs.LG"])
        )
        assert len(results) == 1

    async def test_a_paper_with_many_categories_is_not_duplicated(self, factory) -> None:  # noqa: ANN001
        """A join would return the chunk once per matching category row."""
        await seed_paper(
            factory, "2401.00001", ("cs.CL", "cs.LG", "stat.ML"), "cs.CL"
        )
        results = await PgVectorStore(factory, SPACE).search(
            unit(1),
            top_k=5,
            filters=VectorFilter(categories=["cs.CL", "cs.LG", "stat.ML"]),
        )
        assert len(results) == 1

    async def test_substring_categories_do_not_leak(self, factory) -> None:  # noqa: ANN001
        """`cs.LG` must not match `cs.LG.MS` — the old LIKE form did."""
        await seed_paper(factory, "2401.00001", ("cs.LG.MS",), "cs.LG.MS")
        results = await PgVectorStore(factory, SPACE).search(
            unit(1), top_k=5, filters=VectorFilter(categories=["cs.LG"])
        )
        assert results == []

    async def test_multiple_categories_are_a_union(self, factory) -> None:  # noqa: ANN001
        await seed_paper(factory, "2401.00001", ("cs.CL",), "cs.CL")
        await seed_paper(factory, "2401.00002", ("cs.AI",), "cs.AI")
        await seed_paper(factory, "2401.00003", ("cs.LG",), "cs.LG")
        results = await PgVectorStore(factory, SPACE).search(
            unit(1), top_k=5, filters=VectorFilter(categories=["cs.CL", "cs.AI"])
        )
        assert len(results) == 2


class TestPaperIdFilter:
    async def test_restricts_to_the_named_papers(self, factory) -> None:  # noqa: ANN001
        keep, keep_chunk = await seed_paper(factory, "2401.00001", ("cs.CL",), "cs.CL", 1)
        await seed_paper(factory, "2401.00002", ("cs.CL",), "cs.CL", 2)
        results = await PgVectorStore(factory, SPACE).search(
            unit(1), top_k=5, filters=VectorFilter(paper_ids=[keep])
        )
        assert [chunk_id for chunk_id, _ in results] == [keep_chunk]


class TestSchema:
    async def test_persists_relationally(self, factory) -> None:  # noqa: ANN001
        assert PgVectorStore(factory, SPACE).persists_relationally is True

    async def test_count_is_visible(self, factory) -> None:  # noqa: ANN001
        store = PgVectorStore(factory, SPACE)
        await seed_paper(factory, "2401.00001", ("cs.CL",), "cs.CL", 1)
        assert await store.count() == 1

    async def test_dimension_mismatch_is_refused_before_the_scan(self, factory) -> None:  # noqa: ANN001
        store = PgVectorStore(factory, SPACE)
        with pytest.raises(ValueError):
            await store.search([0.0] * 7, top_k=1)

    async def test_native_vector_column_type(self, factory) -> None:  # noqa: ANN001
        """A JSON column would silently skip the HNSW index."""
        async with factory() as session:
            rows = await session.execute(
                text(
                    "SELECT udt_name FROM information_schema.columns "
                    "WHERE table_name = :t AND column_name = 'vector'"
                ),
                {"t": SPACE.resolved_table},
            )
            assert rows.scalar() == "vector"

    async def test_hnsw_index_exists(self, factory) -> None:  # noqa: ANN001
        async with factory() as session:
            rows = await session.execute(
                text("SELECT indexdef FROM pg_indexes WHERE tablename = :t"),
                {"t": SPACE.resolved_table},
            )
            definitions = " ".join(row[0] for row in rows)
            assert "hnsw" in definitions.lower()


class TestJunctionTables:
    async def test_links_are_written_by_upsert(self, factory) -> None:  # noqa: ANN001
        from sqlalchemy import func, select

        paper_id, _ = await seed_paper(factory, "2401.00001", ("cs.CL", "cs.LG"), "cs.CL")
        async with factory() as session:
            authors = await session.scalar(
                select(func.count()).select_from(PaperAuthor).where(PaperAuthor.paper_id == paper_id)
            )
            categories = await session.scalar(
                select(func.count())
                .select_from(PaperCategory)
                .where(PaperCategory.paper_id == paper_id)
            )
        assert authors == 1
        assert categories == 2

    async def test_json_columns_are_gone(self, factory) -> None:  # noqa: ANN001
        """The tables replaced them; the columns must not linger."""
        async with factory() as session:
            rows = await session.execute(
                text(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name = 'papers' "
                    "AND column_name IN ('authors_json', 'categories')"
                )
            )
            assert rows.scalars().all() == []