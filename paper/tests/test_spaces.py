"""Embedding spaces: one model, one table.

These tests pin the core promise of the design — two models coexist, neither
disturbs the other, and vectors never mix across an index.
"""

from __future__ import annotations

import pytest
from sqlalchemy import func, select

from app.db.space_repository import EmbeddingSpaceRepository, SpaceConflictError
from app.db.spaces import (
    DEFAULT_SPACE_NAME,
    LEGACY_TABLE_NAME,
    EmbeddingSpace,
    assert_distinct,
    slugify,
)
from app.db.vector_store.memory_store import InMemoryVectorStore
from app.db.vector_store.schema import (
    ddl_statements,
    embedding_table,
    ensure_space_table,
    known_tables,
)
from app.domain.models import VectorRecord


@pytest.fixture
def small() -> EmbeddingSpace:
    return EmbeddingSpace(
        name="small-384", provider="openai", model="text-embedding-3-small", dimensions=384
    )


@pytest.fixture
def large() -> EmbeddingSpace:
    return EmbeddingSpace(
        name="large-1536", provider="openai", model="text-embedding-3-large", dimensions=1536
    )


class TestNaming:
    def test_default_keeps_the_legacy_table(self) -> None:
        """Existing deployments must upgrade without moving rows."""
        space = EmbeddingSpace(
            name=DEFAULT_SPACE_NAME, provider="p", model="m", dimensions=1536
        )
        assert space.resolved_table == LEGACY_TABLE_NAME

    def test_named_spaces_get_their_own_table(self, small, large) -> None:
        assert small.resolved_table == "embeddings__small_384"
        assert large.resolved_table == "embeddings__large_1536"
        assert small.resolved_table != large.resolved_table

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("BGE-M3 (1024d)", "bge_m3_1024d"),
            ("small  384", "small_384"),
            ("--weird--", "weird"),
            ("0dim", "n0dim"),
        ],
    )
    def test_slugify(self, value: str, expected: str) -> None:
        assert slugify(value) == expected

    def test_slugify_rejects_unusable_names(self) -> None:
        with pytest.raises(ValueError):
            slugify("---")

    def test_fingerprint_identifies_the_model(self, small) -> None:
        assert small.fingerprint == "openai:text-embedding-3-small:384"

    def test_explicit_table_name_wins(self) -> None:
        space = EmbeddingSpace(
            name="weird", provider="p", model="m", dimensions=64, table_name="custom_tbl"
        )
        assert space.resolved_table == "custom_tbl"

    def test_rejects_bad_definitions(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            EmbeddingSpace(name=" ", provider="p", model="m", dimensions=64)
        with pytest.raises(ValueError, match="must be positive"):
            EmbeddingSpace(name="a", provider="p", model="m", dimensions=0)
        with pytest.raises(ValueError, match="unknown distance"):
            EmbeddingSpace(name="a", provider="p", model="m", dimensions=64, distance="nope")

    def test_table_collision_is_detected(self) -> None:
        a = EmbeddingSpace(name="x", provider="p", model="m", dimensions=64, table_name="t")
        b = EmbeddingSpace(name="y", provider="p", model="m", dimensions=64, table_name="t")
        with pytest.raises(ValueError, match="both map to table"):
            assert_distinct([a, b])
        assert_distinct([a, EmbeddingSpace(name="z", provider="p", model="m", dimensions=64)])


class TestTableDefinition:
    def test_columns_and_indexes(self, small) -> None:
        table = embedding_table(small)
        assert [c.name for c in table.columns] == [
            "id", "chunk_id", "paper_id", "provider", "model",
            "dimensions", "fingerprint", "vector", "meta", "created_at",
        ]
        index_names = {index.name for index in table.indexes}
        assert f"uq_{small.resolved_table}_chunk" in index_names
        assert f"ix_{small.resolved_table}_hnsw" in index_names

    def test_definitions_are_cached(self, small) -> None:
        assert embedding_table(small) is embedding_table(small)

    def test_hnsw_opclass_follows_distance(self) -> None:
        cosine = EmbeddingSpace(
            name="c", provider="p", model="m", dimensions=8, distance="cosine"
        )
        euclid = EmbeddingSpace(
            name="e", provider="p", model="m", dimensions=8, distance="euclid"
        )
        assert "vector_cosine_ops" in "\n".join(ddl_statements(cosine, "postgresql"))
        assert "vector_l2_ops" in "\n".join(ddl_statements(euclid, "postgresql"))

    def test_width_is_locked_to_the_space(self, small, large) -> None:
        """Two spaces must never share a column: the widths differ."""
        assert embedding_table(small) is not embedding_table(large)
        assert embedding_table(small).name != embedding_table(large).name


class TestPostgresDdl:
    def test_hnsw_index_uses_vector_opclass(self) -> None:
        space = EmbeddingSpace(
            name="pg-384", provider="openai", model="m", dimensions=384
        )
        statements = ddl_statements(space, "postgresql")
        joined = "\n".join(statements)
        assert "USING hnsw" in joined
        assert "vector_cosine_ops" in joined
        assert "CREATE TABLE IF NOT EXISTS embeddings__pg_384" in joined

    def test_sqlite_ddl_has_no_postgres_syntax(self) -> None:
        space = EmbeddingSpace(name="lite-8", provider="p", model="m", dimensions=8)
        joined = "\n".join(ddl_statements(space, "sqlite"))
        assert "USING hnsw" not in joined
        assert "vector_cosine_ops" not in joined

    def test_known_tables_lists_cached(self, small) -> None:
        embedding_table(small)
        assert small.resolved_table in known_tables()


class TestSpaceRegistry:
    async def test_create_and_resolve(self, session) -> None:
        repo = EmbeddingSpaceRepository(session)
        small = EmbeddingSpace(name="small", provider="p", model="m", dimensions=64)
        await repo.create(small, is_active=True)

        assert (await repo.resolve("small")).name == "small"
        assert (await repo.resolve()).name == "small"  # active
        assert (await repo.active()).name == "small"

    async def test_unknown_name_raises(self, session) -> None:
        with pytest.raises(SpaceConflictError, match="unknown embedding space"):
            await EmbeddingSpaceRepository(session).resolve("nope")

    async def test_create_is_idempotent_for_same_definition(self, session) -> None:
        repo = EmbeddingSpaceRepository(session)
        space = EmbeddingSpace(name="s", provider="p", model="m", dimensions=64)
        await repo.create(space)
        await repo.create(space)
        # create_all seeds the settings-derived space, so count only ours.
        assert len([r for r in await repo.list() if r.name == "s"]) == 1

    async def test_redefining_a_space_is_rejected(self, session) -> None:
        """Silently changing a model's width would corrupt the table."""
        repo = EmbeddingSpaceRepository(session)
        await repo.create(EmbeddingSpace(name="s", provider="p", model="m", dimensions=64))
        with pytest.raises(SpaceConflictError, match="already exists with a different"):
            await repo.create(
                EmbeddingSpace(name="s", provider="p", model="m2", dimensions=128)
            )

    async def test_only_one_active_space(self, session) -> None:
        repo = EmbeddingSpaceRepository(session)
        a = EmbeddingSpace(name="a", provider="p", model="m", dimensions=64)
        b = EmbeddingSpace(name="b", provider="p", model="m", dimensions=128)
        await repo.create(a, is_active=True)
        await repo.create(b, is_active=True)

        actives = [r.name for r in await repo.list() if r.is_active]
        assert actives == ["b"]

    async def test_table_collision_rejected(self, session) -> None:
        repo = EmbeddingSpaceRepository(session)
        await repo.create(
            EmbeddingSpace(name="a", provider="p", model="m", dimensions=64, table_name="shared")
        )
        with pytest.raises(SpaceConflictError, match="already used by space"):
            await repo.create(
                EmbeddingSpace(
                    name="b", provider="p", model="m", dimensions=64, table_name="shared"
                )
            )

    async def test_locked_space_cannot_be_deleted(self, session) -> None:
        repo = EmbeddingSpaceRepository(session)
        await repo.create(
            EmbeddingSpace(name="managed", provider="p", model="m", dimensions=64),
            is_locked=True,
        )
        with pytest.raises(SpaceConflictError, match="managed by settings"):
            await repo.delete("managed")

    async def test_delete_removes_unlocked(self, session) -> None:
        repo = EmbeddingSpaceRepository(session)
        await repo.create(EmbeddingSpace(name="tmp", provider="p", model="m", dimensions=64))
        assert await repo.delete("tmp") is True
        assert await repo.get("tmp") is None

    async def test_fallback_used_when_registry_empty(self, session) -> None:
        """create_all seeds the registry, so clear it to exercise this path."""
        from sqlalchemy import delete

        from app.db.models import EmbeddingSpaceRecord

        repo = EmbeddingSpaceRepository(session)
        fallback = EmbeddingSpace(name="default", provider="p", model="m", dimensions=64)

        await session.execute(delete(EmbeddingSpaceRecord))
        resolved = await repo.resolve(None, fallback=fallback)
        assert resolved.name == "default"

    async def test_resolve_without_fallback_or_registry_raises(self, session) -> None:
        from sqlalchemy import delete

        from app.db.models import EmbeddingSpaceRecord

        await session.execute(delete(EmbeddingSpaceRecord))
        with pytest.raises(SpaceConflictError, match="no embedding space registered"):
            await EmbeddingSpaceRepository(session).resolve()


class TestStoreIsolation:
    def test_memory_stores_do_not_share_vectors(self) -> None:
        small = EmbeddingSpace(name="s", provider="p", model="m", dimensions=4)
        large = EmbeddingSpace(name="l", provider="p", model="m", dimensions=8)
        a, b = InMemoryVectorStore(small), InMemoryVectorStore(large)
        assert a is not b
        assert a.space is small and b.space is large

    async def test_wrong_width_is_rejected_per_space(self) -> None:
        space = EmbeddingSpace(name="s", provider="p", model="m", dimensions=8)
        store = InMemoryVectorStore(space)
        with pytest.raises(ValueError, match="space 's' expects 8-dim"):
            await store.upsert([VectorRecord(chunk_id="a", paper_id="p", vector=[1.0, 2.0])])

    async def test_search_rejects_wrong_width_query(self) -> None:
        space = EmbeddingSpace(name="s", provider="p", model="m", dimensions=8)
        store = InMemoryVectorStore(space)
        with pytest.raises(ValueError, match="expects 8-dim vectors"):
            await store.search([1.0, 0.0])


class TestTableCreation:
    """`run_sync` needs a Connection: `Session.run_sync` would pass the Session."""

    async def test_ensure_creates_then_reports_absent(self, session, small) -> None:
        connection = await session.connection()
        created = await connection.run_sync(lambda c: ensure_space_table(c, small))
        assert created is True
        again = await connection.run_sync(lambda c: ensure_space_table(c, small))
        assert again is False

    async def test_created_table_is_queryable(self, session, small) -> None:
        connection = await session.connection()
        await connection.run_sync(lambda c: ensure_space_table(c, small))
        await session.commit()
        table = embedding_table(small)
        count = await session.scalar(select(func.count()).select_from(table))
        assert count == 0