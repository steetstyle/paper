"""Migration tests against a real PostgreSQL.

`create_all` builds the schema from the models, so it cannot catch a migration
that disagrees with them — which is exactly where the bugs were. These run the
real Alembic chain, then assert the resulting schema through SQLAlchemy's
inspector.

Also covers SQLite, because the SQLite branch of a migration is a different
program from the PostgreSQL one and is just as easy to get wrong.

PostgreSQL tests need `TEST_PG_URL`; SQLite ones always run.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import make_url

from paper_app.db.models import Base

BACKEND_ROOT = Path(__file__).resolve().parents[1]
TEST_PG_URL = os.environ.get("TEST_PG_URL", "")


@contextmanager
def alembic_config(database_url: str, monkeypatch: pytest.MonkeyPatch) -> Iterator[Config]:
    """An Alembic config pointed at ``database_url``, not at ``.env``.

    ``command.upgrade`` takes no options, and ``migrations/env.py`` reads the
    URL from either settings or this variable, so the override has to be in the
    environment for the duration of the call.
    """
    monkeypatch.setenv("MIGRATION_DATABASE_URL", database_url)
    config = Config(str(BACKEND_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(BACKEND_ROOT / "migrations"))
    yield config


@pytest.fixture
def sqlite_url() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        yield f"sqlite:///{Path(tmp) / 'migrate.db'}"


@pytest.fixture
def pg_url() -> str:
    if not TEST_PG_URL:
        pytest.skip("set TEST_PG_URL to run migration tests on PostgreSQL")
    # A dedicated database: the migration drops and recreates the public schema.
    # The sync driver is required — create_engine rejects an asyncpg URL.
    url = make_url(TEST_PG_URL).set(drivername="postgresql+psycopg")
    name = "paper_migrate_test"
    _recreate_database(url, name)
    yield url.set(database=name).render_as_string(hide_password=False)
    _recreate_database(url, name)


def _recreate_database(url: object, name: str) -> None:
    """(Re)create ``name`` from scratch.

    CREATE/DROP DATABASE cannot run inside a transaction block, so autocommit is
    required — ``engine.begin()`` would fail with ActiveSqlTransaction.
    """
    engine = create_engine(url, isolation_level="AUTOCOMMIT")
    try:
        with engine.connect() as conn:
            conn.execute(text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
            conn.execute(text(f'CREATE DATABASE "{name}"'))
    finally:
        engine.dispose()


def drop_everything(database_url: str) -> None:
    """Empty the schema so `upgrade head` starts from nothing.

    Uses CASCADE because the per-space vector tables are created at runtime and
    are not in Base.metadata, so drop_all cannot remove them.
    """
    engine = create_engine(database_url)
    try:
        with engine.begin() as conn:
            conn.execute(text("DROP SCHEMA public CASCADE"))
            conn.execute(text("CREATE SCHEMA public"))
    finally:
        engine.dispose()


def inspector_for(database_url: str) -> object:
    engine = create_engine(database_url)
    try:
        with engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()
    finally:
        engine.dispose()
    return create_engine(database_url)


class TestUpgradeHead:
    """`upgrade head` must produce a schema matching the models."""

    def test_sqlite_upgrade_head_succeeds(self, sqlite_url: str, monkeypatch: pytest.MonkeyPatch) -> None:
        with alembic_config(sqlite_url, monkeypatch) as config:
            command.upgrade(config, "head")

    def test_sqlite_creates_every_table(self, sqlite_url: str, monkeypatch: pytest.MonkeyPatch) -> None:
        with alembic_config(sqlite_url, monkeypatch) as config:
            command.upgrade(config, "head")
        engine = create_engine(sqlite_url)
        try:
            names = set(inspect(engine).get_table_names())
        finally:
            engine.dispose()
        for table in (
            "papers",
            "raw_documents",
            "chunks",
            "embeddings",
            "ingestion_runs",
            "pipeline_step_runs",
            "embedding_spaces",
            "embedding_jobs",
            "authors",
            "paper_authors",
            "categories",
            "paper_categories",
            "paper_references",
            "projects",
            "project_papers",
        ):
            assert table in names, f"{table} missing after upgrade head"

    def test_sqlite_papers_stores_no_json_lists(
        self, sqlite_url: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Authors and categories are tables from the start, never columns."""
        with alembic_config(sqlite_url, monkeypatch) as config:
            command.upgrade(config, "head")
        engine = create_engine(sqlite_url)
        try:
            columns = {c["name"] for c in inspect(engine).get_columns("papers")}
        finally:
            engine.dispose()
        assert "authors_json" not in columns
        assert "categories" not in columns
        assert "primary_category" in columns

    def test_sqlite_junction_columns_match_the_models(self, sqlite_url: str, monkeypatch: pytest.MonkeyPatch) -> None:
        with alembic_config(sqlite_url, monkeypatch) as config:
            command.upgrade(config, "head")
        engine = create_engine(sqlite_url)
        try:
            inspector = inspect(engine)
            for table, expected in [
                ("paper_authors", {"id", "paper_id", "author_id", "ordinal", "affiliation"}),
                ("paper_categories", {"id", "paper_id", "category", "ordinal", "is_primary"}),
                ("categories", {"code", "label", "parent", "depth", "created_at"}),
            ]:
                actual = {c["name"] for c in inspector.get_columns(table)}
                assert actual == expected, f"{table}: {actual ^ expected}"
        finally:
            engine.dispose()

    def test_sqlite_vector_column_is_json_not_native(self, sqlite_url: str, monkeypatch: pytest.MonkeyPatch) -> None:
        """SQLite has no vector type; it stores JSON. The model says so too."""
        with alembic_config(sqlite_url, monkeypatch) as config:
            command.upgrade(config, "head")
        engine = create_engine(sqlite_url)
        try:
            columns = {c["name"]: c for c in inspect(engine).get_columns("embeddings")}
        finally:
            engine.dispose()
        assert str(columns["vector"]["type"]).upper().startswith("JSON")


class TestUpgradeAndDowngrade:
    def test_sqlite_full_round_trip(self, sqlite_url: str, monkeypatch: pytest.MonkeyPatch) -> None:
        with alembic_config(sqlite_url, monkeypatch) as config:
            command.upgrade(config, "head")
            command.downgrade(config, "base")
        # Back to a clean slate, so the chain is reversible.
        engine = create_engine(sqlite_url)
        try:
            remaining = {
                name
                for name in inspect(engine).get_table_names()
                if name not in {"alembic_version", "sqlite_sequence"}
            }
        finally:
            engine.dispose()
        assert remaining == set()

    def test_downgrade_to_base_removes_the_reference_tables(
        self, sqlite_url: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        with alembic_config(sqlite_url, monkeypatch) as config:
            command.upgrade(config, "head")
            command.downgrade(config, "base")

        engine = create_engine(sqlite_url)
        try:
            tables = set(inspect(engine).get_table_names())
        finally:
            engine.dispose()
        assert "paper_references" not in tables
        assert "projects" not in tables
        assert "project_papers" not in tables


class TestPostgres:
    @pytest.mark.skipif(not TEST_PG_URL, reason="set TEST_PG_URL")
    def test_upgrade_head_succeeds(self, pg_url: str, monkeypatch: pytest.MonkeyPatch) -> None:
        drop_everything(pg_url)
        with alembic_config(pg_url, monkeypatch) as config:
            command.upgrade(config, "head")

    @pytest.mark.skipif(not TEST_PG_URL, reason="set TEST_PG_URL")
    def test_vector_column_is_native(self, pg_url: str, monkeypatch: pytest.MonkeyPatch) -> None:
        drop_everything(pg_url)
        with alembic_config(pg_url, monkeypatch) as config:
            command.upgrade(config, "head")
        engine = create_engine(pg_url)
        try:
            with engine.connect() as conn:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                conn.commit()
            columns = {
                row[0]
                for row in engine.connect().execute(
                    text(
                        "SELECT udt_name FROM information_schema.columns "
                        "WHERE table_name = 'embeddings' AND column_name = 'vector'"
                    )
                )
            }
        finally:
            engine.dispose()
        assert columns == {"vector"}

    @pytest.mark.skipif(not TEST_PG_URL, reason="set TEST_PG_URL")
    def test_json_columns_are_dropped(self, pg_url: str, monkeypatch: pytest.MonkeyPatch) -> None:
        drop_everything(pg_url)
        with alembic_config(pg_url, monkeypatch) as config:
            command.upgrade(config, "head")
        engine = create_engine(pg_url)
        try:
            columns = {c["name"] for c in inspect(engine).get_columns("papers")}
        finally:
            engine.dispose()
        assert "authors_json" not in columns
        assert "categories" not in columns

    @pytest.mark.skipif(not TEST_PG_URL, reason="set TEST_PG_URL")
    def test_hnsw_index_exists(self, pg_url: str, monkeypatch: pytest.MonkeyPatch) -> None:
        drop_everything(pg_url)
        with alembic_config(pg_url, monkeypatch) as config:
            command.upgrade(config, "head")
        engine = create_engine(pg_url)
        try:
            with engine.connect() as conn:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                conn.commit()
                definitions = " ".join(
                    row[0]
                    for row in conn.execute(
                        text(
                            "SELECT indexdef FROM pg_indexes "
                            "WHERE tablename = 'embeddings'"
                        )
                    )
                )
        finally:
            engine.dispose()
        assert "hnsw" in definitions.lower()

    @pytest.mark.skipif(not TEST_PG_URL, reason="set TEST_PG_URL")
    def test_full_round_trip(self, pg_url: str, monkeypatch: pytest.MonkeyPatch) -> None:
        drop_everything(pg_url)
        with alembic_config(pg_url, monkeypatch) as config:
            command.upgrade(config, "head")
            command.downgrade(config, "base")


class TestSchemaMatchesModels:
    """The migration and the ORM must not drift apart."""

    @pytest.mark.parametrize("database", ["sqlite"])
    def test_column_names_line_up(self, sqlite_url: str, database: str, monkeypatch: pytest.MonkeyPatch) -> None:
        with alembic_config(sqlite_url, monkeypatch) as config:
            command.upgrade(config, "head")
        engine = create_engine(sqlite_url)
        try:
            inspector = inspect(engine)
            for table in Base.metadata.sorted_tables:
                migrated = {c["name"] for c in inspector.get_columns(table.name)}
                declared = {c.name for c in table.columns}
                assert migrated == declared, (
                    f"{table.name}: migration has {migrated - declared}, "
                    f"models have {declared - migrated}"
                )
        finally:
            engine.dispose()