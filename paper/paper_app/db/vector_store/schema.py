"""Per-space embedding tables.

``embedding_table(space)`` returns a SQLAlchemy ``Table`` whose ``vector`` column
is width-locked to that space's dimensions and which carries a matching HNSW
index. Definitions are cached because SQLAlchemy reflects DDL into a shared
``MetaData``, and redefining the same table name would conflict.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import (
    JSON,
    Column,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Table,
)
from sqlalchemy.dialects import postgresql as pg_dialect

from paper_app.db.base import TIMESTAMP, VectorColumn
from paper_app.db.spaces import EmbeddingSpace
from paper_app.logging import get_logger

logger = get_logger(__name__)

# Deliberately separate from the ORM metadata: ``Base.metadata.create_all`` must
# never touch space tables, because they are created per model on demand.
def _fresh_metadata() -> MetaData:
    """A MetaData holding only the chunks/papers FK stubs.

    The stubs exist so `CREATE TABLE` can resolve those references. They are
    never created: every call site passes an explicit ``tables=[...]``.
    """
    metadata = MetaData()
    for target in ("chunks", "papers"):
        Table(target, metadata, Column("id", String(32), primary_key=True))
    return metadata


# Deliberately separate from the ORM metadata: `Base.metadata.create_all` must
# never touch space tables, because they are created per model on demand.
SPACE_METADATA = _fresh_metadata()

_CACHE: dict[tuple[str, int, str], Table] = {}


def embedding_table(space: EmbeddingSpace) -> Table:
    """Return (and cache) the table definition for ``space``."""
    table_name = space.resolved_table
    key = (table_name, space.dimensions, space.distance)
    if key in _CACHE:
        return _CACHE[key]

    table = Table(
        table_name,
        SPACE_METADATA,
        Column("id", String(32), primary_key=True),
        # Both FKs cascade, so re-chunking a paper clears every space at once.
        Column(
            "chunk_id",
            String(32),
            ForeignKey("chunks.id", ondelete="CASCADE"),
            nullable=False,
        ),
        Column(
            "paper_id",
            String(32),
            ForeignKey("papers.id", ondelete="CASCADE"),
            nullable=False,
        ),
        # Denormalised for auditing. The space already implies these values, but
        # they cost ~50 bytes against a multi-KB vector and keep each row
        # self-describing when debugging a mixed-model corpus.
        Column("provider", String(64), nullable=False),
        Column("model", String(255), nullable=False),
        Column("dimensions", Integer, nullable=False),
        Column("fingerprint", String(128), nullable=False),
        Column("vector", VectorColumn(space.dimensions), nullable=True),
        Column("meta", JSON, nullable=False, server_default="{}"),
        Column("created_at", TIMESTAMP, nullable=False),
    )

    table.append_constraint(Index(f"uq_{table_name}_chunk", "chunk_id", unique=True))
    table.append_constraint(Index(f"ix_{table_name}_paper", "paper_id"))
    table.append_constraint(Index(f"ix_{table_name}_fingerprint", "fingerprint"))
    # `postgresql_*` arguments are ignored by other dialects, so this renders as
    # a plain index on SQLite and a real HNSW index on PostgreSQL.
    table.append_constraint(
        Index(
            f"ix_{table_name}_hnsw",
            "vector",
            postgresql_using="hnsw",
            postgresql_ops={"vector": space.index_ops},
        )
    )

    _CACHE[key] = table
    logger.debug(
        "embedding_table_defined",
        extra={
            "table": table_name,
            "dimensions": space.dimensions,
            "distance": space.distance,
        },
    )
    return table


def ensure_space_table(connection, space: EmbeddingSpace) -> bool:  # noqa: ANN001
    """Create the space's table and index if absent. True if it was created.

    Synchronous because it is handed to ``Connection.run_sync``. Safe to call on
    every ingest: the existence check is one catalogue query, negligible next to
    the cost of embedding the paper.
    """
    from sqlalchemy import inspect as sa_inspect  # noqa: PLC0415

    table = embedding_table(space)
    inspector: Any = sa_inspect(connection)
    if inspector is not None and inspector.has_table(table.name):
        return False
    # Explicit table list: the chunks/papers stubs must never be emitted here.
    table.metadata.create_all(connection, tables=[table], checkfirst=True)
    logger.info("space_table_created", extra={"table": table.name, "space": space.name})
    return True


def clear_cache() -> None:
    """Forget every cached table definition.

    The MetaData is rebuilt as well: SQLAlchemy refuses to define the same
    table name twice within one MetaData, so clearing only the dict would make
    the next `embedding_table()` call raise.
    """
    global SPACE_METADATA  # noqa: PLW0603 - intentional module-level state
    SPACE_METADATA = _fresh_metadata()
    _CACHE.clear()


def known_tables() -> list[str]:
    return sorted({name for name, _, _ in _CACHE})


def ddl_statements(space: EmbeddingSpace, dialect_name: str = "postgresql") -> list[str]:
    """Render the CREATE statements for a space.

    Used by ``paper embedding-space show --sql`` so schema changes can be
    reviewed and applied through Alembic instead of at runtime.
    """
    from sqlalchemy.dialects import sqlite as lite_dialect  # noqa: PLC0415
    from sqlalchemy.schema import CreateIndex, CreateTable  # noqa: PLC0415

    dialect = pg_dialect.dialect() if dialect_name == "postgresql" else lite_dialect.dialect()
    table = embedding_table(space)
    statements = [str(CreateTable(table, if_not_exists=True).compile(dialect=dialect)).strip()]
    for index in sorted(table.indexes, key=lambda i: i.name or ""):
        statements.append(str(CreateIndex(index).compile(dialect=dialect)).strip() + ";")
    return statements
