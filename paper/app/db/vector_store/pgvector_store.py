"""pgvector-backed store, one table per embedding space.

Vectors live in the same database as the relational rows, so a chunk's text and
its vector share one transactional boundary. The HNSW index is created with the
table (idempotently), using the opclass that matches the space's distance.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Any

from sqlalchemy import Float, and_, delete, exists, func, inspect, or_, select, type_coerce
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.models import Chunk, Paper, PaperCategory
from app.db.spaces import EmbeddingSpace
from app.db.vector_store.base import VectorFilter, VectorStore
from app.db.vector_store.schema import embedding_table
from app.domain.models import VectorRecord
from app.logging import get_logger

logger = get_logger(__name__)

_META_KEYS = {"provider", "model", "fingerprint"}


class PgVectorStore(VectorStore):
    name = "pgvector"
    persists_relationally = True

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        space: EmbeddingSpace,
    ) -> None:
        super().__init__(space)
        self._session_factory = session_factory
        self._table = embedding_table(space)

    # ------------------------------------------------------------------ setup
    @property
    def table(self):  # noqa: ANN201 - the space's SQLAlchemy Table
        return self._table

    async def ensure_ready(self) -> None:
        """Create the space table and its HNSW index if they do not exist."""
        async with self._session_factory() as session, session.begin():
            connection = await session.connection()
            await connection.run_sync(self._create_if_missing)

    def _create_if_missing(self, connection) -> bool:  # noqa: ANN001, ANN201
        inspector: Any = inspect(connection)
        if inspector is not None and inspector.has_table(self._table.name):
            return False
        # Explicit table list: chunks/papers stubs in SPACE_METADATA must never
        # be emitted here.
        self._table.metadata.create_all(
            connection, tables=[self._table], checkfirst=True
        )
        return True

    async def table_exists(self) -> bool:
        def _has_table(conn) -> bool:  # noqa: ANN001, ANN202
            inspector = inspect(conn)
            return inspector is not None and inspector.has_table(self._table.name)

        async with self._session_factory() as session:
            connection = await session.connection()
            return bool(await connection.run_sync(_has_table))

    # ------------------------------------------------------------------ write
    async def upsert(self, records: Sequence[VectorRecord]) -> int:
        if not records:
            return 0
        rows = [
            {
                "id": uuid.uuid4().hex,
                "chunk_id": record.chunk_id,
                "paper_id": record.paper_id,
                "provider": str(record.payload.get("provider", self.space.provider)),
                "model": str(record.payload.get("model", self.space.model)),
                "dimensions": len(record.vector),
                "fingerprint": str(
                    record.payload.get("fingerprint", self.space.fingerprint)
                ),
                "vector": record.vector,
                "meta": {
                    k: v for k, v in record.payload.items() if k not in _META_KEYS
                },
                "created_at": _now(),
            }
            for record in records
        ]

        async with self._session_factory() as session, session.begin():
            chunk_ids = [record.chunk_id for record in records]
            existing = await session.execute(
                select(self._table.c.chunk_id).where(self._table.c.chunk_id.in_(chunk_ids))
            )
            stale = set(existing.scalars().all())
            if stale:
                await session.execute(
                    delete(self._table).where(self._table.c.chunk_id.in_(list(stale)))
                )
            await session.execute(self._table.insert(), rows)

        logger.debug(
            "pgvector_upsert",
            extra={"space": self.space.name, "count": len(records), "replaced": len(stale)},
        )
        return len(records)

    async def delete_for_papers(self, paper_ids: list[str]) -> int:
        if not paper_ids:
            return 0
        async with self._session_factory() as session, session.begin():
            result = await session.execute(
                delete(self._table).where(self._table.c.paper_id.in_(paper_ids))
            )
            return int(getattr(result, "rowcount", 0) or 0)

    # ------------------------------------------------------------------- read
    async def search(
        self,
        vector: list[float],
        *,
        top_k: int = 10,
        filters: VectorFilter | None = None,
        min_score: float | None = None,
    ) -> list[tuple[str, float]]:
        self._check_dimensions(vector)
        filters = filters or VectorFilter()
        if filters.matches_nothing:
            # An empty value set is an explicit "match nothing" (see
            # VectorFilter.paper_ids). Emitting `IN ()` and hoping the dialect
            # reads it as false is not a contract.
            return []

        distance = self._table.c.vector.op(self.space.distance_operator)(vector)
        # `1.0` must be typed as Float explicitly. Without it SQLAlchemy infers
        # the whole subtraction as VECTOR (the right operand's type) and then
        # runs `1.0` through VectorColumn's bind processor, which hands pgvector
        # a bare float: "expected list or ndarray".
        similarity = (type_coerce(1.0, Float()) - distance).label("similarity")

        stmt = (
            select(self._table.c.chunk_id, similarity)
            .join(Chunk, Chunk.id == self._table.c.chunk_id)
            .join(Paper, Paper.id == self._table.c.paper_id)
        )
        if filters.paper_ids is not None:
            stmt = stmt.where(self._table.c.paper_id.in_(filters.paper_ids))
        if filters.content_kinds is not None:
            # Filtered in SQL rather than after ranking: a pgvector scan already
            # visits every matching row, so post-filtering would quietly cap
            # top_k instead of returning top_k *matching* rows.
            stmt = stmt.where(Chunk.content_kind.in_(filters.content_kinds))
        if filters.categories is not None:
            # EXISTS, not a join: one paper has many category rows, so joining
            # would duplicate the chunk it is matched against.
            stmt = stmt.where(
                exists().where(
                    and_(
                        PaperCategory.paper_id == Paper.id,
                        PaperCategory.category.in_(filters.categories),
                    )
                )
            )
        if filters.sources is not None:
            stmt = stmt.where(Chunk.source.in_(filters.sources))
        if filters.sections is not None:
            # A disjunction of pairs, not a filter on the ordinal alone: ordinals
            # restart at zero in every document, so `IN (3, 7)` would match two
            # unrelated sections in two different books.
            stmt = stmt.where(
                or_(
                    *(
                        and_(
                            Chunk.paper_id == paper_id,
                            Chunk.section_ordinal == ordinal,
                        )
                        for paper_id, ordinal in filters.sections
                    )
                )
            )
        if min_score is not None:
            # WHERE, not HAVING: there is no GROUP BY, so HAVING makes
            # PostgreSQL reject the whole statement ("chunk_id must appear in the
            # GROUP BY clause"). The scan is over one table with no aggregation,
            # so a row filter is the correct translation.
            stmt = stmt.where(similarity >= min_score)
        stmt = stmt.order_by(distance).limit(top_k)

        async with self._session_factory() as session:
            result = await session.execute(stmt)
            return [(chunk_id, float(score)) for chunk_id, score in result.all()]

    async def count(self, paper_id: str | None = None) -> int:
        async with self._session_factory() as session:
            stmt = select(func.count()).select_from(self._table)
            if paper_id:
                stmt = stmt.where(self._table.c.paper_id == paper_id)
            return int(await session.scalar(stmt) or 0)

    async def count_by_paper(self) -> dict[str, int]:
        """Vector counts grouped by paper — used by ``paper spaces show``."""
        async with self._session_factory() as session:
            rows = await session.execute(
                select(self._table.c.paper_id, func.count())
                .group_by(self._table.c.paper_id)
            )
            return {paper_id: int(count) for paper_id, count in rows.all()}

    async def load_vectors(self, chunk_ids: Sequence[str]) -> dict[str, list[float]]:
        """Fetch stored vectors for specific chunks (used by the sweep tooling)."""
        if not chunk_ids:
            return {}
        async with self._session_factory() as session:
            rows = await session.execute(
                select(self._table.c.chunk_id, self._table.c.vector).where(
                    self._table.c.chunk_id.in_(list(chunk_ids))
                )
            )
            return {chunk_id: list(vector or []) for chunk_id, vector in rows.all()}


def _now():  # noqa: ANN202
    from app.db.base import utcnow  # noqa: PLC0415

    return utcnow()