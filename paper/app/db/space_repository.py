"""Repository for the embedding-space registry."""

from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.base import utcnow
from app.db.models import EmbeddingSpaceRecord
from app.db.spaces import EmbeddingSpace, assert_distinct
from app.logging import get_logger

logger = get_logger(__name__)


class SpaceConflictError(RuntimeError):
    """A space with that name or table already exists."""


class EmbeddingSpaceRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    # ------------------------------------------------------------------ reads
    async def get(self, name: str) -> EmbeddingSpace | None:
        row = await self._session.get(EmbeddingSpaceRecord, name)
        return row.as_space() if row else None

    async def get_record(self, name: str) -> EmbeddingSpaceRecord | None:
        return await self._session.get(EmbeddingSpaceRecord, name)

    async def list(self) -> list[EmbeddingSpaceRecord]:
        result = await self._session.execute(
            select(EmbeddingSpaceRecord).order_by(
                EmbeddingSpaceRecord.is_active.desc(), EmbeddingSpaceRecord.name
            )
        )
        return list(result.scalars().all())

    async def active(self) -> EmbeddingSpace | None:
        row = await self._session.scalar(
            select(EmbeddingSpaceRecord)
            .where(EmbeddingSpaceRecord.is_active.is_(True))
            .limit(1)
        )
        return row.as_space() if row else None

    async def resolve(
        self, name: str | None = None, *, fallback: EmbeddingSpace | None = None
    ) -> EmbeddingSpace:
        """Find a space by name, else the active one, else ``fallback``."""
        if name:
            space = await self.get(name)
            if space is None:
                raise SpaceConflictError(f"unknown embedding space {name!r}")
            return space
        active = await self.active()
        if active is not None:
            return active
        if fallback is not None:
            return fallback
        raise SpaceConflictError("no embedding space registered and no fallback given")

    # ----------------------------------------------------------------- writes
    async def create(
        self,
        space: EmbeddingSpace,
        *,
        is_active: bool = False,
        is_locked: bool = False,
        set_exclusive: bool = True,
    ) -> EmbeddingSpace:
        """Register a space. Idempotent when the definition is unchanged."""
        existing = await self.get_record(space.name)
        if existing is not None:
            await self._assert_same_definition(existing, space)
            if is_active and not existing.is_active:
                await self.activate(space.name)
            return existing.as_space()

        table_taken = await self._session.scalar(
            select(EmbeddingSpaceRecord).where(
                EmbeddingSpaceRecord.table_name == space.resolved_table
            )
        )
        if table_taken is not None:
            raise SpaceConflictError(
                f"table {space.resolved_table!r} is already used by space {table_taken.name!r}"
            )

        # Guard against two new spaces colliding on one physical table.
        await self._assert_no_table_collision(space)

        record = EmbeddingSpaceRecord(
            name=space.name,
            provider=space.provider,
            model=space.model,
            dimensions=space.dimensions,
            distance=space.distance,
            table_name=space.resolved_table,
            description=space.description,
            is_active=is_active,
            is_locked=is_locked,
        )
        self._session.add(record)
        await self._session.flush()
        if is_active and set_exclusive:
            await self._deactivate_others(space.name)
        logger.info(
            "space_created",
            extra={
                "space": space.name,
                "table": space.resolved_table,
                "dimensions": space.dimensions,
                "active": is_active,
            },
        )
        return record.as_space()

    async def activate(self, name: str) -> EmbeddingSpace:
        space = await self.get(name)
        if space is None:
            raise SpaceConflictError(f"unknown embedding space {name!r}")
        await self._deactivate_others(name)
        await self._session.execute(
            update(EmbeddingSpaceRecord)
            .where(EmbeddingSpaceRecord.name == name)
            .values(is_active=True, last_used_at=utcnow())
        )
        await self._session.flush()
        logger.info("space_activated", extra={"space": name})
        return space

    async def touch(self, name: str) -> None:
        await self._session.execute(
            update(EmbeddingSpaceRecord)
            .where(EmbeddingSpaceRecord.name == name)
            .values(last_used_at=utcnow())
        )
        await self._session.flush()

    async def set_vector_count(self, name: str, count: int) -> None:
        await self._session.execute(
            update(EmbeddingSpaceRecord)
            .where(EmbeddingSpaceRecord.name == name)
            .values(vector_count=count)
        )
        await self._session.flush()

    async def delete(self, name: str, *, drop_table: bool = False) -> bool:
        """Remove a space (and optionally its table). Locked spaces are kept."""
        record = await self.get_record(name)
        if record is None:
            return False
        if record.is_locked:
            raise SpaceConflictError(
                f"space {name!r} is managed by settings (EMBEDDING_SPACE) and cannot be removed"
            )
        await self._session.delete(record)
        await self._session.flush()
        logger.info("space_deleted", extra={"space": name, "drop_table": drop_table})
        return True

    # ----------------------------------------------------------------- guards
    async def _assert_same_definition(
        self, record: EmbeddingSpaceRecord, space: EmbeddingSpace
    ) -> None:
        mismatched = []
        if record.provider != space.provider:
            mismatched.append(f"provider {record.provider!r} != {space.provider!r}")
        if record.model != space.model:
            mismatched.append(f"model {record.model!r} != {space.model!r}")
        if record.dimensions != space.dimensions:
            mismatched.append(f"dimensions {record.dimensions} != {space.dimensions}")
        if record.distance != space.distance:
            mismatched.append(f"distance {record.distance!r} != {space.distance!r}")
        if mismatched:
            raise SpaceConflictError(
                f"embedding space {space.name!r} already exists with a different "
                f"definition ({'; '.join(mismatched)}). Vectors from different models "
                "cannot share a table — create a new space instead."
            )

    async def _assert_no_table_collision(self, space: EmbeddingSpace) -> None:
        existing = await self.list()
        assert_distinct([row.as_space() for row in existing] + [space])

    async def _deactivate_others(self, keep: str) -> None:
        await self._session.execute(
            update(EmbeddingSpaceRecord)
            .where(EmbeddingSpaceRecord.name != keep)
            .values(is_active=False)
        )

    async def bulk_counts(self, counters: dict[str, int]) -> None:
        for name, count in counters.items():
            await self.set_vector_count(name, count)


async def seed_from_settings(
    repo: EmbeddingSpaceRepository, spaces: Sequence[EmbeddingSpace], active: str
) -> EmbeddingSpace:
    """Ensure the configured spaces exist and one of them is active."""
    for space in spaces:
        await repo.create(
            space, is_active=space.name == active, is_locked=True, set_exclusive=False
        )
    return await repo.resolve(active)