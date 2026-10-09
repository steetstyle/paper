"""Async engine / session lifecycle.

One engine per process, created lazily. ``session_scope`` is the only sanctioned
way to obtain a session so commit/rollback semantics stay in one place.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from sqlalchemy import select
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from paper_app.config import DatabaseSettings, get_settings

if TYPE_CHECKING:
    from sqlalchemy import Table
from paper_app.db.base import utcnow
from paper_app.db.models import Base
from paper_app.logging import get_logger

logger = get_logger(__name__)

_engines: dict[str, AsyncEngine] = {}
_session_factories: dict[str, async_sessionmaker[AsyncSession]] = {}


def _engine_kwargs(settings: DatabaseSettings) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"echo": settings.echo, "future": True, "pool_pre_ping": settings.pool_pre_ping}
    if not settings.url.startswith("sqlite"):
        kwargs.update(
            pool_size=settings.pool_size,
            max_overflow=settings.max_overflow,
            pool_recycle=1800,
        )
    else:
        kwargs.pop("pool_pre_ping", None)
    return kwargs


def get_engine(settings: DatabaseSettings | None = None) -> AsyncEngine:
    settings = settings or get_settings().database
    if settings.url not in _engines:
        ensure_sqlite_directory(settings.url)
        logger.info("db_engine_created", extra={"url": _redact(settings.url)})
        _engines[settings.url] = create_async_engine(settings.url, **_engine_kwargs(settings))
    return _engines[settings.url]


def ensure_sqlite_directory(url: str) -> None:
    """SQLite cannot create intermediate directories; do it for the caller.

    Called by both the async engine factory and Alembic's ``env.py`` so that
    ``paper db upgrade`` works on a fresh checkout.
    """
    if not url.startswith("sqlite"):
        return
    _, _, path = url.partition(":///")
    if not path or path.startswith(":memory:"):
        return
    directory = Path(path).expanduser().parent
    if str(directory) not in {"", "."}:
        directory.mkdir(parents=True, exist_ok=True)


def get_session_factory(
    settings: DatabaseSettings | None = None,
) -> async_sessionmaker[AsyncSession]:
    settings = settings or get_settings().database
    if settings.url not in _session_factories:
        _session_factories[settings.url] = async_sessionmaker(
            bind=get_engine(settings),
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )
    return _session_factories[settings.url]


@contextlib.asynccontextmanager
async def session_scope(
    settings: DatabaseSettings | None = None,
    factory: async_sessionmaker[AsyncSession] | None = None,
) -> AsyncIterator[AsyncSession]:
    """Transactional scope: commits on success, rolls back on error."""
    factory = factory or get_session_factory(settings)
    session = factory()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


async def create_all(settings: DatabaseSettings | None = None) -> None:
    """Create tables directly (dev only; production uses Alembic).

    Also registers the settings-derived embedding space, so the registry ends up
    the same whichever path created the schema.
    """
    engine = get_engine(settings)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
        await connection.run_sync(seed_default_space)


def seed_default_space(connection: Any) -> bool:
    """Register the active space if it is not there yet. True if inserted."""
    from paper_app.db.models import EmbeddingSpaceRecord  # noqa: PLC0415
    from paper_app.db.spaces import EmbeddingSpace  # noqa: PLC0415

    # `__table__` is typed as a read-only FromClause; Core insert() needs Table.
    table = cast("Table", EmbeddingSpaceRecord.__table__)
    space = EmbeddingSpace.default_for(get_settings())
    existing = connection.execute(select(table.c.name).where(table.c.name == space.name)).first()
    if existing is not None:
        return False
    connection.execute(
        table.insert().values(
            name=space.name,
            provider=space.provider,
            model=space.model,
            dimensions=space.dimensions,
            distance=space.distance,
            table_name=space.resolved_table,
            description=space.description,
            is_active=True,
            is_locked=True,
            created_at=utcnow(),
            vector_count=0,
        )
    )
    logger.info("embedding_space_seeded", extra={"space": space.name})
    return True


async def drop_all(settings: DatabaseSettings | None = None) -> None:  # pragma: no cover
    engine = get_engine(settings)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.drop_all)


async def dispose_engines() -> None:
    for engine in _engines.values():
        await engine.dispose()
    _engines.clear()
    _session_factories.clear()


async def check_connection(settings: DatabaseSettings | None = None) -> bool:
    from sqlalchemy import text  # noqa: PLC0415

    try:
        engine = get_engine(settings)
        async with engine.connect() as connection:
            await connection.execute(text("SELECT 1"))
        return True
    except Exception as exc:  # noqa: BLE001 - health probes must not raise
        logger.warning("db_unreachable", extra={"error": str(exc)})
        return False


def _redact(url: str) -> str:
    if "@" not in url:
        return url
    scheme, _, rest = url.partition("://")
    credentials, _, host = rest.rpartition("@")
    user = credentials.split(":", 1)[0]
    return f"{scheme}://{user}:***@{host}"