"""FastAPI dependencies."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import AppSettings, get_settings
from app.container import Container, get_container
from app.domain.ids import parse_arxiv_id


def container_dependency(request: Request) -> Container:
    """Prefer the app-lifespan container; fall back to the process singleton."""
    container = getattr(request.app.state, "container", None)
    return container or get_container()


def settings_dependency() -> AppSettings:
    return get_settings()


async def session_dependency(
    container: ContainerDep,
) -> AsyncIterator[AsyncSession]:
    """Transactional session: commits on success, rolls back on error."""
    factory = container.session_factory
    session = factory()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


ContainerDep = Annotated[Container, Depends(container_dependency)]
SettingsDep = Annotated[AppSettings, Depends(settings_dependency)]
SessionDep = Annotated[AsyncSession, Depends(session_dependency)]


def normalize_arxiv_id_or_400(value: str) -> str:
    try:
        return parse_arxiv_id(value).id
    except ValueError as exc:
        from fastapi import HTTPException  # noqa: PLC0415

        raise HTTPException(status_code=422, detail=str(exc)) from exc

async def resolve_space(
    container: Container,
    session: AsyncSession,
    name: str | None = None,
):
    """Resolve an embedding space: the named one, else the active one, else settings.

    Centralised here because every endpoint that touches vectors needs the same
    precedence, and getting it wrong silently compares the wrong index.
    """
    from fastapi import HTTPException  # noqa: PLC0415

    from app.db.space_repository import (  # noqa: PLC0415
        EmbeddingSpaceRepository,
        SpaceConflictError,
    )

    repo = EmbeddingSpaceRepository(session)
    try:
        return await repo.resolve(name, fallback=container.default_space)
    except SpaceConflictError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
