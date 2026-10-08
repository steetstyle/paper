"""FastAPI dependencies."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, Query, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import AppSettings, get_settings
from app.container import Container, get_container


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


def device_dependency(
    device: str | None = Query(default=None),
) -> str | None:
    """Validate the ``?device=`` query argument of an embedding route.

    Query-shaped counterpart to :data:`app.api.devices.DeviceField`: a request
    body is validated by pydantic, but a query argument arrives as a plain string
    and would otherwise reach torch misspelled.
    """
    from fastapi import HTTPException  # noqa: PLC0415

    from app.api.devices import parse_device  # noqa: PLC0415

    try:
        return parse_device(device)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


ContainerDep = Annotated[Container, Depends(container_dependency)]
SettingsDep = Annotated[AppSettings, Depends(settings_dependency)]
SessionDep = Annotated[AsyncSession, Depends(session_dependency)]
DeviceDep = Annotated[str | None, Depends(device_dependency)]


def normalize_arxiv_id_or_400(value: str) -> str:
    """Accept any handle a row can be addressed by, arXiv-shaped or not.

    Used to reject anything that was not an arXiv id, which is right for a corpus
    that is only arXiv papers and wrong for one that also holds books: a local
    document is addressed by its ``doc_key``, and refusing that here meant
    ``GET /papers/solid-state-basics`` was a 422 rather than a 404 — telling the
    caller the document does not exist, only with the wrong status.

    Now it only rejects an empty handle; whether the identifier names anything is
    the repository's question, and it answers 404 when it does not.
    """
    text = (value or "").strip()
    if not text:
        from fastapi import HTTPException  # noqa: PLC0415

        raise HTTPException(status_code=422, detail="empty identifier")
    return text

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
