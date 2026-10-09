"""Liveness and readiness probes."""

from __future__ import annotations

import contextlib

from fastapi import APIRouter, Response, status

from paper_app import __version__
from paper_app.api.deps import ContainerDep, SessionDep
from paper_app.api.schemas import HealthResponse
from paper_app.clients.content.mineru_resolver import installed_version
from paper_app.db.session import check_connection
from paper_app.db.space_repository import EmbeddingSpaceRepository

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse, summary="Liveness")
async def health(
    container: ContainerDep, session: SessionDep, response: Response
) -> HealthResponse:
    database_ok = await check_connection(container.settings.database)
    vector_ok = await container.vector_store.health()

    status_value = "ok" if (database_ok and vector_ok) else "degraded"
    if not database_ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    extractor = container.mineru
    space_count = 0
    with contextlib.suppress(Exception):
        space_count = len(await EmbeddingSpaceRepository(session).list())

    return HealthResponse(
        active_space=container.active_space.name,
        space_count=space_count,
        status=status_value,
        version=__version__,
        database="up" if database_ok else "down",
        vector_store=f"{container.vector_store.name}:{'up' if vector_ok else 'down'}",
        embeddings=f"{container.embeddings.name}/{container.embeddings.model}"
        f"@{container.embeddings.dimensions}",
        mineru_backends=extractor.available_backends,
        mineru_detail=extractor.describe_backends(),
        mineru_version=installed_version(),
    )


@router.get("/health/ready", summary="Readiness")
async def ready(container: ContainerDep, response: Response) -> dict[str, object]:
    database_ok = await check_connection(container.settings.database)
    vector_ok = await container.vector_store.health()
    ready_flag = database_ok and vector_ok
    if not ready_flag:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {
        "ready": ready_flag,
        "database": database_ok,
        "vector_store": vector_ok,
        "mineru_backends": container.mineru.available_backends,
        "mineru_version": installed_version(),
    }