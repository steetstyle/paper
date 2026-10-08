"""Embedding-space registry endpoints.

One model, one table. These endpoints make the set of models a deployment knows
about visible and manageable without a redeploy.
"""

from __future__ import annotations

import contextlib

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.api.deps import ContainerDep, SessionDep
from app.db.space_repository import EmbeddingSpaceRepository, SpaceConflictError
from app.db.spaces import EmbeddingSpace
from app.db.vector_store.schema import ddl_statements

router = APIRouter(prefix="/embedding-spaces", tags=["embedding-spaces"])


class SpaceOut(BaseModel):
    name: str
    provider: str
    model: str
    dimensions: int
    distance: str
    table: str
    fingerprint: str
    description: str | None = None
    is_active: bool = False
    is_locked: bool = False
    vector_count: int = 0
    last_used_at: str | None = None


class SpaceCreate(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    dimensions: int = Field(ge=1)
    distance: str = "cosine"
    description: str | None = None
    activate: bool = False


class SpaceDetail(SpaceOut):
    ddl: list[str] = Field(default_factory=list)


def _to_out(record, count: int | None = None) -> SpaceOut:
    space = record.as_space()
    return SpaceOut(
        **space.to_dict(),
        is_active=record.is_active,
        is_locked=record.is_locked,
        vector_count=record.vector_count if count is None else count,
        last_used_at=record.last_used_at.isoformat() if record.last_used_at else None,
    )


async def _counts(container: ContainerDep, spaces: list) -> dict[str, int]:  # noqa: ANN001
    """Live vector counts, so the list reflects reality rather than bookkeeping."""
    counts: dict[str, int] = {}
    for record in spaces:
        store = container.vector_store_for(record.as_space())
        with contextlib.suppress(Exception):
            counts[record.name] = await store.count()
        counts.setdefault(record.name, 0)
    return counts


@router.get("", response_model=list[SpaceOut], summary="List embedding spaces")
async def list_spaces(container: ContainerDep, session: SessionDep) -> list[SpaceOut]:
    records = await EmbeddingSpaceRepository(session).list()
    counts = await _counts(container, records)
    return [_to_out(r, counts.get(r.name)) for r in records]


@router.post("", response_model=SpaceOut, status_code=201, summary="Register a model")
async def create_space(
    payload: SpaceCreate, container: ContainerDep, session: SessionDep
) -> SpaceOut:
    """Register a model. Its table is created on first ingest (or on demand).

    Two models can never share a table: their vectors are not comparable, and
    pgvector's column width is fixed.
    """
    repo = EmbeddingSpaceRepository(session)
    try:
        space = EmbeddingSpace(
            name=payload.name,
            provider=payload.provider,
            model=payload.model,
            dimensions=payload.dimensions,
            distance=payload.distance,
            description=payload.description,
        )
        await repo.create(space, is_active=payload.activate)
    except (ValueError, SpaceConflictError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    if container.settings.vector.auto_create:
        await container.vector_store_for(space).ensure_ready()

    record = await repo.get_record(space.name)
    return _to_out(record, await container.vector_store_for(space).count())


@router.get("/{name}", response_model=SpaceDetail, summary="Space detail")
async def get_space(name: str, container: ContainerDep, session: SessionDep) -> SpaceDetail:
    repo = EmbeddingSpaceRepository(session)
    record = await repo.get_record(name)
    if record is None:
        raise HTTPException(status_code=404, detail=f"unknown embedding space {name!r}")
    space = record.as_space()
    count = 0
    # A space whose table was never created simply has no vectors yet.
    with contextlib.suppress(Exception):
        count = await container.vector_store_for(space).count()
    return SpaceDetail(
        **_to_out(record, count).model_dump(),
        ddl=ddl_statements(space),
    )


@router.post("/{name}/activate", response_model=SpaceOut, summary="Make a space active")
async def activate_space(name: str, container: ContainerDep, session: SessionDep) -> SpaceOut:
    """Set the default space used when a request does not name one.

    Vectors are not moved: this only changes which model new work uses, so
    switching is instant and reversible.
    """
    repo = EmbeddingSpaceRepository(session)
    try:
        space = await repo.activate(name)
    except SpaceConflictError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    container.use_space(space)
    record = await repo.get_record(name)
    return _to_out(record)


@router.delete("/{name}", summary="Remove a space")
async def delete_space(
    name: str,
    session: SessionDep,
    drop_table: bool = Query(default=False, description="Also DROP the vector table."),
) -> dict:
    """Remove a space from the registry.

    ``drop_table`` destroys its vectors, so it is opt-in and irreversible.
    Spaces derived from ``EMBEDDING_*`` settings are locked and cannot be
    removed here.
    """
    repo = EmbeddingSpaceRepository(session)
    record = await repo.get_record(name)
    if record is None:
        raise HTTPException(status_code=404, detail=f"unknown embedding space {name!r}")
    space = record.as_space()

    if drop_table:
        from sqlalchemy import text as sa_text  # noqa: PLC0415

        # Identifier is validated by slugify() in EmbeddingSpace, so it is safe
        # to interpolate here; parameters are not allowed for identifiers.
        await session.execute(sa_text(f'DROP TABLE IF EXISTS "{space.resolved_table}"'))
        await session.execute(
            sa_text(f'DROP INDEX IF EXISTS "ix_{space.resolved_table}_hnsw"')
        )
        await session.execute(
            sa_text(f'DROP INDEX IF EXISTS "uq_{space.resolved_table}_chunk"')
        )

    try:
        await repo.delete(name)
    except SpaceConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"deleted": name, "table_dropped": drop_table, "table": space.resolved_table}