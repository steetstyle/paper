"""Backfill ``chunks.content_kind`` for chunks written before it existed.

The classifier in :mod:`paper_app.services.chunk_kinds` is cheap and pure — it needs
only the chunk's own text plus the heading breadcrumb that is already on the
row — so an older corpus can be labelled without re-downloading or re-parsing a
single paper. That matters because the column was added after the corpus was
ingested, and every ``body`` default would otherwise silently defeat the
"only equations" filter.

Only rows still at the default are touched, so this is safe to re-run after a
classifier fix only if the default changes; to force a full re-label, pass
``only_body=False``.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import select, update

from paper_app.db.models import Chunk
from paper_app.domain.enums import ChunkKind
from paper_app.logging import get_logger
from paper_app.services.chunk_kinds import classify_chunk

logger = get_logger(__name__)


async def backfill_chunk_kinds(
    session: Any,
    *,
    only_body: bool = True,
    paper_id: str | None = None,
) -> dict[str, int]:
    """Classify stored chunks in one pass. Returns counts per kind.

    Args:
        only_body: restrict to rows whose kind is still the ``body`` default.
        paper_id: restrict to one paper.
    """
    stmt = select(Chunk)
    if only_body:
        stmt = stmt.where(Chunk.content_kind == ChunkKind.BODY.value)
    if paper_id:
        stmt = stmt.where(Chunk.paper_id == paper_id)

    rows = (await session.execute(stmt)).scalars().all()
    counts: dict[str, int] = {}
    pending: list[tuple[str, str]] = []
    for row in rows:
        kind = classify_chunk(row.text, heading=row.heading, section_path=row.section_path or ())
        counts[kind.value] = counts.get(kind.value, 0) + 1
        if kind.value != row.content_kind:
            pending.append((kind.value, row.id))

    if pending:
        # One statement per distinct kind rather than per row: 3k chunks become
        # 7 UPDATE ... WHERE id IN (...) instead of 3k round trips.
        by_kind: dict[str, list[str]] = {}
        for kind_value, chunk_id in pending:
            by_kind.setdefault(kind_value, []).append(chunk_id)
        for kind_value, ids in by_kind.items():
            await session.execute(
                update(Chunk).where(Chunk.id.in_(ids)).values(content_kind=kind_value)
            )
    await session.flush()
    logger.info(
        "chunk_kind_backfill",
        extra={"scanned": len(rows), "changed": len(pending), **counts},
    )
    return counts


__all__ = ["backfill_chunk_kinds"]
