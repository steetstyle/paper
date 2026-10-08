"""Storage and retrieval for figures, tables and equations.

Writes replace wholesale per paper. ``ordinal`` is unique per paper, and a
re-extraction of the same source must not leave a second copy of every figure
behind — the alternative is a table that grows on every ingest.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Sequence
from typing import Any

from sqlalchemy import Select, delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.clients.content.assets import Assets
from app.db.asset_models import PaperEquation, PaperFigure, PaperTable
from app.db.base import utcnow
from app.db.models import Paper
from app.infra.storage import BlobStore
from app.logging import get_logger

logger = get_logger(__name__)

Row = PaperFigure | PaperTable | PaperEquation


def _bbox_json(value: Sequence[int] | None) -> str | None:
    return json.dumps(list(value)) if value else None


def _table_shape(html: str | None) -> tuple[int | None, int | None]:
    """Row and column count of a ``<table>`` body, for display and filtering.

    Measured on the header row rather than the widest row: a row count from
    ``<tr>`` tags is correct, but a "column count" taken from the first row lies
    whenever a cell spans columns.
    """
    if not html:
        return None, None
    rows = html.count("<tr")
    first_row = html[html.find("<tr") : html.find("</tr>")]
    columns = first_row.count("<td") + first_row.count("<th") if "<tr" in html else 0
    return (rows or None), (columns or None)


class AssetRepository:
    def __init__(self, session: AsyncSession, blobs: BlobStore | None = None) -> None:
        self._session = session
        self._blobs = blobs

    # ------------------------------------------------------------------ write
    async def replace_for_paper(
        self,
        paper_id: str,
        assets: Assets,
        *,
        store_images: bool = True,
    ) -> dict[str, int]:
        """Write every asset of one paper, replacing whatever was there.

        Images present on disk (the PDF path) are hashed into the blob store so
        they survive MinerU's temp directory; images we only have a URL for are
        recorded as URLs and fetched on demand.
        """
        counts: dict[str, int] = {}
        counts["figures"] = await self._write_figures(paper_id, assets, store_images)
        counts["tables"] = await self._write_tables(paper_id, assets, store_images)
        counts["equations"] = await self._write_equations(paper_id, assets, store_images)
        logger.info("assets_replaced", extra={"paper_id": paper_id, **counts})
        return counts

    async def _write_figures(
        self, paper_id: str, assets: Assets, store_images: bool
    ) -> int:
        await self._session.execute(
            delete(PaperFigure).where(PaperFigure.paper_id == paper_id)
        )
        rows = [
            PaperFigure(
                id=uuid.uuid4().hex,
                paper_id=paper_id,
                ordinal=figure.ordinal,
                label=figure.label,
                caption=figure.caption,
                image_sha256=self._store_image(figure.image_path)
                if store_images
                else None,
                image_url=figure.image_url,
                page_idx=figure.page_idx,
                bbox=_bbox_json(figure.bbox),
                width=figure.width,
                height=figure.height,
                source=figure.source,
                created_at=utcnow(),
            )
            for figure in assets.figures
        ]
        if rows:
            self._session.add_all(rows)
        await self._session.flush()
        return len(rows)

    async def _write_tables(
        self, paper_id: str, assets: Assets, store_images: bool
    ) -> int:
        await self._session.execute(
            delete(PaperTable).where(PaperTable.paper_id == paper_id)
        )
        rows = []
        for table in assets.tables:
            row_count, column_count = _table_shape(table.body_html)
            rows.append(
                PaperTable(
                    id=uuid.uuid4().hex,
                    paper_id=paper_id,
                    ordinal=table.ordinal,
                    label=table.label,
                    caption=table.caption,
                    body_html=table.body_html,
                    image_sha256=self._store_image(table.image_path)
                    if store_images
                    else None,
                    image_url=table.image_url,
                    page_idx=table.page_idx,
                    bbox=_bbox_json(table.bbox),
                    row_count=row_count,
                    column_count=column_count,
                    source=table.source,
                    created_at=utcnow(),
                )
            )
        if rows:
            self._session.add_all(rows)
        await self._session.flush()
        return len(rows)

    async def _write_equations(
        self, paper_id: str, assets: Assets, store_images: bool
    ) -> int:
        await self._session.execute(
            delete(PaperEquation).where(PaperEquation.paper_id == paper_id)
        )
        rows = [
            PaperEquation(
                id=uuid.uuid4().hex,
                paper_id=paper_id,
                ordinal=equation.ordinal,
                latex=equation.latex,
                text_format="latex",
                is_display=equation.is_display,
                image_sha256=self._store_image(equation.image_path)
                if store_images
                else None,
                page_idx=equation.page_idx,
                bbox=_bbox_json(equation.bbox),
                source=equation.source,
                created_at=utcnow(),
            )
            for equation in assets.equations
        ]
        if rows:
            self._session.add_all(rows)
        await self._session.flush()
        return len(rows)

    def _store_image(self, image_path: str | None) -> str | None:
        """Hash a local image into the blob store. Best effort, never fatal.

        A missing file must not lose the figure: the caption and the label are
        the useful part, and the image is the part we can go and get again.
        """
        if not image_path or self._blobs is None:
            return None
        from pathlib import Path  # noqa: PLC0415

        path = Path(image_path)
        if not path.exists():
            logger.warning("asset_image_missing", extra={"path": image_path})
            return None
        try:
            return self._blobs.put_file(path, prefix="assets").sha256
        except OSError as exc:
            logger.warning("asset_image_unreadable", extra={"path": image_path, "error": str(exc)})
            return None

    # ------------------------------------------------------------------- read
    async def count(self, paper_id: str) -> dict[str, int]:
        return {
            "figures": await self._count(PaperFigure, paper_id),
            "tables": await self._count(PaperTable, paper_id),
            "equations": await self._count(PaperEquation, paper_id),
            "display_equations": await self._count(PaperEquation, paper_id, display_only=True),
        }

    async def _count(self, model: type[Any], paper_id: str, *, display_only: bool = False) -> int:
        stmt = select(func.count()).select_from(model).where(model.paper_id == paper_id)
        if display_only:
            stmt = stmt.where(model.is_display.is_(True))
        return int(await self._session.scalar(stmt) or 0)

    async def figures(
        self, paper_id: str, *, limit: int | None = None, offset: int = 0
    ) -> list[PaperFigure]:
        stmt = (
            select(PaperFigure)
            .where(PaperFigure.paper_id == paper_id)
            .order_by(PaperFigure.ordinal)
        )
        if limit is not None:
            stmt = stmt.limit(limit).offset(offset)
        return list((await self._session.execute(stmt)).scalars().all())

    async def tables(
        self, paper_id: str, *, limit: int | None = None, offset: int = 0
    ) -> list[PaperTable]:
        stmt = (
            select(PaperTable)
            .where(PaperTable.paper_id == paper_id)
            .order_by(PaperTable.ordinal)
        )
        if limit is not None:
            stmt = stmt.limit(limit).offset(offset)
        return list((await self._session.execute(stmt)).scalars().all())

    async def equations(
        self,
        paper_id: str,
        *,
        limit: int | None = None,
        offset: int = 0,
        display_only: bool = False,
    ) -> list[PaperEquation]:
        stmt = (
            select(PaperEquation)
            .where(PaperEquation.paper_id == paper_id)
            .order_by(PaperEquation.ordinal)
        )
        if display_only:
            stmt = stmt.where(PaperEquation.is_display.is_(True))
        if limit is not None:
            stmt = stmt.limit(limit).offset(offset)
        return list((await self._session.execute(stmt)).scalars().all())

    async def search_equations(
        self,
        needle: str,
        *,
        limit: int = 20,
        display_only: bool = False,
        paper_ids: Sequence[str] | None = None,
    ) -> list[PaperEquation]:
        """Equations whose LaTeX contains ``needle``.

        Args:
            paper_ids: restrict to these papers. Empty means "no paper matches",
                matching :class:`~app.db.vector_store.base.VectorFilter`; the
                distinction matters because the callers of this endpoint can pass
                a project whose papers hold no equations at all.
        """
        pattern = f"%{needle.lower()}%"
        stmt: Select[Any] = (
            select(PaperEquation)
            .where(func.lower(func.coalesce(PaperEquation.latex, "")).like(pattern))
            .order_by(PaperEquation.paper_id, PaperEquation.ordinal)
            .limit(limit)
        )
        if display_only:
            stmt = stmt.where(PaperEquation.is_display.is_(True))
        if paper_ids is not None:
            stmt = stmt.where(PaperEquation.paper_id.in_(list(paper_ids)))
        return list((await self._session.execute(stmt)).scalars().all())

    async def papers_missing_figures(self, *, limit: int = 50) -> list[Paper]:
        """Ingested papers with no figures recorded.

        This is the set worth re-extracting: either the source predates asset
        extraction, or the rendering had no figures to find. It is deliberately
        "no figures" rather than "no assets", because a paper with 140 inline
        equations and no figures still has nothing to show here.
        """
        have_figures = select(PaperFigure.paper_id).where(
            PaperFigure.paper_id.is_not(None)
        )
        stmt = (
            select(Paper)
            .where(
                Paper.ingested_at.is_not(None),
                ~Paper.id.in_(have_figures),
            )
            .order_by(Paper.ingested_at.desc().nullslast())
            .limit(limit)
        )
        return list((await self._session.execute(stmt)).scalars().unique().all())


__all__ = ["AssetRepository", "Row"]