"""Citation graph: the references of a paper, and the papers citing it.

Kept separate from :mod:`app.db.repositories` because the two are unrelated: one
is about papers, the other about the edges between them.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Any

from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased

from app.db.base import utcnow
from app.db.models import Paper, PaperReference
from app.domain.ids import normalize_arxiv_id as domain_normalize_arxiv_id
from app.domain.models import Reference
from app.logging import get_logger

logger = get_logger(__name__)

def normalize_arxiv_id(value: str | None) -> str | None:
    """Versionless canonical arXiv id, or None when unparseable.

    Delegates to :mod:`app.domain.ids` rather than re-implementing it: a second,
    looser copy silently mangled old-style ids, whose archive part is
    case-sensitive (``math.CO/0309136``).
    """
    if not value:
        return None
    try:
        return domain_normalize_arxiv_id(value)
    except ValueError:
        return None


class ReferenceRepository:
    """Reads and writes :class:`~app.db.models.PaperReference`."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    # ------------------------------------------------------------------ write
    async def replace_for_paper(
        self,
        paper_id: str,
        references: Sequence[Reference],
        *,
        source: str = "",
    ) -> dict[str, int]:
        """Replace a paper's bibliography wholesale.

        Wholesale rather than merged because re-ingestion must not accumulate
        duplicates: the bibliography is a list, and ``ordinal`` is unique per
        citing paper, so replacing is both correct and idempotent.
        """
        await self._session.execute(delete(PaperReference).where(PaperReference.citing_paper_id == paper_id))
        rows = [
            PaperReference(
                id=uuid.uuid4().hex,
                citing_paper_id=paper_id,
                cited_arxiv_id=normalize_arxiv_id(ref.cited_arxiv_id),
                title=ref.title,
                authors=ref.authors,
                year=ref.year,
                venue=ref.venue,
                raw_text=ref.raw_text,
                ordinal=ref.ordinal,
                source=source,
            )
            for ref in references
        ]
        if rows:
            self._session.add_all(rows)
        await self._session.flush()
        await self.link_to_papers(paper_id)

        stats = {"total": len(rows), "linked": 0}
        linked = await self._session.scalar(
            select(func.count())
            .select_from(PaperReference)
            .where(
                PaperReference.citing_paper_id == paper_id,
                PaperReference.cited_paper_id.is_not(None),
            )
        )
        stats["linked"] = int(linked or 0)
        logger.info(
            "references_replaced",
            extra={"paper_id": paper_id, **stats},
        )
        return stats

    async def link_to_papers(self, citing_paper_id: str | None = None) -> int:
        """Point references at corpus rows for the arXiv ids we now hold.

        The bibliography usually cites papers we do not have, so resolution is
        opportunistic and rerunnable: it is called after every ingestion, which
        means a reference written months before the cited paper arrived gets
        linked the moment it does.
        """
        stmt = (
            select(PaperReference.id, Paper.arxiv_id, Paper.id)
            .join(Paper, Paper.arxiv_id == PaperReference.cited_arxiv_id)
            .where(
                PaperReference.cited_paper_id.is_(None),
                PaperReference.cited_arxiv_id.is_not(None),
                Paper.id != PaperReference.citing_paper_id,
            )
        )
        if citing_paper_id is not None:
            stmt = stmt.where(PaperReference.citing_paper_id == citing_paper_id)

        pairs = (await self._session.execute(stmt)).all()
        for reference_id, _cited_arxiv, paper_id in pairs:
            await self._session.execute(
                update(PaperReference)
                .where(PaperReference.id == reference_id)
                .values(cited_paper_id=paper_id)
                .execution_options(synchronize_session=False)
            )
        if pairs:
            await self._session.flush()
            logger.info("references_linked", extra={"count": len(pairs)})
        return len(pairs)

    async def delete_for_paper(self, paper_id: str) -> int:
        result = await self._session.execute(
            delete(PaperReference).where(PaperReference.citing_paper_id == paper_id)
        )
        await self._session.flush()
        return int(getattr(result, "rowcount", 0) or 0)

    # ------------------------------------------------------------------- read
    async def count(self, paper_id: str, *, resolved_only: bool = False) -> int:
        stmt = select(func.count()).select_from(PaperReference).where(
            PaperReference.citing_paper_id == paper_id
        )
        if resolved_only:
            stmt = stmt.where(PaperReference.cited_paper_id.is_not(None))
        return int(await self._session.scalar(stmt) or 0)

    async def list_references(
        self,
        paper_id: str,
        *,
        limit: int | None = None,
        offset: int = 0,
        resolved_only: bool = False,
    ) -> list[PaperReference]:
        stmt = select(PaperReference).where(PaperReference.citing_paper_id == paper_id)
        if resolved_only:
            stmt = stmt.where(PaperReference.cited_paper_id.is_not(None))
        stmt = stmt.order_by(PaperReference.ordinal)
        if limit is not None:
            stmt = stmt.limit(limit).offset(offset)
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def list_citations(self, paper_id: str) -> list[PaperReference]:
        """Papers in our corpus that cite this one (incoming edges)."""
        stmt = (
            select(PaperReference)
            .where(PaperReference.cited_paper_id == paper_id)
            .order_by(PaperReference.citing_paper_id)
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def citing_papers(self, paper_id: str, *, limit: int = 50) -> list[Paper]:
        """Distinct citing papers, most recently created first."""
        stmt = (
            select(Paper)
            .join(PaperReference, PaperReference.citing_paper_id == Paper.id)
            .where(PaperReference.cited_paper_id == paper_id)
            .order_by(Paper.created_at.desc())
            .limit(limit)
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().unique().all())

    async def referenced_papers(self, paper_id: str, *, limit: int = 50) -> list[Paper]:
        """Corpus papers this one cites (outgoing edges that resolved)."""
        stmt = (
            select(Paper)
            .join(PaperReference, PaperReference.cited_paper_id == Paper.id)
            .where(PaperReference.citing_paper_id == paper_id)
            .order_by(PaperReference.ordinal)
            .limit(limit)
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().unique().all())

    async def most_cited(self, *, limit: int = 20) -> list[dict[str, Any]]:
        """Corpus papers ranked by how many of our papers cite them."""
        cited = aliased(Paper)
        stmt = (
            select(
                cited.arxiv_id,
                cited.title,
                func.count(PaperReference.id).label("citation_count"),
            )
            .select_from(PaperReference)
            .join(cited, cited.id == PaperReference.cited_paper_id)
            .group_by(cited.id)
            .order_by(func.count(PaperReference.id).desc(), cited.title)
            .limit(limit)
        )
        result = await self._session.execute(stmt)
        return [
            {
                "arxiv_id": arxiv_id,
                "title": title,
                "citation_count": int(count),
            }
            for arxiv_id, title, count in result.all()
        ]


__all__ = ["ReferenceRepository", "normalize_arxiv_id", "utcnow"]