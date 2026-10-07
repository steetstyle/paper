"""Removing sections that are the renderer's furniture rather than structure.

A section row exists so a reader can be told where to read. A heading that cannot
be pointed at on a page and does not state its own place in the structure is not
one: measured on this corpus, an arXiv paper ingested from ar5iv produced fourteen
"sections" that were the abs page's chrome — ``Submission history``,
``Access Paper:``, ``BibTeX formatted citation``, ``Demos``, ``arXivLabs:
experimental projects`` — and across 88 documents those outweighed the genuine ones
4103 to 322.

``paper sections prune`` removes them. Dry run by default, for the same reason
``paper runs --reap`` is: a row here is part of the record of what the corpus
believes, and ending that belief is a decision rather than a side effect.

The predicate is :func:`app.services.sections._only_navigable`'s own rule, imported
rather than restated. The builder and the prune agreeing is the whole point: a
prune with its own idea of what furniture is would eventually delete something the
pipeline had just decided to keep.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Chunk, DocumentSection, Paper
from app.logging import get_logger
from app.services.sections import _is_furniture

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class PruneReport:
    """What a prune would do, or did."""

    sections: int
    """Headings that are furniture and would go."""

    papers: int
    """Documents that had at least one."""

    chunks_unlinked: int
    """Chunks whose ``section_ordinal`` pointed at one and now points at nothing.

    Not a loss: the chunk and its text stay, only the navigation label goes.
    Leaving the pointer would make a chunk claim a section that does not exist.
    """

    kept: int
    """Sections that survive — the real structure."""

    applied: bool
    """Whether the delete was *requested*, not whether it changed anything: a
    caller asking to prune an already-clean corpus should not have to infer from
    ``sections == 0`` that it was honoured."""


async def _furniture(session: AsyncSession) -> list[DocumentSection]:
    """Every stored heading that is furniture, by the builder's own rule."""
    rows = (
        await session.execute(
            select(DocumentSection).where(DocumentSection.page_start.is_(None))
        )
    ).scalars().all()
    return [row for row in rows if _is_furniture(row)]


async def prune_unplaceable_sections(
    session: AsyncSession, *, apply: bool = False
) -> PruneReport:
    """Delete sections that are furniture. Dry run unless ``apply``."""
    doomed = await _furniture(session)
    total = int(await session.scalar(select(func.count()).select_from(DocumentSection)) or 0)
    papers = len({row.paper_id for row in doomed})

    if not apply or not doomed:
        logger.info(
            "sections_prune_noop" if apply else "sections_prune_dry_run",
            extra={"sections": len(doomed), "papers": papers},
        )
        return PruneReport(
            sections=len(doomed),
            papers=papers,
            chunks_unlinked=0,
            kept=total,
            applied=apply,
        )

    # Unlink first: the chunks are the corpus, the labels are the navigation.
    # The other way round would leave pointers to rows already gone.
    unlinked = 0
    for row in doomed:
        unlinked += int(
            await session.scalar(
                select(func.count())
                .select_from(Chunk)
                .where(Chunk.paper_id == row.paper_id, Chunk.section_ordinal == row.ordinal)
            )
            or 0
        )
        await session.execute(
            update(Chunk)
            .where(Chunk.paper_id == row.paper_id, Chunk.section_ordinal == row.ordinal)
            .values(section_ordinal=None)
        )
        await session.execute(
            delete(DocumentSection).where(
                DocumentSection.paper_id == row.paper_id,
                DocumentSection.ordinal == row.ordinal,
            )
        )
    await session.commit()
    logger.info("sections_pruned", extra={"sections": len(doomed), "chunks_unlinked": unlinked})
    return PruneReport(
        sections=len(doomed),
        papers=papers,
        chunks_unlinked=unlinked,
        kept=total - len(doomed),
        applied=True,
    )


async def documents_with_unplaceable_sections(session: AsyncSession) -> list[dict]:
    """Which documents would lose sections, with their titles.

    Reported so a prune can be checked before it is applied: "4103 rows across 88
    papers" is a number, and "the Phononics paper loses 14, all of them arXiv's own
    page furniture" is a decision.
    """
    doomed = await _furniture(session)
    counts: dict[str, int] = {}
    for row in doomed:
        counts[row.paper_id] = counts.get(row.paper_id, 0) + 1
    if not counts:
        return []

    papers = (
        await session.execute(
            select(Paper.id, Paper.doc_key, Paper.title, Paper.kind).where(
                Paper.id.in_(list(counts))
            )
        )
    ).all()
    return sorted(
        (
            {
                "doc_key": row.doc_key,
                "title": row.title,
                "kind": row.kind,
                "sections": counts[row.id],
            }
            for row in papers
        ),
        key=lambda entry: -int(entry["sections"]),
    )
