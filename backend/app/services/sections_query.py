"""Finding sections by what a reader calls them.

The vector store cannot do this: it knows chunk metadata and nothing about what a
heading *is*. Section titles live in ``document_sections``, and resolving "Debye" to
"section 2.2, pages 24-31" is a database question that has to be answered before a
filter is built.

Matching happens in Python rather than in SQL, which costs a table scan over the
section rows — a few thousand for a corpus of books — and buys three things that
matter more:

* **Word boundaries.** ``Exercises`` must not come back for ``ex``. A substring
  match hands a reader a chapter they did not ask for, with total confidence and
  no way to tell it was a guess.
* **Ranking.** Asking for "Debye" should rank ``2.2 Debye's Calculation`` above
  ``2 Specific Heat of Solids: Boltzmann, Einstein, and Debye``, which merely
  contains the word. Without an order the corpus answers in whatever order the
  database returns.
* **Dialect portability.** The suite runs on SQLite, which has no regex operator;
  ``ILIKE`` is Postgres-only and ``LIKE`` is case-sensitive on Postgres. A word
  boundary spelled in SQL is three dialects' problem.

Two spellings are accepted, because both are how people refer to a passage: a
book's own numbering (``7``, ``2.2``) and words (``"Debye"``, ``"abrikosov"``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import DocumentSection, Paper
from app.logging import get_logger

logger = get_logger(__name__)

#: A book writes its own numbering as ``2``, ``2.2`` or ``2.2.1``; a chapter may
#: also be ``IV`` or ``A``. Anything else is words.
_NUMBERING_RE = re.compile(r"^(?:[IVXLCDM]+|[A-Z]|\d+(?:\.\d+)*)$")

#: The numbering prefix of a title, e.g. ``2.2`` in ``2.2 Debye's Calculation``.
_LEADING_NUMBER_RE = re.compile(r"^\s*(?:chapter\s+|appendix\s+|section\s+|part\s+)?"
                                r"(\d+(?:\.\d+)*|[IVXLCDM]+|[A-Z])[.):]?\s*", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class SectionMatch:
    """One section a query resolved to, with the pages it covers."""

    paper_id: str
    ordinal: int
    title: str
    level: int
    page_start: int | None
    page_end: int | None
    source: str
    paper_title: str = ""
    display_id: str = ""
    match_kind: str = "contains"
    """How the title matched: ``exact``, ``prefix`` or ``contains``.

    Reported rather than hidden, because it is the honest answer to "why did it
    pick that one" — and because a caller narrowing a search to one section should
    be able to see it matched by substring only.
    """

    @property
    def key(self) -> tuple[str, int]:
        """The identity a :class:`~app.db.vector_store.base.VectorFilter` wants."""
        return (self.paper_id, self.ordinal)

    @property
    def pages(self) -> str:
        if self.page_start is None:
            return "-"
        if self.page_end is None or self.page_end == self.page_start:
            return str(self.page_start)
        return f"{self.page_start}-{self.page_end}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "doc_key": self.display_id,
            "title": self.paper_title,
            "ordinal": self.ordinal,
            "section": self.title,
            "level": self.level,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "pages": self.pages,
            "source": self.source,
            "matched": self.match_kind,
        }


def is_section_number(value: str) -> bool:
    """Whether ``value`` is a book's own numbering rather than words."""
    return bool(_NUMBERING_RE.match(value.strip()))


def _match_kind(title: str, needle: str) -> str | None:
    """How ``needle`` matches ``title``, or ``None`` if it does not.

    Word boundaries throughout: a term matches on a word edge, never inside one.
    """
    haystack = title.lower()
    if haystack == needle:
        return "exact"
    pattern = rf"(?<![0-9a-z]){re.escape(needle)}(?![0-9a-z])"
    found = re.search(pattern, haystack)
    if not found:
        return None
    # A prefix of the *title*, ignoring any numbering in front of it: asking for
    # "debounce" should rank `2.3 Debounce Theory` above `2.4 Damped Waves`.
    stripped = _LEADING_NUMBER_RE.sub("", title).lower()
    return "prefix" if stripped.startswith(needle) else "contains"


async def find_sections(
    session: AsyncSession,
    query: str,
    *,
    doc_keys: list[str] | None = None,
    max_level: int | None = None,
    limit: int = 50,
) -> list[SectionMatch]:
    """Sections matching ``query``, best match first.

    ``doc_keys`` narrows to particular documents. Left off, the answer is a question
    about the whole corpus and will have hundreds of rows, so a caller who means one
    book should say so.
    """
    text = query.strip()
    if not text:
        return []

    stmt = select(DocumentSection, Paper).join(Paper, Paper.id == DocumentSection.paper_id)
    if doc_keys:
        paper_ids = select(Paper.id).where(Paper.doc_key.in_(list(doc_keys)))
        stmt = stmt.where(DocumentSection.paper_id.in_(paper_ids))
    if max_level is not None:
        stmt = stmt.where(DocumentSection.level <= max_level)

    rows = (await session.execute(stmt)).all()

    if is_section_number(text):
        return _numbered(rows, text, limit)
    return _by_words(rows, text.lower(), limit)


def _numbered(rows, wanted: str, limit: int) -> list[SectionMatch]:  # noqa: ANN001
    """Match a book's own numbering: ``2.2`` finds ``2.2 Debye's Calculation``."""
    needle = wanted.strip().rstrip(".)").lower()
    out: list[tuple[int, SectionMatch]] = []
    for section, paper in rows:
        leading = _LEADING_NUMBER_RE.match(section.title)
        number = leading.group(1).lower() if leading else None
        if number != needle:
            continue
        # Exact number wins; then the shortest title, which is the section rather
        # than the chapter that happens to carry the same number.
        rank = len(section.title)
        out.append((rank, _match(section, paper, "exact")))
    out.sort(key=lambda pair: pair[0])
    return [match for _rank, match in out[:limit]]


def _by_words(rows, needle: str, limit: int) -> list[SectionMatch]:  # noqa: ANN001
    """Match words, best first: exact, then prefix, then contains.

    Ties break on level and then on page, so ``1 Introduction`` at page 4 comes
    before ``9.2 Introduction`` at page 400 — the reader almost always means the
    earlier, shallower one.
    """
    ranked: list[tuple[int, int, int, SectionMatch]] = []
    for section, paper in rows:
        kind = _match_kind(section.title, needle)
        if kind is None:
            continue
        tier = {"exact": 0, "prefix": 1, "contains": 2}[kind]
        ranked.append((tier, section.level, section.page_start or 0,
                       _match(section, paper, kind)))
    ranked.sort(key=lambda row: row[:3])
    return [match for *_rank, match in ranked[:limit]]


def _match(section, paper, kind: str) -> SectionMatch:  # noqa: ANN001, ANN202
    return SectionMatch(
        paper_id=section.paper_id,
        ordinal=section.ordinal,
        title=section.title,
        level=section.level,
        page_start=section.page_start,
        page_end=section.page_end,
        source=section.source,
        paper_title=paper.title,
        display_id=paper.display_id,
        match_kind=kind,
    )


def sections_within(match: SectionMatch, sections) -> list:  # noqa: ANN001
    """Sections nested inside ``match``, in document order.

    Decided by **page range**, not by level. The two structural sources disagree
    about depth in the same book — measured on *Solid State Basics*: the PDF
    bookmarks put ``2.2 Debye's Calculation`` at level 3 while the markdown read
    ``2.2.1 Periodic…`` as level 3 too — so a level comparison finds no children
    for a section that visibly has them. Pages are what a reader sees as
    containment, and both sources agree on pages even when they disagree on
    hierarchy.

    Level is the fallback for sections with no page, rather than dropping them: an
    unlocated heading is still under something.
    """
    rows = sorted(sections, key=lambda row: row.ordinal)
    end = match.page_end if match.page_end is not None else match.page_start
    inside: list = []
    for row in rows:
        if row.ordinal <= match.ordinal:
            continue
        if match.page_start is not None and row.page_start is not None:
            # Stops at the first section that leaves the range: everything after it
            # belongs to a sibling or an ancestor, not to this one.
            if end is not None and row.page_start > end:
                break
            if row.page_start < match.page_start:
                continue
            inside.append(row)
        elif row.level > match.level:
            if inside and row.level <= inside[-1].level:
                break
            inside.append(row)
    return inside
