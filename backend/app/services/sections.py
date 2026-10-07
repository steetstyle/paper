"""Recover a document's own structure: chapters, sections, page ranges.

There is no single source of truth, which is the whole problem. Measured on the
three books this was built for:

============================  ======  ========  ============================
book                          pages   outline   headings in MinerU markdown
============================  ======  ========  ============================
Girvin, Condensed Matter       721      341     sparse, and all at ATX depth 1
Oxford, Solid State Basics     305      174     sparse
Pethick, Superconductivity     578        0     8 over a 13-page slice
============================  ======  ========  ============================

So neither source can be treated as the fallback for the other: one book in three
has no bookmarks at all, and MinerU emits every heading at the same ATX depth,
which means ``#`` count carries no information about hierarchy.

Depth is therefore taken from the heading's own *numbering* — ``2.3.1`` is level 3
— because that is the one place a book states its own structure in a way that
survives extraction. Unnumbered headings inherit the level of the numbered heading
that precedes them, which is a guess; it is recorded as such in the ``source``
column rather than presented as fact.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Protocol

from app.clients.content.pdf_info import OutlineNode
from app.logging import get_logger
from app.services.page_map import PageMap, normalise

logger = get_logger(__name__)

#: ``2.4.3 Title`` or ``Chapter 7 Title`` or ``Appendix B Title``.
# The optional bracket matters: a physics textbook writes its figure captions and
# section titles as "(2.2) Debye Theory I", and without it every one of those
# counted as unnumbered and inherited a level from whatever came before — which
# chained (2.1) < (b) < (2.2) < (2.3) four levels deep inside one appendix.
_NUMBERING_RE = re.compile(
    r"^[\(\[]?\s*(?:chapter\s+|appendix\s+|section\s+|part\s+)?"
    r"(\d+(?:\.\d+)*|[IVXLC]+|[A-Z])"
    r"[.:)\]]?\s+(\S.*)$",
    re.IGNORECASE,
)

#: Roman-numeral headings (``IV. Title``) have no dots to count, so their level
#: comes from the numeral itself rather than from a component count.
_ROMAN_RE = re.compile(r"^\s*([IVXLC]+)[.:)]?\s+(\S.*)$", re.IGNORECASE)

#: A heading shorter than this with no terminal period is a caption or a stray
#: line of running text; anything that long is a heading whatever its punctuation.
_MIN_HEADING_CHARS = 3

#: ``2 The Ginzburg-Landau model 31`` — a heading with a page number bolted on,
#: which is what a table-of-contents line looks like once it is read as a heading.
_CONTENTS_LINE_RE = re.compile(r"^(.*\S)\s+(\d{1,4})$")

_WHITESPACE_RE = re.compile(r"\s+")
_PUNCTUATION_RE = re.compile(r"[\u2014:;,.!?()\[\]&]+")

#: The numbering a heading puts in front of itself: ``2.4.1 Title``,
#: ``Chapter 7 Title``, ``(2.2) Debye Theory I``. Also accepts a bracketed form,
#: because a physics textbook writes its section titles that way and reading those
#: as unnumbered chained a run of them four levels deep inside one appendix.
_LEADING_NUMBER_RE = re.compile(
    r"^\s*[\(\[]?\s*(?:chapter\s+|appendix\s+|section\s+|part\s+)?"
    r"(\d+(?:\.\d+)*|[IVXLCDM]+|[A-Z])"
    r"[.:)\]]?\s+",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class Section:
    """One structural division, with the pages it spans."""

    ordinal: int
    title: str
    level: int
    page_start: int | None = None
    page_end: int | None = None
    source: str = "outline"
    """``outline``, ``markdown`` or ``merged`` — where this row's evidence came
    from. Kept because the two sources disagree often enough that a reader
    deserves to know which one told them a section exists."""


@dataclass(frozen=True, slots=True)
class Heading:
    """A heading found in the extracted markdown, before pages are known."""

    title: str
    level: int
    char_start: int
    numbered: bool


def parse_markdown_headings(markdown: str) -> list[Heading]:
    """Every ATX heading in MinerU's markdown, with depth inferred from numbering.

    The inference is the point. Reading depth off the count of ``#`` yields a
    flat list for a book — measured, MinerU writes ``#`` for a subsection and a
    chapter alike — so ``2.3.1`` is read as level 3 and an unnumbered heading
    inherits from whatever numbered heading preceded it.
    """
    raw: list[tuple[str, int, int]] = []
    for match in re.finditer(r"^#{1,6}[ \t]+(.+?)[ \t]*#*[ \t]*$", markdown, re.MULTILINE):
        title = match.group(1).strip()
        if len(title) < _MIN_HEADING_CHARS:
            continue
        raw.append((title, match.start(), match.end(1)))

    headings: list[Heading] = []
    previous_level = 1
    for title, start, _end in raw:
        level, numbered = infer_level(title, previous_level)
        previous_level = level
        headings.append(Heading(title=title, level=level, char_start=start, numbered=numbered))
    return headings


def infer_level(title: str, fallback: int = 1) -> tuple[int, bool]:
    """Depth of a heading from its own numbering.

    ``2.4`` is level 2 and ``2.4.3`` is level 3, because a component count *is*
    the book's own depth convention. Returns ``(level, numbered)`` so the caller
    knows whether the answer was read or inherited.
    """
    roman = _ROMAN_RE.match(title)
    if roman and not title[:1].isdigit():
        return _roman_level(roman.group(1)), True

    numbered = _NUMBERING_RE.match(title)
    if not numbered:
        return fallback, False
    number = numbered.group(1)
    if number.replace(".", "").isdigit():
        return min(number.count(".") + 1, 6), True
    if len(number) == 1 and number.isalpha():
        return 1, True
    return 1, True


def _roman_level(numeral: str) -> int:
    """Depth from a roman numeral: ``II`` is a part, ``IV`` a chapter."""
    return 1 if len(numeral) <= 2 else 2


def build_sections(
    *,
    outline: tuple[OutlineNode, ...] = (),
    markdown: str = "",
    page_map: PageMap | None = None,
    document_pages: int | None = None,
) -> list[Section]:
    """Merge the two structural sources into one ordered, page-bounded list.

    Which source is the spine depends on which one exists, because they are not
    two views of the same thing:

    * **Outline present** — it is the book's own table of contents, ordered and
      paged by the publisher, so it is the spine. Headings the markdown found are
      attached underneath whichever outline section contains them, which is how
      the outline gains detail it did not have without being reordered by an
      extractor.
    * **Outline empty** — the markdown is all there is. Its headings become the
      spine, in the order they appear, which is document order.

    Either way, a section survives only if it can be given a page. See
    :func:`_only_navigable` for why that rule exists and what it caught.

    The outline is kept whole even after a partial ingest that covered only part
    of the book: it describes the document, not the slice, and a reader asking
    "what is chapter 9?" should not be told it does not exist because only
    chapters 1-2 were indexed. Chunks only ever link to sections whose pages
    overlap what was actually extracted.
    """
    if outline:
        candidates = _outline_with_details(
            outline=outline,
            markdown=markdown,
            page_map=page_map,
            document_pages=document_pages,
        )
    else:
        candidates = _markdown_sections(markdown, page_map)
    placed = _only_navigable(candidates)
    if not placed:
        return []
    return _fill_ranges(placed, document_pages)


class _HasTitleAndPage(Protocol):
    """Anything with a title and a page: a built ``Section`` or a stored row."""

    @property
    def title(self) -> str: ...

    @property
    def page_start(self) -> int | None: ...


#: Headings that are a *renderer's* furniture rather than a document's structure,
#: taken from what arXiv's page actually produced here — the fourteen that
#: ``paper outline 1211.4482v1`` listed under the paper's title.
#:
#: A blocklist, and the reasoning for accepting one: the two rules tried before it
#: both destroyed real structure, which is the worse error. Requiring a page threw
#: out genuine numbered sections of HTML-ingested papers (an HTML page has no
#: pages). Requiring a number then threw out ``II.1 The Assortative Skeleton
#: Network`` and ``Results`` and ``Discussion``, because plenty of real headings are
#: unnumbered names and plenty are numbered with Roman numerals.
#:
#: The asymmetry decides it. An unknown future heading is an extra row in a table of
#: contents; a deleted one is structure the reader cannot find. So this list is
#: consulted only for headings with no page to point at, and matched on the whole
#: normalised title — ``References & Citations`` is furniture and ``References`` is
#: not, which no substring rule could tell apart.
_FURNITURE = frozenset({
    "title",
    "submission history",
    "access paper",
    "current browse context",
    "references citations",
    "references and citations",
    "bibtex formatted citation",
    "bookmark",
    "bibliographic and citation tools",
    "code data and media associated with this article",
    "demos",
    "recommenders and search tools",
    "arxivlabs experimental projects",
    "arxivlabs experimental projects with community collaborators",
    "keywords",
})


def _only_navigable(sections: list[Section]) -> list[Section]:
    """Drop headings that are the renderer's furniture rather than structure.

    A heading is kept if it can be pointed at on a page — which every book section
    and every PDF bookmark can — or if it is not recognisable as page furniture.
    Only the second path can drop anything, and only for a heading with no page.

    Measured: across 88 documents the chrome headings of one arXiv paper
    outweighed the 322 genuine sections in the corpus, and ``paper show`` announced
    "sections: 14" for a paper that has none.
    """
    kept = [section for section in sections if not _is_furniture(section)]
    if len(kept) != len(sections):
        logger.info(
            "sections_dropped_furniture",
            extra={
                "kept": len(kept),
                "dropped": len(sections) - len(kept),
                "examples": [
                    section.title[:60] for section in sections if _is_furniture(section)
                ][:3],
            },
        )
    return kept


#: Furniture whose text varies per document, so a fixed string cannot match it.
#: The abs page writes ``Title: <the paper's own title>`` and a subject breadcrumb
#: as ``Condensed Matter > Mesoscale and Nanoscale Physics`` — both are the page
#: describing the paper, not the paper describing itself.
_FURNITURE_PATTERNS = (
    re.compile(r"^title\b", re.IGNORECASE),
    re.compile(r">"),  # the subject breadcrumb
)


def _is_furniture(section: _HasTitleAndPage) -> bool:
    """Whether a heading is a renderer's page furniture rather than structure.

    Takes either a :class:`Section` or a ``DocumentSection`` row, because the same
    question is asked twice — once while building a document, once while pruning
    rows stored by an earlier build — and two structurally identical predicates
    would eventually disagree. Typed structurally rather than by union so neither
    module has to import the other's class.
    """
    if section.page_start is not None:
        # Navigable: whatever it is called, the reader can be sent there.
        return False
    title = section.title.strip()
    if any(pattern.search(title) for pattern in _FURNITURE_PATTERNS):
        return True
    return _normalise_heading(title) in _FURNITURE


def _normalise_heading(title: str) -> str:
    """Fold a heading to the form the furniture list is written in."""
    return _WHITESPACE_RE.sub(" ", _PUNCTUATION_RE.sub(" ", title)).strip().lower()


def _outline_with_details(
    *,
    outline: tuple[OutlineNode, ...],
    markdown: str,
    page_map: PageMap | None,
    document_pages: int | None,
) -> list[Section]:
    """The outline, with markdown headings attached beneath it."""
    from_markdown = _markdown_sections(markdown, page_map)
    sections = [
        Section(
            ordinal=index,
            title=node.title,
            # The outline tree is 0-based; a reader counts from 1.
            level=node.depth + 1,
            page_start=node.page,
            page_end=node.page_end,
            source="outline",
        )
        for index, node in enumerate(outline)
    ]
    seen = {_section_key(section) for section in sections}

    # Hosts are searched in the outline only. Searching the growing list would let
    # one markdown heading host the next, and a run of them would nest as deep as
    # the run is long — a shape no book has.
    hosts = list(sections)
    for extra in from_markdown:
        if _section_key(extra) in seen:
            continue
        host = _containing_section(hosts, extra.page_start)
        if host is None:
            # Outside every outline range: the outline already describes the
            # whole book, so this is either an artefact or a heading the
            # bookmarks omit. Keeping it would put a section in a place the book
            # itself does not have one.
            logger.debug(
                "section_outside_outline",
                extra={"title": extra.title[:60], "page": extra.page_start},
            )
            continue
        seen.add(_section_key(extra))
        # Its own numbering when it has any, otherwise one deeper than the
        # outline entry it sits inside: it is a division *of* that section, but
        # not six levels deep just because five headings preceded it.
        numbered_level, numbered = infer_level(extra.title, fallback=host.level + 1)
        sections.append(
            Section(
                ordinal=-1,
                title=extra.title,
                level=numbered_level if numbered else host.level + 1,
                page_start=extra.page_start,
                source="markdown",
            )
        )

    return sorted(sections, key=_outline_order)


def _containing_section(sections: list[Section], page: int | None) -> Section | None:
    """The tightest section whose page range holds ``page``.

    A missing end means *unbounded*, not "a one-page section": the last bookmark
    in a book has no successor to bound it, and reading its open end as its start
    put every page of the final chapter outside every section.
    """
    if page is None:
        return None
    best: Section | None = None
    for section in sections:
        start = section.page_start
        if start is None or page < start:
            continue
        if section.page_end is not None and page > section.page_end:
            continue
        if best is None or section.level > best.level:
            best = section
    return best


def _markdown_sections(markdown: str, page_map: PageMap | None) -> list[Section]:
    """Headings from the markdown, in document order, with pages where known.

    A heading's page is found by searching for the heading *and the sentence that
    follows it*, never the heading text alone. A section title also appears in the
    body as a cross-reference — "see 2.4.1 Condensed phase" — and searching for
    the title alone finds those first, putting the heading on the wrong page.
    Anchoring to what comes after disambiguates them.
    """
    headings = _without_contents_lines(parse_markdown_headings(markdown))
    if not headings:
        return []

    out: list[Section] = []
    cursor = 0
    for index, heading in enumerate(headings):
        page: int | None = None
        if page_map is not None and page_map.usable:
            page, cursor = _page_of_heading(markdown, heading, page_map, cursor)
        out.append(
            Section(
                ordinal=index,
                title=heading.title,
                level=heading.level,
                page_start=page,
                source="markdown",
            )
        )
    return out


def _without_contents_lines(headings: list[Heading]) -> list[Heading]:
    """Drop headings that are a table-of-contents line rather than a section.

    A book's contents page is a list of headings, and the extractor reads it as
    headings: "2 The Ginzburg-Landau model 31" is a section title with its page
    number attached. Left in, the outline lists every chapter twice — once pointing
    at page 2 and once pointing at page 33.

    The test is deliberately strict: a heading qualifies only if *another* heading
    has exactly its title without the trailing number. Measured across the four
    books used to build this: 3 matches on Annett (the three contents lines, all
    real) and **0** on the other three, so a section that merely happens to end in
    a digit survives.
    """
    titles = {heading.title.strip() for heading in headings}
    kept: list[Heading] = []
    for heading in headings:
        match = _CONTENTS_LINE_RE.match(heading.title.strip())
        if match and match.group(1).strip() in titles:
            logger.info("section_skipped_contents_line", extra={"title": heading.title[:70]})
            continue
        kept.append(heading)
    return kept


def _page_of_heading(
    markdown: str, heading: Heading, page_map: PageMap, cursor: int
) -> tuple[int | None, int]:
    """Locate one heading by the text that follows it."""
    tail = markdown[heading.char_start : heading.char_start + 400]
    return page_map.page_for(tail, cursor)


#: Greek letters by name. NFKD decomposes an accented Latin letter to its base
#: (``ş`` -> ``s``) but leaves ``ζ`` as ``ζ``, which ASCII-encoding then drops
#: entirely — so a heading written in Unicode and the same heading written as
#: LaTeX came out as ``4`` and ``zeta4``. This is the one mapping that has to be
#: spelled out.
_GREEK = str.maketrans(
    {
        "α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta", "ε": "epsilon",
        "ζ": "zeta", "η": "eta", "θ": "theta", "ι": "iota", "κ": "kappa",
        "λ": "lambda", "μ": "mu", "ν": "nu", "ξ": "xi", "ο": "omicron",
        "π": "pi", "ρ": "rho", "σ": "sigma", "ς": "sigma", "τ": "tau",
        "υ": "upsilon", "φ": "phi", "χ": "chi", "ψ": "psi", "ω": "omega",
        "Α": "Alpha", "Β": "Beta", "Γ": "Gamma", "Δ": "Delta", "Ε": "Epsilon",
        "Ζ": "Zeta", "Η": "Eta", "Θ": "Theta", "Ι": "Iota", "Κ": "Kappa",
        "Λ": "Lambda", "Μ": "Mu", "Ν": "Nu", "Ξ": "Xi", "Ο": "Omicron",
        "Π": "Pi", "Ρ": "Rho", "Σ": "Sigma", "Τ": "Tau", "Υ": "Upsilon",
        "Φ": "Phi", "Χ": "Chi", "Ψ": "Psi", "Ω": "Omega",
    }
)


def _section_key(section: Section) -> str:
    r"""Identity for de-duplication: the title, folded hard.

    Deliberately title-only. The same section found at page 41 by the markdown
    and at page 40 by the outline is one section, and the one-page disagreement
    between two extractors is not evidence that there are two of anything.

    Three things are folded because both sources spell them differently:

    * the numbering prefix — "2.4 Thermodynamic quantities" vs
      "Thermodynamic quantities";
    * punctuation and brackets — the outline writes ``ζ(4)``, the extraction
      writes ``$\zeta ( 4 )$``, and leaving them distinct put the same appendix
      in the table of contents twice;
    * the symbols themselves, so the two above meet: ``ζ`` has no NFKD
      decomposition to ASCII, and ``\zeta`` in the extraction is its name, so
      Greek and LaTeX spellings of one letter are folded to one string.

    Not used for locating text, only for deciding whether two headings are the
    same heading: folding this hard would make distinct titles collide.
    """
    folded = normalise(section.title).lower()
    folded = re.sub(r"^(?:chapter|appendix|section|part)\s+", "", folded)
    folded = re.sub(r"^\d+(?:\.\d+)*[.:)\]]?\s+", "", folded)
    # Translate *before* dropping non-ASCII: ``encode(errors="ignore")`` would
    # delete ``ζ`` outright, leaving the LaTeX spelling with nothing to meet.
    spelled = unicodedata.normalize("NFKD", folded).translate(_GREEK)
    ascii_only = spelled.encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "", ascii_only)


def _outline_order(section: Section) -> tuple[int, int]:
    """Outline order for the sort: page, then depth, then origin.

    Markdown headings sort after the outline entry they were attached to at the
    same page and level, so a chapter's own title precedes its subdivisions.
    """
    return (
        section.page_start if section.page_start is not None else 10**9,
        section.level if section.source == "outline" else section.level,
    )


def _fill_ranges(sections: list[Section], document_pages: int | None) -> list[Section]:
    """Give every section an end: the next one's start, closing at the last page.

    A section's end is implied by its successor, which is why a bookmark tree
    alone cannot answer "pages 377-402" — it only ever says where things start.
    """
    result: list[Section] = []
    for index, section in enumerate(sections):
        end = section.page_end
        if end is None:
            end = _next_start(sections, index)
        if end is None and document_pages is not None:
            end = document_pages
        if (
            end is not None
            and section.page_start is not None
            and end < section.page_start
        ):
            # Out-of-order bookmarks happen in hand-made PDFs. Keep the start,
            # drop the end: a negative-length range is worse than no range.
            end = None
        result.append(
            Section(
                ordinal=index,
                title=section.title,
                level=section.level,
                page_start=section.page_start,
                page_end=end,
                source=section.source,
            )
        )
    return result


def _next_start(sections: list[Section], index: int) -> int | None:
    """Where this section ends: the next section at the same or shallower level.

    A deeper section that follows belongs inside this one, so it does not close
    it. That is what makes a chapter span its subsections.
    """
    level = sections[index].level
    for later in sections[index + 1 :]:
        if later.level <= level and later.page_start is not None:
            return later.page_start
    return None
