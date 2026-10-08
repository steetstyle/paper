"""Bibliography extraction.

ArXiv's HTML rendering prints every reference as three ``ltx_bibblock`` spans in
a fixed order — authors, title, then venue and year — inside
``<li class="ltx_bibitem">``. Measured on 1706.03762v7: 40 references, 22 of them
naming an arXiv id, which is what makes a reference resolvable to a corpus row.

This runs on the HTML already downloaded for the paper. The markdown extractor
deliberately *drops* ``.ltx_bibliography`` because a bibliography is not body
text, so the references have to be read from the source separately.

Parsed with :mod:`html.parser` rather than a regex: the blocks contain *nested*
spans (the venue is wrapped in an italic one), and a non-greedy ``</span>`` stops
at the inner tag and silently loses the trailing year.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from html.parser import HTMLParser
from typing import Any

from app.domain.models import Reference
from app.logging import get_logger

logger = get_logger(__name__)

_ARXIV_RE = re.compile(
    r"arXiv[:\s/]*(?:preprint\s*)?"
    r"(\d{4}\.\d{4,5}(?:v\d+)?|[a-z-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?)",
    re.IGNORECASE,
)
_ABS_RE = re.compile(
    r"\babs/(\d{4}\.\d{4,5}(?:v\d+)?|[a-z-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?)\b",
    re.IGNORECASE,
)
_YEAR_RE = re.compile(r"\b(1[89]\d{2}|20\d{2})\b")
_REFNUM_RE = re.compile(r"\[\s*(\d+)\s*\]")
_BIB_HEADING_RE = re.compile(
    r"^#{1,3}\s*(?:\d+[.)]\s*)?(references|bibliography|works cited|literature cited)\s*$",
    re.MULTILINE | re.IGNORECASE,
)
_ENTITY_RE = re.compile(r"&(amp|lt|gt|quot|nbsp|#x27|apos);")


def _unescape(text: str) -> str:
    replacements = {
        "amp": "&", "lt": "<", "gt": ">", "quot": '"',
        "nbsp": " ", "#x27": "'", "apos": "'",
    }
    return _ENTITY_RE.sub(lambda m: replacements[m.group(1)], text)


def _collapse(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


class _BibliographyParser(HTMLParser):
    """Collect the text of each ``ltx_bibitem`` and of each bibblock within it.

    Depth counting is what makes nesting work: an italic venue inside a bibblock
    does not open a new block, and the trailing ", 2014." that sits outside that
    inner span still lands in the block it belongs to.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.items: list[list[str]] = []
        self.refnums: list[str] = []
        self._item: list[str] | None = None
        self._block: list[str] | None = None
        # The [n] marker is a separate span and would otherwise land in the
        # first block, making "authors" read as "[1] Jimmy Lei Ba, ...".
        self._refnum: list[str] = []
        self._in_refnum = False
        # True once a bibblock has been seen in this item, so the refnum marker
        # is not confused for the authors block.
        self._in_body: bool = False
        # (item_depth, block_depth) at which each list started.
        self._item_depth: int = 0
        self._block_depth: int = 0

    # -- helpers
    def _flush_block(self) -> None:
        if self._block is not None and self._item is not None:
            text = _collapse("".join(self._block))
            if text:
                self._item.append(text)
        self._block = None

    def _flush_item(self) -> None:
        self._flush_block()
        if self._item is not None and self._item:
            self.items.append(self._item)
            self.refnums.append("".join(self._refnum))
        self._item = None
        self._refnum = []
        self._in_refnum = False
        self._in_body = False

    @staticmethod
    def _classes(attrs: list[tuple[str, str | None]]) -> str:
        for name, value in attrs:
            if name == "class" and value:
                return value
        return ""

    # -- HTMLParser hooks
    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:  # noqa: D102
        classes = self._classes(attrs)
        if "ltx_bibitem" in classes:
            self._flush_item()
            self._item = []
            self._item_depth = self._depth
            return
        if self._item is None:
            return
        if "ltx_role_refnum" in classes:
            # Only text inside this span is the entry's own marker. Some
            # renderings instead print citations inline in the body
            # ("Krizhevsky et al. [2012]"), and that year must not be mistaken
            # for a reference number.
            self._in_refnum = True
            return
        if "ltx_bibblock" in classes:
            self._flush_block()
            self._block = []
            self._block_depth = self._depth
            self._in_body = True

    def handle_endtag(self, tag: str) -> None:  # noqa: D102
        if self._item is None:
            return
        if self._in_refnum and self._depth <= self._block_depth:
            self._in_refnum = False
        if self._block is not None and self._depth <= self._block_depth:
            self._flush_block()
        if self._depth <= self._item_depth:
            self._flush_item()

    def handle_data(self, data: str) -> None:  # noqa: D102
        if self._item is None:
            return
        if self._block is not None:
            self._block.append(data)
            return
        if self._in_refnum:
            self._refnum.append(data)
            return
        if not self._in_body:
            # Stray text before the first bibblock: part of the surrounding
            # prose, so keep it out of the fields entirely.
            return
        # Between blocks the source has only indentation. Keeping it would make
        # whitespace its own "block" and shift every field by one.
        if data.strip():
            self._item.append(data)

    _depth = 0

    def _track_depth(self) -> None:  # pragma: no cover - trivial
        self._depth += 1


class _DepthTrackingParser(_BibliographyParser):
    """Adds depth bookkeeping, which HTMLParser does not provide."""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:  # noqa: D102
        super().handle_starttag(tag, attrs)
        self._depth += 1

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:  # noqa: D102
        super().handle_starttag(tag, attrs)

    def handle_endtag(self, tag: str) -> None:  # noqa: D102
        self._depth = max(0, self._depth - 1)
        super().handle_endtag(tag)


def _parse_bibliography(html: str) -> list[tuple[list[str], str]]:
    """``(blocks, refnum)`` per bibliography entry."""
    parser = _DepthTrackingParser()
    parser.feed(html)
    parser._flush_item()  # noqa: SLF001 - closing the last item is ours to do
    return list(zip(parser.items, parser.refnums, strict=False))


def _pick_arxiv_id(text: str) -> str | None:
    for pattern in (_ARXIV_RE, _ABS_RE):
        match = pattern.search(text)
        if match:
            return match.group(1)
    return None


def _pick_year(text: str) -> int | None:
    match = _YEAR_RE.search(text)
    return int(match.group(1)) if match else None


def _from_blocks(blocks: Sequence[str], ordinal: int) -> Reference:
    """Map arXiv's fixed block order onto the reference fields.

    Positional, because that is what the rendering guarantees. Anything heuristic
    here would misfire on titles that end in a question mark or contain a comma,
    which is most of them.
    """
    authors = blocks[0] if len(blocks) > 1 else None
    title = blocks[1] if len(blocks) > 1 else (blocks[0] if blocks else None)
    venue_block = blocks[-1] if len(blocks) > 2 else None
    venue = None
    year = None
    if venue_block is not None:
        year = _pick_year(venue_block)
        venue = venue_block
        if title == venue_block:
            venue, title = None, blocks[0] if blocks else None
    raw = " ".join(blocks)
    return Reference(
        raw_text=raw,
        ordinal=ordinal,
        title=title,
        authors=authors,
        year=year,
        venue=venue,
        cited_arxiv_id=_pick_arxiv_id(raw),
    )


def extract_references(html: str | None) -> list[Reference]:
    """Parse a bibliography out of an arXiv/ar5iv HTML rendering.

    An empty list means "no bibliography found", which is normal for PDF-only
    papers and for rendering styles this does not understand — callers treat it
    as an outcome, not a failure.
    """
    if not html or "ltx_bibitem" not in html:
        return []

    references: list[Reference] = []
    used: set[int] = set()
    for index, (blocks, refnum_text) in enumerate(_parse_bibliography(html), start=1):
        if not blocks:
            continue
        # Ordinal must be unique per citing paper (it is part of the primary
        # key of the uniqueness constraint), so a printed marker is only trusted
        # when it is plausible *and* unused. Document order is always a safe
        # fallback: the bibliography is printed in citation order.
        refnum = _REFNUM_RE.search(refnum_text)
        candidate = int(refnum.group(1)) if refnum else None
        ordinal = candidate if candidate is not None and candidate < 1000 and candidate not in used else index
        used.add(ordinal)
        references.append(_from_blocks(blocks, ordinal))

    logger.debug(
        "references_extracted",
        extra={
            "count": len(references),
            "with_arxiv_id": sum(1 for r in references if r.cited_arxiv_id),
        },
    )
    return references


def extract_references_from_text(markdown: str | None) -> list[Reference]:
    """Fallback when only markdown or plain text is available.

    Much weaker: the block structure is gone, so authors and title cannot be
    separated and only ``raw_text`` and the year are recovered.
    """
    if not markdown:
        return []
    heading = _BIB_HEADING_RE.search(markdown)
    if heading is None:
        return []
    body = markdown[heading.end():]
    next_heading = re.search(r"^#{1,3}\s+\S", body, re.MULTILINE)
    if next_heading:
        body = body[: next_heading.start()]

    references: list[Reference] = []
    for line in body.splitlines():
        text = line.strip()
        if not text or text.startswith("#") or not re.match(r"^[\[(]?\d+[\]).]", text):
            continue
        raw = re.sub(r"^[\[(]?\d+[\]).]\s*", "", text)
        references.append(
            Reference(
                raw_text=raw,
                ordinal=len(references) + 1,
                year=_pick_year(raw),
                cited_arxiv_id=_pick_arxiv_id(raw),
            )
        )
    return references


def summarise(references: Sequence[Any]) -> dict[str, int]:  # noqa: ANN401 - Reference
    """Counts for run metadata and CLI/API summaries."""
    return {
        "total": len(references),
        "with_arxiv_id": sum(1 for r in references if r.cited_arxiv_id),
    }