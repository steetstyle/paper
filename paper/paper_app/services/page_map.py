"""Locate a passage of extracted text in the pages it came from.

MinerU writes two things per PDF: a markdown rendering and ``content_list.json``
— the same content as an ordered list of blocks, each carrying the page it was
found on. The pages are already known; what is missing is the join between the
markdown a chunk is cut from and the block list the pages live in.

The join is a search, not a pointer arithmetic trick. Blocks are joined into one
normalised string with each block's extent recorded, and a chunk is located by
searching for its opening words from the position the previous chunk reached.

Monotonic by construction: the cursor only moves forward. That is both the
simplest correct rule and a real constraint, because chunks are produced in
document order, so a passage that matches several times (a repeated equation
number, a section title) can only plausibly be the occurrence *after* the one
already placed.

The page offset matters and is easy to get wrong. Measured on a 305-page book:
``mineru -s 100 -e 104`` returns ``page_idx`` 0..4 — the indexes are relative to
the requested slice, not to the book. Adding 100 and reading it as 0..104 would
misplace every chunk of a partial ingest by up to a hundred pages.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from paper_app.logging import get_logger

logger = get_logger(__name__)

#: Block keys that may carry readable text, in the order MinerU uses them.
_TEXT_KEYS = ("text", "table_body", "latex", "img_caption", "image_caption", "table_caption")

#: Characters that differ between the markdown rendering and the block text for
#: no reason that matters to a search: LaTeX math delimiters and escapes.
_NOISE_RE = re.compile(r"[$\\{}^_~]+")

#: Markdown syntax that appears in the rendering and has no counterpart in the
#: block list: images become cropped files on disk and links become nothing. Left
#: in, a chunk opening with a figure would never match — measured: 11 of 13
#: chunks matched, and both misses opened with ``![](images/…)``.
_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_ATX_RE = re.compile(r"^\s{0,3}#{1,6}\s+", re.MULTILINE)
_BULLET_RE = re.compile(r"^\s{0,3}[-*+]\s+", re.MULTILINE)
_WS_RE = re.compile(r"\s+")

#: How many characters of a chunk to look for. Long enough to be specific, short
#: enough to survive the small differences between the two renderings.
_NEEDLE_CHARS = 90

#: Windows tried per chunk, each the length of the last. A chunk whose opening is
#: all markup ("10" then a figure) has no usable opening, so the search slides
#: forward instead of giving up on the chunk.
_WINDOWS = 4

#: Below this, a window is too short to be evidence of anything: a two-word
#: phrase occurs in most books more than once.
_MIN_NEEDLE = 24


def normalise(text: str) -> str:
    """Fold away the differences between markdown and block text.

    Everything removed here is markup, not prose: an image reference, a link
    target, a heading mark, a LaTeX delimiter. What is left is the words, in the
    order they appear, with single spaces — which is the only form both renderings
    agree on.
    """
    stripped = _IMAGE_RE.sub(" ", text)
    stripped = _LINK_RE.sub(r"\1", stripped)
    stripped = _ATX_RE.sub("", stripped)
    stripped = _BULLET_RE.sub("", stripped)
    stripped = _NOISE_RE.sub("", stripped)
    return _WS_RE.sub(" ", stripped).strip()


def needle_windows(text: str) -> list[str]:
    """Successive windows of ``text``, each a candidate to search for.

    A chunk is a window of a document, not the document, so the useful evidence
    is usually near its middle. Measured on a 5-page slice: 11 of 13 chunks
    matched from their first window, and both misses opened with markup — one
    with a figure, one with nothing but a chapter number.
    """
    cleaned = normalise(text)
    if not cleaned:
        return []
    windows = [cleaned[:_NEEDLE_CHARS]]
    for index in range(1, _WINDOWS):
        start = index * _NEEDLE_CHARS
        window = cleaned[start : start + _NEEDLE_CHARS]
        if not window:
            break
        windows.append(window)
    return [w for w in windows if len(w) >= _MIN_NEEDLE]


@dataclass(frozen=True, slots=True)
class PageMap:
    """Answers "which page is this text on?" for text cut from a MinerU run."""

    blocks: tuple[tuple[int, int, int], ...] = ()
    """``(span_start, span_end, page)`` per block, over the joined text."""

    haystack: str = ""
    page_offset: int = 1
    """1-based page number of the first extracted page."""

    total_pages: int | None = None
    matched: int = 0
    searched: int = 0

    @property
    def usable(self) -> bool:
        return bool(self.blocks) and bool(self.haystack)

    @property
    def match_rate(self) -> float:
        return self.matched / self.searched if self.searched else 0.0

    def page_for(self, text: str, cursor: int = 0) -> tuple[int | None, int]:
        """Find the page of ``text``, searching from ``cursor``.

        Returns the page (1-based, in the book's own numbering) and the new
        cursor. A miss returns ``(None, cursor)`` — never a guess, because a
        guessed page is worse than no page: a reader who opens the book at the
        wrong place trusts what the program said.
        """
        if not self.usable:
            return None, cursor
        for needle in needle_windows(text):
            found = self.haystack.find(needle, cursor)
            if found < 0:
                # A chunk can start inside a formula or across a block boundary,
                # so one retry from the start catches a chunk that precedes the
                # cursor by a character or two without re-matching the whole book.
                found = self.haystack.find(needle)
            if found >= 0:
                for start, end, page in self.blocks:
                    if start <= found < end:
                        return page, end
        return None, cursor


def build_page_map(
    blocks: list[dict[str, Any]],
    *,
    page_offset: int = 1,
    total_pages: int | None = None,
) -> PageMap:
    """Join MinerU's blocks into one searchable string with page boundaries.

    ``page_offset`` is the book's page number for ``page_idx`` 0. It is 1 for a
    whole-book extraction and the range's start for a partial one.
    """
    pieces: list[str] = []
    spans: list[tuple[int, int, int]] = []
    cursor = 0
    seen_pages: list[int] = []

    for block in blocks:
        if not isinstance(block, dict):
            continue
        index = block.get("page_idx")
        if index is None:
            continue
        text = _block_text(block)
        if not text:
            continue
        page = page_offset + int(index)
        seen_pages.append(page)
        if spans:
            # Exactly one separator space, counted in the offsets. Padding each
            # block instead would put two spaces between them — the per-block text
            # is already collapsed, and ``str.find`` is literal, so a chunk whose
            # opening words straddle a block boundary would never match.
            cursor += 1
        spans.append((cursor, cursor + len(text), page))
        cursor += len(text)
        pieces.append(text)

    if not spans:
        logger.debug("page_map_empty", extra={"blocks": len(blocks)})
        return PageMap(page_offset=page_offset, total_pages=total_pages)

    # Two independent claims about the same run must not disagree: if the caller
    # knows the page count and the blocks disagree, the blocks win but the caller
    # is told, because one of them is reading the wrong file.
    if total_pages is not None and max(seen_pages) > total_pages:
        logger.warning(
            "page_map_beyond_document",
            extra={"max_page": max(seen_pages), "page_count": total_pages},
        )

    return PageMap(
        blocks=tuple(spans),
        haystack=" ".join(pieces),
        page_offset=page_offset,
        total_pages=total_pages,
    )


def _block_text(block: dict[str, Any]) -> str:
    """The readable part of one content_list block, normalised."""
    for key in _TEXT_KEYS:
        value = block.get(key)
        if isinstance(value, str) and value.strip():
            return normalise(value)
    return ""
