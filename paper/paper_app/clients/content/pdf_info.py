"""What a PDF says about itself, before anything tries to extract it.

A separate module because two callers need it for different reasons and neither
owns the other: the fetcher needs the page count to size a local ingest, and the
section builder needs the outline. Both want the same pass over the file, and
opening a 700-page book twice for two integers is not free.

Everything here is best-effort and never raises on a malformed document: a PDF
with a broken xref table should still reach the extractor, which has its own
recovery, rather than being rejected at the door with a stack trace.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from paper_app.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class PdfInfo:
    """Facts about a PDF that do not require extracting its text."""

    page_count: int | None
    """Real page count, or None if the file could not be opened."""

    meta: dict[str, Any] = field(default_factory=dict)
    """The ``/Info`` dictionary verbatim: ``/Title``, ``/Author``, ``/CreationDate``…

    Unfiltered on purpose. Measured on three textbooks used to design this:
    ``/Title`` was ``'Modern Condensed Matter Physics'`` for one and
    ``'pethick.dvi'`` for another — a LaTeX job name. Filtering belongs to
    whoever needs a title, who can see the raw string; filtering here would hide
    the evidence.
    """

    outline: tuple[OutlineNode, ...] = ()
    """The bookmark tree, flattened depth-first with its depth kept per entry.

    Empty for books that have none. Measured: 341 entries for Girvin, 174 for
    Oxford's *Solid State Basics*, **0** for Pethick's *Superconductivity* — one
    book in three ships without bookmarks, so this can never be the only source
    of a document's structure.
    """

    encrypted: bool = False


@dataclass(frozen=True, slots=True)
class OutlineNode:
    """One bookmark: its title, its depth, and where it lands in the file.

    ``page`` is 1-based to match what a reader sees when they open the PDF, and
    ``page_end`` is filled in by walking the flat list: a bookmark covers up to
    the page before the next bookmark at the same or shallower depth. That is why
    this is a flat list with a ``depth`` rather than a nested tree — the nesting
    is recoverable, the ranges are not, without it.
    """

    title: str
    depth: int
    page: int | None = None
    page_end: int | None = None


def read_pdf_info(path: Path) -> PdfInfo:
    """Open a PDF and report its structure. Never raises for a readable file."""
    try:
        from pypdf import PdfReader  # noqa: PLC0415 - optional heavy import
    except ImportError:  # pragma: no cover - pypdf is a hard dependency in practice
        logger.warning("pypdf_missing", extra={"path": str(path)})
        return PdfInfo(page_count=None)

    try:
        reader = PdfReader(str(path))
    except Exception as exc:  # noqa: BLE001 - any parse failure means "unknown"
        logger.warning("pdf_unreadable", extra={"path": str(path), "error": str(exc)})
        return PdfInfo(page_count=None)

    try:
        page_count = len(reader.pages)
    except Exception:  # noqa: BLE001
        page_count = None

    meta: dict[str, Any] = {}
    try:
        for key, value in (reader.metadata or {}).items():
            meta[str(key)] = str(value)
    except Exception:  # noqa: BLE001
        logger.debug("pdf_info_unreadable", extra={"path": str(path)})

    try:
        encrypted = bool(reader.is_encrypted)
    except Exception:  # noqa: BLE001
        encrypted = False

    outline = _flatten_outline(reader)
    return PdfInfo(
        page_count=page_count,
        meta=meta,
        outline=with_page_ranges(outline),
        encrypted=encrypted,
    )


def _flatten_outline(reader: Any) -> tuple[OutlineNode, ...]:
    """Depth-first flattening of the bookmark tree, resolving destinations.

    pypdf returns a list whose elements are either ``Destination`` objects or
    nested lists of them; both levels are walked, and a destination that points
    nowhere (``None``) still becomes a node with ``page=None`` because its title
    is real even when its target is not.
    """
    try:
        raw = reader.outline
    except Exception:  # noqa: BLE001 - a missing outline is normal, not an error
        return ()
    if not raw:
        return ()

    nodes: list[OutlineNode] = []
    resolver = getattr(reader, "get_destination_page_number", None)

    def walk(items: list[Any], depth: int) -> None:
        for item in items:
            if isinstance(item, list):
                walk(item, depth + 1)
                continue
            title = getattr(item, "title", None)
            if not title:
                continue
            nodes.append(
                OutlineNode(
                    title=str(title).strip(),
                    depth=depth,
                    page=_destination_page(item, resolver),
                )
            )

    walk(raw, 0)
    return tuple(nodes)


def _destination_page(destination: Any, resolver: Any) -> int | None:
    """Resolve a bookmark to a 1-based page number, or None if it has none.

    Delegated to pypdf rather than computed here. ``Destination.page.idnum`` is
    the *indirect object id* of the page, not its position in the file: reading
    it as an index put the last bookmark of a 721-page book on page 2562. Only
    the library knows how to turn an indirect reference into a position.
    """
    if resolver is None:
        return None
    try:
        return int(resolver(destination)) + 1
    except Exception:  # noqa: BLE001 - a broken destination is not fatal
        return None

def with_page_ranges(nodes: tuple[OutlineNode, ...]) -> tuple[OutlineNode, ...]:
    """Fill each node's ``page_end`` from the next sibling of equal-or-lower depth.

    A bookmark tree states where each section *starts*; the end is implied by
    whatever comes next. Without this a search hit on page 400 could only say
    "somewhere in chapter 9" rather than "pages 377-402", which is the difference
    between navigating to a passage and navigating to a chapter.

    The last section in the book keeps ``page_end=None``: there is nothing after
    it to bound it, and inventing an end from a page count that may itself be
    wrong is worse than admitting the range is open.
    """
    if not nodes:
        return ()

    out: list[OutlineNode] = []
    for index, node in enumerate(nodes):
        end: int | None = None
        for later in nodes[index + 1 :]:
            if later.depth <= node.depth:
                end = later.page
                break
        out.append(
            OutlineNode(
                title=node.title,
                depth=node.depth,
                page=node.page,
                # An end before its own start means the bookmark tree is out of
                # order, which happens in hand-made PDFs. Keep the start and
                # drop the end rather than report a negative-length range.
                page_end=end if end is None or node.page is None or end >= node.page else None,
            )
        )
    return tuple(out)
