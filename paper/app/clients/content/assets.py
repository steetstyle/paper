"""Figures, tables and equations, extracted from whichever source we have.

The two sources are complementary rather than redundant, which is why both are
supported:

===============  ==========================  ===========================
                 arXiv HTML                   MinerU (PDF path)
===============  ==========================  ===========================
figures          5, with caption + image URL  5, with page + bbox + crop
tables           4, with caption + body      4, with page + bbox + crop
equations        142 (139 inline + 3 block)  5 (block only), page + bbox
image bytes      only by fetching the URL    already on disk
===============  ==========================  ===========================

So the HTML path wins on equation coverage and cost, the PDF path wins on
locating things on the page. Measured on 1706.03762v7.

Every field except ``caption``/``latex`` is optional, and nothing here fails a
paper: a rendering without figures simply yields none.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

from app.logging import get_logger

logger = get_logger(__name__)

# `Figure 3: the caption` -> label="Figure 3", caption="the caption"
_LABEL_RE = re.compile(r"^\s*(figure|table)\s*([0-9]+[A-Za-z]?)\s*[:.]\s*", re.IGNORECASE)
_WS_RE = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class FigureAsset:
    """A figure. ``image_url`` is only set for HTML-sourced ones."""

    def has_image(self) -> bool:
        return bool(self.image_url or self.image_path or self.image_sha256)

    ordinal: int
    caption: str | None = None
    label: str | None = None
    image_url: str | None = None
    image_sha256: str | None = None
    image_path: str | None = None
    """Local file, for sources that already have the bytes on disk."""
    page_idx: int | None = None
    bbox: tuple[int, int, int, int] | None = None
    width: int | None = None
    height: int | None = None
    source: str = "html"


@dataclass(frozen=True, slots=True)
class TableAsset:
    """A table. ``body_html`` is the raw ``<table>`` subtree."""

    def has_image(self) -> bool:
        return bool(self.image_url or self.image_path or self.image_sha256)

    ordinal: int
    caption: str | None = None
    label: str | None = None
    body_html: str | None = None
    image_url: str | None = None
    image_sha256: str | None = None
    image_path: str | None = None
    page_idx: int | None = None
    bbox: tuple[int, int, int, int] | None = None
    source: str = "html"


@dataclass(frozen=True, slots=True)
class EquationAsset:
    """A formula.

    ``is_display`` separates a numbered display equation from inline math. Both
    are stored because they are different things: a paper has ~5 real equations
    and ~140 inline fragments, and a caller usually wants the first kind.
    """

    ordinal: int
    latex: str | None = None
    is_display: bool = False
    image_sha256: str | None = None
    image_path: str | None = None
    page_idx: int | None = None
    bbox: tuple[int, int, int, int] | None = None
    source: str = "html"


@dataclass(slots=True)
class Assets:
    figures: list[FigureAsset] = field(default_factory=list)
    tables: list[TableAsset] = field(default_factory=list)
    equations: list[EquationAsset] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not (self.figures or self.tables or self.equations)

    def merge(self, other: Assets) -> Assets:
        """Combine two extractions.

        Ordinals are renumbered across the merged set rather than kept, because
        the same figure can arrive from both sources and two rows numbered 1 would
        violate ``unique(paper_id, ordinal)``.
        """
        figures = [*self.figures, *other.figures]
        tables = [*self.tables, *other.tables]
        equations = [*self.equations, *other.equations]
        return Assets(
            figures=[_renumber(f, i) for i, f in enumerate(figures, start=1)],
            tables=[_renumber(t, i) for i, t in enumerate(tables, start=1)],
            equations=[_renumber(e, i) for i, e in enumerate(equations, start=1)],
        )

    def deduplicate(self) -> Assets:
        """Drop assets both sources reported, keeping the better record.

        **Only cross-source duplicates are removed.** A formula that appears
        twice in a paper is two real occurrences, and collapsing them would throw
        away a position in the document. What must not happen is one figure being
        stored twice because both extractors saw it.

        Which copy survives is per kind, because the two sources differ in what
        they know:

        - figures and tables prefer **MinerU**: it has the cropped image bytes and
          the page, where HTML has only a URL.
        - equations prefer **HTML**: it reports every formula including inline
          math (142 on 1706.03762v7 against MinerU's 5), so dropping HTML rows to
          keep MinerU's would lose most of them.
        """
        return Assets(
            figures=_prefer(self.figures, _figure_key, _SOURCE_PREFERENCE["figure"]),
            tables=_prefer(self.tables, _table_key, _SOURCE_PREFERENCE["table"]),
            equations=_prefer(
                self.equations, _equation_key, _SOURCE_PREFERENCE["equation"]
            ),
        )

    def count(self) -> dict[str, int]:
        return {
            "figures": len(self.figures),
            "tables": len(self.tables),
            "equations": len(self.equations),
            "display_equations": sum(1 for e in self.equations if e.is_display),
        }


def _renumber(asset: Any, ordinal: int) -> Any:  # noqa: ANN401
    from dataclasses import replace  # noqa: PLC0415

    return replace(asset, ordinal=ordinal)


def _figure_key(asset: FigureAsset) -> str:
    """Figures and tables match on their printed label, not their caption.

    Captions are re-transcribed differently by the two extractors, so comparing
    them misses duplicates that are obviously the same figure. ``Figure 2`` is
    the same figure in both.
    """
    if asset.label:
        return f"label:{asset.label.strip().lower()}"
    return f"caption:{(asset.caption or '').strip().lower()[:120]}"


def _table_key(asset: TableAsset) -> str:
    if asset.label:
        return f"label:{asset.label.strip().lower()}"
    return f"caption:{(asset.caption or '').strip().lower()[:120]}"


def _equation_key(asset: EquationAsset) -> str:
    """Whitespace-insensitive LaTeX.

    MinerU transcribes letter-spaced LaTeX (``\\mathrm { A t t e n t i o n }``)
    where arXiv's HTML emits ``\\mathrm{Attention}``, so the two spellings of one
    formula only match once every space is dropped.
    """
    latex = re.sub(r"\s+", "", asset.latex or "")
    return f"tex:{latex}" if latex else f"pos:{asset.ordinal}:{asset.page_idx}"


_SOURCE_PREFERENCE: dict[str, str] = {
    "figure": "mineru",
    "table": "mineru",
    "equation": "html",
}
"""Which source wins when both reported the same asset. See `deduplicate`."""


def _prefer(items: list[Any], key, preferred_source: str) -> list[Any]:  # noqa: ANN001
    """Remove cross-source duplicates, keeping every distinct occurrence.

    The distinction that matters: a key seen from **one** source is kept in full,
    because repeated inline math in a paper is genuinely repeated. A key seen from
    **both** sources means two extractors reported one asset, and only the
    preferred source's copies survive.
    """
    groups: dict = {}
    order: list = []
    for item in items:
        marker = key(item)
        if marker not in groups:
            groups[marker] = []
            order.append(marker)
        groups[marker].append(item)

    kept: list[Any] = []
    for marker in order:
        bucket = groups[marker]
        sources = {item.source for item in bucket}
        if len(sources) == 1:
            kept.extend(bucket)
        else:
            kept.extend(item for item in bucket if item.source == preferred_source)
    return kept


def _collapse(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()


def _split_label(caption: str | None) -> tuple[str | None, str | None]:
    """``"Figure 2: x"`` -> ``("Figure 2", "x")``."""
    if not caption:
        return None, None
    match = _LABEL_RE.match(caption)
    if not match:
        return None, _collapse(caption)
    return f"{match.group(1).title()} {match.group(2)}", _collapse(caption[match.end() :])


class _AssetParser(HTMLParser):
    """One pass over an arXiv HTML rendering, collecting all three kinds.

    A single pass rather than three regexes because a table *is* a ``<figure>``,
    and a caption may itself contain ``<math>`` — matching either kind in
    isolation would misclassify both.
    """

    def __init__(self, base_url: str | None) -> None:
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.figures: list[dict[str, Any]] = []
        self.tables: list[dict[str, Any]] = []
        self.equations: list[tuple[str, bool]] = []

        self._figure_depth: int | None = None
        self._table_depth: int | None = None
        self._in_caption = False
        self._caption_parts: list[str] = []
        self._image_url: str | None = None
        self._width: int | None = None
        self._height: int | None = None
        # Table bodies are collected per figure. Pairing a caption with a body by
        # document order is wrong: arXiv emits layout tables (`width:0.0pt`, a
        # single cell) and equation tables alongside the data tables, so the
        # first N `<table>` elements in the file are not the captioned ones.
        self._body_stack: list[list[list[str]]] = []
        self._table_open = 0
        self._caption: str | None = None

    # -- helpers
    @staticmethod
    def _classes(attrs: list[tuple[str, str | None]]) -> str:
        for name, value in attrs:
            if name == "class" and value:
                return value
        return ""

    @staticmethod
    def _attr(attrs: list[tuple[str, str | None]], name: str) -> str | None:
        for key, value in attrs:
            if key == name:
                return value
        return None

    def _absolute(self, url: str | None) -> str | None:
        if not url:
            return None
        if url.startswith(("http://", "https://", "data:")):
            return url
        if not self.base_url:
            return url
        return urljoin(self.base_url, url)

    # -- HTMLParser hooks
    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "math":
            latex = self._attr(attrs, "alttext")
            if latex:
                self.equations.append(
                    (latex, self._attr(attrs, "display") == "block")
                )
            return
        if tag in {"img", "object"}:
            # Figures ship as <img src> for bitmaps and <object data> for SVG.
            url = self._attr(attrs, "src") or self._attr(attrs, "data")
            if url and self._figure_depth is not None and self._image_url is None:
                self._image_url = self._absolute(url)
                self._width = _as_int(self._attr(attrs, "width"))
                self._height = _as_int(self._attr(attrs, "height"))
            return
        if tag == "figure":
            classes = self._classes(attrs)
            if "ltx_figure" in classes and self._figure_depth is None:
                self._figure_depth = 0
            elif "ltx_table" in classes and self._table_depth is None:
                self._table_depth = 0
                self._body_stack.append([])
            return
        if tag == "figcaption":
            self._in_caption = True
            self._caption_parts = []
            return
        if tag == "table" and self._table_depth is not None:
            # Nested tables stay inside the one body, so depth is tracked.
            self._table_open += 1
            if self._table_open == 1:
                self._body_stack[-1].append([])
            self._body_stack[-1][-1].append(self.get_starttag_text() or "<table>")
            return
        if self._table_open and self._body_stack:
            self._body_stack[-1][-1].append(self.get_starttag_text() or f"<{tag}>")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "figcaption":
            self._in_caption = True
            self._caption_parts = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "figcaption" and self._in_caption:
            self._in_caption = False
            self._caption = _collapse("".join(self._caption_parts))
            self._caption_parts = []
            return
        if tag == "table" and self._table_open:
            self._body_stack[-1][-1].append("</table>")
            self._table_open -= 1
            return
        if self._table_open and self._body_stack and self._body_stack[-1]:
            # Closing tags belong in the stored markup too: without them the
            # first `</tr>` is never emitted, so anything parsing the body (our
            # own row/column count included) reads past the end of the row.
            self._body_stack[-1][-1].append(f"</{tag}>")
            return
        if tag == "figure":
            self._close_figure()
            return

    def flush(self) -> None:
        """Emit anything left open by a truncated document."""
        self._close_figure()

    def _close_figure(self) -> None:
        """Finish whichever figure was open and emit it with its own body."""
        label, caption = _split_label(getattr(self, "_caption", None))
        if self._figure_depth is not None:
            self.figures.append(
                {
                    "label": label,
                    "caption": caption,
                    "image_url": self._image_url,
                    "width": self._width,
                    "height": self._height,
                }
            )
            self._figure_depth = None
            self._image_url = None
            self._width = self._height = None
        elif self._table_depth is not None:
            self.tables.append(
                {
                    "label": label,
                    "caption": caption,
                    "body_html": _best_body(self._body_stack),
                }
            )
            self._table_depth = None
            self._body_stack.pop() if self._body_stack else None
        self._caption = None

    def handle_data(self, data: str) -> None:
        if self._in_caption:
            self._caption_parts.append(data)
        elif self._table_open and self._body_stack and self._body_stack[-1]:
            # Without this the stored markup keeps the tags and loses every
            # cell's text, which is the part a caller actually wants.
            self._body_stack[-1][-1].append(data)

    def handle_entityref(self, name: str) -> None:  # noqa: D102
        if self._in_caption:
            self._caption_parts.append(f"&{name};")
        elif self._table_open and self._body_stack and self._body_stack[-1]:
            self._body_stack[-1][-1].append(f"&{name};")

    def handle_charref(self, name: str) -> None:  # noqa: D102
        if self._in_caption:
            self._caption_parts.append(f"&#{name};")
        elif self._table_open and self._body_stack and self._body_stack[-1]:
            self._body_stack[-1][-1].append(f"&#{name};")


def _as_int(value: str | None) -> int | None:
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None


def _best_body(stack: list[list[list[str]]]) -> str | None:
    """The real data table among the tables inside one captioned figure.

    A captioned ``<figure class="ltx_table">`` holds exactly one data table, but
    the choice is made by size rather than by position: a table figure can still
    contain a nested layout table, and the largest subtree is the data.
    """
    bodies = ["".join(parts) for group in stack for parts in group if parts]
    if not bodies:
        return None
    return max(bodies, key=len)


def extract_assets_from_html(html: str | None, *, page_url: str | None = None) -> Assets:
    """Figures, tables and formulas from an arXiv/ar5iv HTML rendering.

    ``page_url`` is the URL the HTML was fetched from, used to resolve relative
    ``src`` attributes. It must be the page URL **without** a trailing slash:
    arXiv writes image sources as ``<version>/Figures/x.png``, so a trailing
    slash turns the resolution into ``.../v7/v7/Figures/x.png``.
    """
    if not html:
        return Assets()
    parser = _AssetParser(page_url.rstrip("/") if page_url else None)
    try:
        parser.feed(html)
        # A truncated page leaves a figure unclosed; its caption is still good.
        parser.close()
        parser.flush()
    except Exception as exc:  # noqa: BLE001 - a bad page must not lose the paper
        logger.warning("asset_html_parse_failed", extra={"error": str(exc)})

    figures = [
        FigureAsset(
            ordinal=index,
            label=record["label"],
            caption=record["caption"],
            image_url=record["image_url"],
            width=record["width"],
            height=record["height"],
            source="html",
        )
        for index, record in enumerate(parser.figures, start=1)
    ]
    tables = [
        TableAsset(
            ordinal=index,
            label=record["label"],
            caption=record["caption"],
            body_html=record["body_html"],
            source="html",
        )
        for index, record in enumerate(parser.tables, start=1)
    ]
    equations = [
        EquationAsset(
            ordinal=index, latex=latex, is_display=is_display, source="html"
        )
        for index, (latex, is_display) in enumerate(parser.equations, start=1)
    ]

    assets = Assets(figures=figures, tables=tables, equations=equations)
    logger.debug("assets_extracted_html", extra=assets.count())
    return assets


def content_list_path(output_dir: Path) -> Path | None:
    """MinerU's structured block list, if this run produced one."""
    for pattern in ("*/*_content_list.json", "*_content_list.json", "**/*_content_list.json"):
        matches = sorted(output_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def strip_display_delimiters(latex: str | None) -> str | None:
    """Remove a surrounding ``$$…$$``.

    MinerU stores display formulas wrapped in ``$$``, arXiv's HTML stores the
    same content bare. Storing one shape means both sources are comparable and
    the column holds a formula rather than a fragment of a document.
    """
    if not latex:
        return latex
    text = latex.strip()
    if text.startswith("$$") and text.endswith("$$") and len(text) >= 4:
        text = text[2:-2]
    return text.strip() or None


def _bbox(value: Any) -> tuple[int, int, int, int] | None:  # noqa: ANN401
    if isinstance(value, list) and len(value) == 4:
        try:
            return (int(value[0]), int(value[1]), int(value[2]), int(value[3]))
        except (TypeError, ValueError):
            return None
    return None


def extract_assets_from_mineru(
    output_dir: Path | None = None,
    *,
    blocks: list[dict[str, Any]] | None = None,
    image_root: Path | None = None,
    stem: str | None = None,
) -> Assets:
    """Figures, tables and formulas from a MinerU PDF run.

    ``content_list.json`` is the structured form of a run: every figure, table and
    formula carries the page and bounding box it came from. That is the one thing
    the HTML path cannot give us, and the reason this exists.

    ``blocks`` may be supplied directly, which is how the pipeline calls it: the
    extractor loads the block list while the output directory still exists and
    deletes the directory before returning, so reading it afterwards found
    nothing. Measured on the live corpus: 691 figures, 288 tables and 62 391
    equations, **all** of them from the HTML path — the MinerU path had produced
    zero rows for every paper ingested.
    """
    if blocks is None:
        if output_dir is None:
            return Assets()
        path = content_list_path(output_dir)
        if path is None:
            logger.debug("mineru_content_list_absent", extra={"dir": str(output_dir)})
            return Assets()
        try:
            blocks = json.loads(path.read_text(errors="replace"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("mineru_content_list_unreadable", extra={"error": str(exc)})
            return Assets()
        image_root = image_root or path.parent
    if not isinstance(blocks, list):
        return Assets()

    root = image_root
    figures: list[FigureAsset] = []
    tables: list[TableAsset] = []
    equations: list[EquationAsset] = []

    def local_image(img_path: str | None) -> str | None:
        """Resolve a cropped image, when the run's directory is still around.

        ``root`` is None when the blocks came from memory, which is the normal
        case now: the extractor has already deleted its scratch directory. An
        asset then keeps its caption, page and bounding box and simply has no
        image file — a degraded asset beats a missing one.
        """
        if not img_path or root is None:
            return None
        candidate = root / img_path
        return str(candidate) if candidate.exists() else None

    for block in blocks:
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        image = local_image(block.get("img_path"))
        page = block.get("page_idx")
        page_idx = page if isinstance(page, int) else None
        box = _bbox(block.get("bbox"))

        if kind == "image":
            caption = " ".join(block.get("image_caption") or []).strip() or None
            label, body = _split_label(caption)
            figures.append(
                FigureAsset(
                    ordinal=len(figures) + 1,
                    label=label,
                    caption=body,
                    image_path=image,
                    page_idx=page_idx,
                    bbox=box,
                    source="mineru",
                )
            )
        elif kind == "table":
            caption = " ".join(block.get("table_caption") or []).strip() or None
            label, body_text = _split_label(caption)
            tables.append(
                TableAsset(
                    ordinal=len(tables) + 1,
                    label=label,
                    caption=body_text,
                    body_html=block.get("table_body") or None,
                    image_path=image,
                    page_idx=page_idx,
                    bbox=box,
                    source="mineru",
                )
            )
        elif kind in {"equation", "interline_equation", "inline_equation"}:
            equations.append(
                EquationAsset(
                    ordinal=len(equations) + 1,
                    latex=strip_display_delimiters(block.get("text")),
                    is_display=kind != "inline_equation",
                    image_path=image,
                    page_idx=page_idx,
                    bbox=box,
                    source="mineru",
                )
            )

    assets = Assets(figures=figures, tables=tables, equations=equations)
    logger.debug("assets_extracted_mineru", extra=assets.count())
    return assets
