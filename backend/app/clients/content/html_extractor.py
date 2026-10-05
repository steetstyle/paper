"""HTML source -> Markdown.

Uses ``beautifulsoup4`` + ``markdownify`` when installed (higher fidelity) and
otherwise falls back to a dependency-free ``html.parser`` based converter so the
pipeline never hard-depends on an optional extra.
"""

from __future__ import annotations

import re
from html import unescape
from html.parser import HTMLParser
from typing import Any

from app.logging import get_logger

logger = get_logger(__name__)

# ArXiv/chrome wrappers that carry no paper content.
_DROP_SELECTORS = (
    "script", "style", "nav", "footer", "header", "aside", "form", "noscript",
    "svg", ".ltx_bibliography", ".ltx_page_footer",
    ".ltx_role_document_title", ".ltx_dates", ".ltx_authors", "#abstract",
)
# `figure.ltx_figure` is deliberately absent: it is reduced to its caption by
# :func:`_reduce_figures_to_captions` instead of being dropped outright.
_TARGET_SELECTORS = (
    "article.ltx_document", "article", "main", ".ltx_document", "#main", "body",
)

_HEADING_TAGS = {"h1": "#", "h2": "##", "h3": "###", "h4": "####", "h5": "#####", "h6": "######"}

_ATX_HEADING_RE = re.compile(r"^(#{1,6})(\s+\S.*)$", re.MULTILINE)


class _MarkdownFallbackParser(HTMLParser):
    """Minimal, dependency-free HTML -> Markdown converter."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip_depth = 0
        self._skip_tags: set[str] = set()
        self._list_stack: list[str] = []
        self._in_link = False
        self._link_href: str | None = None
        self._link_text: list[str] = []
        self._pre_depth = 0

    # -- helpers
    def _emit(self, text: str) -> None:
        if self._skip_depth:
            return
        if self._in_link:
            self._link_text.append(text)
            return
        self._parts.append(text)

    # -- HTMLParser hooks
    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _DROP_SELECTORS:
            self._skip_depth += 1
            self._skip_tags.add(tag)
            return
        if self._skip_depth:
            return
        if tag == "br":
            self._emit("\n")
        elif tag == "hr":
            self._emit("\n\n---\n\n")
        elif tag in _HEADING_TAGS:
            self._emit(f"\n\n{_HEADING_TAGS[tag]} ")
        elif tag == "pre":
            self._pre_depth += 1
            self._emit("\n\n```\n")
        elif tag == "code" and not self._pre_depth:
            self._emit("`")
        elif tag == "li":
            marker = "1. " if self._list_stack and self._list_stack[-1] == "ol" else "- "
            self._emit(f"\n{marker}")
        elif tag in {"p", "div", "section", "tr"}:
            self._emit("\n\n")
        elif tag in {"ul", "ol"}:
            self._list_stack.append(tag)
            self._emit("\n\n")
        elif tag == "a":
            self._in_link = True
            self._link_href = dict(attrs).get("href")
            self._link_text = []
        elif tag in {"strong", "b"}:
            self._emit("**")
        elif tag in {"em", "i"}:
            self._emit("*")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._skip_tags and self._skip_depth:
            self._skip_depth -= 1
            self._skip_tags.discard(tag)
            return
        if self._skip_depth:
            return
        if tag in _HEADING_TAGS or tag in {"p", "div", "section", "tr"}:
            self._emit("\n\n")
        elif tag == "pre":
            self._pre_depth = max(0, self._pre_depth - 1)
            self._emit("\n```\n\n")
        elif tag == "code" and not self._pre_depth:
            self._emit("`")
        elif tag == "a":
            self._in_link = False
            text = "".join(self._link_text).strip()
            href = self._link_href
            if text and href and not href.startswith("#"):
                self._parts.append(f"[{text}]({href})")
            elif text:
                self._parts.append(text)
            self._link_text = []
        elif tag in {"strong", "b"}:
            self._emit("**")
        elif tag in {"em", "i"}:
            self._emit("*")
        elif tag in {"ul", "ol"} and self._list_stack:
            self._list_stack.pop()
            self._emit("\n")
        elif tag == "li":
            self._emit("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._pre_depth:
            self._parts.append(data)
            return
        if not data.strip():
            self._emit(" ")
            return
        self._emit(re.sub(r"\s+", " ", data))

    @property
    def text(self) -> str:
        joined = "".join(self._parts)
        joined = re.sub(r"[ \t]{2,}", " ", joined)
        return re.sub(r"\n{3,}", "\n\n", joined).strip()


def html_to_markdown(html: str) -> str:
    """Convert an HTML document to markdown-ish text.

    Uses bs4 + markdownify when installed; otherwise falls back to a
    dependency-free ``html.parser`` converter. Both paths normalise heading
    depth so downstream chunking sees a dense, consistent outline.
    """
    if not html.strip():
        return ""
    try:
        return _html_to_markdown_with_bs4(html)
    except ImportError:
        logger.debug("bs4_unavailable", extra={"fallback": "stdlib_parser"})
        parser = _MarkdownFallbackParser()
        parser.feed(html)
        parser.close()
        return _normalise_heading_levels(_tidy(parser.text))


def _html_to_markdown_with_bs4(html: str) -> str:
    from bs4 import BeautifulSoup  # noqa: PLC0415
    from markdownify import markdownify  # noqa: PLC0415

    soup = BeautifulSoup(html, "lxml" if _has_lxml() else "html.parser")
    for selector in _DROP_SELECTORS:
        for node in soup.select(selector):
            node.decompose()

    _promote_section_titles(soup)
    # Equations first: a figure can contain an equation table, and replacing the
    # figure before the table is lifted out would detach it from the tree.
    equations = _extract_equation_tables(soup)
    _reduce_figures_to_captions(soup)
    _drop_empty_layout_tables(soup)

    root: Any = None
    for selector in _TARGET_SELECTORS:
        candidate = soup.select_one(selector)
        if candidate is not None and len(candidate.get_text(strip=True)) > 400:
            root = candidate
            break
    if root is None:
        body = soup.find("body")
        root = body if body is not None else soup

    # Retitle: ArXiv wraps the paper title in ltx_role_document_title which we drop.
    title_node = soup.select_one(".ltx_title_document, h1.title, .ltx_authors + h1")
    markdown = markdownify(str(root), heading_style="ATX", bullets="-", strip=["script", "style"])
    # Sentinels, not markdown: markdownify escapes `_` (LaTeXML's `d_{k}`) and
    # renders <pre> as a fenced block, so the LaTeX has to bypass it entirely.
    markdown = _restore_equations(markdown, equations)
    if title_node:
        title_text = title_node.get_text(" ", strip=True)
        if title_text and title_text.lower() not in markdown[:400].lower():
            markdown = f"# {title_text}\n\n{markdown}"
    return _normalise_heading_levels(_tidy(markdown))


def _reduce_figures_to_captions(soup: Any) -> None:
    """Replace each ``<figure>`` with its caption text, keeping any equations.

    The image itself cannot go into markdown, but the caption is exactly what a
    reader searches for ("show me the figures about attention heads"), and the
    caption is the only place it appears: LaTeXML writes it inside the figure,
    and the surrounding prose merely links to it ("see Figure 2").

    Dropping the whole figure — which is what this used to do — left the corpus
    with **zero** figure chunks, so "only figures" returned nothing at all for
    every arXiv paper. The structured figure record (image URL, page, bounding
    box) is extracted separately by
    :func:`app.clients.content.assets.extract_assets_from_html` from the raw
    HTML, so nothing is lost here.

    Runs after :func:`_extract_equation_tables`, so a display equation sitting
    inside a figure arrives as a sentinel paragraph and is carried over with the
    caption rather than discarded along with the image.
    """
    for figure in soup.select("figure"):
        caption = figure.select_one("figcaption")
        keep: list[Any] = []
        if caption is not None:
            # The "Figure N:" tag is a sibling span inside the caption.
            text = caption.get_text(" ", strip=True)
            if text:
                node = soup.new_tag("p")
                node.string = text
                keep.append(node)
        keep.extend(
            child
            for child in figure.find_all("p", recursive=True)
            if _EQUATION_SENTINEL.format(index="") in child.get_text()
        )
        if not keep:
            # An image with no caption and no equation carries no searchable text.
            figure.decompose()
            continue
        figure.replace_with(keep[0] if len(keep) == 1 else _wrapper(soup, keep))


def _wrapper(soup: Any, nodes: list[Any]) -> Any:  # noqa: ANN401
    """A bare container so several nodes can replace one in place."""
    holder = soup.new_tag("div")
    for node in nodes:
        holder.append(node.extract())
    return holder


_EQUATION_SENTINEL = "xPAPEREQx{index}x"


def _extract_equation_tables(soup: Any) -> list[str]:
    """Replace LaTeXML's equation *layout tables* with sentinels; return the LaTeX.

    ArXiv's HTML rendering positions every display equation in a three-column
    ``<table class="ltx_equation ltx_eqn_table">`` — a centre pad, the formula,
    and the equation number. Run through a generic HTML-to-markdown converter
    that comes out as::

        |  |  |  |
        | --- | --- | --- |
        |  | Attention(Q,K,V)=softmax(...)V |  | (1) |

    which is then indistinguishable from a real data table. It is not one: this
    is a positioning artefact. Left alone it makes every display equation in an
    arXiv paper look like a table, so "only equations" and "only tables" both
    return nonsense.

    The ``alttext`` attribute on ``<math>`` carries the original LaTeX, so the
    formula is recovered rather than scraped from rendered glyphs — which also
    drops the MathML/annotation duplication LaTeXML emits.

    The formula is *not* put back as ``$$ ... $$`` here: markdownify escapes
    ``_`` (LaTeXML's ``d_{k}`` becomes an escaped underscore) and turns ``<pre>``
    into a fenced code block, either of which would corrupt the LaTeX.
    :func:`_restore_equations` substitutes it after the conversion instead.
    """
    found: list[str] = []
    for table in soup.select("table.ltx_equation, table.ltx_eqn_table"):
        # Equation groups hold several numbered rows; take each `math` in order.
        pieces = [
            latex
            for latex in (
                (math.get("alttext") or "").strip() for math in table.select("math[alttext]")
            )
            if latex
        ]
        if not pieces:
            continue
        # Exactly one replacement per table. An equation group holds several
        # `math` nodes, and `replace_with` detaches the node it is given — a
        # second call on the same table raises "not part of a tree", which is
        # how this only showed up on real papers (1706.03762v7 has groups of
        # four and two) and not on a single-equation fixture.
        sentinels = []
        for _latex in pieces:
            index = len(found)
            found.append(_latex)
            sentinels.append(_sentinel_node(soup, f"{index}"))
        table.replace_with(sentinels[0] if len(sentinels) == 1 else _wrapper(soup, sentinels))
    return found


def _sentinel_node(soup: Any, index: str) -> Any:
    """A paragraph holding only the marker; markdownify passes it through."""
    node = soup.new_tag("p")
    node.string = _EQUATION_SENTINEL.format(index=index)
    return node


def _restore_equations(markdown: str, equations: list[str]) -> str:
    """Swap sentinels back for ``$$ ... $$`` display math."""
    for index, latex in enumerate(equations):
        marker = _EQUATION_SENTINEL.format(index=index)
        markdown = markdown.replace(marker, "$$" + chr(10) + latex + chr(10) + "$$")
    return markdown


def _drop_empty_layout_tables(soup: Any) -> None:
    """Remove one-cell ``ltx_tabular`` spacers left behind by LaTeXML.

    Measured on 1706.03762v7: of five ``ltx_tabular`` tables, four are real data
    (6, 12, 22 and 13 rows) and one is a 1x1 cell with no text at all. It becomes
    an empty three-line markdown table, which then reads as a table with a header
    and no data. A table with no text cannot carry data, so it goes.
    """
    for table in soup.select("table.ltx_tabular"):
        cells = table.select("td, th")
        if len(cells) > 1:
            continue
        if table.get_text(strip=True):
            continue
        table.decompose()


def _promote_section_titles(soup: Any) -> None:
    """LaTeXML marks the abstract as ``<h6>`` even though it is a top-level section.

    Without this the extracted markdown starts with ``###### Abstract`` and the
    chunker nests the whole paper underneath it.
    """
    for node in soup.select(".ltx_title_abstract, h6.ltx_title_abstract"):
        node.name = "h2"


def _normalise_heading_levels(markdown: str) -> str:
    """Renumber ATX headings to a dense, gap-free hierarchy.

    Sources (LaTeXML in particular) use inconsistent levels — ``h6`` for the
    abstract, ``h2``/``h3`` for sections. Levels are remapped by order of first
    appearance so the outline stays dense and the relative nesting is preserved.
    """
    seen: dict[int, int] = {}

    def assign(match: re.Match[str]) -> str:
        level = len(match.group(1))
        if level not in seen:
            seen[level] = len(seen) + 1
        return "#" * seen[level] + match.group(2)

    return _ATX_HEADING_RE.sub(assign, markdown)


def _has_lxml() -> bool:
    try:
        import lxml  # noqa: F401, PLC0415

        return True
    except ImportError:
        return False


def _tidy(markdown: str) -> str:
    markdown = re.sub(r"\n{3,}", "\n\n", markdown)
    markdown = re.sub(r"[ \t]+\n", "\n", markdown)
    return unescape(markdown).strip()


def extract_title(html: str) -> str | None:
    match = re.search(
        r"<meta[^>]+name=[\"']citation_title[\"'][^>]+content=[\"']([^\"']+)[\"']", html
    )
    if match:
        return match.group(1).strip()
    match = re.search(r"<title[^>]*>(.*?)</title>", html, re.S | re.I)
    if match:
        return re.sub(r"\s+", " ", match.group(1)).replace("arXiv:", "").strip()
    return None