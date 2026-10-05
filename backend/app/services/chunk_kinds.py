"""Classifying a chunk by what it contains.

Retrieval filters ("only equations", "only figures") are only useful if the kind
is decided **once**, while the chunk is being cut and the surrounding markdown is
still in hand. Re-guessing it later from the chunk text alone gets the common
cases wrong: a figure caption inside a body paragraph reads as prose, and a
reference list reads as body text.

The signals, strongest first:

1. the heading breadcrumb, which the chunker already tracks
2. a caption marker at the start of the text (``Figure 2:``)
3. markdown shape — a fence, a table pipe row, display-math delimiters
4. density, as a last resort for a chunk that is nothing but symbols

Ambiguity is resolved towards ``BODY``: a chunk that only *mentions* a figure is
still body text, and quietly classifying it as a figure would make the filter lie.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from app.domain.enums import ChunkKind

_HEADING_KINDS: tuple[tuple[re.Pattern[str], ChunkKind], ...] = (
    (re.compile(r"\b(references?|bibliography|works cited|literature cited)\b", re.I), ChunkKind.REFERENCE),
    (re.compile(r"\b(abstract|summary)\b", re.I), ChunkKind.ABSTRACT),
)
"""Heading breadcrumb -> kind.

Deliberately narrow. An earlier version also mapped ``appendix`` here, which was
wrong and expensive: measured over the corpus it claimed 505 chunks for the
References section that were really proofs ("Appendix C Proofs of our theorems >
Proof.") — dense with display equations, the opposite of a bibliography.
Appendix and Proof are ordinary body text, so they get no special case and fall
through to the markdown-shape rules below.
"""

_CAPTION_RE = re.compile(r"^\s*(figure|fig\.?|table|algorithm|listing)\s*\d+\s*[:.]", re.I)
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$", re.MULTILINE)
_FENCE_RE = re.compile(r"^\s*```", re.MULTILINE)
_DISPLAY_MATH_RE = re.compile(r"\$\$.*?\$\$", re.DOTALL)
_INLINE_MATH_RE = re.compile(r"(?<!\$)\\\([^\\]*\\\)|(?<!\$)\$[^$\n]+\$(?!\$)")
_LATEX_COMMAND_RE = re.compile(r"\\[a-zA-Z]+")

#: A pipe row that ends in a bare parenthesised number is LaTeXML's equation
#: *layout* table, not data. Measured shape from 1706.03762v7:
#:
#:     |  |  |  |
#:     | --- | --- | --- |
#:     |  | Attention(Q,K,V)=softmax(...)V |  | (1) |
#:
#: A real table row ends with a data cell, never with "(1)". Without this the
#: classifier calls every display equation in an arXiv paper a table, and
#: "only equations" then returns nothing at all.
_EQUATION_ROW_RE = re.compile(r"^\s*\|.*\|\s*\(\d+[a-z]?\)\s*\|\s*$", re.MULTILINE | re.I)

#: Equation *groups* carry no number, so the row above misses them. Their layout
#: is the giveaway instead: a pad cell that is empty at both ends, with LaTeX in
#: between. Requiring the LaTeX matters — measured over 2970 chunks, the pad-cell
#: rule on its own claimed 22 real data tables (an empty first *and* last column,
#: e.g. a 16-cell benchmark table); with the LaTeX requirement none of those 22
#: flip.
_CELL_RE = re.compile(r"^\s*\|(.*)\|\s*$")


def _is_latex_equation_table(body: str) -> bool:
    """True when a pipe block is a LaTeXML equation layout, not a data table."""
    for line in body.splitlines():
        match = _CELL_RE.match(line)
        if match is None:
            continue
        cells = [cell.strip() for cell in match.group(1).split("|")]
        if len(cells) < 3:
            continue
        # A separator row (`| --- | --- |`) is not content.
        if all(set(cell) <= set("-: ") for cell in cells):
            continue
        first, last = cells[0], cells[-1]
        pad_left = first == ""
        pad_right = last == "" or bool(re.fullmatch(r"\(\d+[a-z]?\)", last))
        if pad_left and pad_right and any(_LATEX_COMMAND_RE.search(c) for c in cells[1:-1]):
            return True
    return False

# A chunk is "mostly maths" when symbols outnumber prose words. Tuned by reading
# output, not guessed: real equation chunks sit well above 0.15, prose below 0.02.
_SYMBOL_DENSITY = 0.15
_MIN_LENGTH_FOR_EQUATION = 24


_LEADING_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$", re.MULTILINE)
#: The chunker composes its breadcrumb into the chunk text as
#: ``Paper Title > 3.2.1 Attention``, so the last ``>`` segment is the section.
_BREADCRUMB_SEGMENT_RE = re.compile(r"^[^>]{0,120}$")
_MAX_BREADCRUMB_SEGMENTS = 6


def _leading_headings(body: str, limit: int = 3) -> list[str]:
    """Heading trail found at the very start of the chunk's own text.

    Two forms are recognised, because the chunker emits both depending on the
    source: an ATX line (``## References``) and a composed breadcrumb
    (``Paper Title > 3.2.1 Attention``). Only the *last* breadcrumb segment is
    used, so a paper whose title starts with a heading word ("Abstracts of
    Everything > Methods") is not mistaken for its own abstract.

    Bounded by ``limit`` and by the segment rules: a heading quoted deep inside
    a chunk is body content mentioning a section, not the chunk's location.
    """
    head = body.lstrip()
    found: list[str] = []
    first_paragraph_end = head.find("\n\n")

    for match in _LEADING_HEADING_RE.finditer(head):
        if found and match.start() > max(first_paragraph_end, 0):
            break
        found.append(match.group(1))
        if len(found) >= limit:
            break

    if not found:
        first_line = head.split("\n", 1)[0].strip()
        segments = [segment.strip() for segment in first_line.split(">")]
        if (
            1 < len(segments) <= _MAX_BREADCRUMB_SEGMENTS
            and all(_BREADCRUMB_SEGMENT_RE.match(segment) for segment in segments)
            and segments[-1]
        ):
            found.append(segments[-1])
    return found


def _strip_leading_trail(body: str) -> str:
    """Remove the heading/breadcrumb block the chunker composes onto a chunk.

    Essential, not cosmetic. Every chunk produced by the chunker starts with
    ``Paper Title > 3.2.1 Attention`` (or an ATX line), so a caption check
    anchored at the start of the raw text never matches anything: measured over
    the live corpus this silently produced **zero** figure chunks, because
    ``Figure 2: ...`` is never the first thing in the text. Shape rules have to
    run on the chunk's own body.
    """
    text = body.lstrip()
    consumed = 0
    lines = text.split("\n")
    index = 0
    for line in lines:
        stripped = line.strip()
        if not stripped:
            index += 1
            # One blank line after the trail; anything more is content.
            if consumed:
                break
            continue
        is_heading = bool(_LEADING_HEADING_RE.match(line))
        segments = [segment.strip() for segment in stripped.split(">")]
        is_breadcrumb = (
            not is_heading
            and 1 < len(segments) <= _MAX_BREADCRUMB_SEGMENTS
            and all(_BREADCRUMB_SEGMENT_RE.match(segment) for segment in segments)
        )
        if not (is_heading or is_breadcrumb):
            break
        consumed += len(line) + 1
        index += 1
    return text[consumed:].lstrip("\n")


def _symbol_density(text: str) -> float:
    """Fraction of whitespace-separated tokens that look like maths."""
    tokens = text.split()
    if not tokens:
        return 0.0
    symbolic = sum(
        1
        for token in tokens
        if _LATEX_COMMAND_RE.search(token)
        or token.strip("()[]{}.,;:") in {"=", "+", "-", "*", "/", "<", ">"}
        or token.startswith("\\")
        or bool(re.fullmatch(r"[∇∑∏∫α-ωΑ-Ω\\^_{}]{1,3}", token))
    )
    return symbolic / len(tokens)


def classify_chunk(
    text: str,
    *,
    heading: str | None = None,
    section_path: Sequence[str] = (),
) -> ChunkKind:
    """Decide what one chunk is. See the module docstring for the signals."""
    body = text or ""
    stripped = body.strip()
    if not stripped:
        return ChunkKind.BODY

    # 1. Heading breadcrumb: strongest, and free.
    #    The chunker composes the trail into the chunk text itself
    #    ("Paper Title > 3.2.1 Attention"), so an ATX line at the top of the text
    #    counts too — otherwise the same chunk classifies differently depending
    #    on whether the caller remembered to pass `heading`.
    trail = " ".join([heading or "", *section_path, *_leading_headings(body)])
    for pattern, kind in _HEADING_KINDS:
        if pattern.search(trail):
            return kind

    # 2. A caption marker at the start of the chunk's *own* body.
    #    Not at the start of the raw text: the breadcrumb comes first.
    content = _strip_leading_trail(body)
    stripped = content.strip()
    if not stripped:
        return ChunkKind.BODY
    caption = _CAPTION_RE.match(stripped)
    if caption:
        word = caption.group(1).lower()
        if word.startswith("table"):
            return ChunkKind.TABLE
        return ChunkKind.FIGURE

    # 3. Markdown shape.
    if _FENCE_RE.search(content):
        return ChunkKind.CODE
    # LaTeXML's equation layout tables, checked *before* the table rule below.
    if _EQUATION_ROW_RE.search(content) or _is_latex_equation_table(content):
        return ChunkKind.EQUATION
    # Two pipe rows means a real table, not a stray pipe in prose.
    if len(_TABLE_ROW_RE.findall(content)) >= 2:
        return ChunkKind.TABLE
    display = _DISPLAY_MATH_RE.findall(content)
    if display:
        inline = _INLINE_MATH_RE.findall(content)
        # Display maths that dominates the chunk: it *is* an equation chunk. A
        # formula embedded in a paragraph stays body text.
        math_chars = sum(len(m) for m in display) + sum(len(m) for m in inline)
        if len(stripped) >= _MIN_LENGTH_FOR_EQUATION and math_chars / len(stripped) > 0.35:
            return ChunkKind.EQUATION

    # 4. Density, for a chunk that is nothing but symbols.
    if len(stripped) >= _MIN_LENGTH_FOR_EQUATION and _symbol_density(stripped) >= _SYMBOL_DENSITY:
        return ChunkKind.EQUATION

    # 5. A standalone formula line. Token-density below misses the commonest
    #    shape of all, because a formula has no spaces and is therefore *one*
    #    token: `Attention(Q,K,V)=softmax(QK^T/sqrt(d_k))V` scores 0.
    #    Measured over the corpus this recovers 90 chunks that were labelled
    #    body (75) or table (15) and are every one of them a display equation.
    #    After the table rules on purpose: a table cell holding a formula is
    #    still a table.
    if _has_formula_line(stripped):
        return ChunkKind.EQUATION

    return ChunkKind.BODY


_FORMULA_MARKER_RE = re.compile(
    r"(?<![A-Za-z])="           # assignment, not part of a word
    r"|\\[a-zA-Z]+"             # a LaTeX command
    r"|[\^_{}]"                 # sub/superscript or a braced group
    r"|‖|∑|∏|∫|√|∂|∇",
    re.I,
)


def _has_formula_line(content: str) -> bool:
    """A line that is a formula rather than prose: short, dense, marker-bearing.

    "Short" is two words or fewer, which no English sentence reaches, and pipe
    rows are excluded so a table cell cannot masquerade as one.
    """
    for line in content.splitlines():
        stripped = line.strip()
        if len(stripped) < 12 or stripped.startswith("|"):
            continue
        if len(stripped.split()) <= 2 and _FORMULA_MARKER_RE.search(stripped):
            return True
    return False


def merge_run(values: Sequence[ChunkKind]) -> ChunkKind:
    """One kind for a set of chunks: the most structured wins.

    Used when several chunks are returned together and the caller needs a single
    label — BODY loses to anything more specific, so a result set of "one figure,
    three paragraphs" is described as a figure.
    """
    if not values:
        return ChunkKind.BODY
    order = [
        ChunkKind.REFERENCE,
        ChunkKind.EQUATION,
        ChunkKind.TABLE,
        ChunkKind.FIGURE,
        ChunkKind.CODE,
        ChunkKind.ABSTRACT,
        ChunkKind.BODY,
    ]
    present = set(values)
    for kind in order:
        if kind in present:
            return kind
    return ChunkKind.BODY


__all__ = ["classify_chunk", "merge_run", "parse_kinds"]


def parse_kinds(values: Sequence[str] | str | None) -> list[str] | None:
    """Turn user input into canonical kind values, or raise.

    Accepts plurals and abbreviations (``equations``, ``eq``, ``fig``) because
    people type them, and validates against :class:`ChunkKind` so a typo is an
    error instead of a filter that silently matches nothing — a search that
    quietly returns zero rows is worse than one that refuses to start.

    Raises:
        ValueError: on an unknown kind, naming the accepted set.
    """
    if values is None:
        return None
    raw = [values] if isinstance(values, str) else list(values)
    wanted = [item.strip().lower() for item in raw if item and item.strip()]
    if not wanted:
        return None

    known = {kind.value: kind for kind in ChunkKind}
    aliases = {
        "eq": ChunkKind.EQUATION,
        "eqs": ChunkKind.EQUATION,
        "equations": ChunkKind.EQUATION,
        "formula": ChunkKind.EQUATION,
        "formulas": ChunkKind.EQUATION,
        "math": ChunkKind.EQUATION,
        "fig": ChunkKind.FIGURE,
        "figs": ChunkKind.FIGURE,
        "figures": ChunkKind.FIGURE,
        "image": ChunkKind.FIGURE,
        "images": ChunkKind.FIGURE,
        "tables": ChunkKind.TABLE,
        "abstract": ChunkKind.ABSTRACT,
        "abstracts": ChunkKind.ABSTRACT,
        "summary": ChunkKind.ABSTRACT,
        "ref": ChunkKind.REFERENCE,
        "refs": ChunkKind.REFERENCE,
        "references": ChunkKind.REFERENCE,
        "bibliography": ChunkKind.REFERENCE,
        "code": ChunkKind.CODE,
        "snippet": ChunkKind.CODE,
        "body": ChunkKind.BODY,
        "prose": ChunkKind.BODY,
        "text": ChunkKind.BODY,
    }

    resolved: list[str] = []
    unknown: list[str] = []
    for word in wanted:
        kind = known.get(word) or aliases.get(word)
        if kind is None:
            unknown.append(word)
        elif kind.value not in resolved:
            resolved.append(kind.value)
    if unknown:
        raise ValueError(
            f"unknown content kind(s): {', '.join(unknown)}; "
            f"choose from: {', '.join(sorted(known))}"
        )
    return resolved