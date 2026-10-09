"""Sentence segmentation with character-exact locations.

The report is only useful if it can say *where* a problem is, so the segmenter
never loses offsets: every :class:`~checker_app.domain.models.Sentence` carries its
character range, line range, page, paragraph number, heading path and the slice
of the document token stream it owns.

Turkish and English share one rule set, because the differences are lexical
(abbreviations, initials) rather than structural:

* ``3,14`` and ``1.000,50`` never end a sentence
* ``vb.``, ``bkz.``, ``örn.``, ``Dr.``, ``Fig.``, ``e.g.``, ``et al.`` never do either
* ``J. Smith`` and ``B. Yılmaz`` are initials, not sentences
* URLs, DOIs, e-mails and arXiv ids never are
* sentence quotes stay whole unless ``split_inside_quotes`` is set
* document structure (headings, list items, code fences, tables, math) is kept,
  so a line inside a code block is not mistaken for prose
"""

from __future__ import annotations

import re
from bisect import bisect_right
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

from checker_app.config import SegmentationSettings
from checker_app.domain.enums import BlockType
from checker_app.domain.models import Location, Sentence
from checker_app.domain.sections import classify_section
from checker_app.domain.text import Token, language_confidence, tokenize

__all__ = ["Block", "Line", "SegmentationResult", "SentenceSplitter"]

_TERMINATORS = frozenset(".!?…")
_CLOSERS = "\"'’”»«)]}"
_BOUNDARY_RE = re.compile(r"([.!?…]+[\"'’”»«\)\]]*)(?=\s|$)")
_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.*\S)\s*$")
_LIST_RE = re.compile(r"^\s*(?:[-*+•]|\(?\d{1,3}[.)]|[a-zA-ZÇĞİÖŞÜçğıöşü][.)])\s+")
_FENCE_RE = re.compile(r"^\s*(```|~~~)")
_TABLE_RE = re.compile(r"^\s*\|")
_QUOTE_RE = re.compile(r"^\s*>")
_MATH_ENV_RE = re.compile(r"\\begin\{(equation|align|gather|multline|displaymath)\*?\}")
_CAPTION_RE = re.compile(
    r"^\s*(?:[ŞS]ekil|Tablo|Resim|Figure|Table|Chart|Grafik|Alg|Algorithm)\s*\.?\s*\d+",
    re.IGNORECASE,
)
_DIGIT_RUN_RE = re.compile(r"\d+$")
_TOKEN_CHARS = "ÇĞİÖŞÜçğıöşü"
_WORD_TAIL_RE = re.compile(rf"([\w{_TOKEN_CHARS}]+(?:['’][\w{_TOKEN_CHARS}]+)*)\.?$", re.UNICODE)
_INITIALS_RE = re.compile(rf"(?:^|[\s(\[\"“«])([\w{_TOKEN_CHARS}])\.$")
_DOTTED_INITIALS_RE = re.compile(rf"(?:[A-Za-z{_TOKEN_CHARS}]\.){{1,3}}[A-Za-z{_TOKEN_CHARS}]")
_URLISH_RE = re.compile(r"(https?://|www\.|@\w|10\.\d{4,}|arxiv|\bdoi\b)", re.IGNORECASE)

# Tokens that end in a period without ending a sentence.
_ABBREVIATIONS = frozenset(
    {
        # English
        "e.g",
        "i.e",
        "etc",
        "vs",
        "cf",
        "ca",
        "viz",
        "resp",
        "approx",
        "al",
        "fig",
        "figs",
        "eq",
        "eqs",
        "ref",
        "refs",
        "sec",
        "secs",
        "ch",
        "chap",
        "tab",
        "tabs",
        "no",
        "nos",
        "vol",
        "vols",
        "pp",
        "p",
        "ed",
        "eds",
        "dr",
        "prof",
        "mr",
        "mrs",
        "ms",
        "st",
        "jr",
        "sr",
        "inc",
        "ltd",
        "dept",
        "univ",
        "max",
        "min",
        "avg",
        "std",
        "def",
        "fn",
        "et",
        "ibid",
        "op",
        "cit",
        "n.b",
        "seq",
        "drs",
        "prof.",
        "sgt",
        # Turkish
        "vb",
        "vd",
        "örn",
        "bkz",
        "yy",
        "bk",
        "sf",
        "s",
        "ş",
        "c",
        "c.",
        "doç",
        "doc",
        "drl",
        "ing",
        "av",
        "tuğ",
        "alb",
        "yr",
        "sn",
        "bkz.",
        "örn.",
        "vb.",
        "vd.",
        "yy.",
        "bk.",
        "sf.",
        "çev",
        "no:",
        "maddesi",
        "fıkra",
        "madde",
        "bent",
        "hazırlayan",
        "gerekirse",
    }
)

# Whole "sentences" that are structural noise but must survive segmentation.
_SENTENCE_ALLOW = frozenset({"#", "..."})


@dataclass(frozen=True, slots=True)
class Line:
    """One physical line, with its offset in the document."""

    start: int
    end: int
    text: str
    page: int


@dataclass(frozen=True, slots=True)
class Block:
    """A structural unit of the document (paragraph, heading, code fence…)."""

    kind: BlockType
    text: str
    char_start: int
    char_end: int
    line_start: int
    line_end: int
    paragraph_index: int
    page: int
    section_path: tuple[str, ...] = ()


@dataclass(slots=True)
class SegmentationResult:
    """Everything the rest of the pipeline needs about the document shape."""

    sentences: tuple[Sentence, ...]
    tokens: tuple[Token, ...]
    blocks: tuple[Block, ...]
    page_count: int
    language: str
    language_confidence: float
    char_count: int
    word_count: int
    heading_trail: tuple[str, ...] = field(default=())


class SentenceSplitter:
    """Cut a document into located sentences."""

    def __init__(self, settings: SegmentationSettings | None = None) -> None:
        self._settings = settings or SegmentationSettings()

    # ------------------------------------------------------------------ public
    def split(self, text: str) -> SegmentationResult:
        settings = self._settings
        if not text or not text.strip():
            return SegmentationResult(
                sentences=(),
                tokens=(),
                blocks=(),
                page_count=1,
                language="unknown",
                language_confidence=0.0,
                char_count=len(text),
                word_count=0,
            )

        page_marks = (
            [i for i, ch in enumerate(text) if ch == "\f"]
            if settings.treat_formfeed_as_page_break
            else []
        )
        lines = _lines(text, page_marks)

        blocks = self._blocks(text, lines)
        line_starts = [line.start for line in lines]
        sentences: list[Sentence] = []
        tokens: list[Token] = []

        for block in blocks:
            for start, end, kind in self._units(block):
                abs_start, abs_end = _trim_span(
                    text, block.char_start + start, block.char_start + end
                )
                if abs_end <= abs_start:
                    continue
                sentence_tokens = tokenize(
                    text[abs_start:abs_end],
                    offset=abs_start,
                    start_index=len(tokens),
                )
                sentences.append(
                    Sentence(
                        index=len(sentences),
                        text=text[abs_start:abs_end],
                        location=_location(
                            block,
                            abs_start - block.char_start,
                            abs_end - block.char_start,
                            kind,
                            line_starts,
                            page_marks,
                        ),
                        token_start=len(tokens),
                        token_end=len(tokens) + len(sentence_tokens),
                    )
                )
                tokens.extend(sentence_tokens)

        if settings.detect_language:
            language, confidence = language_confidence(text)
        else:
            language, confidence = "unknown", 0.0

        trail: tuple[str, ...] = ()
        if blocks:
            trail = max(blocks, key=lambda b: b.char_start).section_path

        return SegmentationResult(
            sentences=tuple(sentences),
            tokens=tuple(tokens),
            blocks=tuple(blocks),
            page_count=len(page_marks) + 1,
            language=language,
            language_confidence=confidence,
            char_count=len(text),
            word_count=len(tokens),
            heading_trail=trail,
        )

    # ----------------------------------------------------------------- private
    def _blocks(self, text: str, lines: Sequence[Line]) -> list[Block]:
        """Group lines into paragraphs, tracking the heading trail."""
        blocks: list[Block] = []
        heading_stack: list[tuple[int, str]] = []
        paragraph_index = 0
        i = 0
        total = len(lines)

        while i < total:
            if not lines[i].text.strip():
                i += 1
                continue
            paragraph_index += 1
            start_i = i
            fence: str | None = None
            while i < total:
                line = lines[i]
                if fence is None:
                    if not line.text.strip():
                        break
                    fence_match = _FENCE_RE.match(line.text)
                    if fence_match:
                        fence = fence_match.group(1)
                elif line.text.strip().startswith(fence):
                    fence = None
                i += 1

            first, last = lines[start_i], lines[i - 1]
            body = text[first.start : last.end]
            kind = _classify_block(body, fence is not None)
            trail = heading_stack
            if kind is BlockType.HEADING:
                heading_match = _HEADING_RE.match(body.strip())
                level = len(heading_match.group(1)) if heading_match else 1
                title = heading_match.group(2).strip() if heading_match else body.strip()
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                heading_stack = [*heading_stack, (level, title)]
                trail = heading_stack
            blocks.append(
                Block(
                    kind=kind,
                    text=body,
                    char_start=first.start,
                    char_end=last.end,
                    line_start=start_i + 1,
                    line_end=i,
                    paragraph_index=paragraph_index,
                    page=first.page,
                    section_path=tuple(h for _, h in trail),
                )
            )
        return blocks

    def _units(self, block: Block) -> Iterator[tuple[int, int, BlockType]]:
        """Split one block into reportable units, as relative spans."""
        kind = block.kind
        if kind in {BlockType.CODE, BlockType.MATH, BlockType.HEADING}:
            yield (0, len(block.text), kind)
            return
        if kind is BlockType.TABLE:
            for line_start, line_end in _iter_lines(block.text):
                yield (line_start, line_end, kind)
            return
        if kind is BlockType.LIST_ITEM:
            for line_start, line_end in _iter_lines(block.text):
                marker = _LIST_RE.match(block.text[line_start:line_end])
                content_start = line_start + (marker.end() if marker else 0)
                found = False
                for span in self._sentence_spans(block.text, start=content_start, end=line_end):
                    found = True
                    yield (*span, kind)
                if not found:
                    yield (line_start, line_end, kind)
            return
        for span in self._sentence_spans(block.text):
            yield (*span, kind)

    def _sentence_spans(
        self, text: str, start: int = 0, end: int | None = None
    ) -> Iterator[tuple[int, int]]:
        """Sentence spans inside ``text[start:end]``."""
        end = len(text) if end is None else end
        if end <= start:
            return
        region = text[start:end]
        track_quotes = not self._settings.split_inside_quotes and _quotes_balanced(region)
        depths = _quote_depths(region) if track_quotes else None

        spans: list[tuple[int, int]] = []
        cursor = start
        for match in _BOUNDARY_RE.finditer(region):
            punct_end = start + match.end(1)
            depth = depths[match.start(1)] if depths is not None else 0
            if not _is_boundary(text, start, match, depth):
                continue
            spans.append((cursor, punct_end))
            cursor = start + match.end(0)

        if cursor < end:
            spans.append((cursor, end))

        min_chars = self._settings.min_sentence_chars
        merged: list[tuple[int, int]] = []
        for span in spans:
            if merged and (span[1] - span[0]) < min_chars:
                merged[-1] = (merged[-1][0], span[1])
            else:
                merged.append(span)

        for span_start, span_end in merged:
            trimmed = _trim_span(text, span_start, span_end)
            if trimmed[1] > trimmed[0]:
                yield trimmed


# --------------------------------------------------------------------- helpers
def _is_boundary(
    text: str,
    region_start: int,
    match: re.Match[str],
    depth: int,
) -> bool:
    """Guard rules that make a full stop not end a sentence."""
    group = match.group(1)
    terminators = group.rstrip(_CLOSERS)
    if not terminators or set(terminators) - _TERMINATORS:
        return False
    if depth > 0:
        return False

    punct_start = region_start + match.start(1)
    head = text[:punct_start]
    next_char = text[region_start + match.end(1) :].lstrip()[:1]

    if terminators == ".":
        tail = _last_token(head)
        tail = tail.rstrip(".") if tail else ""

        # "U.S.", "e.g." style dotted initials.
        if tail and _DOTTED_INITIALS_RE.fullmatch(tail):
            return False

        # Decimals, thousands separators, ordinals: "3.14", "1.000,50", "3rd."
        digits = _DIGIT_RUN_RE.search(head)
        if digits:
            run_start = digits.start()
            if run_start > 0 and head[run_start - 1] in ".,":
                return False
            if next_char.isdigit():
                return False
            if run_start > 0 and head[run_start - 1].isalpha():
                return False

        # Initial such as "J. Smith" / "B. Yılmaz". The window must include the
        # period, otherwise "J." can never match.
        initials = _INITIALS_RE.search(f"{head[-5:]}{terminators}")
        if initials:
            letter = initials.group(1)
            if letter.lower() == "i" and next_char.islower():
                return True  # English pronoun: "I. think".
            if not next_char.isdigit():
                return False

        # URL / DOI / e-mail / arXiv id.
        if tail and _URLISH_RE.search(tail):
            return False

        # Known abbreviation.
        key = tail.lower().rstrip(".") if tail else ""
        if key and key in _ABBREVIATIONS:
            return False

    return True


def _last_token(head: str) -> str:
    """The last whitespace-delimited token of ``head``, punctuation stripped."""
    if not head:
        return ""
    return re.split(r"[\s(\[\"“«]", head)[-1].rstrip("\"'’”)").strip()


def _quote_depths(region: str) -> list[int]:
    """Unclosed quote/bracket depth before every offset in ``region``.

    One linear pass: the boundary scanner asks for the depth at each candidate
    and re-scanning the prefix per candidate would be quadratic.
    """
    pairs = (("(", ")"), ("[", "]"), ("{", "}"), ("“", "”"), ("«", "»"), ("‘", "’"))
    depths = [0] * (len(region) + 1)
    depth = 0
    straight = 0
    for i, ch in enumerate(region):
        depths[i] = depth + straight
        for opener, closer in pairs:
            if ch == opener:
                depth += 1
            elif ch == closer:
                depth = max(0, depth - 1)
        if ch == '"':
            straight ^= 1
    depths[len(region)] = depth + straight
    return depths


def _quotes_balanced(region: str) -> bool:
    """Only track quote depth in regions whose quotes actually pair up.

    One stray ``"`` in a paragraph would otherwise suppress every sentence
    boundary after it, which is a silent, total failure mode.
    """
    straight = region.count('"') % 2
    curly = sum(
        abs(region.count(a) - region.count(b)) for a, b in (("“", "”"), ("«", "»"), ("‘", "’"))
    )
    parens = abs(region.count("(") - region.count(")"))
    return straight == 0 and curly <= 2 and parens <= 1


def _location(
    block: Block,
    start: int,
    end: int,
    kind: BlockType,
    line_starts: Sequence[int],
    page_marks: Sequence[int] = (),
) -> Location:
    """Resolve a unit span to an absolute, page-aware location."""
    abs_start = block.char_start + start
    abs_end = block.char_start + end
    return Location(
        char_start=abs_start,
        char_end=abs_end,
        line_start=bisect_right(line_starts, abs_start),
        line_end=bisect_right(line_starts, max(abs_start, abs_end - 1)),
        paragraph_index=block.paragraph_index,
        # Pages come from the character offset, not the block: a form feed can
        # sit in the middle of a paragraph.
        page=_page_of(list(page_marks), abs_start) if page_marks else block.page,
        section_path=block.section_path,
        block_type=kind,
        # Thesis structure, inferred once per block: it decides whether the
        # sentence is AI-scored at all and which baseline it is judged against.
        section_role=classify_section(block.section_path),
    )


def _page_of(page_marks: Sequence[int], offset: int) -> int:
    """1-based page number for a character offset."""
    return bisect_right(list(page_marks), offset) + 1


def _lines(text: str, page_marks: Sequence[int]) -> list[Line]:
    """Physical lines including blank ones.

    Empty lines must be kept: the paragraph structure (and therefore heading,
    list and code-fence detection) is defined by the blank lines.
    """
    lines: list[Line] = []
    pos = 0
    while True:
        nl = text.find("\n", pos)
        end = len(text) if nl == -1 else nl
        lines.append(
            Line(start=pos, end=end, text=text[pos:end], page=bisect_right(page_marks, pos) + 1)
        )
        if nl == -1:
            return lines
        pos = nl + 1


def _iter_lines(text: str) -> Iterator[tuple[int, int]]:
    pos = 0
    n = len(text)
    while pos < n:
        nl = text.find("\n", pos)
        end = n if nl == -1 else nl
        if end > pos:
            yield (pos, end)
        pos = end + 1 if nl != -1 else n


def _trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _classify_block(text: str, inside_fence: bool) -> BlockType:
    stripped = text.strip()
    if not stripped:
        return BlockType.PROSE
    if inside_fence or _FENCE_RE.match(stripped):
        return BlockType.CODE
    if _HEADING_RE.match(stripped):
        return BlockType.HEADING
    if _MATH_ENV_RE.search(stripped) or (
        stripped.startswith("$$") and stripped.endswith("$$") and len(stripped) > 4
    ):
        return BlockType.MATH
    lines = [line for line in stripped.split("\n") if line.strip()]
    if len(lines) > 1 and all(_TABLE_RE.match(line) for line in lines):
        return BlockType.TABLE
    if len(lines) > 1 and all(_QUOTE_RE.match(line) for line in lines):
        return BlockType.QUOTE
    if lines and all(_LIST_RE.match(line) for line in lines):
        return BlockType.LIST_ITEM
    if len(lines) == 1 and _CAPTION_RE.match(lines[0]):
        return BlockType.CAPTION
    return BlockType.PROSE
