"""Attribution: is a matched passage quoted, cited, both, or neither?

The most important number in this whole literature is not a detector score. A
2025-26 corpus-scale study of semantic reuse in 26,400 papers found that only
**6.4%** of matched paper pairs carried a citation link (human-verified 7.0%,
95% CI 3.5–10.6), while matched pairs grew 4x and citation-linked pairs stayed
flat at 59 (arXiv:2609.32963). The other half of that finding is the one that
matters for a thesis: **a match count is not a misconduct count**, so the
attribution status of every match has to be reported, not just its existence.

Turnitin's four groups are the de-facto standard and are reproduced here:

======================  ==================================================
Group                   Meaning
======================  ==================================================
``not_cited_or_quoted``  no citation, no quotation marks - the serious one
``missing_citation``    quoted, but no reference
``missing_quotation``   cited, but copied without quotation marks
``cited_and_quoted``    legitimate: reference given and marked as a quote
======================  ==================================================

Citation and quotation *detection* is pattern-based (TR + EN), so it is
deliberately conservative: a false "cited" is safer than a false "not cited",
because it downgrades the finding. Every classification keeps the evidence that
produced it so a reviewer can check it.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from checker_app.domain.enums import CitationStatus

__all__ = [
    "CitationStatus",
    "sentence_bounds",
    "CitationEvidence",
    "Span",
    "find_quotation_spans",
    "find_citation_spans",
    "classify_attribution",
    "is_quoted",
    "is_cited",
]


@dataclass(frozen=True, slots=True)
class Span:
    """A character range with the evidence that produced it."""

    char_start: int
    char_end: int
    evidence: str = ""


@dataclass(frozen=True, slots=True)
class CitationEvidence:
    """Everything attribution looked at for one sentence."""

    has_quote: bool
    has_citation: bool
    quote_evidence: str = ""
    citation_evidence: str = ""
    status: CitationStatus = CitationStatus.NOT_CITED_OR_QUOTED


# --- quotations -------------------------------------------------------------
_QUOTE_PAIRS = (
    ('"', '"'),
    ("“", "”"),
    ("„", "”"),
    ("«", "»"),
    ("‘", "’"),
    ("”", "”"),
    ("'", "'"),
)
# Block quotes: a quoted line in Markdown or an indented verbatim block.
_BLOCK_QUOTE_RE = re.compile(r"(?m)^[ \t]{0,3}>[ \t]*\S.*$")
_QUOTE_RE = re.compile(
    r"(?P<open>“|«|„)[^”»]{4,}?(?P<close>”|»)|\"[^\"]{4,}?\"|«[^»]{4,}?»",
    re.DOTALL,
)

# --- citations --------------------------------------------------------------
_BRACKET_NUMBER_RE = re.compile(r"\[\s*\d{1,3}(?:\s*[,;–-]\s*\d{1,3})*\s*\]")
_PAREN_YEAR_RE = re.compile(r"\([^()]{0,80}?\b(?:19|20)\d{2}[a-z]?[^()]{0,25}?\)")
_NARRATIVE_YEAR_RE = re.compile(
    r"\b[A-ZÇĞİÖŞÜ][\wÇĞİÖŞÜ'’.-]*(?:\s+(?:ve|and|vd\.?|et\s+al\.?|[A-ZÇĞİÖŞÜ][\wÇĞİÖŞÜ'’.-]*)){0,4}"
    r"\s*\(?\b(?:19|20)\d{2}[a-z]?\)?"
)
_URL_RE = re.compile(r"https?://\S+|doi:\s*10\.\S+", re.IGNORECASE)
_ACTARAN_RE = re.compile(r"\b(?:akt\.?|aktar[ıi]lan|alıntılayan)\b", re.IGNORECASE)
_SUPERSCRIPT_RE = re.compile(r"[¹²³⁴⁵⁶⁷⁸⁹⁰]+")
_FOOTNOTE_LANG_RE = re.compile(r"\b(?:bkz\.|see|cf\.|p{1,2}\.\s*\d)", re.IGNORECASE)


def find_quotation_spans(text: str) -> list[Span]:
    """Character ranges that are explicitly marked as quotations."""
    spans: list[Span] = []
    for match in _QUOTE_RE.finditer(text):
        if match.group(0).strip('"“”«»„') == match.group(0):
            continue  # an empty or single-quote artefact
        spans.append(Span(match.start(), match.end(), match.group(0)[:60]))
    for match in _BLOCK_QUOTE_RE.finditer(text):
        spans.append(Span(match.start(), match.end(), "tırnak bloğu (markdown '>')"))
    return _merge(spans)


def find_citation_spans(text: str) -> list[Span]:
    """Character ranges that look like a reference to a source."""
    spans: list[Span] = []
    patterns = (
        _BRACKET_NUMBER_RE,
        _PAREN_YEAR_RE,
        _NARRATIVE_YEAR_RE,
        _URL_RE,
        _ACTARAN_RE,
        _SUPERSCRIPT_RE,
    )
    for pattern in patterns:
        for match in pattern.finditer(text):
            spans.append(Span(match.start(), match.end(), match.group(0)[:60]))
    return _merge(spans)


def is_quoted(spans: Sequence[Span], char_start: int, char_end: int) -> tuple[bool, str]:
    for span in spans:
        if span.char_start < char_end and span.char_end > char_start:
            return True, span.evidence
    return False, ""


def is_cited(spans: Sequence[Span], char_start: int, char_end: int) -> tuple[bool, str]:
    """A citation elsewhere in the sentence counts for the match.

    This is the standard reading: ``... (Yılmaz, 2021) ...`` placed at the end
    of the sentence attributes the whole sentence.
    """
    for span in spans:
        if span.char_start < char_end and span.char_end > char_start:
            return True, span.evidence
    return False, ""


def classify_attribution(
    text: str,
    match_start: int,
    match_end: int,
    *,
    quote_spans: Sequence[Span] | None = None,
    citation_spans: Sequence[Span] | None = None,
) -> CitationEvidence:
    """Classify one match inside ``text`` (absolute character offsets)."""
    quotes = quote_spans if quote_spans is not None else find_quotation_spans(text)
    citations = citation_spans if citation_spans is not None else find_citation_spans(text)

    quoted, quote_evidence = is_quoted(quotes, match_start, match_end)
    cited, citation_evidence = is_cited(citations, match_start, match_end)
    if not cited:
        # A reference at the end of the sentence attributes the whole sentence
        # and it usually sits *after* the quoted span: "..." (Yilmaz, 2021).
        #
        # Two subtleties, both of which produced wrong verdicts:
        #   * the window is the sentence of the match *start* - using the match
        #     end lets a multi-sentence match absorb the whole document;
        #   * sentence bounds ignore punctuation *inside* quotations, or
        #     "Veri temizligi gerceklestirilmis." (Yilmaz, 2021) looks like a
        #     sentence that ends before its own reference.
        window_start, window_end = sentence_bounds(text, match_start, quotes)
        cited, citation_evidence = is_cited(citations, window_start, max(match_end, window_end))

    status = classify_flags(cited=cited, quoted=quoted)
    return CitationEvidence(
        has_quote=quoted,
        has_citation=cited,
        quote_evidence=quote_evidence,
        citation_evidence=citation_evidence,
        status=status,
    )


def sentence_bounds(text: str, offset: int, quote_spans: Sequence[Span] = ()) -> tuple[int, int]:
    """Character range of the sentence containing ``offset``.

    Quotation-aware: a full stop inside "..." does not end the sentence.
    """

    def _inside_quote(position: int) -> bool:
        return any(s.char_start <= position < s.char_end for s in quote_spans)

    start = 0
    cursor = 0
    while cursor < offset:
        positions = [pos for pos in (text.find(ch, cursor) for ch in ".!?\n") if pos != -1]
        nxt = min((p for p in positions if p >= offset or p < offset), default=-1)
        candidates = sorted({p for p in positions if p < offset})
        step = candidates[-1] + 1 if candidates else offset
        if not candidates:
            start = offset
            break
        if _inside_quote(candidates[-1]):
            cursor = step
            continue
        start = step
        _ = nxt
        break
    else:
        start = start or 0

    end = len(text)
    cursor = offset
    while True:
        positions = [pos for pos in (text.find(ch, cursor) for ch in ".!?\n") if pos != -1]
        if not positions:
            end = len(text)
            break
        nxt = min(positions)
        if _inside_quote(nxt):
            cursor = nxt + 1
            continue
        end = nxt + 1
        break
    return start, end


def classify_flags(*, cited: bool, quoted: bool) -> CitationStatus:
    if cited and quoted:
        return CitationStatus.CITED_AND_QUOTED
    if cited:
        return CitationStatus.MISSING_QUOTATION
    if quoted:
        return CitationStatus.MISSING_CITATION
    return CitationStatus.NOT_CITED_OR_QUOTED


def summarise(statuses: Iterable[CitationStatus]) -> dict[str, int]:
    counts = {status.value: 0 for status in CitationStatus}
    for status in statuses:
        counts[status.value] += 1
    return counts


def _merge(spans: Sequence[Span]) -> list[Span]:
    ordered = sorted(spans, key=lambda s: (s.char_start, s.char_end))
    merged: list[Span] = []
    for span in ordered:
        if merged and span.char_start <= merged[-1].char_end:
            previous = merged[-1]
            merged[-1] = Span(previous.char_start, max(previous.char_end, span.char_end))
        else:
            merged.append(span)
    return merged
