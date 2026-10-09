"""Plagiarism detection: exact n-gram shingles plus reworded alignment.

Two complementary mechanisms:

**Exact shingles.** The document and every reference source are tokenised and
reduced to word n-grams. Shared n-grams that follow each other on the same
"diagonal" (same offset in both texts) are merged into runs, which is what makes
a report readable - "these 11 words are verbatim from X" - instead of a pile of
n-grams. A digit-folded second index catches reused *templates*: same wording,
different numbers, the most common form of academic self-plagiarism.

**Near matches.** ``difflib.SequenceMatcher`` (Myers diff - the family Google
publishes as diff-match-patch) aligns the two word streams and keeps blocks
above a similarity ratio, which catches translation and light paraphrasing.

Every match carries offsets on both sides, so it says *where* in the document and
*where* in the source it was found. Code and math blocks are compared separately
by :mod:`checker_app.services.code_ast` and credited through :meth:`credit`.
"""

from __future__ import annotations

import difflib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from checker_app.config import PlagiarismSettings
from checker_app.domain.enums import MatchKind
from checker_app.domain.models import MatchSpan, PlagiarismStats, Sentence, SourceDocument
from checker_app.domain.text import Token, fold_text, shingles
from checker_app.logging import get_logger

__all__ = [
    "PlagiarismResult",
    "drop_shadowed",
    "SourceSimilarity",
    "PlagiarismService",
    "content_bearing",
    "build_span",
]

logger = get_logger("checker.plagiarism")

# Tokens are compared in folded form, so the stopword list must be folded too
# ("için" -> "icin"); otherwise an all-stopword n-gram looks content bearing
# and matches everywhere.
_STOPWORDS_RAW = frozenset(
    {
        # Turkish
        "bir",
        "bu",
        "ve",
        "ile",
        "için",
        "olarak",
        "daha",
        "ancak",
        "çok",
        "gibi",
        "kadar",
        "sonra",
        "değil",
        "olan",
        "var",
        "ise",
        "ya",
        "her",
        "veya",
        "ki",
        "ne",
        "nasıl",
        "göre",
        "herhangi",
        "şey",
        # English
        "the",
        "and",
        "of",
        "to",
        "in",
        "is",
        "that",
        "for",
        "with",
        "this",
        "are",
        "was",
        "we",
        "have",
        "which",
        "using",
        "used",
        "can",
        "these",
        "been",
        "from",
        "our",
        "more",
        "than",
        "such",
        "their",
        "there",
        "when",
        "what",
        "also",
        "both",
        "each",
        "other",
        "into",
        "about",
    }
)
_STOPWORDS = frozenset(fold_text(word) for word in _STOPWORDS_RAW)
_INDEX_SOFT_LIMIT = 50_000
_NEAR_MATCH_BUDGET = 40_000_000
# How far apart two matching blocks may be and still count as one passage.
_NEAR_BLOCK_GAP = 2


@dataclass(frozen=True, slots=True)
class SourceSimilarity:
    """How much of the document overlaps one reference source."""

    source_id: str
    name: str
    shared_ngrams: int
    total_ngrams: int
    ratio: float


@dataclass(slots=True)
class PlagiarismResult:
    """Everything the plagiarism layer found."""

    sentences: dict[int, PlagiarismStats] = field(default_factory=dict)
    matches: tuple[MatchSpan, ...] = ()
    similarity: tuple[SourceSimilarity, ...] = ()
    degraded: tuple[str, ...] = ()

    @property
    def matched_sentences(self) -> int:
        return sum(1 for stats in self.sentences.values() if stats.has_match)


class PlagiarismService:
    """Compare a document against reference sources, per sentence."""

    def __init__(self, settings: PlagiarismSettings | None = None) -> None:
        self._settings = settings or PlagiarismSettings()

    # ------------------------------------------------------------------ public
    def compare(
        self,
        sentences: Sequence[Sentence],
        tokens: Sequence[Token],
        sources: Sequence[SourceDocument],
        doc_text: str = "",
    ) -> PlagiarismResult:
        settings = self._settings
        result = PlagiarismResult()
        if not sources or not tokens:
            result.sentences = {s.index: PlagiarismStats() for s in sentences}
            return result

        n = settings.ngram_size
        doc_grams = list(shingles(tokens, n))
        doc_gram_set = {gram for _, gram in doc_grams}
        exact_index, template_index, degraded = self._index_sources(sources)

        matches: list[MatchSpan] = []
        matches.extend(
            self._runs(
                doc_grams,
                exact_index,
                settings.min_match_words,
                sources,
                MatchKind.EXACT,
                tokens,
                doc_text,
            )
        )
        if settings.template_match and template_index:
            doc_folded = [
                (i, gram)
                for i, gram in _shingles_raw(
                    [
                        fold_text(t.text, strip_diacritics=False, normalize_digits=True)
                        for t in tokens
                    ],
                    n,
                )
            ]
            matches.extend(
                self._runs(
                    doc_folded,
                    template_index,
                    settings.template_min_words,
                    sources,
                    MatchKind.TEMPLATE,
                    tokens,
                    doc_text,
                )
            )
        if settings.near_match:
            matches.extend(self._near_matches(tokens, doc_text, sources, degraded))

        similarity = []
        for source in sources:
            shared = len({gram for _, gram in _source_grams(source, n) if gram in doc_gram_set})
            total = max(1, len(tokens) - n + 1)
            similarity.append(
                SourceSimilarity(
                    source_id=source.source_id,
                    name=source.name,
                    shared_ngrams=shared,
                    total_ngrams=total,
                    ratio=round(shared / total, 4),
                )
            )

        result.matches = tuple(sorted(matches, key=lambda m: (m.doc_char_start, -m.matched_words)))
        result.sentences = self.credit(sentences, result.matches)
        result.similarity = tuple(similarity)
        result.degraded = tuple(degraded)
        return result

    def credit(
        self, sentences: Sequence[Sentence], matches: Sequence[MatchSpan]
    ) -> dict[int, PlagiarismStats]:
        """Attribute matches to sentences and turn them into per-sentence stats.

        Public because code/math comparisons produce matches too and must be
        credited the same way.
        """
        by_sentence: dict[int, list[MatchSpan]] = {}
        for span in matches:
            for sentence in sentences:
                if (
                    sentence.location.char_start < span.doc_char_end
                    and sentence.location.char_end > span.doc_char_start
                ):
                    by_sentence.setdefault(sentence.index, []).append(span)

        stats: dict[int, PlagiarismStats] = {}
        for sentence in sentences:
            spans = by_sentence.get(sentence.index)
            if not spans:
                stats[sentence.index] = PlagiarismStats()
                continue
            covered = _union_length([(s.doc_char_start, s.doc_char_end) for s in spans])
            total_chars = max(1, sentence.location.char_end - sentence.location.char_start)
            longest = max(spans, key=lambda s: s.matched_words)
            stats[sentence.index] = PlagiarismStats(
                containment=round(min(1.0, covered / total_chars), 4),
                longest_match_words=longest.matched_words,
                best_source=longest.source_name,
                matches=tuple(sorted(spans, key=lambda s: -s.matched_words)),
                code_matches=sum(1 for s in spans if s.kind is MatchKind.CODE),
                math_matches=sum(1 for s in spans if s.kind is MatchKind.MATH),
            )
        return stats

    def score(self, stats: PlagiarismStats) -> float:
        """Turn per-sentence plagiarism into a 0..1 risk score.

        The attribution status scales the result, not just the per-match
        confidence: a passage that is quoted *and* referenced is legitimate
        reuse, and only 6.4% of matched pairs in a 2025-26 corpus study carry a
        citation at all (arXiv:2609.32963), so ignoring this would inflate the
        score of correct academic writing.
        """
        if not stats.has_match:
            return 0.0
        coverage = min(1.0, stats.containment / max(1e-6, self._settings.containment_threshold))
        length = min(1.0, stats.longest_match_words / max(8.0, self._settings.min_match_words * 3))
        confidence = max((m.confidence for m in stats.matches), default=0.0)
        base = 0.55 * coverage + 0.25 * length + 0.20 * confidence
        attribution = min(
            (
                m.citation_status.severity_weight
                for m in stats.matches
                if m.citation_status is not None
            ),
            default=1.0,
        )
        return round(min(1.0, base * attribution), 4)

    # ----------------------------------------------------------------- private
    def _index_sources(
        self, sources: Sequence[SourceDocument]
    ) -> tuple[
        dict[tuple[str, ...], list[tuple[int, int]]],
        dict[tuple[str, ...], list[tuple[int, int]]],
        list[str],
    ]:
        """n-gram -> [(source position, start token index)] for both foldings."""
        exact: dict[tuple[str, ...], list[tuple[int, int]]] = {}
        template: dict[tuple[str, ...], list[tuple[int, int]]] = {}
        degraded: list[str] = []
        limit = self._settings.max_source_chars
        n = self._settings.ngram_size
        for pos, source in enumerate(sources):
            if len(source.text) > limit:
                logger.warning(
                    "kaynak çok büyük, atlandı: %s (%d karakter)", source.name, len(source.text)
                )
                degraded.append(f"source_too_large:{source.name}")
                continue
            for start, gram in _source_grams(source, n):
                if not content_bearing(gram):
                    continue
                exact.setdefault(gram, []).append((pos, start))
            if self._settings.template_match:
                folded = [
                    fold_text(t.text, strip_diacritics=False, normalize_digits=True)
                    for t in source.tokens
                ]
                for start, gram in _shingles_raw(folded, n):
                    template.setdefault(gram, []).append((pos, start))
        return exact, template, degraded

    def _runs(
        self,
        doc_grams: Sequence[tuple[int, tuple[str, ...]]],
        index: dict[tuple[str, ...], list[tuple[int, int]]],
        min_words: int,
        sources: Sequence[SourceDocument],
        kind: MatchKind,
        doc_tokens: Sequence[Token],
        doc_text: str,
    ) -> list[MatchSpan]:
        """Merge shared n-grams that continue each other into verbatim runs."""
        n = self._settings.ngram_size
        # (doc_start, source_pos, source_start) -> the run it can extend.
        # Runs are mutable lists so a continuation updates them in place.
        last: dict[tuple[int, int, int], list[int]] = {}
        # [doc_start, doc_end, source_start, source_end, source_pos]
        runs: list[list[int]] = []

        for doc_start, gram in doc_grams:
            if not content_bearing(gram):
                self._prune(last, doc_start)
                continue
            positions = index.get(gram)
            if not positions:
                self._prune(last, doc_start)
                continue
            for source_pos, source_start in positions:
                # Overlapping n-grams start one token apart, so a run continues
                # at (doc_start - 1, source_start - 1) - not doc_start - n.
                chain = last.get((doc_start - 1, source_pos, source_start - 1))
                if chain is None:
                    runs.append(
                        [doc_start, doc_start + n, source_start, source_start + n, source_pos]
                    )
                    chain = runs[-1]
                else:
                    chain[1] = doc_start + n
                    chain[3] = source_start + n
                last[(doc_start, source_pos, source_start)] = chain
            self._prune(last, doc_start)

        spans: list[MatchSpan] = []
        for doc_start, doc_end, src_start, src_end, source_pos in runs:
            if doc_end - doc_start < min_words:
                continue
            spans.append(
                build_span(
                    doc_tokens=doc_tokens,
                    doc_text=doc_text,
                    doc_start=doc_start,
                    doc_end=doc_end,
                    source=sources[source_pos],
                    src_start=src_start,
                    src_end=src_end,
                    matched_words=doc_end - doc_start,
                    kind=kind,
                )
            )
        return _merge_touching(spans)

    @staticmethod
    def _prune(last: dict[tuple[int, int, int], list[int]], doc_start: int) -> None:
        """Forget chains that can no longer be continued (keeps memory flat)."""
        if len(last) < _INDEX_SOFT_LIMIT:
            return
        cutoff = doc_start - 1
        for key in [k for k in last if k[0] < cutoff]:
            del last[key]

    def _near_matches(
        self,
        tokens: Sequence[Token],
        doc_text: str,
        sources: Sequence[SourceDocument],
        degraded: list[str],
    ) -> list[MatchSpan]:
        settings = self._settings
        doc_words = [t.norm for t in tokens]
        spans: list[MatchSpan] = []
        for source in sources:
            src_words = [t.norm for t in source.tokens]
            if not src_words or len(doc_words) * len(src_words) > _NEAR_MATCH_BUDGET:
                logger.warning("yakın eşleştirme atlandı (boyut sınırı): %s", source.name)
                degraded.append(f"near_match_skipped:{source.name}")
                continue
            matcher = difflib.SequenceMatcher(None, doc_words, src_words, autojunk=False)
            for a, b, size in _merge_blocks(
                [blk for blk in matcher.get_matching_blocks() if blk.size],
                gap=_NEAR_BLOCK_GAP,
            ):
                if size < settings.near_match_min_words:
                    continue
                ratio = _block_ratio(tokens, source.tokens, a, b, size)
                if ratio < settings.near_match_ratio:
                    continue
                spans.append(
                    build_span(
                        doc_tokens=tokens,
                        doc_text=doc_text,
                        doc_start=a,
                        doc_end=a + size,
                        source=source,
                        src_start=b,
                        src_end=b + size,
                        matched_words=size,
                        kind=MatchKind.NEAR,
                        ratio=ratio,
                    )
                )
        return spans


# --------------------------------------------------------------------- helpers
def drop_shadowed(matches: Sequence[MatchSpan], coverage: float = 0.6) -> list[MatchSpan]:
    """Remove verbatim/near matches already explained by a code or math match.

    A copied LaTeX formula produces a ``math`` match *and* verbatim n-gram
    matches over its tokens. Reporting all of them inflates the sentence and
    buries the finding that matters.
    """
    structural = [m for m in matches if m.kind in {MatchKind.CODE, MatchKind.MATH}]
    keep: list[MatchSpan] = []
    for match in matches:
        if match.kind not in {MatchKind.EXACT, MatchKind.NEAR, MatchKind.TEMPLATE}:
            keep.append(match)
            continue
        shadowed = any(
            other.source_id == match.source_id
            and min(match.doc_char_end, other.doc_char_end)
            - max(match.doc_char_start, other.doc_char_start)
            >= coverage * max(1, match.doc_char_end - match.doc_char_start)
            for other in structural
        )
        if not shadowed:
            keep.append(match)
    return keep


def content_bearing(gram: tuple[str, ...]) -> bool:
    """Reject all-stopword n-grams: they match everywhere and prove nothing."""
    return sum(1 for word in gram if word not in _STOPWORDS and len(word) > 2) >= 2


def _source_grams(source: SourceDocument, n: int) -> Iterable[tuple[int, tuple[str, ...]]]:
    """``(start_token_index, n_gram)`` pairs for a source."""
    return shingles(source.tokens, n)


def _shingles_raw(words: Sequence[str], n: int) -> Iterable[tuple[int, tuple[str, ...]]]:
    if n <= 0 or len(words) < n:
        return
    for i in range(len(words) - n + 1):
        yield i, tuple(words[i : i + n])


def build_span(
    *,
    doc_tokens: Sequence[Token],
    doc_text: str,
    doc_start: int,
    doc_end: int,
    source: SourceDocument,
    src_start: int,
    src_end: int,
    matched_words: int,
    kind: MatchKind,
    ratio: float | None = None,
) -> MatchSpan:
    doc_char_start = doc_tokens[doc_start].char_start
    doc_char_end = doc_tokens[doc_end - 1].char_end
    src_char_start = source.tokens[src_start].char_start
    src_char_end = source.tokens[src_end - 1].char_end
    return MatchSpan(
        kind=kind,
        source_id=source.source_id,
        source_name=source.name,
        sentence_indices=(),
        doc_char_start=doc_char_start,
        doc_char_end=doc_char_end,
        source_char_start=src_char_start,
        source_char_end=src_char_end,
        matched_words=matched_words,
        ratio=1.0 if ratio is None else round(ratio, 4),
        snippet=doc_text[doc_char_start:doc_char_end][:400],
        source_snippet=source.text[src_char_start:src_char_end][:400],
        context=_context(doc_text, doc_char_start, doc_char_end),
        source_context=_context(source.text, src_char_start, src_char_end),
    )


def _merge_blocks(
    blocks: Sequence[tuple[int, int, int]], gap: int = 2
) -> list[tuple[int, int, int]]:
    """Join blocks that are separated by a few tokens on *both* sides.

    A sentence with one word changed arrives as two matching blocks; without
    this they would each be too short to report. The merged span is then scored
    on token overlap, so the changed word lowers the ratio instead of hiding it.
    """
    merged: list[tuple[int, int, int]] = []
    for a, b, size in sorted(blocks):
        if merged:
            prev_a, prev_b, prev_size = merged[-1]
            prev_end = prev_a + prev_size
            if a - prev_end <= gap and b - (prev_b + prev_size) <= gap:
                new_end = max(prev_end, a + size)
                merged[-1] = (prev_a, prev_b, new_end - prev_a)
                continue
        merged.append((a, b, size))
    return merged


def _block_ratio(doc: Sequence[Token], src: Sequence[Token], a: int, b: int, size: int) -> float:
    """Similarity of an aligned block: token overlap over the larger side."""
    left = {t.norm for t in doc[a : a + size]}
    right = {t.norm for t in src[b : b + size]}
    if not left or not right:
        return 0.0
    return round(len(left & right) / max(len(left), len(right)), 4)


def _merge_touching(spans: Sequence[MatchSpan]) -> list[MatchSpan]:
    """Merge runs that are contiguous on both sides (same source and kind)."""
    ordered = sorted(spans, key=lambda s: (s.source_id, s.kind.value, s.doc_char_start))
    merged: list[MatchSpan] = []
    for span in ordered:
        if merged:
            previous = merged[-1]
            if (
                previous.source_id == span.source_id
                and previous.kind is span.kind
                and previous.doc_char_end == span.doc_char_start
                and previous.source_char_end == span.source_char_start
            ):
                merged[-1] = MatchSpan(
                    kind=previous.kind,
                    source_id=previous.source_id,
                    source_name=previous.source_name,
                    sentence_indices=previous.sentence_indices,
                    doc_char_start=previous.doc_char_start,
                    doc_char_end=span.doc_char_end,
                    source_char_start=previous.source_char_start,
                    source_char_end=span.source_char_end,
                    matched_words=previous.matched_words + span.matched_words,
                    ratio=1.0,
                    snippet=_join(previous.snippet, span.snippet),
                    source_snippet=_join(previous.source_snippet, span.source_snippet),
                    context=previous.context,
                    source_context=previous.source_context,
                )
                continue
        merged.append(span)
    return merged


def _union_length(intervals: Sequence[tuple[int, int]]) -> int:
    total = 0
    for start, end in sorted(intervals):
        if end > start:
            total += end - start
    return total


def _join(left: str, right: str) -> str:
    return f"{left} {right}"[:400]


def _context(text: str, start: int, end: int, radius: int = 90) -> str:
    left = max(0, start - radius)
    right = min(len(text), end + radius)
    prefix = "…" if left > 0 else ""
    suffix = "…" if right < len(text) else ""
    return prefix + text[left:right].strip() + suffix
