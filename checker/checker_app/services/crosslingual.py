"""Cross-lingual reuse, found through what survives translation.

The problem this solves
-----------------------
A word-level matcher sees *zero* overlap between a Turkish passage and the
English paper it was translated from, so the report renders a near-verbatim
translation as "0% overlap, clean". That is the failure mode this module exists
to remove, and it is not hypothetical: cross-language precision is measured to
collapse from **80%** untranslated to **26.7%** translated to **16.7%**
translated-then-paraphrased (DOI 10.33806/ijaes1026), while PAN 2013 measured a
single translation step costing **11-15 PlagDet points**.

Why anchors rather than embeddings
----------------------------------
An embedding model would work better, and the tool already refuses to use one:
English-only models are catastrophic on Turkish, scoring **37.02** on EN-TR
bitext against **73.07** for a multilingual model (TR-MTEB,
DOI 10.18653/v1/2025.findings-emnlp.471), and a 0.6-1.1 GB second model is a
price this tool charges per sentence for a result it refuses to over-read.
Translation also destroys the thing embeddings rely on - Kobak et al. measured
TF-IDF cosine between a Turkish source and its LLM rewrite at **0.531**, and
**0.299** for completion.

So: anchors. What a translator leaves alone.

====================  =========================================================
anchor                why it survives translation
====================  =========================================================
**numbers**           sample sizes, thresholds, coefficients, percentages.
                      412 records, 0.80, 23 participants all stay put.
**decimal precision** 0.80 and 0.8 are different anchors; the number of
                      significant digits is a fingerprint.
**identifiers**       DOIs, arXiv ids, instrument names, gene labels.
**Latin-script proper  Turkish text keeps foreign surnames in Latin script:
nouns**               "Smith", "Cohen", "Kaufman" are not translated into
                      "Düşünce" however the prose around them is.
**author-year pairs** Smith (2019), Smith ve Jones (2019) - a citation shape.
====================  =========================================================

What counts as evidence
-----------------------
A single anchor is a coincidence; an academic paper mentions 412 or 0.8 without
it being anyone's. So a match requires a **cluster**: at least
:data:`MIN_CLUSTER` distinct anchors inside :data:`WINDOW_TOKENS` tokens, **in the
same relative order** in both texts. This is the same greedy-tiling idea as
HyPlag's GIT applied to a different alphabet - count ordered reused *units*,
not shared words - which is the construction that reached MRR **0.79** against
the order-agnostic **0.58**.

The order requirement is what separates a translated passage from two unrelated
sentences that happen to share numbers.

The honest ceiling, stated in every payload
-------------------------------------------
Recoverability is set by the **register** of the source, not the language pair:
**95.25% ± 1.76** on Wikipedia against **74.10% ± 1.29** on a scientific
conference corpus (DOI 10.18653/v1/E17-2066). A cross-lingual *hit* is evidence;
a cross-lingual *miss* is not evidence of absence. Reports say both.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from checker_app.domain.enums import MatchKind
from checker_app.domain.models import MatchSpan
from checker_app.domain.text import Token, fold_text

__all__ = [
    "AnchorCluster",
    "CrossLingualReport",
    "MIN_CLUSTER",
    "WINDOW_TOKENS",
    "CROSS_LINGUAL_EVIDENCE",
    "find_cross_lingual",
]

#: A cluster needs this many distinct anchors before it counts. Two is where
#: coincidences start; three is where an ordered run of a number, a decimal and
#: a proper noun becomes hard to explain by chance.
MIN_CLUSTER = 3

#: ...and at least this many of them must be non-year. A year appears in almost
#: every citation and in every methods section, so on its own it is not evidence;
#: requiring two numbers or identifiers keeps a cluster from being "2019, 2019,
#: 2019".
MIN_STRONG_ANCHORS = 2

#: Anchors must fall inside this many tokens of each other, in both texts.
#: Wide enough for a translated sentence, narrow enough that two unrelated
#: methods sections do not fuse.
WINDOW_TOKENS = 90

CROSS_LINGUAL_EVIDENCE = (
    "Çapraz dil tespiti tavanı: geri kazanılabilirlik dil çiftinden çok kaynaktaki "
    "KAYIT TÜRÜNE bağlıdır — Wikipedia'da %95.25 ± 1.76, bilimsel konferans "
    "korpusunda %74.10 ± 1.29 (DOI 10.18653/v1/E17-2066). PAN 2013'te tek çeviri "
    "adımı PlagDet'i 11-15 puan düşürüyor. Yani bulunan küme elle doğrulamayı "
    "gerektiren bir işarettir; bulunamayan küme hiçbir şey kanıtlamaz."
)

_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")
_DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:a-z0-9]+", re.IGNORECASE)
_CITATION_RE = re.compile(
    r"\((?:\d{4})\)|\b(?:19[5-9]\d|20[0-4]\d)\s*\)"
)
#: A Latin-script capitalised token that is not sentence-initial. Turkish prose
#: keeps foreign surnames in Latin script, so these survive translation while the
#: surrounding words do not.
_PROPER_RE = re.compile(r"^[A-Z][a-zÀ-ÿ'’-]{2,}$")


@dataclass(frozen=True, slots=True)
class AnchorCluster:
    """A co-located, order-consistent set of translation-resistant anchors."""

    source_id: str
    source_name: str
    anchors: tuple[str, ...]
    doc_token_start: int
    doc_token_end: int
    source_token_start: int
    source_token_end: int

    @property
    def size(self) -> int:
        return len(self.anchors)

    @property
    def strong_count(self) -> int:
        """Numbers and identifiers."""
        return sum(1 for a in self.anchors if _is_strong(a))

    @property
    def named_count(self) -> int:
        """Proper nouns and DOIs - the anchors that are hard to coincide on."""
        return sum(1 for a in self.anchors if a.startswith(("name:", "doi:")))

    def confidence(self) -> str:
        """How much this cluster is worth, calibrated against its own weakness.

        Calibrated against a control, not by intuition: a Turkish text that
        happens to contain "412 katılımcı, 2019, Cohen (1988), 0.80" in that
        order fires a four-anchor cluster against an unrelated English paper.
        Numbers alone are cheap - sample sizes and thresholds recur across a
        field. A proper noun or a DOI is not.
        """
        if self.named_count >= 1 and self.size >= 6:
            return "high"
        if self.size >= 6:
            return "medium"
        if self.named_count >= 1:
            return "medium"
        # Numbers only, and few of them: a flag, not a finding.
        return "review"

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source_name,
            "anchors": list(self.anchors),
            "size": self.size,
            "confidence": self.confidence(),
            "strong_anchors": self.strong_count,
            "named_anchors": self.named_count,
            "document_token_range": [self.doc_token_start, self.doc_token_end],
            "source_token_range": [self.source_token_start, self.source_token_end],
        }


@dataclass(frozen=True, slots=True)
class CrossLingualReport:
    """Clusters found, and the two sentences a reader needs with them."""

    clusters: tuple[AnchorCluster, ...] = ()
    document_language: str = ""
    cross_language_sources: int = 0
    lexical_matches: int = 0
    """How many matches the word-level matcher already found. When this is
    non-zero the documents are largely same-language and the cross-lingual pass
    adds little, which is worth saying rather than implying extra power."""

    evidence: str = CROSS_LINGUAL_EVIDENCE
    caveat: str = (
        "Yapısal çapadır; gömme ya da makine çevirisi yoktur. Yöntem, kelime "
        "eşleştirmesinin göremediği aktarımları yakalamaya çalışır ve bu yüzden "
        "bir ALT SINIRDIR: kaçırdığı her şeyi bilmez. Sayı tabanlı kümeler "
        "alan içinde tekrar eden örneklem büyüklükleri yüzünden tesadüfen "
        "oluşabilir; 'review' seviyesi tam olarak bu uyarıdır."
    )

    @property
    def clusters_found(self) -> int:
        return len(self.clusters)

    def reading(self) -> str:
        if not self.clusters:
            if self.cross_language_sources == 0:
                return "Tüm kaynaklar belge diliyle aynı; çapraz dil taraması yapılmadı."
            return (
                f"{self.cross_language_sources} farklı dilli kaynak tarandı ve "
                f"≥{MIN_CLUSTER} çapadan oluşan küme bulunamadı. "
                "**Bu, çapraz dil kullanımının olmadığı anlamına gelmez** — "
                "tavan %74'tür (bilimsel kayıt) ve paraphrase edilmiş metin "
                "bu yöntemin dışındadır."
            )
        by_confidence: dict[str, int] = {}
        for cluster in self.clusters:
            level = cluster.confidence()
            by_confidence[level] = by_confidence.get(level, 0) + 1
        parts = ", ".join(f"{count} {level}" for level, count in sorted(by_confidence.items()))
        return (
            f"{len(self.clusters)} çap kümesi bulundu ({parts}). Bunlar kelime "
            "eşleştirmesinin yapısal olarak göremediği aktarımlardır, ancak "
            "**kanıt değil işarettir**: yalnızca sayılardan oluşan kısa bir küme, "
            "alan içinde tekrar eden örneklem büyüklükleri ve eşikler yüzünden "
            "tesadüfen oluşabilir. 'review' seviyesindeki küme ancak kaynak metin "
            "elle karşılaştırıldığında anlamlıdır."
        )

    def to_dict(self, limit: int = 25) -> dict[str, object]:
        return {
            "clusters_found": self.clusters_found,
            "min_cluster": MIN_CLUSTER,
            "min_strong_anchors": MIN_STRONG_ANCHORS,
            "window_tokens": WINDOW_TOKENS,
            "document_language": self.document_language,
            "cross_language_sources": self.cross_language_sources,
            "lexical_matches_already_found": self.lexical_matches,
            "largest_cluster": max((c.size for c in self.clusters), default=0),
            "confidence_counts": {
                level: sum(1 for c in self.clusters if c.confidence() == level)
                for level in ("high", "medium", "review")
            },
            "sources_involved": sorted({c.source_name for c in self.clusters}),
            "clusters": [c.to_dict() for c in self.clusters[: max(1, limit)]],
            "truncated": self.clusters_found > max(1, limit),
            "reading": self.reading(),
            "evidence": self.evidence,
            "caveat": self.caveat,
        }


def _anchors(tokens: Sequence[Token], text: str = "") -> list[tuple[int, str]]:
    """Translation-resistant anchors as ``(token_index, label)`` pairs."""
    found: list[tuple[int, str]] = []
    initial = _sentence_initial_indices(tokens, text)
    for index, token in enumerate(tokens):
        surface = token.text
        if _DOI_RE.search(surface):
            doi = _DOI_RE.search(surface)
            if doi is not None:
                found.append((index, f"doi:{doi.group(0).lower()}"))
            continue
        for number in _NUMBER_RE.findall(surface):
            # Significant digits are the fingerprint: 0.80 and 0.8 differ, and a
            # bare 0 or 1 carries no information. A decimal keeps its structure
            # - stripping leading zeros would turn 0.80 into ".80".
            if "." in number or "," in number:
                value = number.replace(",", "")
            else:
                stripped = number.lstrip("0")
                if stripped in {"", "1"}:
                    continue
                value = stripped
            label = f"num:{value}"
            if label not in {label_ for _, label_ in found}:
                found.append((index, label))
        citation = _CITATION_RE.search(surface)
        if citation is not None:
            found.append((index, f"year:{citation.group(0).strip('()')}"))
        # A capital is free at the start of a sentence in *any* language, so
        # sentence-initial tokens are not proper nouns and carry no signal. This
        # was a real false-positive source: "Veriler temizlendi." yielded
        # ``name:veriler``, and "Her katılımcı..." yielded ``name:her``.
        if index in initial or not _PROPER_RE.match(surface):
            continue
        folded = fold_text(surface, strip_diacritics=True)
        if folded.lower() in _NOT_NAMES:
            continue
        found.append((index, f"name:{folded.lower()}"))
    return found


_SENTENCE_END = ".!?:;…،۔"
_NOT_NAMES = frozenset(
    {
        "the", "this", "that", "these", "those", "however", "therefore",
        "moreover", "thus", "furthermore", "while", "although", "because",
        "since", "results", "method", "methods", "discussion", "conclusion",
        "introduction", "abstract", "references", "data", "study", "table",
        "figure", "both", "all", "each", "when", "where", "there", "here",
        "one", "two", "first", "second",
    }
)


def _sentence_initial_indices(tokens: Sequence[Token], text: str = "") -> set[int]:
    """Token positions that begin a sentence.

    Detected from the **gap between tokens**, not from capitalisation and not
    from ``token.text``: the tokenizer strips punctuation, so a sentence-final
    period sits in the character range between one token's ``char_end`` and the
    next token's ``char_start``. Checking ``token.text[-1]`` therefore never
    fires, which is exactly how "Veriler" and "Her" slipped through as proper
    nouns and became fake anchors.

    With no source text the check degrades to the first token only, rather than
    guessing.
    """
    if not tokens:
        return set()
    initial: set[int] = {0}
    if not text:
        return initial
    for index in range(1, len(tokens)):
        gap = text[tokens[index - 1].char_end : tokens[index].char_start]
        # A newline counts as a boundary too: after a heading there is no full
        # stop, so "GİRİŞ" -> "Veriler" produced a fake name anchor. Erring this
        # way only ever *discards* a candidate, never adds a false one.
        if gap and (any(ch in _SENTENCE_END for ch in gap) or "\n" in gap):
            initial.add(index)
    return initial


def find_cross_lingual(
    *,
    document_tokens: Sequence[Token],
    sources: Sequence,
    document_language: str,
    document_text: str = "",
    lexical_matches: int = 0,
) -> CrossLingualReport:
    """Look for ordered anchor clusters between the document and each source.

    Only sources detected as a *different* language are considered, and only
    when the word-level matcher found nothing there - otherwise this is a second
    opinion on something already reported.
    """
    document = _anchors(document_tokens, document_text)
    cross_sources = [
        source
        for source in sources
        if source.language not in (document_language, "unknown")
    ]
    clusters: list[AnchorCluster] = []

    for source in cross_sources:
        source_anchors = _anchors(source.tokens, source.text)
        if len(document) < MIN_CLUSTER or len(source_anchors) < MIN_CLUSTER:
            continue
        clusters.extend(
            _cluster(document, document_tokens, source_anchors, source)
        )

    clusters.sort(key=lambda c: (-c.size, c.doc_token_start))
    return CrossLingualReport(
        clusters=tuple(clusters),
        document_language=document_language,
        cross_language_sources=len(cross_sources),
        lexical_matches=lexical_matches,
    )


def _is_strong(label: str) -> bool:
    """Evidence worth two anchors' worth.

    A number or an identifier, but not a bare single digit: "7 exclusions" is a
    real detail while "3 conditions" recurs in every methods section, and the
    first token of a document must not smuggle a capital into the anchor set.
    """
    if label.startswith("doi:"):
        return True
    if label.startswith("num:"):
        digits = label[4:].replace(".", "")
        return len(digits) >= 2
    return False


def _cluster(
    document: Sequence[tuple[int, str]],
    document_tokens: Sequence[Token],
    source_anchors: Sequence[tuple[int, str]],
    source,
) -> list[AnchorCluster]:  # noqa: ANN001
    """Greedy ordered clustering of anchors present on both sides.

    Longest-first, exactly as HyPlag tiles: find the largest ordered run, keep
    it, then look for the next one among the anchors it did not consume. Order is
    required because two unrelated sentences both mentioning "412" and "0.80" is
    a coincidence, while the *same* pair in the *same* order is a translation.

    Each candidate run is grown forward from one anchor while the order holds and
    the window fits, which keeps this linear in the anchor count rather than
    quadratic - a 20 000-word thesis has thousands of anchors.
    """
    by_label: dict[str, list[int]] = {}
    for index, label in source_anchors:
        by_label.setdefault(label, []).append(index)

    candidates = [(i, lab) for i, lab in document if lab in by_label]
    if len(candidates) < MIN_CLUSTER:
        return []

    # Grow the best ordered run from every starting position.
    runs: list[list[tuple[int, str]]] = []
    for start in range(len(candidates)):
        chosen: list[int] = []
        last_source = -1
        used_labels: set[str] = set()
        for index, label in candidates[start:]:
            if index - candidates[start][0] > WINDOW_TOKENS:
                break
            if label in used_labels:
                continue  # a repeated anchor is not independent evidence
            position = _next_position(by_label[label], last_source)
            if position is None:
                break  # order broken: not a translation
            chosen.append(position)
            used_labels.add(label)
            last_source = position
        if len(chosen) >= MIN_CLUSTER:
            window = candidates[start : start + len(chosen)]
            if sum(1 for _, label in window if _is_strong(label)) >= MIN_STRONG_ANCHORS:
                runs.append(window)

    # Longest first, then non-overlapping, the way HyPlag greedily tiles.
    runs.sort(key=len, reverse=True)
    used_document: set[int] = set()
    found: list[AnchorCluster] = []
    for run in runs:
        doc_indices = [index for index, _ in run]
        if len(set(doc_indices)) != len(doc_indices) or used_document.intersection(doc_indices):
            continue
        source_positions = _ordered_run(by_label, [label for _, label in run]) or []
        found.append(
            AnchorCluster(
                source_id=source.source_id,
                source_name=source.name,
                anchors=tuple(label for _, label in run),
                doc_token_start=min(doc_indices),
                doc_token_end=max(doc_indices),
                source_token_start=min(source_positions),
                source_token_end=max(source_positions),
            )
        )
        used_document.update(doc_indices)
    return found


def _next_position(options: Sequence[int], after: int) -> int | None:
    """Earliest occurrence strictly after ``after``."""
    for position in options:
        if position > after:
            return position
    return None


def _ordered_run(
    by_label: dict[str, list[int]], labels: Sequence[str]
) -> list[int] | None:
    """Pick one source position per label so the sequence is non-decreasing.

    Greedy left-to-right: take the earliest unused occurrence. If the order
    cannot be satisfied, the window is a coincidence rather than a translation.
    """
    chosen: list[int] = []
    last = -1
    for label in labels:
        options = [pos for pos in by_label[label] if pos > last and pos not in chosen]
        if not options:
            return None
        picked = min(options)
        chosen.append(picked)
        last = picked
    return chosen


def to_match_spans(report: CrossLingualReport, snippet_for) -> list[MatchSpan]:  # noqa: ANN001
    """Turn clusters into :class:`MatchSpan` objects for the normal pipeline.

    ``snippet_for`` maps ``(source_name, doc_token_start, doc_token_end)`` to the
    document text for that range. Kept as a callback so this module needs no
    knowledge of the sentence model.
    """
    spans: list[MatchSpan] = []
    for cluster in report.clusters:
        text = snippet_for(cluster.source_name, cluster.doc_token_start, cluster.doc_token_end)
        spans.append(
            MatchSpan(
                kind=MatchKind.CROSS_LINGUAL,
                source_id=cluster.source_id,
                source_name=cluster.source_name,
                sentence_indices=(),
                doc_char_start=text[0],
                doc_char_end=text[1],
                source_char_start=0,
                source_char_end=0,
                matched_words=cluster.size,
                ratio=0.0,
                snippet=text[2][:220],
                source_snippet="",
            )
        )
    return spans