"""Pure domain objects for the checker.

No I/O, no model imports, no framework types: the same dataclasses are produced
by the segmenter, the signals, the scorer and consumed by the reporters and the
CLI.

Location is the point of this tool. Every sentence, every matched run and every
finding carries character-exact offsets plus line/page/section breadcrumbs, so a
report can answer "which sentence, where in the document" without re-reading the
source.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from checker_app.domain.enums import (
    BlockType,
    CitationStatus,
    DiffOp,
    MatchKind,
    RiskLevel,
    Severity,
    Signal,
)
from checker_app.domain.sections import SectionRole
from checker_app.domain.text import Token

SCHEMA_VERSION = 1

__all__ = [
    "SCHEMA_VERSION",
    "Location",
    "Sentence",
    "SourceDocument",
    "StyleHit",
    "StyleStats",
    "PerplexityStats",
    "RatioStats",
    "ClassifierStats",
    "MatchSpan",
    "PlagiarismStats",
    "Finding",
    "SentenceReport",
    "SectionReport",
    "DocumentReport",
    "SentenceDiff",
    "risk_level_for",
]


@dataclass(frozen=True, slots=True)
class Location:
    """Where a fragment sits in its source document."""

    char_start: int
    char_end: int
    line_start: int
    line_end: int
    paragraph_index: int
    page: int | None = None
    section_path: tuple[str, ...] = ()
    block_type: BlockType = BlockType.PROSE
    section_role: SectionRole = SectionRole.OTHER

    def label(self, *, with_paragraph: bool = True) -> str:
        """Human readable breadcrumb, e.g. ``s.3 ¶12 · Giriş > Yöntem``."""
        parts: list[str] = [f"satır {self.line_start}"]
        if self.line_end != self.line_start:
            parts[0] = f"satır {self.line_start}-{self.line_end}"
        if self.page:
            parts.append(f"s.{self.page}")
        if with_paragraph:
            parts.append(f"¶{self.paragraph_index}")
        if self.section_path:
            parts.append(" › ".join(self.section_path))
        if self.block_type not in {BlockType.PROSE}:
            parts.append(f"[{self.block_type}]")
        return " · ".join(parts)

    def citation(self) -> str:
        return f"char {self.char_start}-{self.char_end}, line {self.line_start}-{self.line_end}"


@dataclass(frozen=True, slots=True)
class Sentence:
    """One located sentence with its slice of the document token stream."""

    index: int
    text: str
    location: Location
    token_start: int
    token_end: int
    """Half-open range into :attr:`SourceDocument.tokens`."""

    @property
    def word_count(self) -> int:
        return max(0, self.token_end - self.token_start)

    def snippet(self, limit: int = 160) -> str:
        flat = " ".join(self.text.split())
        return flat if len(flat) <= limit else flat[: limit - 1] + "…"


@dataclass(frozen=True, slots=True)
class SourceDocument:
    """A reference text the document is compared against."""

    source_id: str
    name: str
    kind: str
    """file | dir | url | stdin | text"""

    text: str
    tokens: tuple[Token, ...]
    page_count: int = 1
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class StyleHit:
    """One triggered stylometric rule."""

    code: str
    severity: Severity
    weight: float
    detail: str
    evidence: str = ""


@dataclass(frozen=True, slots=True)
class StyleStats:
    """Rule-based features for one sentence."""

    word_count: int
    char_count: int
    avg_word_length: float = 0.0
    comma_density: float = 0.0
    """Commas per clause: dense punctuation is a model tell."""

    type_token_ratio: float = 0.0
    hapax_ratio: float = 0.0
    """Share of word types that occur exactly once.

    Second of the three lexical-richness features that carry this literature.
    Dropping the trio costs 13.1 F1 in-domain and **27.7 F1 out-of-domain**,
    and the trio *alone* beats all 284 features of the same study out-of-domain
    by 14.29 F1 (El Attar vd., arXiv:2606.04177). TTR and hapax move together, so
    only one of the three is reported as a per-sentence feature; the other two
    live in the document-level profile, where token counts are trustworthy.
    """

    lexical_density: float = 0.0
    """Share of tokens that are content words. Third of the same trio."""

    informal_rate: float = 0.0
    """Contractions / slang markers: 0 means a suspiciously clean register."""

    digits_ratio: float = 0.0
    upper_ratio: float = 0.0
    hits: tuple[StyleHit, ...] = ()

    @property
    def hit_codes(self) -> tuple[str, ...]:
        return tuple(h.code for h in self.hits)

    @property
    def total_weight(self) -> float:
        return sum(h.weight for h in self.hits)


@dataclass(frozen=True, slots=True)
class PerplexityStats:
    """Perplexity/burstiness numbers for one sentence."""

    perplexity: float
    mean_nll: float
    token_count: int
    context_tokens: int = 0
    log10_perplexity: float = 0.0
    z_score: float | None = None
    """Relative position inside the document (lower than typical = predictable)."""

    truncated: bool = False
    model: str = ""


@dataclass(frozen=True, slots=True)
class RatioStats:
    """Two-model likelihood ratio (Binoculars family) for one sentence.

    ``score = log(PPL_observer) / log(PPL_performer)``. Human text is hard for
    both models and they disagree; machine text is easy for both and they agree,
    so the ratio falls toward 1.
    """

    score: float
    observer_perplexity: float
    performer_perplexity: float
    token_count: int
    observer_model: str
    performer_model: str
    usable: bool = True

    @property
    def human_likelihood(self) -> float:
        """0..1 human-likeness, **increasing** in ``score``.

        In this family a *low* ratio means machine-like: observer and performer
        agree on an easy, predictable text. Human text is hard for both and they
        disagree, so the ratio is larger.
        """
        return 1.0 - self.machine_likelihood(0.90, 0.25)

    def machine_likelihood(self, center: float = 0.90, scale: float = 0.25) -> float:
        """0..1 machine-likeness around a documented operating point.

        ``center`` is the published decision threshold of the family (0.901 in
        Binoculars); ``scale`` controls how sharp the transition is. A flat
        sigmoid around 0 would never move the score, so the threshold - not a
        guess - is what anchors the mapping.
        """
        import math

        return 1.0 / (1.0 + math.exp((self.score - center) / max(1e-3, scale)))


@dataclass(frozen=True, slots=True)
class ClassifierStats:
    """Output of the AI/Human sequence classifier."""

    p_ai: float
    model: str = ""
    context_sentences: int = 0


@dataclass(frozen=True, slots=True)
class MatchSpan:
    """A passage found in both the document and a reference source."""

    kind: MatchKind
    source_id: str
    source_name: str
    sentence_indices: tuple[int, ...]
    doc_char_start: int
    doc_char_end: int
    source_char_start: int
    source_char_end: int
    matched_words: int
    ratio: float
    snippet: str
    source_snippet: str = ""
    context: str = ""
    source_context: str = ""
    citation_status: CitationStatus | None = None
    """Turnitin's four groups. Only 6.4% of matched pairs in a 2025-26 corpus
    study carried a citation link (arXiv:2609.32963): a match count is not a
    misconduct count, and this field is what keeps them apart."""

    citation_evidence: str = ""

    @property
    def confidence(self) -> float:
        base = self.kind.base_confidence * min(1.0, 0.55 + 0.45 * self.ratio)
        if self.citation_status is not None:
            return base * self.citation_status.severity_weight
        return base


@dataclass(frozen=True, slots=True)
class PlagiarismStats:
    """Per-sentence plagiarism numbers."""

    containment: float = 0.0
    """Share of the sentence covered by matched runs."""

    longest_match_words: int = 0
    best_source: str | None = None
    matches: tuple[MatchSpan, ...] = ()
    code_matches: int = 0
    math_matches: int = 0

    @property
    def has_match(self) -> bool:
        return bool(self.matches or self.code_matches or self.math_matches)


@dataclass(frozen=True, slots=True)
class Finding:
    """One explanation attached to a sentence, in report language."""

    code: str
    signal: Signal
    severity: Severity
    score: float
    message: str
    evidence: str = ""


@dataclass(frozen=True, slots=True)
class SentenceReport:
    """Everything known about one sentence."""

    index: int
    text: str
    location: Location
    word_count: int
    style: StyleStats
    section_role: SectionRole = SectionRole.OTHER
    perplexity: PerplexityStats | None = None
    ratio: RatioStats | None = None
    classifier: ClassifierStats | None = None
    plagiarism: PlagiarismStats | None = None
    ai_score: float = 0.0
    plagiarism_score: float = 0.0
    risk_score: float = 0.0
    risk: RiskLevel = RiskLevel.NONE
    confidence: float = 0.0
    """How much of the scoring evidence was actually available (0..1)."""

    findings: tuple[Finding, ...] = ()

    @property
    def ai_percent(self) -> int:
        return int(round(self.ai_score * 100))

    @property
    def plagiarism_percent(self) -> int:
        return int(round(self.plagiarism_score * 100))

    @property
    def unscored_for_ai(self) -> bool:
        """True when this sentence was too short or structurally excluded."""
        return self.ratio is None and self.classifier is None

    def snippet(self, limit: int = 160) -> str:
        flat = " ".join(self.text.split())
        return flat if len(flat) <= limit else flat[: limit - 1] + "…"


@dataclass(frozen=True, slots=True)
class SectionReport:
    """Aggregated risk for one heading section."""

    title: str
    depth: int
    sentence_indices: tuple[int, ...]
    word_count: int
    ai_score: float
    plagiarism_score: float
    high_risk_count: int

    @property
    def risk(self) -> RiskLevel:
        return risk_level_for(max(self.ai_score, self.plagiarism_score))


@dataclass(frozen=True, slots=True)
class SentenceDiff:
    """One sentence-level change between two revisions."""

    op: DiffOp
    left_index: int | None
    right_index: int | None
    left_text: str | None
    right_text: str | None
    similarity: float = 0.0
    left_location: Location | None = None
    right_location: Location | None = None


@dataclass(frozen=True, slots=True)
class DocumentReport:
    """The scan result: every sentence, the sections, and the verdict."""

    path: str
    language: str
    language_confidence: float
    char_count: int
    word_count: int
    sentence_count: int
    page_count: int
    sentences: tuple[SentenceReport, ...]
    sections: tuple[SectionReport, ...] = ()
    ai_score: float = 0.0
    plagiarism_score: float = 0.0
    risk_score: float = 0.0
    risk: RiskLevel = RiskLevel.NONE
    flagged_sentences: int = 0
    high_risk_sentences: int = 0
    matched_sentences: int = 0
    mean_perplexity: float | None = None
    perplexity_burstiness: float | None = None
    length_burstiness: float | None = None
    models: dict[str, str] = field(default_factory=dict)
    degraded_signals: tuple[str, ...] = ()
    """Signals that could not run, e.g. ``classifier: model unavailable``."""

    sources: tuple[dict[str, Any], ...] = ()
    settings: dict[str, Any] = field(default_factory=dict)
    unscored_sentences: int = 0
    """Sentences the AI signals deliberately skipped (too short or structural)."""

    ai_share: float = 0.0
    """Word-weighted share of sentences at MEDIUM AI risk or above. Compared
    against the 20% reporting floor before any document-level AI claim."""

    ai_flagged_words: int = 0
    below_ai_reporting_floor: bool = False
    """Under the floor, no document-level AI verdict is issued - only the located
    per-sentence findings. Turnitin's own numbers for why: on 719 877 pre-2019
    human student writings it reports documents at FPR 0.51% and sentences at
    0.33% (AIW-2, Aug 2024)."""

    similarity: Any = None
    """Turnitin-style institutional similarity report, when references were given."""

    patchwork: Any = None
    """Quantity-sensitive reuse shape (Greedy Identifier Tiles), with its measured
    chance baseline. Lives beside ``similarity`` because it is the second half of
    the same question: the percentage says how much, this says how assembled."""

    discourse: Any = None
    """Relation-mapped connective profile for the document."""

    style_stats: Any = None
    """Document-level style context: the lexical-richness trio and the length
    confound. Reported, never scored - see ``LITERATURE.md`` §1.10 and §1.12."""

    references: Any = None
    """Structural audit of the reference list: which entries a committee should
    look up by hand, and the measured rates that make it worth doing."""

    integrity: Any = None
    """Byte-level health of the document (homoglyphs, zero-width characters)."""

    duration_seconds: float = 0.0
    disclaimer: str = ""
    degree_caveat: str = ""
    """Why no output of this tool states how *much* of a text was machine
    written. Present in every report rather than only the README, because the
    sentence-level output is the part people screenshot."""
    human_baseline: str = ""
    """What lecturers achieve at this task, so the number has a comparison."""
    operating_point: dict[str, Any] = field(default_factory=dict)

    def sentence(self, index: int) -> SentenceReport:
        return self.sentences[index]

    def sorted_by_risk(self, limit: int | None = None) -> list[SentenceReport]:
        ordered = sorted(self.sentences, key=lambda s: (-s.risk_score, s.index))
        return ordered[:limit] if limit else ordered

    def flagged(self, level: RiskLevel = RiskLevel.MEDIUM) -> list[SentenceReport]:
        return [s for s in self.sentences if s.risk.at_least(level)]

    def sections_of(self, path: Sequence[str]) -> SectionReport | None:
        title = " › ".join(path)
        for section in self.sections:
            if section.title == title:
                return section
        return None

    def to_dict(self) -> dict[str, Any]:
        """Stable JSON shape (``schema_version`` first)."""
        return {
            "schema_version": SCHEMA_VERSION,
            "disclaimer": self.disclaimer,
            "degree_caveat": self.degree_caveat,
            "human_baseline": self.human_baseline,
            "document": {
                "path": self.path,
                "language": self.language,
                "language_confidence": round(self.language_confidence, 3),
                "char_count": self.char_count,
                "word_count": self.word_count,
                "sentence_count": self.sentence_count,
                "page_count": self.page_count,
                "duration_seconds": round(self.duration_seconds, 3),
            },
            "verdict": {
                "ai_score": round(self.ai_score, 4),
                "plagiarism_score": round(self.plagiarism_score, 4),
                "risk_score": round(self.risk_score, 4),
                "risk": self.risk.value,
                "flagged_sentences": self.flagged_sentences,
                "high_risk_sentences": self.high_risk_sentences,
                "matched_sentences": self.matched_sentences,
                "mean_perplexity": _round_opt(self.mean_perplexity),
                "perplexity_burstiness": _round_opt(self.perplexity_burstiness),
                "length_burstiness": _round_opt(self.length_burstiness),
                "ai_share": self.ai_share,
                "ai_flagged_words": self.ai_flagged_words,
                "below_ai_reporting_floor": self.below_ai_reporting_floor,
            },
            "models": self.models,
            "degraded_signals": list(self.degraded_signals),
            "unscored_sentences": self.unscored_sentences,
            "operating_point": self.operating_point,
            "similarity": self.similarity.to_dict() if self.similarity else None,
            "patchwork": self.patchwork.to_dict() if self.patchwork else None,
            "discourse": self.discourse.to_dict() if self.discourse else None,
            "style_stats": self.style_stats.to_dict() if self.style_stats else None,
            "references": self.references.to_dict() if self.references else None,
            "integrity": self.integrity.to_dict() if self.integrity else None,
            "sources": [dict(s) for s in self.sources],
            "settings": self.settings,
            "sections": [
                {
                    "title": s.title,
                    "depth": s.depth,
                    "sentence_indices": list(s.sentence_indices),
                    "word_count": s.word_count,
                    "ai_score": round(s.ai_score, 4),
                    "plagiarism_score": round(s.plagiarism_score, 4),
                    "risk": s.risk.value,
                    "high_risk_count": s.high_risk_count,
                }
                for s in self.sections
            ],
            "sentences": [_sentence_to_dict(s) for s in self.sentences],
        }


def _sentence_to_dict(s: SentenceReport) -> dict[str, Any]:
    out: dict[str, Any] = {
        "index": s.index,
        "text": s.text,
        "location": {
            "char_start": s.location.char_start,
            "char_end": s.location.char_end,
            "line_start": s.location.line_start,
            "line_end": s.location.line_end,
            "paragraph_index": s.location.paragraph_index,
            "page": s.location.page,
            "section_path": list(s.location.section_path),
            "block_type": s.location.block_type.value,
            "label": s.location.label(),
        },
        "word_count": s.word_count,
        "scores": {
            "ai_score": round(s.ai_score, 4),
            "plagiarism_score": round(s.plagiarism_score, 4),
            "risk_score": round(s.risk_score, 4),
            "risk": s.risk.value,
            "confidence": round(s.confidence, 4),
        },
        "style": {
            "avg_word_length": round(s.style.avg_word_length, 3),
            "comma_density": round(s.style.comma_density, 3),
            "type_token_ratio": round(s.style.type_token_ratio, 3),
            "informal_rate": round(s.style.informal_rate, 3),
            "hits": [
                {
                    "code": h.code,
                    "severity": h.severity.value,
                    "weight": h.weight,
                    "detail": h.detail,
                    "evidence": h.evidence,
                }
                for h in s.style.hits
            ],
        },
        "section_role": s.section_role.value,
        "ratio": (
            None
            if s.ratio is None
            else {
                "score": round(s.ratio.score, 4),
                "observer_perplexity": s.ratio.observer_perplexity,
                "performer_perplexity": s.ratio.performer_perplexity,
                "machine_likelihood": round(s.ratio.machine_likelihood(), 4),
                "observer_model": s.ratio.observer_model,
                "performer_model": s.ratio.performer_model,
                "token_count": s.ratio.token_count,
            }
        ),
        "perplexity": (
            None
            if s.perplexity is None
            else {
                "perplexity": _round_opt(s.perplexity.perplexity, 2),
                "mean_nll": round(s.perplexity.mean_nll, 4),
                "token_count": s.perplexity.token_count,
                "context_tokens": s.perplexity.context_tokens,
                "log10_perplexity": round(s.perplexity.log10_perplexity, 4),
                "z_score": _round_opt(s.perplexity.z_score),
                "model": s.perplexity.model,
            }
        ),
        "classifier": (
            None
            if s.classifier is None
            else {
                "p_ai": round(s.classifier.p_ai, 4),
                "model": s.classifier.model,
                "context_sentences": s.classifier.context_sentences,
            }
        ),
        "plagiarism": None
        if s.plagiarism is None
        else {
            "containment": round(s.plagiarism.containment, 4),
            "longest_match_words": s.plagiarism.longest_match_words,
            "best_source": s.plagiarism.best_source,
            "code_matches": s.plagiarism.code_matches,
            "math_matches": s.plagiarism.math_matches,
            "matches": [
                {
                    "kind": m.kind.value,
                    "source_id": m.source_id,
                    "source_name": m.source_name,
                    "sentence_indices": list(m.sentence_indices),
                    "doc_char_start": m.doc_char_start,
                    "doc_char_end": m.doc_char_end,
                    "source_char_start": m.source_char_start,
                    "source_char_end": m.source_char_end,
                    "matched_words": m.matched_words,
                    "ratio": round(m.ratio, 4),
                    "confidence": round(m.confidence, 4),
                    "snippet": m.snippet,
                    "source_snippet": m.source_snippet,
                }
                for m in s.plagiarism.matches
            ],
        },
        "findings": [
            {
                "code": f.code,
                "signal": f.signal.value,
                "severity": f.severity.value,
                "score": round(f.score, 4),
                "message": f.message,
                "evidence": f.evidence,
            }
            for f in s.findings
        ],
    }
    return out


def _round_opt(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(value, digits)


def risk_level_for(score: float, thresholds: dict[str, float] | None = None) -> RiskLevel:
    """Map a 0..1 score onto :class:`RiskLevel`."""
    t = thresholds or {"low": 0.35, "medium": 0.55, "high": 0.75, "critical": 0.90}
    if score >= t.get("critical", 0.90):
        return RiskLevel.CRITICAL
    if score >= t.get("high", 0.75):
        return RiskLevel.HIGH
    if score >= t.get("medium", 0.55):
        return RiskLevel.MEDIUM
    if score >= t.get("low", 0.35):
        return RiskLevel.LOW
    return RiskLevel.NONE
