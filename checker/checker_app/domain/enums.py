"""Enumerations shared by every signal layer."""

from __future__ import annotations

from enum import StrEnum


class BlockType(StrEnum):
    """What a block of the source document is."""

    PROSE = "prose"
    HEADING = "heading"
    LIST_ITEM = "list_item"
    QUOTE = "quote"
    CODE = "code"
    MATH = "math"
    TABLE = "table"
    CAPTION = "caption"


class RiskLevel(StrEnum):
    """Sentence or document risk band."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @property
    def rank(self) -> int:
        return _RISK_RANK[self]

    def at_least(self, other: RiskLevel) -> bool:
        return self.rank >= other.rank

    @property
    def color(self) -> str:
        return _RISK_COLOR[self]


_RISK_RANK: dict[RiskLevel, int] = {
    RiskLevel.NONE: 0,
    RiskLevel.LOW: 1,
    RiskLevel.MEDIUM: 2,
    RiskLevel.HIGH: 3,
    RiskLevel.CRITICAL: 4,
}

_RISK_COLOR: dict[RiskLevel, str] = {
    RiskLevel.NONE: "green",
    RiskLevel.LOW: "cyan",
    RiskLevel.MEDIUM: "yellow",
    RiskLevel.HIGH: "red",
    RiskLevel.CRITICAL: "bold red",
}


class Signal(StrEnum):
    """Where a number in the report came from."""

    PERPLEXITY = "perplexity"
    BURSTINESS = "burstiness"
    CLASSIFIER = "classifier"
    STYLOMETRY = "stylometry"
    UNIFORMITY = "uniformity"
    LIKELIHOOD_RATIO = "likelihood_ratio"
    """Two-model disagreement (Binoculars family): the only family with ~0% FPR
    on non-native/formal academic writing (arXiv:2608.26710)."""

    PLAGIARISM = "plagiarism"
    CODE_AST = "code_ast"

    @property
    def is_ai_signal(self) -> bool:
        return self in _AI_SIGNALS


_AI_SIGNALS = frozenset(
    {
        Signal.LIKELIHOOD_RATIO,
        Signal.PERPLEXITY,
        Signal.BURSTINESS,
        Signal.CLASSIFIER,
        Signal.STYLOMETRY,
        Signal.UNIFORMITY,
    }
)


class MatchKind(StrEnum):
    """How a plagiarism match relates to the source."""

    EXACT = "exact"
    """A verbatim run of >= ``min_match_words`` words."""

    NEAR = "near"
    """Same passage, reworded: aligned with difflib above ``near_match_ratio``."""

    TEMPLATE = "template"
    """Same run after digit folding: a reused form/template with new numbers."""

    CODE = "code"
    """Code block matched on its syntax tree or token stream."""

    MATH = "math"
    """LaTeX/math block matched after normalisation."""

    @property
    def base_confidence(self) -> float:
        return _MATCH_CONFIDENCE[self]


_MATCH_CONFIDENCE: dict[MatchKind, float] = {
    MatchKind.EXACT: 0.97,
    MatchKind.TEMPLATE: 0.85,
    MatchKind.NEAR: 0.65,
    MatchKind.CODE: 0.80,
    MatchKind.MATH: 0.75,
}


class Severity(StrEnum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class CitationStatus(StrEnum):
    """Turnitin's four attribution groups for a matched passage.

    Only 6.4% of matched paper pairs in a 2025-26 corpus-scale reuse study
    carried a citation link (human-verified 7.0%, 95% CI 3.5-10.6,
    arXiv:2609.32963), and matched pairs grew 4x while citation-linked pairs
    stayed flat: **a match count is not a misconduct count**. Reporting the
    status of each match is what keeps that distinction visible.
    """

    NOT_CITED_OR_QUOTED = "not_cited_or_quoted"
    """No citation, no quotation marks - the serious case."""

    MISSING_CITATION = "missing_citation"
    """Quoted, but no reference given."""

    MISSING_QUOTATION = "missing_quotation"
    """Reference given, but copied without quotation marks."""

    CITED_AND_QUOTED = "cited_and_quoted"
    """Legitimate: referenced and marked as a quotation."""

    @property
    def label(self) -> str:
        return {
            CitationStatus.NOT_CITED_OR_QUOTED: "Alıntılanmamış, kaynak gösterilmemiş",
            CitationStatus.MISSING_CITATION: "Alıntılanmış ama kaynak gösterilmemiş",
            CitationStatus.MISSING_QUOTATION: "Kaynak gösterilmiş ama tırnak içinde değil",
            CitationStatus.CITED_AND_QUOTED: "Kaynak gösterilmiş ve tırnak içinde (meşru)",
        }[self]

    @property
    def severity_weight(self) -> float:
        """How much this status counts toward the plagiarism score."""
        return {
            CitationStatus.NOT_CITED_OR_QUOTED: 1.0,
            CitationStatus.MISSING_CITATION: 0.75,
            CitationStatus.MISSING_QUOTATION: 0.55,
            CitationStatus.CITED_AND_QUOTED: 0.10,
        }[self]

    @property
    def attributed(self) -> bool:
        return self is not CitationStatus.NOT_CITED_OR_QUOTED


class DiffOp(StrEnum):
    """Result of comparing two revisions of the same text."""

    EQUAL = "equal"
    INSERT = "insert"
    DELETE = "delete"
    REPLACE = "replace"
