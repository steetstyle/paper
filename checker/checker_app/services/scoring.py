"""Fusing the signals into one explainable risk number per sentence.

Design rules:

* **No single signal decides.** Perplexity, the classifier, stylometry and
  length uniformity vote; each contributes a term in 0..1 with a configurable
  weight, and the weights of *unavailable* signals are dropped instead of
  counted as zero - so an offline run is not silently "less risky", it is
  *less certain*, and says so through ``confidence``.
* **Relative and absolute perplexity.** A calibrated absolute term catches text
  that is machine-flat everywhere; a within-document relative term catches one
  suspiciously predictable sentence among human ones. Both are needed.
* **Every number carries a sentence.** :class:`~checker_app.domain.models.Finding`
  explains what fired, with the evidence value, so a score can be argued with.

This is a heuristic reading aid for a human reviewer, not a verdict.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field, replace

from checker_app.config import CheckerSettings
from checker_app.domain.enums import (
    BlockType,
    MatchKind,
    RiskLevel,
    Severity,
    Signal,
)
from checker_app.domain.models import (
    ClassifierStats,
    Finding,
    PerplexityStats,
    PlagiarismStats,
    RatioStats,
    SectionReport,
    Sentence,
    SentenceReport,
    StyleStats,
    risk_level_for,
)
from checker_app.domain.sections import SectionRole, role_prior
from checker_app.logging import get_logger
from checker_app.services.perplexity import burstiness as stdev
from checker_app.services.perplexity import cv
from checker_app.services.plagiarism import PlagiarismService

__all__ = ["Scorer", "DocumentScores", "DISCLAIMER", "is_prose"]

logger = get_logger("checker.scoring")

DISCLAIMER = (
    "Bu rapor bir TAHMİN sıralamasıdır, kesin bir hüküm değildir. AI tespit araçları "
    "dilbilimsel örüntülere bakar ve özellikle kendi ana dilinde yazmayan, çok resmî veya "
    "teknik metinlerde yanlış pozitif üretir. Tek bir cümleyi kanıt saymak mümkün değildir; "
    "bulguları bir insan gözden geçirmelidir."
)

#: Turnitin's AIW-2 white paper (Aug 2024) does not report an AI score at all
#: below this document share, because false-positive incidence rises under it.
#: Measured there on 719,877 pre-2019 human student writings: document FPR
#: **0.51%**, sentence FPR **0.33%**, document recall **91.18%**. We adopt the
#: same floor, so a thesis with a few flagged sentences cannot acquire a
#: document-level AI verdict by accumulation alone.
AI_REPORTING_FLOOR = 0.20

#: Severity means "this needs a human to read it", never "this much of the text
#: was written by a machine". The evidence for refusing the stronger claim:
#: GLTR's false-positive rate goes from **6.83%** on untouched human text to
#: **26.85%** when only **1% of words** were edited, and to **40.87%** after a
#: cosmetic GPT-4o polish (Saha & Feizi, APT-Eval, arXiv:2502.15666). Detectors
#: also cannot grade the degree: RoBERTa moves only 4.3 points between "minor"
#: and "major" polishing, and Fast-DetectGPT is inverted (10.07% at 1% of words
#: edited vs 9.59% at 75%). A flagged sentence is compatible with a 1% edit.
AI_DEGREE_CAVEAT = (
    "İşaretlenen cümle 'tamamen makine yazıldı' demek değildir: kelimelerin yalnızca "
    "%1'inin düzenlenmesi GLTR'nin yanlış pozitif oranını %6.83'ten %26.85'e çıkarıyor, "
    "kozmetik bir GPT-4o cilası %40.87'ye (APT-Eval, arXiv:2502.15666). Dedektörler "
    "düzeltme derecesini ölçemez. Bu nedenle hiçbir yerde 'AI tarafından yazıldı' "
    "denmez; yalnızca 'insan gözden geçirmeli' denir."
)

#: What the strongest available human baseline is, because a reader should know
#: this is not a task people are good at either. 63 lecturers read 200-300 word
#: excerpts from German university-of-applied-sciences theses: **57%** correct on
#: AI texts, **64%** on human texts, no significant gap between human and
#: machine detectors, and **under 20%** correct on professional-level AI text
#: (Fiedler & Döpke, Int. Review of Economics Education 49:100321, 2025,
#: 10.1016/j.iree.2025.100321). Sentence localization with a Binoculars-family
#: score is F1@K **0.608** on TriBERT (2025.findings-ijcnlp.48).
AI_HUMAN_BASELINE = (
    "Jüri de bu işte zayıf: 63 öğretim üyesi Alman tez parçalarında %57 (AI metni) ve %64 "
    "(insan metni) doğru tanıdı, profesyonel düzeydeki AI metninde doğruluk %20'nin altında "
    "kaldı (10.1016/j.iree.2025.100321). Cümle düzeyinde yerelleştirme için Binoculars "
    "ailesinin F1@K değeri 0.608'dir (2025.findings-ijcnlp.48). Tez alanı diğer "
    "alanlardan da zordur: DeBERTa F1 85.88 (tez), 90.05 (hikâye), 60.66 (deneme); "
    "arXiv:2410.14259."
)


@dataclass(slots=True)
class DocumentScores:
    """Scored sentences plus document and section aggregates."""

    sentences: tuple[SentenceReport, ...]
    sections: tuple[SectionReport, ...]
    ai_score: float
    plagiarism_score: float
    risk_score: float
    risk: RiskLevel
    flagged: int
    high_risk: int
    matched: int
    mean_perplexity: float | None
    perplexity_burstiness: float | None
    length_burstiness: float | None
    degraded: tuple[str, ...] = field(default=())
    ai_share: float = 0.0
    """Word-weighted share of sentences at MEDIUM AI risk or above."""
    ai_flagged_words: int = 0
    below_ai_reporting_floor: bool = False
    """True when :data:`AI_REPORTING_FLOOR` is not reached, so no document-level
    AI claim is made. Per-sentence findings are still reported - Turnitin reports
    sentences (FPR 0.33%) long before it reports documents (FPR 0.51%)."""


class Scorer:
    """Turn signals into :class:`SentenceReport` objects."""

    def __init__(self, settings: CheckerSettings, language: str) -> None:
        self._settings = settings
        self._language = language
        self._plagiarism = PlagiarismService(settings.plagiarism)

    # ------------------------------------------------------------------ public
    def score(
        self,
        sentences: Sequence[Sentence],
        styles: dict[int, StyleStats],
        perplexities: dict[int, PerplexityStats],
        classifiers: dict[int, ClassifierStats],
        plagiarism: dict[int, PlagiarismStats],
        degraded: Sequence[str] = (),
        ratios: dict[int, RatioStats] | None = None,
    ) -> DocumentScores:
        scoring = self._settings.scoring
        lengths = [float(s.word_count) for s in sentences]
        length_cv = cv(lengths)
        mean_length = sum(lengths) / len(lengths) if lengths else 0.0
        std_length = stdev(lengths)
        z_scores = self._z_scores(perplexities)
        ratios = ratios or {}
        min_words = self._settings.ai.min_ai_words

        reports: list[SentenceReport] = []
        for sentence in sentences:
            style = styles.get(sentence.index, _empty_style(sentence))
            role = sentence.location.section_role
            # Short sentences are where detectors fail and where false positives
            # live (AUROC 0.62 at 10 tokens, FP modal length 34 words).
            scorable = is_prose(sentence) and role.scored and sentence.word_count >= min_words
            perplexity = perplexities.get(sentence.index) if scorable else None
            ratio = ratios.get(sentence.index) if scorable else None
            if perplexity is not None and sentence.index in z_scores:
                # Frozen dataclass: rebuild with the relative score attached.
                perplexity = replace(perplexity, z_score=z_scores[sentence.index])
            classifier = classifiers.get(sentence.index) if scorable else None
            plag = plagiarism.get(sentence.index)

            terms: dict[Signal, tuple[float, float]] = {}
            findings: list[Finding] = []

            if ratio is not None:
                # A low observer/performer ratio means the models agree: the
                # text is predictable for both, which is the machine signature.
                machine_likelihood = clamp01(
                    ratio.machine_likelihood(
                        self._settings.ratio_threshold(self._language),
                        scoring.ratio_scale,
                    )
                )
                terms[Signal.LIKELIHOOD_RATIO] = (
                    machine_likelihood,
                    scoring.weight_ratio,
                )
                # 0.5 is the decision boundary by construction; 0.6 is roughly
                # one logistic scale-width past it, i.e. a sentence the two
                # models agree is predictable.
                if machine_likelihood >= 0.6:
                    findings.append(
                        Finding(
                            code="models_agree",
                            signal=Signal.LIKELIHOOD_RATIO,
                            severity=_severity(machine_likelihood, 0.6, 0.8, 0.92),
                            score=round(machine_likelihood, 4),
                            message=(
                                "İki dil modeli bu cümlede anlaşıyor "
                                "(oran "
                                f"{ratio.score:.3f}); üretilmiş metin örüntüsü."
                            ),
                            evidence=(
                                f"gözlemci {ratio.observer_model} ppl "
                                f"{ratio.observer_perplexity:.1f} / icra "
                                f"{ratio.performer_model} ppl "
                                f"{ratio.performer_perplexity:.1f}"
                            ),
                        )
                    )
            elif is_prose(sentence) and role.scored and sentence.word_count < min_words:
                findings.append(
                    Finding(
                        code="too_short_for_ai",
                        signal=Signal.STYLOMETRY,
                        severity=Severity.INFO,
                        score=0.0,
                        message=(
                            f"Cümle {sentence.word_count} kelime ({min_words} eşiğinin altında); "
                            "AI ölçümü yapılmadı - kısa metinde tespit güvenilmez."
                        ),
                        evidence="",
                    )
                )

            ppl_term, ppl_findings = self._perplexity_terms(
                perplexity, z_scores.get(sentence.index)
            )
            if ppl_term is not None:
                terms[Signal.PERPLEXITY] = (ppl_term, scoring.weight_perplexity)
            findings.extend(ppl_findings)

            if classifier is not None:
                terms[Signal.CLASSIFIER] = (clamp01(classifier.p_ai), scoring.weight_classifier)
                findings.append(
                    Finding(
                        code="classifier_ai",
                        signal=Signal.CLASSIFIER,
                        severity=_severity(classifier.p_ai, 0.5, 0.7, 0.9),
                        score=round(classifier.p_ai, 4),
                        message=(
                            f"Sınıflandırıcı bu cümleyi %{round(classifier.p_ai * 100)} "
                            "olasılıkla üretim olarak işaretliyor."
                        ),
                        evidence=classifier.model,
                    )
                )

            style_score, style_findings = self._style_term(style)
            terms[Signal.STYLOMETRY] = (style_score, scoring.weight_stylometry)
            findings.extend(style_findings)

            uniformity = self._uniformity(sentence.word_count, mean_length, std_length, length_cv)
            terms[Signal.UNIFORMITY] = (uniformity, scoring.weight_uniformity)
            if uniformity >= 0.75 and is_prose(sentence):
                findings.append(
                    Finding(
                        code="uniform_rhythm",
                        signal=Signal.UNIFORMITY,
                        severity=Severity.LOW,
                        score=round(uniformity, 4),
                        message=(
                            "Cümle uzunluğu belge ortalamasına çok yakın; metin ritmi tek düze."
                        ),
                        evidence=f"{sentence.word_count} kelime, ortalama {mean_length:.1f}",
                    )
                )

            ai_score = _weighted(terms)
            ai_score = self._apply_role_prior(ai_score, role, sentence)
            plagiarism_score = self._plagiarism.score(plag) if plag else 0.0
            if plag and plag.has_match:
                findings.extend(self._plagiarism_findings(plag))

            confidence = _confidence(terms, self._settings)
            if not scorable:
                confidence = min(confidence, 0.5)
            risk_score = _fuse(ai_score, plagiarism_score, scoring.plagiarism_weight)

            reports.append(
                SentenceReport(
                    index=sentence.index,
                    text=sentence.text,
                    location=sentence.location,
                    word_count=sentence.word_count,
                    style=style,
                    section_role=role,
                    perplexity=perplexity,
                    ratio=ratio,
                    classifier=classifier,
                    plagiarism=plag,
                    ai_score=round(ai_score, 4),
                    plagiarism_score=round(plagiarism_score, 4),
                    risk_score=round(risk_score, 4),
                    risk=risk_level_for(risk_score, scoring.risk_thresholds),
                    confidence=round(confidence, 4),
                    findings=tuple(findings),
                )
            )

        return self._aggregate(reports, perplexities, degraded)

    # ----------------------------------------------------------------- private
    def _aggregate(
        self,
        reports: Sequence[SentenceReport],
        perplexities: dict[int, PerplexityStats],
        degraded: Sequence[str],
    ) -> DocumentScores:
        total_words = sum(r.word_count for r in reports) or 1
        ai_score = sum(r.ai_score * r.word_count for r in reports) / total_words
        plagiarism_score = sum(r.plagiarism_score * r.word_count for r in reports) / total_words

        # The 20% floor, applied the way Turnitin applies it: the document gets
        # no AI verdict below it, however high the individual sentences scored.
        # Counted on ``ai_score`` alone - the fused risk also contains plagiarism,
        # and a document full of uncited matches must not inherit an AI share.
        ai_threshold = self._settings.scoring.risk_thresholds.get("medium", 0.55)
        flagged_words = sum(
            r.word_count for r in reports if r.ai_score >= ai_threshold and r.word_count
        )
        ai_share = flagged_words / total_words
        below_floor = ai_share < AI_REPORTING_FLOOR
        risk_score = _fuse(
            0.0 if below_floor else ai_score,
            plagiarism_score,
            self._settings.scoring.plagiarism_weight,
        )
        # Only sentences the models actually judged contribute to the
        # perplexity statistics; a heading must not drag the mean apart.
        ppl_values = [
            p.perplexity
            for p in perplexities.values()
            if math.isfinite(p.perplexity) and p.token_count >= 5
        ]

        return DocumentScores(
            sentences=tuple(reports),
            sections=self._sections(reports),
            ai_score=round(ai_score, 4),
            plagiarism_score=round(plagiarism_score, 4),
            risk_score=round(risk_score, 4),
            risk=risk_level_for(risk_score, self._settings.scoring.risk_thresholds),
            flagged=sum(1 for r in reports if r.risk.at_least(RiskLevel.MEDIUM)),
            high_risk=sum(1 for r in reports if r.risk.at_least(RiskLevel.HIGH)),
            matched=sum(1 for r in reports if r.plagiarism and r.plagiarism.has_match),
            mean_perplexity=(round(sum(ppl_values) / len(ppl_values), 3) if ppl_values else None),
            perplexity_burstiness=round(stdev(ppl_values), 3) if len(ppl_values) > 1 else None,
            length_burstiness=round(stdev([float(r.word_count) for r in reports]), 3),
            degraded=tuple(degraded),
            ai_share=round(ai_share, 4),
            ai_flagged_words=flagged_words,
            below_ai_reporting_floor=below_floor,
        )

    def _sections(self, reports: Sequence[SentenceReport]) -> tuple[SectionReport, ...]:
        buckets: dict[str, list[SentenceReport]] = {}
        depth_by_key: dict[str, int] = {}
        for report in reports:
            key = " › ".join(report.location.section_path) or "(giriş)"
            buckets.setdefault(key, []).append(report)
            depth_by_key[key] = len(report.location.section_path)
        sections: list[SectionReport] = []
        for key, items in buckets.items():
            words = sum(i.word_count for i in items) or 1
            sections.append(
                SectionReport(
                    title=key,
                    depth=depth_by_key.get(key, 0),
                    sentence_indices=tuple(i.index for i in items),
                    word_count=words,
                    ai_score=round(sum(i.ai_score * i.word_count for i in items) / words, 4),
                    plagiarism_score=round(
                        sum(i.plagiarism_score * i.word_count for i in items) / words, 4
                    ),
                    high_risk_count=sum(1 for i in items if i.risk.at_least(RiskLevel.HIGH)),
                )
            )
        return tuple(sections)

    def _perplexity_terms(
        self, perplexity: PerplexityStats | None, z_score: float | None
    ) -> tuple[float | None, list[Finding]]:
        if perplexity is None or not math.isfinite(perplexity.perplexity):
            return None, []
        scoring = self._settings.scoring
        center = self._settings.perplexity_center(self._language)
        scale = max(1e-3, self._settings.perplexity_scale(self._language))
        absolute = _sigmoid((center - perplexity.log10_perplexity) / scale)

        findings: list[Finding] = []
        if perplexity.token_count < 5:
            relative = 0.4
            findings.append(
                Finding(
                    code="perplexity_unreliable",
                    signal=Signal.PERPLEXITY,
                    severity=Severity.INFO,
                    score=round(absolute, 4),
                    message="Cümle çok kısa; perplexity ölçümü güvenilir değil.",
                    evidence=f"{perplexity.token_count} token",
                )
            )
        elif z_score is None:
            relative = absolute
        else:
            relative = _sigmoid(-(z_score + 0.5))

        term = (1 - scoring.ppl_relative_weight) * absolute + scoring.ppl_relative_weight * relative

        if absolute >= 0.75 or (z_score is not None and z_score <= -1.5):
            findings.append(
                Finding(
                    code="perplexity_low",
                    signal=Signal.PERPLEXITY,
                    severity=_severity(absolute, 0.6, 0.78, 0.9),
                    score=round(absolute, 4),
                    message=(
                        f"Perplexity düşük ({perplexity.perplexity:.1f}); model bu cümleyi "
                        "çok öngörülebilir buluyor."
                    ),
                    evidence=(
                        f"log10 ppl={perplexity.log10_perplexity:.2f}"
                        + (f", z={z_score:.2f}" if z_score is not None else "")
                        + f", {perplexity.token_count} token"
                    ),
                )
            )
        if perplexity.truncated:
            findings.append(
                Finding(
                    code="perplexity_truncated",
                    signal=Signal.PERPLEXITY,
                    severity=Severity.INFO,
                    score=0.0,
                    message="Cümle model penceresinden uzun; yalnızca ilk kısım ölçüldü.",
                    evidence=perplexity.model,
                )
            )
        return term, findings

    def _style_term(self, style: StyleStats) -> tuple[float, list[Finding]]:
        findings = [
            Finding(
                code=hit.code,
                signal=Signal.STYLOMETRY,
                severity=hit.severity,
                score=round(hit.weight, 4),
                message=hit.detail,
                evidence=hit.evidence,
            )
            for hit in style.hits
        ]
        return clamp01(style.total_weight * 1.7), findings

    def _apply_role_prior(self, score: float, role: SectionRole, sentence: Sentence) -> float:
        """Abstracts really are more formulaic than a methods section.

        The nudges are small and ordinal; they exist so that a *document* score is
        not dominated by the abstract, not to make any single sentence conclusive.
        """
        if role is SectionRole.OTHER or not is_prose(sentence):
            return score
        prior = role_prior(role) * self._settings.scoring.section_prior_strength
        return clamp01(score + prior)

    def _uniformity(
        self, words: int, mean_length: float, std_length: float, length_cv: float
    ) -> float:
        """How typical this sentence's length is, damped by document evenness."""
        if mean_length <= 0:
            return 0.0
        z = (words - mean_length) / std_length if std_length > 0 else 0.0
        closeness = math.exp(-0.5 * z * z)
        smoothness = 1.0 / (1.0 + max(0.0, length_cv))
        return round(0.6 * closeness + 0.4 * smoothness, 4)

    def _plagiarism_findings(self, stats: PlagiarismStats) -> list[Finding]:
        findings: list[Finding] = []
        for match in stats.matches[:3]:
            if match.kind is MatchKind.EXACT:
                message = (
                    f"{match.source_name} içinden {match.matched_words} kelime birebir "
                    "kopyalanmış görünüyor."
                )
            elif match.kind is MatchKind.TEMPLATE:
                message = (
                    f"{match.source_name} içinde aynı kalıp, rakamlar değiştirilerek "
                    f"({match.matched_words} kelime)."
                )
            elif match.kind is MatchKind.CODE:
                message = f"Kod bloğu {match.source_name} içinde aynı yapıda (%{round(match.ratio * 100)})."
            elif match.kind is MatchKind.MATH:
                message = f"Denklem {match.source_name} içinde birebir bulundu."
            else:
                message = (
                    f"{match.source_name} ile %{round(match.ratio * 100)} yakınlıkta, "
                    f"{match.matched_words} kelimelik ortak blok."
                )
            status = match.citation_status
            if status is not None:
                message = f"{message} Atıf durumu: {status.label}."
            findings.append(
                Finding(
                    code=f"match_{match.kind.value}",
                    signal=Signal.PLAGIARISM,
                    severity=_severity(match.confidence, 0.5, 0.75, 0.92),
                    score=round(match.confidence, 4),
                    message=message,
                    evidence=(
                        f"karakter {match.doc_char_start}-{match.doc_char_end} "
                        f"(kaynakta {match.source_char_start}-{match.source_char_end})"
                    ),
                )
            )
        return findings

    def _z_scores(self, perplexities: dict[int, PerplexityStats]) -> dict[int, float]:
        """Within-document position of each sentence's log-perplexity."""
        usable = {index: p for index, p in perplexities.items() if p.token_count >= 5}
        if len(usable) < 3:
            return {}
        values = [p.log10_perplexity for p in usable.values()]
        mean = sum(values) / len(values)
        std = stdev(values)
        if std <= 0:
            return {index: 0.0 for index in usable}
        return {index: (p.log10_perplexity - mean) / std for index, p in usable.items()}


# --------------------------------------------------------------------- helpers
def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def _weighted(terms: dict[Signal, tuple[float, float]]) -> float:
    total = sum(weight for _, weight in terms.values())
    if total <= 0:
        return 0.0
    return clamp01(sum(value * weight for value, weight in terms.values()) / total)


def _confidence(terms: dict[Signal, tuple[float, float]], settings: CheckerSettings) -> float:
    """Share of the configured evidence that actually ran."""
    scoring = settings.scoring
    configured = (
        scoring.weight_ratio
        + scoring.weight_perplexity
        + scoring.weight_classifier
        + scoring.weight_stylometry
        + scoring.weight_uniformity
    )
    if configured <= 0:
        return 1.0
    return clamp01(sum(weight for _, weight in terms.values()) / configured)


def _fuse(ai_score: float, plagiarism_score: float, plagiarism_weight: float) -> float:
    """Dominant signal wins, with a bonus when both fire."""
    plag = clamp01(plagiarism_score * plagiarism_weight)
    return clamp01(max(ai_score, plag) + 0.25 * min(ai_score, plag))


def _severity(value: float, low: float, medium: float, high: float) -> Severity:
    if value >= high:
        return Severity.HIGH
    if value >= medium:
        return Severity.MEDIUM
    if value >= low:
        return Severity.LOW
    return Severity.INFO


def _empty_style(sentence: Sentence) -> StyleStats:
    return StyleStats(word_count=sentence.word_count, char_count=len(sentence.text))


def is_prose(sentence: Sentence) -> bool:
    """Headings, code and math are reported but not judged for authorship."""
    return sentence.location.block_type not in {
        BlockType.HEADING,
        BlockType.CODE,
        BlockType.MATH,
        BlockType.TABLE,
    }
