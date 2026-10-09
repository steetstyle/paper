"""Scoring: fusion, confidence, sections and explanations."""

from __future__ import annotations

import math

import pytest

from checker_app.config import CheckerSettings
from checker_app.domain.enums import (
    BlockType,
    CitationStatus,
    MatchKind,
    RiskLevel,
    Severity,
    Signal,
)
from checker_app.domain.models import (
    ClassifierStats,
    MatchSpan,
    PerplexityStats,
    PlagiarismStats,
    RatioStats,
    Sentence,
    StyleHit,
    StyleStats,
    risk_level_for,
)
from checker_app.domain.sections import SectionRole
from checker_app.domain.segmentation import SentenceSplitter
from checker_app.services.scoring import Scorer

# Cümleler bilinçli olarak uzun: 12 kelimenin altındaki cümleler artık
# AI-ölçümüne girmiyor (arXiv:2509.18880: 10 token'da AUROC 0.62).
TEXT = (
    "Ayrıca, bu çalışmada kullanılan yöntemin bütün ayrıntıları oldukça kapsamlı bir şekilde "
    "sunulmakta ve bölümün devamında ayrıntılı olarak tartışılmaktadır. "
    "Veri temizliği gerçekleştirilmiş ve standart bir protokol bütün deneylerde "
    "değişmeden uygulanmıştır. "
    "Model eğitimi üç ayrı koşul altında titizlikle ve tekrarlanabilir biçimde "
    "yürütülmüştür. "
    "Elde edilen sonuçlar bölüm içinde kapsamlı biçimde ve detaylı olarak "
    "değerlendirilmiştir. "
    "Bu çalışmada yöntemin bütün ayrıntıları oldukça kapsamlı bir şekilde sunulmakta "
    "ve tartışılmaktadır. "
    "Veri temizliği gerçekleştirilmiş ve standart bir protokol bütün deneylerde "
    "kullanılmıştır. "
    "Model eğitimi üç ayrı koşul altında titizlikle ve tekrarlanabilir biçimde "
    "yürütülmüştür ve kaydedilmiştir."
)


def sentences(text: str = TEXT) -> list[Sentence]:
    return list(SentenceSplitter().split(text).sentences)


def perplexity(value: float, tokens: int = 20) -> PerplexityStats:
    return PerplexityStats(
        perplexity=value,
        mean_nll=math.log(value),
        token_count=tokens,
        log10_perplexity=math.log10(value),
    )


def span(kind: MatchKind = MatchKind.EXACT, words: int = 25, ratio: float = 1.0) -> MatchSpan:
    return MatchSpan(
        kind=kind,
        source_id="s",
        source_name="kaynak.md",
        sentence_indices=(0,),
        doc_char_start=0,
        doc_char_end=120,
        source_char_start=0,
        source_char_end=120,
        matched_words=words,
        ratio=ratio,
        snippet="metin",
    )


def _span(
    kind: MatchKind,
    words: int,
    ratio: float = 1.0,
    status: CitationStatus | None = None,
):
    from checker_app.domain.models import MatchSpan

    return MatchSpan(
        kind=kind,
        source_id="s",
        source_name="k.md",
        sentence_indices=(),
        doc_char_start=0,
        doc_char_end=10,
        source_char_start=0,
        source_char_end=10,
        matched_words=words,
        ratio=ratio,
        snippet="",
        citation_status=status,
    )


def style(*hits: StyleHit) -> StyleStats:
    return StyleStats(word_count=20, char_count=120, hits=hits)


HIT = StyleHit(
    code="discourse_marker",
    severity=Severity.MEDIUM,
    weight=0.16,
    detail="geçiş kalıbı",
    evidence="Ayrıca",
)


def test_risk_level_thresholds() -> None:
    assert risk_level_for(0.0) is RiskLevel.NONE
    assert risk_level_for(0.4) is RiskLevel.LOW
    assert risk_level_for(0.6) is RiskLevel.MEDIUM
    assert risk_level_for(0.8) is RiskLevel.HIGH
    assert risk_level_for(0.95) is RiskLevel.CRITICAL


def test_ai_score_grows_with_every_agreeing_signal() -> None:
    settings = CheckerSettings()
    scorer = Scorer(settings, "tr")
    rows = sentences()
    quiet = scorer.score(rows, {i: style() for i in range(len(rows))}, {}, {}, {})
    loud = scorer.score(
        rows,
        {i: style(HIT, HIT) for i in range(len(rows))},
        {i: perplexity(8.0) for i in range(len(rows))},
        {i: ClassifierStats(p_ai=0.95) for i in range(len(rows))},
        {},
    )
    assert loud.ai_score > quiet.ai_score
    assert all(0.0 <= s.ai_score <= 1.0 for s in loud.sentences)


def test_confidence_drops_when_signals_are_missing() -> None:
    settings = CheckerSettings()
    scorer = Scorer(settings, "tr")
    rows = sentences()
    ratios = {
        i: RatioStats(
            score=1.2,
            observer_perplexity=60.0,
            performer_perplexity=55.0,
            token_count=30,
            observer_model="a",
            performer_model="b",
        )
        for i in range(len(rows))
    }
    full = scorer.score(
        rows,
        {i: style() for i in range(len(rows))},
        {i: perplexity(30.0) for i in range(len(rows))},
        {i: ClassifierStats(p_ai=0.5) for i in range(len(rows))},
        {},
        ratios=ratios,
    )
    degraded = scorer.score(rows, {i: style() for i in range(len(rows))}, {}, {}, {})
    assert full.sentences[0].confidence == pytest.approx(1.0)
    assert degraded.sentences[0].confidence < full.sentences[0].confidence


def test_plagiarism_dominates_and_explains_itself() -> None:
    settings = CheckerSettings()
    scorer = Scorer(settings, "tr")
    rows = sentences()
    plag = {i: PlagiarismStats() for i in range(len(rows))}
    plag[0] = PlagiarismStats(
        containment=1.0,
        longest_match_words=40,
        best_source="kaynak.md",
        matches=(span(words=40),),
    )
    scores = scorer.score(rows, {i: style() for i in range(len(rows))}, {}, {}, plag)
    first = scores.sentences[0]

    assert first.plagiarism_score > 0.5
    assert first.risk.at_least(RiskLevel.MEDIUM)
    codes = {f.code for f in first.findings}
    assert "match_exact" in codes
    message = next(f.message for f in first.findings if f.code == "match_exact")
    assert "kaynak.md" in message and "birebir" in message


def test_relative_perplexity_catches_the_outlier_sentence() -> None:
    """One very predictable sentence among human ones: the z-score term fires."""
    settings = CheckerSettings()
    scorer = Scorer(settings, "tr")
    rows = sentences()
    values = {i: perplexity(60.0) for i in range(len(rows))}
    outlier = len(rows) - 1
    values[outlier] = perplexity(6.0)
    scores = scorer.score(rows, {i: style() for i in range(len(rows))}, values, {}, {})

    assert scores.sentences[outlier].perplexity is not None
    assert scores.sentences[outlier].perplexity.z_score is not None
    assert scores.sentences[outlier].perplexity.z_score < -1.0
    assert any(f.code == "perplexity_low" for f in scores.sentences[outlier].findings)


def test_short_sentences_are_not_scored_for_ai() -> None:
    """Evidence: AUROC 0.62 at 10 tokens, false-positive modal length 34 words."""
    scorer = Scorer(CheckerSettings(), "tr")
    rows = [s for s in sentences("Veri temizliği yapıldı ve incelendi.")]
    values = {rows[0].index: perplexity(30.0, tokens=8)}
    scores = scorer.score(rows, {rows[0].index: style()}, values, {}, {})

    sentence = scores.sentences[0]
    assert sentence.perplexity is None
    assert sentence.unscored_for_ai
    codes = {f.code for f in sentence.findings}
    assert "too_short_for_ai" in codes
    assert sentence.confidence <= 0.5


def test_likelihood_ratio_is_the_heaviest_ai_term() -> None:
    """Binoculars family: 0.0% FPR on edited non-native academic writing.

    The direction matters and is easy to get backwards: in this family a *low*
    observer/performer ratio means machine-like, because both models agree the
    text is predictable.
    """
    from dataclasses import replace as dc_replace

    from checker_app.domain.models import RatioStats

    settings = CheckerSettings()
    scorer = Scorer(settings, "tr")
    center = settings.ratio_threshold("tr")
    rows = sentences()

    machine_like = RatioStats(
        score=center - 0.15, observer_perplexity=12.0, performer_perplexity=14.0,
        token_count=30, observer_model="a", performer_model="b",
    )
    human_like = RatioStats(
        score=center + 0.25, observer_perplexity=90.0, performer_perplexity=70.0,
        token_count=30, observer_model="a", performer_model="b",
    )

    loud = scorer.score(rows, {i: style() for i in range(len(rows))}, {}, {}, {},
                        ratios={i: machine_like for i in range(len(rows))})
    calm = scorer.score(rows, {i: style() for i in range(len(rows))}, {}, {}, {},
                        ratios={i: human_like for i in range(len(rows))})
    none = scorer.score(rows, {i: style() for i in range(len(rows))}, {}, {}, {})

    assert loud.ai_score > calm.ai_score > none.ai_score
    assert "models_agree" in {f.code for f in loud.sentences[0].findings}
    assert "models_agree" not in {f.code for f in calm.sentences[0].findings}
    _ = dc_replace


def test_section_role_prior_moves_the_abstract_up() -> None:
    """Abstracts are generated last and read formulaic; methods are not."""
    body = (
        "Bu çalışmada konu şu şekilde ele alınmış ve ayrıntılı olarak incelenmiştir. "
        "Yöntem üç aşamadan oluşmakta ve her aşama titizlikle uygulanmaktadır."
    )
    rows = sentences(f"# Özet\n\n{body}\n\n# Yöntem\n\n{body}")
    scorer = Scorer(CheckerSettings(), "tr")
    scores = scorer.score(rows, {r.index: style() for r in rows}, {}, {}, {})
    prose = [s for s in scores.sentences if s.location.block_type is BlockType.PROSE]
    abstract = next(s for s in prose if s.section_role is SectionRole.ABSTRACT)
    method = next(s for s in prose if s.section_role is SectionRole.METHOD)
    assert abstract.section_role is SectionRole.ABSTRACT
    assert method.section_role is SectionRole.METHOD
    # A fixed 0.10 / -0.02 prior on identical (absent) evidence.
    assert abstract.ai_score > method.ai_score


def test_cited_and_quoted_match_is_almost_not_penalised() -> None:
    """Only 6.4% of matched pairs carry a citation (arXiv:2609.32963)."""
    scorer = Scorer(CheckerSettings(), "tr")
    rows = sentences()
    cited = _span(kind=MatchKind.EXACT, words=40, status=CitationStatus.CITED_AND_QUOTED)
    uncited = _span(kind=MatchKind.EXACT, words=40, status=CitationStatus.NOT_CITED_OR_QUOTED)
    quoted_plag = {i: PlagiarismStats() for i in range(len(rows))}
    clean_plag = {i: PlagiarismStats() for i in range(len(rows))}
    quoted_plag[rows[0].index] = PlagiarismStats(
        containment=1.0, longest_match_words=40, best_source="k.md", matches=(cited,)
    )
    clean_plag[rows[0].index] = PlagiarismStats(
        containment=1.0, longest_match_words=40, best_source="k.md", matches=(uncited,)
    )

    first = scorer.score(rows, {i: style() for i in range(len(rows))}, {}, {}, quoted_plag)
    second = scorer.score(rows, {i: style() for i in range(len(rows))}, {}, {}, clean_plag)
    assert first.plagiarism_score < second.plagiarism_score
    assert first.sentences[0].plagiarism_score < 0.2


def test_sections_aggregate_by_heading() -> None:
    text = "# Giriş\n\nBirinci cümle burada bulunmaktadır.\n\n## Yöntem\n\nİkinci cümle burada bulunmaktadır."
    scorer = Scorer(CheckerSettings(), "tr")
    rows = sentences(text)
    scores = scorer.score(rows, {i: style() for i in range(len(rows))}, {}, {}, {})
    titles = [s.title for s in scores.sections]
    assert "Giriş" in titles
    assert "Giriş › Yöntem" in titles


def test_document_aggregates_are_word_weighted() -> None:
    scorer = Scorer(CheckerSettings(), "tr")
    rows = sentences()
    styles = {i: style() for i in range(len(rows))}
    scores = scorer.score(rows, styles, {}, {}, {})
    assert scores.ai_score >= 0.0
    assert scores.length_burstiness is not None
    assert scores.mean_perplexity is None


def test_findings_carry_the_signal_they_came_from() -> None:
    scorer = Scorer(CheckerSettings(), "tr")
    rows = sentences()
    plag = {i: PlagiarismStats() for i in range(len(rows))}
    plag[0] = PlagiarismStats(containment=0.9, longest_match_words=30, matches=(span(),))
    scores = scorer.score(rows, {i: style(HIT) for i in range(len(rows))}, {}, {}, plag)
    signals = {f.signal for f in scores.sentences[0].findings}
    assert Signal.PLAGIARISM in signals
    assert Signal.STYLOMETRY in signals


def test_non_prose_blocks_are_not_flagged_for_uniformity() -> None:
    text = "# Başlık\n\nGövde cümlesi burada bulunmaktadır ve oldukça uzundur."
    rows = sentences(text)
    heading = rows[0]
    assert heading.location.block_type is BlockType.HEADING
