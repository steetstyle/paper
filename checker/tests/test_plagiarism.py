"""Plagiarism: exact runs, templates, near matches, offsets and attribution."""

from __future__ import annotations

import pytest

from checker_app.config import PlagiarismSettings, SegmentationSettings
from checker_app.domain.enums import MatchKind
from checker_app.domain.models import SourceDocument
from checker_app.domain.segmentation import SentenceSplitter
from checker_app.domain.text import tokenize
from checker_app.services.plagiarism import PlagiarismService, content_bearing

SOURCE_TEXT = (
    "Öğrencilerin akademik yazım performansı ölçülmüştür. "
    "Veri temizliği gerçekleştirilmiş ve standart bir protokol kullanılmıştır. "
    "Deneyler üç ayrı koşul altında yürütülmüştür."
)
DOC_TEXT = (
    "Giriş bölümünde çalışmanın kapsamı tanımlanmıştır. "
    "Veri temizliği gerçekleştirilmiş ve standart bir protokol kullanılmıştır. "
    "Sonuç bölümünde bulgular tartışılmıştır."
)
VERBATIM = "Veri temizliği gerçekleştirilmiş ve standart bir protokol kullanılmıştır"


def make_source(text: str, name: str = "kaynak.md") -> SourceDocument:
    return SourceDocument(
        source_id="src-1",
        name=name,
        kind="file",
        text=text,
        tokens=tuple(tokenize(text)),
    )


def run(text: str = DOC_TEXT, **overrides):
    settings = PlagiarismSettings(**overrides)
    service = PlagiarismService(settings)
    result = SentenceSplitter(SegmentationSettings()).split(text)
    sources = [make_source(SOURCE_TEXT)]
    return service.compare(result.sentences, result.tokens, sources, text), result


def test_verbatim_run_is_found_with_offsets_on_both_sides() -> None:
    plagiarism, segmentation = run()
    assert plagiarism.matches, "expected at least one verbatim match"

    match = plagiarism.matches[0]
    assert match.kind is MatchKind.EXACT
    assert match.source_id == "src-1"
    assert match.matched_words >= 4

    # Offsets must point at the matched words in both texts.
    doc_slice = DOC_TEXT[match.doc_char_start : match.doc_char_end]
    src_slice = SOURCE_TEXT[match.source_char_start : match.source_char_end]
    assert doc_slice == src_slice == VERBATIM


def test_match_is_attributed_to_the_sentence_it_lives_in() -> None:
    plagiarism, segmentation = run()
    index = next(s.index for s in segmentation.sentences if "Veri temizliği" in s.text)
    stats = plagiarism.sentences[index]
    assert stats.has_match
    assert stats.longest_match_words >= 4
    assert stats.best_source == "kaynak.md"
    assert stats.containment > 0


def test_unrelated_text_has_no_matches() -> None:
    plagiarism, _ = run(
        "Farklı bir belge tamamen farklı içerik taşımaktadır. "
        "Hava durumu bugün oldukça yağmurlu görünüyor ve rüzgâr sert esiyor."
    )
    assert not [m for m in plagiarism.matches if m.kind is MatchKind.EXACT]


def test_template_match_ignores_changed_numbers() -> None:
    """Same wording, new numbers: the classic reused academic template."""
    source = (
        "Model 2020 yılında eğitildi ve 15.3 hata oranıyla değerlendirildi. "
        "Deneyler üç ayrı koşul altında yürütülmüştür."
    )
    document = (
        "Model 2024 yılında eğitildi ve 8.7 hata oranıyla değerlendirildi. "
        "Deneyler üç ayrı koşul altında yürütülmüştür."
    )
    settings = PlagiarismSettings(ngram_size=3, template_min_words=6)
    service = PlagiarismService(settings)
    segmentation = SentenceSplitter(SegmentationSettings()).split(document)
    sources = [make_source(source)]
    result = service.compare(segmentation.sentences, segmentation.tokens, sources, document)

    kinds = {m.kind for m in result.matches}
    assert MatchKind.TEMPLATE in kinds, "digit-folded template reuse must be caught"


def test_near_match_catches_a_reworded_passage() -> None:
    source = (
        "The cleaning of the recordings was completed before the model was trained, "
        "which changed the reported error rate substantially in the final experiments."
    )
    document = (
        "Intro burada. The recorded data were cleaned before the model was fitted, "
        "which changed the reported error rate substantially."
    )
    settings = PlagiarismSettings(near_match=True, near_match_min_words=8, near_match_ratio=0.6)
    service = PlagiarismService(settings)
    segmentation = SentenceSplitter(SegmentationSettings()).split(document)
    result = service.compare(
        segmentation.sentences, segmentation.tokens, [make_source(source)], document
    )
    assert any(m.kind is MatchKind.NEAR for m in result.matches)


def test_similarity_is_reported_per_source() -> None:
    plagiarism, _ = run()
    assert plagiarism.similarity[0].source_id == "src-1"
    assert plagiarism.similarity[0].shared_ngrams > 0
    assert 0.0 < plagiarism.similarity[0].ratio <= 1.0


def test_sentences_without_matches_get_empty_stats() -> None:
    plagiarism, segmentation = run()
    first = segmentation.sentences[0]
    stats = plagiarism.sentences[first.index]
    assert not stats.has_match
    assert stats.containment == 0.0


def test_no_sources_returns_empty_result() -> None:
    settings = PlagiarismSettings()
    service = PlagiarismService(settings)
    segmentation = SentenceSplitter(SegmentationSettings()).split(DOC_TEXT)
    result = service.compare(segmentation.sentences, segmentation.tokens, [], DOC_TEXT)
    assert result.matches == ()
    assert all(not s.has_match for s in result.sentences.values())


def test_credit_works_for_matches_from_other_layers() -> None:
    """Code/math matches are credited exactly like text matches."""
    from checker_app.domain.models import MatchSpan

    settings = PlagiarismSettings()
    service = PlagiarismService(settings)
    segmentation = SentenceSplitter(SegmentationSettings()).split(DOC_TEXT)
    sentence = segmentation.sentences[1]
    span = MatchSpan(
        kind=MatchKind.CODE,
        source_id="src-9",
        source_name="kod.py",
        sentence_indices=(),
        doc_char_start=sentence.location.char_start,
        doc_char_end=sentence.location.char_end,
        source_char_start=0,
        source_char_end=10,
        matched_words=8,
        ratio=1.0,
        snippet="def f(): ...",
    )
    stats = service.credit(segmentation.sentences, [span])[sentence.index]
    assert stats.code_matches == 1
    assert stats.has_match
    assert stats.containment == pytest.approx(1.0, abs=0.01)


def test_score_grows_with_coverage() -> None:
    from checker_app.domain.models import PlagiarismStats

    service = PlagiarismService(PlagiarismSettings())
    assert service.score(PlagiarismStats()) == 0.0
    weak = service.score(
        PlagiarismStats(
            containment=0.4, longest_match_words=4, matches=(_span(MatchKind.EXACT, 4, 0.2),)
        )
    )
    strong = service.score(
        PlagiarismStats(
            containment=0.9, longest_match_words=30, matches=(_span(MatchKind.EXACT, 30, 1.0),)
        )
    )
    assert 0.0 < weak < strong <= 1.0


def _span(
    kind: MatchKind,
    words: int,
    ratio: float = 1.0,
    doc_start: int = 0,
    doc_end: int = 10,
):
    from checker_app.domain.models import MatchSpan

    return MatchSpan(
        kind=kind,
        source_id="s",
        source_name="n",
        sentence_indices=(),
        doc_char_start=doc_start,
        doc_char_end=doc_end,
        source_char_start=0,
        source_char_end=10,
        matched_words=words,
        ratio=ratio,
        snippet="",
    )


def test_content_bearing_rejects_stopword_only_ngrams() -> None:
    """Diacritics are folded, so "için" has to be matched as "icin"."""
    assert content_bearing(("veri", "temizligi", "gerceklestirilmis", "protokol"))
    assert not content_bearing(("ve", "ile", "icin", "bir"))
    assert not content_bearing(("the", "and", "of", "to"))


def test_drop_shadowed_removes_matches_explained_by_code_or_math() -> None:
    """A copied formula yields a math match plus noisy n-gram matches."""
    from checker_app.services.plagiarism import drop_shadowed

    math_span = _span(MatchKind.MATH, words=12, doc_start=0, doc_end=120)
    shadow = _span(MatchKind.EXACT, words=8, doc_start=10, doc_end=110)
    unrelated = _span(MatchKind.EXACT, words=8, doc_start=500, doc_end=560)

    kept = drop_shadowed([math_span, shadow, unrelated])
    kinds = [m.kind for m in kept]
    assert MatchKind.MATH in kinds
    assert kinds.count(MatchKind.EXACT) == 1
    assert unrelated in kept
