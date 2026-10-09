"""Lexical richness, length confounding, and probability calibration.

Each block pins a number from the literature that justifies the code's shape, so
the reasoning cannot be quietly edited away:

* lexical richness is the strongest feature area (dropping it costs 13.1 F1
  in-domain, 27.7 out-of-domain) but its **sign is not fixed** - light AI
  editing raises it above the human value while generation lowers it;
* word count alone reaches ROC-AUC 0.68, so a short document is not neutral;
* calibration is separable from accuracy: temperature scaling changes ECE
  0.06 -> 0.12 while F1 moves 80.17 -> 80.16.
"""

from __future__ import annotations

import pytest

from checker_app.config import get_settings
from checker_app.domain.segmentation import SentenceSplitter
from checker_app.services.calibration import CalibrationResult
from checker_app.services.reliability import (
    CALIBRATION_EVIDENCE,
    MIN_SAMPLES,
    brier_score,
    expected_calibration_error,
    fit_temperature,
    reliability_report,
)
from checker_app.services.stylometry import (
    _FUNCTION_WORDS,
    LEXICAL_RICHNESS_DIRECTION,
    StylometryService,
)


def _split(text: str):
    return SentenceSplitter(get_settings().segmentation).split(text)


def _rich_document(sentences: int = 40) -> str:
    """Prose long enough for document-level ratios to mean something.

    Vocabulary is drawn from a wide synthetic pool rather than repeated: a
    fixture built from a handful of recycled sentences has almost no hapax
    words, which would make every richness test degenerate.
    """
    subjects = [
        "Katılımcılar", "Örneklem", "Model", "Değerlendirici", "Kurul", "Ölçüm",
        "Görüşme", "Yakınlık", "Kayıp", "Analiz", "Deney", "Tutarlılık",
        "Doğrulama", "Sözlük", "Betik", "Atama", "Aralık", "Değişken", "Bulgu",
    ]
    verbs = [
        "onayladı", "doğruladı", "kaydetti", "raporladı", "denetledi",
        "hesapladı", "karşılaştırdı", "sınıflandırdı", "ölçtü", "seçti",
    ]
    objects = [
        "formu", "protokolü", "eğriyi", "tablosunu", "sözlüğü", "betiği",
        "katmanları", "katsayısı", "aralığı", "varsayımı", "kaydı", "metni",
    ]
    tails = [
        "ve sonuçlar kayda geçirildi",
        "ancak sınırlılıklar tartışmada belirtildi",
        "dolayısıyla yorum temkinli tutuldu",
        "ve bu bulgu genel bulgularla birlikte sunuldu",
        "çünkü prosedür önceden kayıt altına alınmıştı",
        "böylece tekrarlanabilirlik sağlanmış oldu",
        "ve karşılaştırma grubu eşit büyüklükte tutuldu",
        "sonuçlar ilgili bölümde ayrıntılı biçimde aktarıldı",
    ]
    body_parts = []
    for i in range(sentences):
        body_parts.append(
            f"{subjects[i % len(subjects)]} "
            f"{verbs[(i * 3) % len(verbs)]} "
            f"{objects[(i * 5) % len(objects)]} "
            f"{tails[(i * 7) % len(tails)]}, "
            f"{tails[(i * 5 + 3) % len(tails)]} ve "
            f"{objects[(i * 11 + 1) % len(objects)]} "
            f"{verbs[(i * 3 + 4) % len(verbs)]}."
        )
    return "# Giriş\n\n" + "\n\n".join(body_parts)


# ------------------------------------------------------------ lexical richness
def test_lexical_richness_is_not_measured_below_two_hundred_tokens() -> None:
    """A two-sentence fixture has no trustworthy type/token ratio."""
    report = _split(_rich_document(1))
    stats = StylometryService(get_settings().stylometry).document_stats(
        report.sentences, report.tokens
    )
    assert stats.lexical_richness is None


def test_lexical_richness_is_computed_on_a_real_document() -> None:
    report = _split(_rich_document(40))
    stats = StylometryService(get_settings().stylometry).document_stats(
        report.sentences, report.tokens
    )
    richness = stats.lexical_richness
    assert richness is not None
    assert richness.word_count > 200
    assert 0.0 < richness.type_token_ratio <= 1.0
    assert 0.0 < richness.hapax_ratio <= 1.0
    assert 0.0 < richness.lexical_density <= 1.0
    # Structural identity, not an empirical claim: the number of types seen
    # exactly once can never exceed the number of distinct types.
    assert richness.hapax_ratio <= richness.type_token_ratio + 1e-9


def test_repeated_text_lowers_hapax_ratio() -> None:
    """Hapax ratio is the fraction of *types* seen exactly once.

    A document that repeats itself must score lower, which is the whole reason
    the measure is in the trio: it penalises vocabulary recycling that TTR alone
    would flatter.
    """
    service = StylometryService(get_settings().stylometry)
    varied = _split(_rich_document(40))
    repetitive = _split(
        "# Giriş\n\n" + ("Katılımcılar gönüllü olarak katıldı ve onam formu imzaladı. " * 90)
    )
    rich = service.document_stats(varied.sentences, varied.tokens).lexical_richness
    flat = service.document_stats(repetitive.sentences, repetitive.tokens).lexical_richness
    assert rich is not None and flat is not None
    assert flat.hapax_ratio < rich.hapax_ratio


def test_the_trio_direction_is_unstable_and_says_so() -> None:
    """Light AI editing raises lexical richness *above* the human value.

    Measured over 64,304 documents (16,076 human articles x 4 LLM roles), Cheng
    vd., arXiv:2410.14259. This is why the tool reports the profile as context
    and never as a tell: one threshold cannot separate creator from polisher,
    because they push in opposite directions.
    """
    human, human_sd = LEXICAL_RICHNESS_DIRECTION["human_author"]
    polisher, polisher_sd = LEXICAL_RICHNESS_DIRECTION["llm_polisher"]
    creator, _ = LEXICAL_RICHNESS_DIRECTION["llm_creator"]
    assert human == pytest.approx(0.59)
    assert polisher > human, "cila insan değerinin üstüne çıkmalı"
    assert creator < human, "üretim insan değerinin altına inmeli"
    # The ordering is the whole point: polisher > human > creator.
    assert polisher > human > creator
    assert human_sd and polisher_sd


def test_closest_role_is_labelled_as_coarse() -> None:
    report = _split(_rich_document(40))
    stats = StylometryService(get_settings().stylometry).document_stats(
        report.sentences, report.tokens
    )
    payload = stats.lexical_richness.to_dict()
    assert payload["closest_measured_role"] in LEXICAL_RICHNESS_DIRECTION
    assert "bağlam olarak raporlanır" in payload["reading"]
    assert "arXiv:2410.14259" in payload["source"]
    assert "arXiv:2606.04177" in payload["source"]


def test_lexical_density_uses_a_closed_class_list() -> None:
    """The feature must measure content words, not the size of the stop list."""
    assert "ve" in _FUNCTION_WORDS and "the" in _FUNCTION_WORDS
    # Turkish enclitics stay function words, otherwise density rewards a suffix.
    assert "da" in _FUNCTION_WORDS and "mi" in _FUNCTION_WORDS
    assert "model" not in _FUNCTION_WORDS


def test_sentence_stats_carry_the_trio() -> None:
    report = _split(_rich_document(12))
    service = StylometryService(get_settings().stylometry)
    sentence = next(s for s in report.sentences if s.word_count > 10)
    stats = service.analyze(sentence, report.tokens)
    assert stats.hapax_ratio > 0
    assert stats.lexical_density > 0
    assert stats.hapax_ratio <= 1.0


# ------------------------------------------------------------ length confound
def test_word_count_alone_reaches_auroc_zero_point_seven() -> None:
    """Kumar vd., arXiv:2609.26687: CV ROC-AUC 0.68 from word count alone."""
    stats = StylometryService(get_settings().stylometry).document_stats(
        _split(_rich_document(40)).sentences,
        _split(_rich_document(40)).tokens,
    )
    confound = stats.length_confound
    assert confound is not None
    payload = confound.to_dict()
    assert payload["measured"]["word_count_only_auroc"] == 0.68
    assert "t(89)=6.46" in payload["measured"]["paired_t"]
    assert payload["measured"]["longer_in_pairs"] == "72/90"


def test_a_short_document_is_declared_unstable() -> None:
    stats = StylometryService(get_settings().stylometry).document_stats(
        _split(_rich_document(1)).sentences, _split(_rich_document(1)).tokens
    )
    assert stats.length_confound is not None
    assert stats.length_confound.regime == "cok_kisa"
    assert "0.68" in stats.length_confound.note


def test_a_long_document_is_the_safe_direction() -> None:
    stats = StylometryService(get_settings().stylometry).document_stats(
        _split(_rich_document(120)).sentences, _split(_rich_document(120)).tokens
    )
    assert stats.length_confound is not None
    assert stats.length_confound.regime == "uzun"
    assert "arXiv:2402.10586" in stats.length_confound.note


def test_sentence_length_is_excluded_on_purpose() -> None:
    """Sentence-length variability alone performs near chance (arXiv:2609.26687).

    Mean sentence length did not differ between conditions (27.2 vs 27.3,
    p = .97), so it is reported but never weighted.
    """
    stats = StylometryService(get_settings().stylometry).document_stats(
        _split(_rich_document(40)).sentences, _split(_rich_document(40)).tokens
    )
    excluded = stats.length_confound.to_dict()["excluded"]
    assert "p=.97" in excluded
    assert "ağırlıklandırılmaz" in excluded


# ------------------------------------------------------------------ calibration
def test_a_well_calibrated_score_gets_a_low_ece() -> None:
    pairs = [(0.02, 0.0)] * 25 + [(0.98, 1.0)] * 25
    report = reliability_report(pairs)
    assert report.measured is True
    assert report.ece == pytest.approx(0.02, abs=0.01)
    assert report.brier == pytest.approx(0.0004, abs=0.001)
    assert "iyi kalibre" in report.interpretation


def test_overconfidence_is_measured_and_named() -> None:
    """Scores that say 0.95 and are wrong two times in five must be caught."""
    pairs = [(0.05, 0.0)] * 30 + [(0.95, 1.0)] * 18 + [(0.95, 0.0)] * 12
    report = reliability_report(pairs)
    assert report.ece == pytest.approx(0.20, abs=0.01)
    assert "kalibre değil" in report.interpretation
    assert report.temperature is not None and report.temperature > 1.5
    assert "keskin" in report.interpretation


def test_temperature_corrects_overconfidence() -> None:
    pairs = [(0.05, 0.0)] * 30 + [(0.95, 1.0)] * 18 + [(0.95, 0.0)] * 12
    temperature = fit_temperature(pairs)
    assert temperature is not None and temperature > 1.0
    from checker_app.services.reliability import _apply  # noqa: PLC0415

    corrected = [_apply(0.95, temperature) for _ in range(18)]
    assert max(corrected) < 0.95, "ısıtılmış puan orijinalinden keskin olmamalı"


def test_too_few_samples_produce_no_number() -> None:
    """Eight sentences cannot establish calibration, so none is reported."""
    report = reliability_report([(0.5, 0.0), (0.5, 1.0)] * 4)
    assert report.measured is False
    assert report.ece is None and report.brier is None
    assert report.temperature is None
    assert str(MIN_SAMPLES) in report.interpretation
    assert expected_calibration_error([(0.5, 0.0)]) is None
    assert brier_score([(0.5, 0.0)]) is None
    assert fit_temperature([(0.5, 0.0)]) is None


def test_calibration_stays_separable_from_accuracy() -> None:
    """F1 80.17 -> 80.16 while ECE 0.06 -> 0.12 (arXiv:2510.00890)."""
    assert "80.17" in CALIBRATION_EVIDENCE and "80.16" in CALIBRATION_EVIDENCE
    assert "0.06" in CALIBRATION_EVIDENCE and "0.12" in CALIBRATION_EVIDENCE
    assert "0.8" in CALIBRATION_EVIDENCE, "AITDNA eşik salınımı eksik"
    assert "arXiv:2606.04906" in CALIBRATION_EVIDENCE
    # And the literature publishes no agreement ceiling to calibrate against.
    assert "kappa" in CALIBRATION_EVIDENCE


def test_the_report_carries_its_own_limits() -> None:
    payload = reliability_report([(0.02, 0.0)] * 20 + [(0.98, 1.0)] * 20).to_dict()
    caveats = " ".join(payload["caveats"])
    assert "suistimal kanıtı değildir" in caveats
    assert "TR işletim noktası" in caveats, "TR işletim noktasının kaynağı açıklanmalı"
    # The operative claim is *why this tool measures its own point*, not that no
    # Turkish number exists: published ones are given, with their best result
    # and the register it was measured on.
    assert "AUROC %99.31" in caveats
    assert "başka bir çalışmadan alınmaz" in caveats


def test_calibration_result_carries_a_reliability_block() -> None:
    result = CalibrationResult()
    assert result.reliability is None
    assert result.to_dict()["reliability"] is None
    result.reliability = reliability_report([(0.02, 0.0)] * 20 + [(0.98, 1.0)] * 20)
    payload = result.to_dict()
    assert payload["reliability"]["measured"] is True
    assert payload["reliability"]["evidence"]