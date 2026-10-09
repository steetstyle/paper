"""Cross-language coverage: the ordinary case this tool would otherwise hide.

A Turkish thesis citing English literature is not an edge case, it is the normal
one. And reuse detection across that boundary is measurably weaker - precision
**80%** on untranslated text, **26.7%** on translated text, **16.7%** on
translated-then-paraphrased text (DOI 10.33806/ijaes1026). So a low similarity
percentage in that setting is partly a statement about the corpus supplied, not
about the thesis, and the report has to say which case you are in.
"""

from __future__ import annotations

import pytest

from checker_app.config import get_settings
from checker_app.domain.models import SourceDocument
from checker_app.services.similarity import (
    LanguageCoverage,
    build_language_coverage,
)

TR = "Bu çalışmada öğrencilerin akademik yazım performansı ölçülmüştür ve sonuçlar kaydedilmiştir."
EN = (
    "Data cleaning was performed using a standard protocol and all results were "
    "recorded before analysis began in the educational data mining literature."
)


def _document(tmp_path, name: str, text: str) -> str:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


def _sources(tmp_path):
    """A Turkish thesis plus an English and a Turkish reference."""
    loader = _loader()
    thesis = loader.load_document(_document(tmp_path, "tez.md", TR))
    english = loader.load_document(_document(tmp_path, "en.md", EN))
    turkish = loader.load_document(_document(tmp_path, "tr.md", TR))
    return thesis, loader.to_sources([english, turkish])


def _loader():
    from checker_app.services.sources import SourceLoader  # noqa: PLC0415

    return SourceLoader(get_settings())


# --------------------------------------------------------------- detection
def test_each_source_language_is_detected_its_own(tmp_path) -> None:
    """Language is per source, not per scan. That is the whole point."""
    _thesis, sources = _sources(tmp_path)
    by_name = {s.name: s for s in sources}
    assert by_name["en.md"].language == "en"
    assert by_name["tr.md"].language == "tr"
    assert by_name["en.md"].language_confidence > 0.0


def test_a_source_document_without_language_is_unknown() -> None:
    source = SourceDocument(source_id="x", name="x", kind="file", text="", tokens=())
    assert source.language == "unknown"


# ----------------------------------------------------------------- coverage
def test_all_cross_language_sources_are_counted_even_with_zero_matches() -> None:
    """A zero result must still say *why*: all references were English.

    The earlier version derived the counts from matched tokens, so a document
    with no matches reported "language not determined" and hid the single most
    useful thing a reader could learn.
    """
    coverage = build_language_coverage(
        document_language="tr",
        source_languages={"en1": "en", "en2": "en", "en3": "en"},
        per_source_covered={},
    )
    assert coverage.cross_language_sources == 3
    assert coverage.same_language_sources == 0
    assert coverage.cross_language_words == 0
    reading = coverage.reading()
    assert "FARKLI" in reading
    assert "kanıt DEĞİLDİR" in reading


def test_cross_language_reading_points_at_the_cheapest_next_step() -> None:
    """Adding one same-language reference is the cheapest way to test a zero."""
    coverage = build_language_coverage(
        document_language="tr",
        source_languages={"en1": "en"},
        per_source_covered={},
    )
    assert "en ucuz adımdır" in coverage.reading()


def test_same_language_only_reading() -> None:
    coverage = build_language_coverage(
        document_language="tr",
        source_languages={"tr1": "tr"},
        per_source_covered={"tr1": {1, 2, 3}},
    )
    assert coverage.same_language_sources == 1
    assert coverage.same_language_words == 3
    assert "belge diliyle aynı" in coverage.reading()


def test_mixed_sources_split_the_words() -> None:
    coverage = build_language_coverage(
        document_language="tr",
        source_languages={"tr1": "tr", "en1": "en"},
        per_source_covered={"tr1": {1, 2, 3, 4}, "en1": {5, 6}},
    )
    assert coverage.same_language_words == 4
    assert coverage.cross_language_words == 2
    assert coverage.cross_language_ratio == pytest.approx(2 / 6)
    assert "Farklı" in coverage.reading() or "farklı" in coverage.reading()


def test_unknown_sources_are_counted_separately_not_assumed_cross_language() -> None:
    """An undetectable source is not evidence of a cross-language match."""
    coverage = build_language_coverage(
        document_language="tr",
        source_languages={"u1": "unknown"},
        per_source_covered={"u1": {1, 2}},
    )
    assert coverage.unknown_language_sources == 1
    assert coverage.cross_language_sources == 0
    assert coverage.cross_language_words == 0


def test_word_counts_come_from_the_percentage_not_the_match_list() -> None:
    """Only words that survived the institutional filters may be counted.

    Built from ``per_source_covered``, so a match dropped by the quote filter
    cannot inflate this and quietly change the reading.
    """
    coverage = build_language_coverage(
        document_language="tr",
        source_languages={"tr1": "tr"},
        per_source_covered={"tr1": {1, 2}},
    )
    assert coverage.same_language_words == 2


def test_source_and_word_counts_are_tracked_separately() -> None:
    coverage = build_language_coverage(
        document_language="tr",
        source_languages={"en1": "en", "en2": "en", "tr1": "tr"},
        per_source_covered={"en1": {1, 2, 3}, "tr1": {4}},
    )
    payload = coverage.to_dict()
    assert payload["sources_by_language"] == {"en": 2, "tr": 1}
    assert payload["words_by_language"] == {"en": 3, "tr": 1}


def test_no_sources_reads_as_a_fact_not_an_error() -> None:
    coverage = LanguageCoverage()
    assert coverage.reading() == "Kaynak verilmedi."
    assert coverage.to_dict()["cross_language_ratio"] == 0.0


def test_the_presented_rule_is_available_in_both_languages() -> None:
    """Presentative passive is register, not authorship, in English too.

    English academic prose leans on it harder than Turkish does, so leaving the
    rule Turkish-only would have made an English scan score systematically lower
    for the same sentence - a language bias running opposite to the measured
    Turkish one.
    """
    from checker_app.services.lexicon import build_rules  # noqa: PLC0415

    turkish = build_rules(("tr",))
    english = build_rules(("en",))
    assert {r.code for r in turkish} <= {r.code for r in english}, (
        "Türkçedeki her kuralın İngilizce karşılığı olmalı"
    )

    rule = next(r for r in english if r.code == "presentative_passive")
    assert rule.matches("The study was conducted with 240 undergraduates.")
    assert rule.matches("The data were analysed using a standard protocol.")
    assert rule.matches("The aim of this study is to examine revision behaviour.")
    assert not rule.matches("She collected the interviews herself last spring.")

    turkish_rule = next(r for r in turkish if r.code == "presentative_passive")
    assert turkish_rule.matches("Bu çalışmada veriler toplanmıştır.")
    assert turkish_rule.matches("Bu tezde amaç incelemektir.")


def test_english_has_two_measured_lexical_signals_that_turkish_lacks() -> None:
    """The asymmetry is deliberate and is a statement about the evidence.

    Kobak vd. measured an excess-vocabulary set on >15M PubMed abstracts, and
    Thelwall & Kousha measured that LLMs *avoid* ``thus``/``moreover``. There is
    no published Turkish equivalent for either, so Turkish mode carries no such
    rule rather than an invented one.
    """
    from checker_app.services.lexicon import build_rules  # noqa: PLC0415

    turkish_codes = {r.code for r in build_rules(("tr",))}
    english_codes = {r.code for r in build_rules(("en",))}
    assert "llm_vocabulary" in english_codes
    assert "llm_avoided_connective" in english_codes
    assert "llm_vocabulary" not in turkish_codes
    assert "llm_avoided_connective" not in turkish_codes


def test_the_llm_vocabulary_rule_matches_inflections_and_respects_boundaries() -> None:
    """Prose contains ``delves`` and ``underscores``, not bare ``delve``."""
    from checker_app.services.lexicon import build_rules  # noqa: PLC0415

    rule = next(r for r in build_rules(("en",)) if r.code == "llm_vocabulary")
    marked = "pivotal"
    rule = next(
        r for r in build_rules(("en",)) if r.code == "llm_vocabulary" and r.matches(marked)
    )
    assert rule.matches("The study delves into a pivotal realm.")
    assert rule.matches("This work underscores a comprehensive review.")
    assert rule.matches("We provide additional insights across three domains.")
    # Inflected forms must match: this is the form that appears in prose.
    assert rule.matches("The authors delve deeply and underscore the limits.")
    # A word that merely starts the same way must not match.
    assert not rule.matches("The understated results were collected in a classroom.")
    assert not rule.matches("Sonuçlar sınıfta toplandı ve kaydedildi.")


def test_the_avoided_connective_rule_fires_on_what_llms_skip() -> None:
    from checker_app.services.lexicon import build_rules  # noqa: PLC0415

    rule = next(
        r for r in build_rules(("en",)) if r.code == "llm_avoided_connective"
    )
    assert rule.matches("Thus, the effect persisted. Moreover, it generalised.")
    assert not rule.matches("The effect persisted and generalised across sites.")


def test_the_english_rule_does_not_fire_on_turkish_or_ordinary_prose() -> None:
    """A rule that fires on everything measures nothing."""
    from checker_app.services.lexicon import build_rules  # noqa: PLC0415

    rule = next(r for r in build_rules(("en",)) if r.code == "presentative_passive")
    assert not rule.matches("Veriler iki koşul altında toplanmıştır ve kaydedilmiştir.")
    assert not rule.matches("We went to the shop and bought some bread for dinner.")


def test_the_precision_collapse_is_carried_in_the_payload() -> None:
    """80% untranslated -> 26.7% translated -> 16.7% translated-then-paraphrased."""
    evidence = LanguageCoverage().to_dict()["evidence"]
    assert "%80" in evidence
    assert "%26.7" in evidence
    assert "%16.7" in evidence
    assert "10.33806/ijaes1026" in evidence