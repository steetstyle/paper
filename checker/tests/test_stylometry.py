"""Stylometry: the rules must fire on model phrasing and stay quiet otherwise."""

from __future__ import annotations

from checker_app.domain.enums import Severity
from checker_app.domain.segmentation import SentenceSplitter
from checker_app.services.stylometry import StylometryService

SPLITTER = SentenceSplitter()


def analyze(text: str) -> object:
    result = SPLITTER.split(text)
    sentence = result.sentences[0]
    return StylometryService().analyze(
        sentence, result.tokens[sentence.token_start : sentence.token_end]
    )


def test_turkish_discourse_marker() -> None:
    stats = analyze(
        "Ayrıca, bu bulgu sonraki bölümde ayrıntılı olarak ele alınmaktadır ve oldukça önemlidir."
    )
    assert "discourse_marker" in stats.hit_codes


def test_english_hedge_and_favourite_noun() -> None:
    stats = analyze(
        "It is important to note that the framework may effectively leverage a "
        "comprehensive landscape of signals."
    )
    codes = stats.hit_codes
    assert "discourse_marker" in codes
    assert "hedge_stack" in codes
    assert "favourite_noun" in codes


def test_presentative_passive_only_for_turkish() -> None:
    stats = analyze("Bu çalışmada yöntemin tüm ayrıntıları sunulmaktadır.")
    assert "presentative_passive" in stats.hit_codes


def test_human_marker_pulls_the_score_down() -> None:
    human = analyze("Bence asıl sorun veri temizliğinde ve bunu yarın çözeceğim sanırım.")
    assert "human_marker" in human.hit_codes
    assert human.total_weight < 0


def test_typography_hits() -> None:
    stats = analyze("Sonuç — **çok** önemli — burada görülüyor ve “vurgulanmalıdır”.")
    codes = stats.hit_codes
    assert "em_dash" in codes
    assert "markdown_bold" in codes
    assert "typographic_quotes" in codes


def test_plain_neutral_sentence_has_no_hits() -> None:
    stats = analyze("Veri temizliği geçen hafta tamamlandı.")
    assert stats.hit_codes == ()
    assert stats.total_weight == 0.0


def test_self_repetition_is_detected() -> None:
    stats = analyze(
        "Veri temizliği veri temizliği veri temizliği kuralları uygulanmış ve "
        "veri temizliği kaydı tekrar edilmiştir."
    )
    assert "self_repetition" in stats.hit_codes


def test_code_blocks_are_excluded_from_typography() -> None:
    text = "```python\n# yorum — kalın **değil**\nx = 1\n```"
    result = SPLITTER.split(text)
    sentence = result.sentences[0]
    stats = StylometryService().analyze(sentence, ())
    assert stats.hit_codes == ()


def test_severity_of_negative_hits_is_info() -> None:
    stats = analyze("Ben bu yaklaşımı hâlâ beğenmiyorum ama ilginç.")
    hit = next(h for h in stats.hits if h.code == "human_marker")
    assert hit.severity is Severity.INFO
    assert hit.weight < 0


def test_document_stats_summarise_lengths() -> None:
    text = (
        "Birinci cümle burada ve oldukça uzundur. İkinci cümle kısadır. "
        "Üçüncü cümle orta uzunluktadır."
    )
    result = SPLITTER.split(text)
    stats = StylometryService().document_stats(result.sentences)
    assert stats.sentence_count == 3
    assert stats.mean_length > 0
    assert stats.std_length > 0
