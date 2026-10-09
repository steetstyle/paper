"""Institutional similarity metrics, integrity diagnostics and compliance."""

from __future__ import annotations

import pytest

from checker_app.domain.enums import CitationStatus, MatchKind
from checker_app.domain.models import MatchSpan
from checker_app.domain.text import tokenize
from checker_app.services.attribution import Span, find_citation_spans, find_quotation_spans
from checker_app.services.compliance import find_disclosure
from checker_app.services.integrity import check_integrity, folding_cost
from checker_app.services.similarity import (
    INSTITUTION_BANDS,
    TurkishBaseline,
    build_similarity_report,
)

SOURCE = (
    "Veri temizliği gerçekleştirilmiş ve standart bir protokol kullanılmıştır. "
    "Model eğitimi üç ayrı koşul altında titizlikle yürütülmüştür."
)
DOC = (
    "Bu tezde veri temizliği gerçekleştirilmiş ve standart bir protokol kullanılmıştır. "
    "Özgün bir başka cümle burada yer almaktadır ve kaynakla ilgisi yoktur. "
    "Model eğitimi üç ayrı koşul altında titizlikle yürütülmüştür."
)


def span(
    start: int, end: int, words: int = 10, source_id: str = "s1", name: str = "kaynak.md"
) -> MatchSpan:
    return MatchSpan(
        kind=MatchKind.EXACT,
        source_id=source_id,
        source_name=name,
        sentence_indices=(),
        doc_char_start=start,
        doc_char_end=end,
        source_char_start=0,
        source_char_end=end,
        matched_words=words,
        ratio=1.0,
        snippet="",
    )


def build(
    *,
    matches,
    quote_spans=(),
    statuses=None,
    reference_indices=frozenset(),
    total_words: int | None = None,
    min_match_words: int = 5,
    exclude_references: bool = True,
    include_quotes: bool = True,
):
    tokens = tokenize(DOC)
    return build_similarity_report(
        tokens=tokens,
        matches=matches,
        total_words=total_words if total_words is not None else len(tokens),
        reference_token_indices=reference_indices,
        quote_spans=quote_spans,
        statuses=statuses or {},
        min_match_words=min_match_words,
        exclude_references=exclude_references,
        include_quotes=include_quotes,
    )


# ---------------------------------------------------------------- similarity
def test_percentages_use_the_documented_filter_set() -> None:
    first = span(11, 80)
    report = build(matches=[first])
    assert report.similarity_incl_quotes > 0
    assert report.filters_applied["exclude_references"] is True
    assert report.filters_applied["include_quotes"] is True
    assert report.filters_applied["min_match_words"] == 5
    assert report.stats.total_words == len(tokenize(DOC))


def test_quotes_leave_the_report_when_excluded() -> None:
    quotes = [Span(11, 80)]
    included = build(matches=[span(11, 80)], quote_spans=quotes, include_quotes=True)
    excluded = build(matches=[span(11, 80)], quote_spans=quotes, include_quotes=False)
    assert included.stats.matched_words_quoted > 0
    assert included.similarity_incl_quotes > 0
    assert excluded.similarity_incl_quotes == 0.0
    assert excluded.similarity_incl_quotes == excluded.similarity_excl_quotes


def test_matches_inside_the_reference_list_are_dropped() -> None:
    tokens = tokenize(DOC)
    inside_reference = {i for i, token in enumerate(tokens) if token.char_start < 80}
    report = build(matches=[span(11, 70)], reference_indices=inside_reference)
    assert report.stats.match_count == 0
    assert report.similarity_incl_quotes == 0.0


def test_short_matches_are_filtered_by_the_institutional_rule() -> None:
    report = build(matches=[span(11, 40, words=3)], min_match_words=5)
    assert report.stats.match_count == 0
    assert report.stats.matched_words == 0


def test_per_source_shares_are_sorted_and_percentages_are_of_the_document() -> None:
    a = span(11, 80, words=10, source_id="a", name="kaynak1.md")
    b = span(90, 140, words=6, source_id="b", name="kaynak2.md")
    report = build(matches=[a, b])
    assert [s.source_id for s in report.per_source] == ["b", "a"] or report.per_source[
        0
    ].share_percent >= report.per_source[1].share_percent
    assert report.largest_single_source is report.per_source[0]
    assert sum(s.share_percent for s in report.per_source) <= 100.0


def test_attribution_split_is_reported() -> None:
    quoted = span(11, 80)
    plain = span(120, 160, words=8, source_id="s2", name="kaynak2.md")
    report = build(
        matches=[quoted, plain],
        statuses={
            id(quoted): CitationStatus.CITED_AND_QUOTED,
            id(plain): CitationStatus.NOT_CITED_OR_QUOTED,
        },
    )
    assert report.attribution["cited_and_quoted"] == 1
    assert report.attribution["not_cited_or_quoted"] == 1
    assert report.per_source[0].uncited_matches >= 0


def test_granularity_counts_cases_not_fragments() -> None:
    a = span(11, 40)
    # Same region, different source: one copied case, reported twice.
    b = span(11, 40, words=8, source_id="s2", name="kaynak2.md")
    report = build(matches=[a, b])
    assert report.stats.match_count == 2
    assert report.stats.cases == 1
    assert report.stats.granularity == pytest.approx(2.0)


def test_institutional_bands_carry_their_sources() -> None:
    assert len(INSTITUTION_BANDS) >= 4
    for band in INSTITUTION_BANDS:
        assert band.url.startswith("http")
    ytu = next(b for b in INSTITUTION_BANDS if b.institution.startswith("YTÜ"))
    assert ytu.total_excl_quotes == 15.0
    assert ytu.total_incl_quotes == 20.0
    assert ytu.single_source == 2.0


def test_band_evaluation_flags_an_over_threshold_document() -> None:
    small = build(matches=[span(11, 80)])
    over = build(matches=[span(11, 140, words=30)])
    assert any(not band["issues"] for band in small.bands)
    flagged = [b for b in over.bands if b["issues"]]
    assert flagged
    assert all(b["status"] == "gözden geçirilmeli" for b in flagged)


def test_measured_turkish_baseline_is_attached() -> None:
    report = build(matches=[span(11, 80)])
    baseline = report.baseline
    assert isinstance(baseline, TurkishBaseline)
    assert baseline.mean_percent == 28.7
    assert baseline.sd_percent == 11.58
    assert baseline.sample_size == 600
    assert "Toprak" in baseline.source
    assert "z=" in baseline.percentile_note(10.0)
    payload = report.to_dict()
    assert payload["baseline"]["mean_percent"] == 28.7


def test_no_matches_still_produces_a_report() -> None:
    report = build(matches=[])
    assert report.similarity_incl_quotes == 0.0
    assert report.largest_single_source is None
    assert report.stats.granularity == 0.0


# ----------------------------------------------------------------- integrity
def test_clean_text_has_no_findings() -> None:
    report = check_integrity("Türkçe bir metin: ğüşçöı harfleri normal.")
    assert not report.tampered
    assert not report.findings


def test_zero_width_characters_are_flagged() -> None:
    report = check_integrity("Bu metin\u200bgizli karakter içeriyor ve devam ediyor.")
    codes = {f.code for f in report.findings}
    assert "zero_width" in codes
    assert report.tampered
    assert "zero_width" in (report.as_degradation() or "")


def test_homoglyphs_are_flagged_and_folded() -> None:
    # "plagiarism" with a Cyrillic "а" and "о"
    report = check_integrity(
        "Bu metinde \u0440l\u0430g\u0456arism denemesi yap\u0131lm\u0131\u015ft\u0131r."
    )
    codes = {f.code for f in report.findings}
    assert "homoglyph" in codes
    assert report.severity == "high"


def test_soft_hyphen_and_repeated_punctuation() -> None:
    report = check_integrity(
        "kelime\u00ad parça ve birçok!!! nokta??? burada, sonra da!!! bir nokta daha"
    )
    codes = {f.code for f in report.findings}
    assert "soft_hyphen" in codes
    assert "repeat_punctuation" in codes


def test_degradation_reason_cites_the_raid_number() -> None:
    """The RAID figure belongs to the homoglyph finding specifically.

    A zero-width character is an insertion, not a substitution, and the measured
    40.6% accuracy drop is about substitutions. Quoting it for both would describe
    neither, so the reason names the finding it is quoting.
    """
    assert check_integrity("normal metin").as_degradation() is None

    hidden = check_integrity("Bu metin\u200bgizli karakter içeriyor.")
    reason = hidden.as_degradation()
    assert reason and "görünmez karakter" in reason
    assert "güvenilmez" in reason
    assert "40.6" not in reason, "insertion için substitution rakamı alıntılanmamalı"

    substituted = check_integrity("The рареment was sent to units.")
    homoglyph_reason = substituted.as_degradation()
    assert homoglyph_reason and "40.6" in homoglyph_reason


def test_emoji_is_a_style_note_not_tampering() -> None:
    report = check_integrity("Bu çalışmada 🚀 sonuçlar elde edildi ve tartışıldı.")
    codes = {f.code for f in report.findings}
    assert "emoji" in codes
    assert not report.tampered


def test_folding_cost_is_zero_for_ordinary_text() -> None:
    assert folding_cost("Normal Türkçe metin.") == 0.0


def test_quote_and_citation_helpers_agree_with_each_other() -> None:
    text = '"Alıntı." (Kaynak, 2020)'
    assert find_quotation_spans(text)
    assert find_citation_spans(text)


# ---------------------------------------------------------------- compliance
@pytest.mark.parametrize(
    "text,expected",
    [
        ("Bu çalışmada yapay zekâ araçlarından yararlanılmıştır.", True),
        ("Metin tamamen yazılmıştır.", False),
        ("Bu bölümde ChatGPT ile düzenleme yapılmıştır.", True),
        ("Yapay zekâ konusu incelenmiştir.", False),
        ("Metinde büyük dil modeli kullanılmış ve çerçevelenmiştir.", True),
        ("YÖK, üretken yapay zekâ konusunda yeni rehber yayımladı.", False),
        ("Gemini desteğiyle tablo düzenlenmiştir.", True),
    ],
)
def test_disclosure_detection_needs_a_use_verb_not_just_a_mention(text, expected) -> None:
    assert find_disclosure(text)[0] is expected
