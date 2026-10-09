"""Attribution: quoted / cited / neither, in Turkish and English."""

from __future__ import annotations

from checker_app.domain.enums import CitationStatus
from checker_app.services.attribution import (
    CitationEvidence,
    classify_attribution,
    classify_flags,
    find_citation_spans,
    find_quotation_spans,
    is_quoted,
    summarise,
)


def test_quoted_and_cited_is_legitimate() -> None:
    text = 'Bize göre, "Veri temizliği gerçekleştirilmiştir." (Yılmaz ve ark., 2021)'
    start, end = text.index('"'), text.rindex('"') + 1
    evidence = classify_attribution(text, start, end)
    assert evidence.status is CitationStatus.CITED_AND_QUOTED
    assert evidence.has_quote and evidence.has_citation
    assert "Yılmaz" in evidence.citation_evidence


def test_bracket_citation_counts() -> None:
    text = "Veri temizliği gerçekleştirilmiştir [12] ve protokol uygulanmıştır."
    evidence = classify_attribution(text, 0, 40)
    assert evidence.has_citation
    assert evidence.status is CitationStatus.MISSING_QUOTATION


def test_narrative_citation_counts() -> None:
    text = "Yılmaz ve ark. (2021) bu yaklaşımın yetersiz kaldığını göstermiştir."
    evidence = classify_attribution(text, 0, len(text))
    assert evidence.has_citation


def test_no_reference_means_not_cited() -> None:
    text = "Veri temizliği gerçekleştirilmiş ve standart bir protokol kullanılmıştır."
    evidence = classify_attribution(text, 0, len(text))
    assert evidence.status is CitationStatus.NOT_CITED_OR_QUOTED
    assert not evidence.has_citation and not evidence.has_quote


def test_quote_without_reference_is_missing_citation() -> None:
    text = 'Yazar "bu cümle başka bir çalışmadan alınmıştır" biçiminde yazmıştır.'
    start, end = text.index('"'), text.rindex('"') + 1
    evidence = classify_attribution(text, start, end)
    assert evidence.status is CitationStatus.MISSING_CITATION
    assert evidence.has_quote and not evidence.has_citation


def test_url_is_a_reference() -> None:
    text = "Bu tanım https://arxiv.org/abs/1706.03762 adresinde verilmiştir."
    evidence = classify_attribution(text, 0, 40)
    assert evidence.has_citation


def test_classify_flags_covers_the_four_turnitin_groups() -> None:
    assert classify_flags(cited=True, quoted=True) is CitationStatus.CITED_AND_QUOTED
    assert classify_flags(cited=True, quoted=False) is CitationStatus.MISSING_QUOTATION
    assert classify_flags(cited=False, quoted=True) is CitationStatus.MISSING_CITATION
    assert classify_flags(cited=False, quoted=False) is CitationStatus.NOT_CITED_OR_QUOTED


def test_severity_weights_order_the_groups() -> None:
    weights = [status.severity_weight for status in CitationStatus]
    assert weights[0] == max(weights)
    assert CitationStatus.CITED_AND_QUOTED.attributed
    assert not CitationStatus.NOT_CITED_OR_QUOTED.attributed


def test_span_helpers_use_overlap_not_containment() -> None:
    text = 'Öncesi. "Alıntı burada." Sonrası.'
    quotes = find_quotation_spans(text)
    assert quotes
    quoted, _ = is_quoted(quotes, text.index("Alıntı"), text.index("burada"))
    assert quoted
    not_quoted, _ = is_quoted(quotes, 0, 5)
    assert not not_quoted


def test_block_quotes_are_detected() -> None:
    text = "> Bu bir tırnak bloğudur ve iki kelimeden uzundur.\n\nDevamı."
    spans = find_quotation_spans(text)
    assert any("tırnak bloğu" in s.evidence for s in spans)


def test_citation_spans_do_not_overlap_duplicates() -> None:
    text = "Yılmaz (2021) ve Smith (2020) aynı sonuca varmıştır."
    spans = find_citation_spans(text)
    for previous, current in zip(spans, spans[1:], strict=False):
        assert previous.char_end <= current.char_start


def test_evidence_dataclass_defaults() -> None:
    evidence = CitationEvidence(has_quote=False, has_citation=False)
    assert evidence.status is CitationStatus.NOT_CITED_OR_QUOTED


def test_summarise_counts_every_group() -> None:
    counts = summarise(
        [
            CitationStatus.NOT_CITED_OR_QUOTED,
            CitationStatus.CITED_AND_QUOTED,
            CitationStatus.CITED_AND_QUOTED,
        ]
    )
    assert counts["not_cited_or_quoted"] == 1
    assert counts["cited_and_quoted"] == 2
    assert sum(counts.values()) == 3
