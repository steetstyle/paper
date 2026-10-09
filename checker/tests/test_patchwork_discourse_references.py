"""Patchworking, discourse profile and the reference audit.

These three exist because the literature says the obvious measures are not the
informative ones: a percentage says how much was reused, not how it was
assembled; connective counts survive polishing when surprisal does not; and a
bibliography can be the strongest available corroborating evidence. Each test
pins a measured number from the module docstring so the reasoning cannot be
edited away silently.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from checker_app.config import get_settings
from checker_app.domain.enums import BlockType, CitationStatus, MatchKind
from checker_app.domain.models import MatchSpan
from checker_app.domain.sections import SectionRole
from checker_app.domain.segmentation import SentenceSplitter
from checker_app.domain.text import Token
from checker_app.services.attribution import Span
from checker_app.services.discourse import discourse_profile
from checker_app.services.patchwork import CHANCE_BASELINES, build_patchwork_report
from checker_app.services.references import audit_references
from checker_app.services.scoring import AI_REPORTING_FLOOR


def _tokens(text: str) -> list[Token]:
    words: list[Token] = []
    cursor = 0
    for i, word in enumerate(text.split()):
        start = text.index(word, cursor)
        words.append(
            Token(
                index=i,
                text=word,
                norm=word.lower(),
                char_start=start,
                char_end=start + len(word),
                is_number=word.isdigit(),
            )
        )
        cursor = start + len(word)
    return words


def _match(source: str, doc_text: str, phrase: str, occurrence: int = 1) -> MatchSpan:
    """Locate the ``occurrence``-th copy of ``phrase``.

    Without it, three matches meant for three different places all resolved to
    the first one, which would make the fragmentation tests vacuous.
    """
    start = -1
    for _ in range(occurrence):
        start = doc_text.index(phrase, start + 1)
    return MatchSpan(
        kind=MatchKind.EXACT,
        source_id=source,
        source_name=f"{source}.md",
        sentence_indices=(0,),
        doc_char_start=start,
        doc_char_end=start + len(phrase),
        source_char_start=0,
        source_char_end=len(phrase),
        matched_words=len(phrase.split()),
        ratio=1.0,
        snippet=phrase,
    )


# ------------------------------------------------------------------- patchwork
def test_no_tiles_when_everything_is_below_the_five_word_floor() -> None:
    """HyPlag's GIT needs >= 5 matching identifiers; a shorter run is coincidence."""
    text = "Veri temizliği yapıldı ve model eğitildi."
    report = build_patchwork_report(
        tokens=_tokens(text), matches=[_match("k1", text, "Veri temizliği")],
        total_words=len(text.split()),
    )
    assert report.tile_count == 0
    assert report.score == 0.0
    assert "temiz" in report.reading  # says it is not a clean bill of health


def test_one_long_run_and_many_fragments_score_differently() -> None:
    """The whole point of the quantity-sensitive family: shape, not volume."""
    blocks = [
        "bir iki üç dört beş altı yedi",
        "dokuz on onbir oniki on üç",
        "on dört on beş on altı on yedi",
    ]
    text = " ".join(blocks) + " kalan metin burada devam ediyor ve bitiyor"
    whole = " ".join(blocks)
    fragmented = build_patchwork_report(
        tokens=_tokens(text),
        matches=[_match(f"k{i}", text, b) for i, b in enumerate(blocks)],
        total_words=len(text.split()),
    )
    single = build_patchwork_report(
        tokens=_tokens(text),
        matches=[_match("k1", text, whole)],
        total_words=len(text.split()),
    )
    assert fragmented.tile_count == 3
    assert single.tile_count == 1
    # Same reused volume, three times the tile count, three times the score. That
    # gap is what a percentage cannot show and what GIT exists to show.
    assert fragmented.tile_words == single.tile_words
    assert fragmented.matched_words == single.matched_words
    assert fragmented.score == pytest.approx(3 * single.score, rel=1e-9)


def test_greedy_tiling_does_not_double_count_one_span() -> None:
    """The same passage reported by several matchers counts once.

    One copied paragraph found by the shingle, the near-match and the template
    pass is one tile, not three - which is also the granularity behaviour the
    PAN metric punishes, and the reason HyPlag tiles greedily.
    """
    block = "paragraf burada duruyor ve epey uzun bir yerdir"
    text = f"giriş {block} kapanış"
    span = _match("k1", text, block)
    overlapping = [replace(span, source_id=f"k{i}", source_name=f"k{i}.md") for i in range(3)]
    report = build_patchwork_report(
        tokens=_tokens(text), matches=overlapping, total_words=len(text.split())
    )
    assert report.tile_count == 1
    assert report.largest_tile_words == len(block.split())
    assert report.matched_words == 3 * report.tile_words, "ham eşleşme sayısı korunmalı"


def test_the_measured_chance_baselines_are_published() -> None:
    """These come from 1,000,000 random document pairs (HyPlag, arXiv:1906.11761).

    They are the only corpus-derived thresholds in this literature; every
    institutional percentage is a locally chosen number with no derivation.
    """
    assert CHANCE_BASELINES["GIT (greedy identifier tiles, math)"] == 0.15
    assert CHANCE_BASELINES["Encoplot (shared character 16-grams)"] == 0.06
    assert CHANCE_BASELINES["GCT (greedy citation tiles)"] == 0.10


def test_below_chance_is_stated_not_hidden() -> None:
    """A single reused run in a short document cannot clear the chance baseline.

    HyPlag measured that baseline over 1,000,000 random document pairs, so a
    0.10 score is not "some plagiarism" - it is indistinguishable from noise.
    """
    text = " ".join(["bir iki üç dört beş altı yedi sekiz dokuz on onbir on iki"] * 3)
    report = build_patchwork_report(
        tokens=_tokens(text),
        matches=[_match(f"k{i}", text, "bir iki üç dört beş altı yedi sekiz dokuz on onbir on iki", i + 1) for i in range(3)],
        total_words=len(text.split()),
    )
    assert report.score < report.chance_baseline
    assert report.above_chance is False
    assert "şansa ayırt edilemiyor" in report.reading


def test_quoted_and_unattributed_words_are_counted_separately() -> None:
    """Attribution does not change the score; it changes what the score means."""
    block = "alıntılanmış bir bölüm burada yer alıyor ve devam ediyor"
    text = f"giriş {block} kapanış"
    match = _match("k1", text, block)
    quote = Span(char_start=text.index(block), char_end=text.index(block) + len(block))

    cited = build_patchwork_report(
        tokens=_tokens(text),
        matches=[match],
        total_words=len(text.split()),
        quote_spans=[quote],
        statuses={id(match): CitationStatus.CITED_AND_QUOTED},
    )
    bare = build_patchwork_report(
        tokens=_tokens(text), matches=[match], total_words=len(text.split()), quote_spans=[quote]
    )
    assert cited.score == bare.score, "atıf puanı değiştirmemeli"
    assert cited.quoted_tile_words == bare.quoted_tile_words == len(block.split())
    assert cited.unattributed_tile_words == 0
    assert bare.unattributed_tile_words == len(block.split())


def test_cope_concentration_principle_picks_a_reading() -> None:
    """Many short runs are *less* concerning than a few long blocks (COPE)."""
    many = "bir iki üç dört beş"
    text = " ".join([many] * 12)
    spread = build_patchwork_report(
        tokens=_tokens(text),
        matches=[_match(f"k{i}", text, many, i + 1) for i in range(12)],
        total_words=len(text.split()),
    )
    assert spread.shape.startswith("yayılmış")
    assert "COPE" in spread.shape
    assert "daha az" in spread.shape


def test_paraphrase_blind_spot_is_stated_on_every_report() -> None:
    """The limitation belongs in the output, not only in the docstring.

    PAN-2025: the best system reaches precision 0.58 at recall 0.82 and naive
    embedding baselines flag genuine paraphrased text about twice as often as
    actual plagiarism (arXiv:2510.06805).
    """
    report = build_patchwork_report(tokens=_tokens("bir iki üç"), matches=[], total_words=3)
    joined = " ".join(report.caveats)
    assert "0.58" in joined
    assert "arXiv:2510.06805" in joined
    assert "PAN-2012" in joined  # the cross-corpus collapse, 0.61 -> 0.17


# ------------------------------------------------------------------- discourse
def test_human_and_llm_discourse_lean_in_opposite_directions() -> None:
    """Human text attributes and sequences; LLM text elaborates.

    Measured direction: RACE, arXiv:2604.04932 (human creators over-express
    Attribution and Temporal, LLM creators over-express Elaboration and Cause).
    """
    human = discourse_profile(
        "Çünkü ölçüm güvenilmezdir, bu nedenle örneklem genişletildi. "
        "Öncelikle veri temizlendi. Daha sonra model eğitildi. Son olarak test edildi."
    )
    llm = discourse_profile(
        "Ayrıca bu bulgu önemlidir. Bunun yanı sıra başka bir sonuç elde edilmiştir. "
        "Örneğin şu durum geçerlidir. Dahası, bu yaklaşım güçlüdür. Üstelik bu yöntem işe yarar. "
        "Buna ek olarak model güçlüdür."
    )
    assert llm.elaboration_ratio > human.elaboration_ratio
    assert human.elaboration_ratio < 0.40
    assert llm.elaboration_ratio > 0.60
    assert "insan yazar lehine" in human.reading()
    assert "LLM lehine" in llm.reading()


def test_discourse_profile_normalises_per_hundred_words() -> None:
    """One connective in 3 words is not one connective in 60 words.

    Normalisation matters because a thesis chapter and a paragraph would
    otherwise be compared on raw counts.
    """
    short = discourse_profile("Ayrıca bir şey.")
    long = discourse_profile("Ayrıca " + "filler kelime " * 60)
    assert short.counts["elaboration"] == long.counts["elaboration"] == 1
    assert short.per_100_words["elaboration"] > long.per_100_words["elaboration"] > 0


def test_discourse_profile_refuses_to_read_an_empty_document() -> None:
    profile = discourse_profile("Veriler analiz edildi.")
    assert profile.total == 0
    assert "bilgi üretmiyor" in profile.reading()


def test_turkish_and_english_connectives_are_both_mapped() -> None:
    assert discourse_profile("Öncelikle veri toplandı.").counts["temporal"] == 1
    assert discourse_profile("Çünkü örneklem yetersizdi.").counts["attribution"] == 1
    assert discourse_profile("Furthermore, the model was retrained.").counts["elaboration"] == 1
    assert discourse_profile("This leads to higher error rates.").counts["cause"] == 1


# ------------------------------------------------------------------ references
def _reference_document(body: str) -> str:
    """A thesis-shaped document: prose, then a bibliography section."""
    return f"# Giriş\n\nBir cümle.\n\n# Kaynakça\n\n{body}"


def test_reference_audit_reads_whole_entries_not_sentence_fragments() -> None:
    """A bibliography entry is not a sentence, and the splitter knows that.

    ``Yılmaz, A. (2021). Otomatik değerlendirme yöntemleri.`` becomes three
    sentences; the audit re-joins them by paragraph, so the year and the venue
    are seen on the same entry.
    """
    audit = audit_references(
        SentenceSplitter(get_settings().segmentation)
        .split(
            _reference_document(
                "Yılmaz, A. (2021). Otomatik değerlendirme yöntemleri. Eğitim Bilimleri "
                "Dergisi, 12(3), 45-61.\n",
            )
        )
        .sentences
    )
    assert audit.total == 1
    entry = audit.entries[0]
    assert entry.year == 2021
    assert entry.has_authors and entry.has_venue


def test_a_broken_doi_is_review_and_a_clean_one_is_ok() -> None:
    audit = audit_references(
        SentenceSplitter(get_settings().segmentation)
        .split(
            _reference_document(
                "Zhang, W. (2023). Attention mechanisms revisited. Neural Networks. "
                "doi:10.9999\n\n"
                "Smith, J. ve Jones, P. (2019). Deep learning for assessment. Journal "
                "of Learning Analytics. doi:10.1234/jla.2019.5678\n",
            )
        )
        .sentences
    )
    assert audit.total == 2
    by_risk = {e.risk for e in audit.entries}
    assert by_risk == {"review", "ok"}
    broken = next(e for e in audit.entries if e.risk == "review")
    assert "doi_yazilmis_ama_gecersiz" in broken.flags


def test_a_single_missing_identifier_is_not_even_a_review() -> None:
    """No DOI alone does not make a Turkish print-only reference suspicious.

    Plenty of legitimate Turkish and humanities references have no persistent
    identifier, so one missing field must stay silent. Escalation needs a
    cluster - which is also the measured shape of fabrication.
    """
    audit = audit_references(
        SentenceSplitter(get_settings().segmentation)
        .split(
            _reference_document(
                "Yılmaz, A. (2021). Otomatik değerlendirme yöntemleri. Eğitim Bilimleri "
                "Dergisi, 12(3), 45-61.\n",
            )
        )
        .sentences
    )
    entry = audit.entries[0]
    assert entry.flags == ("kalici_kimlik_yok",)
    assert entry.risk == "ok"
    # It is still counted, so a bibliography with no identifiers at all stands out.
    assert audit.summary()["without_identifier"] == 1


def test_future_and_impossible_years_escalate() -> None:
    audit = audit_references(
        SentenceSplitter(get_settings().segmentation)
        .split(
            _reference_document(
                "Doe, A. (2099). Future perspectives on tutoring. Journal of Applied "
                "Research.\n\n"
                "Freeman, T. (1912). Early measurement. Monographs in Education.\n",
            )
        )
        .sentences
    )
    flags = {tuple(e.flags) for e in audit.entries}
    assert ("kalici_kimlik_yok", "gelecek_yil") in flags
    assert any("imkansiz_eski_yil" in f for f in flags)
    assert all(e.risk == "review" for e in audit.entries)


def test_audit_carries_its_own_evidence_and_its_own_limits() -> None:
    audit = audit_references(
        SentenceSplitter(get_settings().segmentation)
        .split(
            _reference_document(
                "Doe, A. (2099). Future perspectives on tutoring. Journal of Applied "
                "Research.\n",
            )
        )
        .sentences
    )
    payload = audit.to_dict()
    evidence = " ".join(payload["evidence"])
    assert "4.046" in evidence and "12 kat" in evidence
    caveats = " ".join(payload["caveats"])
    assert "%91" in caveats, "etkilenen makalelerin çoğunun 1-2 kaynak taşıdığı bilgisi şart"
    assert "DOI çözülmez" in caveats
    assert "tez oranları" in caveats


def test_audit_is_empty_without_a_bibliography() -> None:
    text = "Birinci cümle burada. İkinci cümle şurada."
    audit = audit_references(SentenceSplitter(get_settings().segmentation).split(text).sentences)
    assert audit.total == 0
    assert audit.summary()["references_scanned"] == 0


# ------------------------------------------------------------- reporting floor
def test_the_reporting_floor_is_twenty_percent() -> None:
    """Turnitin AIW-2 (Aug 2024): no document AI score below a 20% share.

    Measured on 719,877 pre-2019 human student writings: document FPR 0.51%,
    sentence FPR 0.33%, document recall 91.18%.
    """
    assert pytest.approx(0.20) == AI_REPORTING_FLOOR


def test_heading_blocks_are_not_counted_as_references() -> None:
    splitter = SentenceSplitter(get_settings().segmentation)
    report = splitter.split(
        _reference_document(
            "Yılmaz, A. (2021). Otomatik değerlendirme yöntemleri. Eğitim Bilimleri "
            "Dergisi, 12(3), 45-61.\n",
        )
    )
    headings = [
        s
        for s in report.sentences
        if s.location.section_role is SectionRole.REFERENCES
        and s.location.block_type is BlockType.HEADING
    ]
    assert headings, "fixture must contain a heading inside the references section"
    audit = audit_references(report.sentences)
    assert all(not e.text.startswith("#") for e in audit.entries)