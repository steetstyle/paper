"""Cross-lingual reuse through translation-resistant anchors.

A word-level matcher reports a translated passage as "0% overlap, clean", which
is the worst possible answer: it reads as exonerating evidence when it is
actually a blind spot. These tests pin both halves of the correction.

The second half matters just as much. A control - a Turkish text that merely
happens to contain the same sample sizes and thresholds in the same order -
fires a cluster too, so the module has to grade what it finds rather than call it
proof. A real translation of the methods paragraph grades ``medium``; the
coincidence grades ``review``.
"""

from __future__ import annotations

import pytest

from checker_app.config import get_settings
from checker_app.domain.enums import MatchKind
from checker_app.domain.segmentation import SentenceSplitter
from checker_app.services.crosslingual import (
    MIN_CLUSTER,
    MIN_STRONG_ANCHORS,
    WINDOW_TOKENS,
    _anchors,
    find_cross_lingual,
)
from checker_app.services.sources import SourceLoader

ENGLISH_METHODS = """# Methods (English)

Participants were recruited from two introductory writing courses and 412
students completed the full protocol. Cohen (1988) reports an inter-rater
agreement of 0.80 as the minimum acceptable value, and all pairs in this study
exceeded 0.80. Data collection ran between 2019 and 2021 and produced 23 usable
response sets after 7 exclusions. The instrument was scored by two blinded
reviewers using the rubric published by Kaufman and Smith (2019).
"""

#: The same paragraph, translated. Every number and every surname survives;
#: almost no content word does.
TURKISH_TRANSLATION = """# YÖNTEM

Katılımcılar iki temel yazma dersinden ve 412 öğrenciden toplandı ve
protokolün tamamını tamamladı. Cohen (1988), kabul edilebilir en düşük
değer olarak 0.80 aralıklar arası uyum raporlamaktadır ve bu çalışmadaki tüm
çiftler 0.80'in üzerindedir. Veri toplama 2019 ile 2021 arasında yürütülmüş ve
7 çıkarım sonrası 23 kullanılabilir yanıt seti üretilmiştir. Ölçek, Kaufman ve
Smith (2019) tarafından yayımlanan rubrik ile iki kör değerlendirici tarafından
puanlanmıştır.
"""

#: Not a translation. Recruits the same numbers, in the same order, which is
#: exactly the coincidence a numbers-only cluster has to survive.
COINCIDENT_TURKISH = """# GİRİŞ

Bu çalışmada 412 katılımcı yer aldı ve veriler 2019 yılında toplandı. Cohen
(1988) çalışmasında 0.80 değerini referans almıştır. Uygulamada 2021 yılına
kadar süren bir süreçte 7 adım tamamlanmış ve 23 sonuç kaydedilmiştir.
"""


def _scan(tmp_path, name: str, text: str):
    loader = SourceLoader(get_settings())
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    document = loader.load_document(str(path))
    segmentation = SentenceSplitter(get_settings().segmentation).split(document.text)
    return document, segmentation


def _run(tmp_path, name: str, text: str, source_text: str):
    document, segmentation = _scan(tmp_path, name, text)
    loader = SourceLoader(get_settings())
    source_path = tmp_path / "source.md"
    source_path.write_text(source_text, encoding="utf-8")
    sources = loader.to_sources([loader.load_document(str(source_path))])
    report = find_cross_lingual(
        document_tokens=segmentation.tokens,
        sources=sources,
        document_language=segmentation.language,
        document_text=document.text,
    )
    return report, sources[0]


# ------------------------------------------------------------------- the gap
def test_the_word_matcher_sees_nothing_in_a_translation(tmp_path) -> None:
    """The failure this module exists to fix, stated as a measurement."""
    document, segmentation = _scan(tmp_path, "tr.md", TURKISH_TRANSLATION)
    source_path = tmp_path / "en.md"
    source_path.write_text(ENGLISH_METHODS, encoding="utf-8")
    loader = SourceLoader(get_settings())
    sources = loader.to_sources([loader.load_document(str(source_path))])

    from checker_app.services.plagiarism import PlagiarismService  # noqa: PLC0415

    result = PlagiarismService(get_settings().plagiarism).compare(
        segmentation.sentences, segmentation.tokens, sources, document.text
    )
    assert not result.matches, "kelime eşleştirmesi gerçekten hiçbir şey görmemeli"
    assert document.text and sources[0].language == "en"


def test_the_anchor_pass_finds_what_the_word_matcher_cannot(tmp_path) -> None:
    report, source = _run(tmp_path, "tr.md", TURKISH_TRANSLATION, ENGLISH_METHODS)
    assert source.language == "en"
    assert report.clusters_found >= 1
    largest = max(report.clusters, key=lambda c: c.size)
    assert largest.size >= MIN_CLUSTER
    labels = set(largest.anchors)
    assert "num:412" in labels
    assert "num:0.80" in labels
    assert "num:1988" in labels


# -------------------------------------------------------------- false positives
def test_a_coincidence_is_graded_below_a_translation(tmp_path) -> None:
    """The control fires too, so grading is not optional.

    Recurring sample sizes and thresholds within a field are cheap anchors; a
    proper noun is not.
    """
    real, _ = _run(tmp_path, "tr.md", TURKISH_TRANSLATION, ENGLISH_METHODS)
    chance, _ = _run(tmp_path, "ctl.md", COINCIDENT_TURKISH, ENGLISH_METHODS)

    assert real.clusters_found >= chance.clusters_found
    best_real = max((c.confidence() for c in real.clusters), default="review")
    best_chance = max((c.confidence() for c in chance.clusters), default="review")
    assert best_real != "review", "gerçek çeviri 'review' olamaz"
    assert best_chance == "review", "tesadüfi sayı kümesi review olmalı"


def test_a_numbers_only_cluster_is_never_high() -> None:
    from checker_app.services.crosslingual import AnchorCluster  # noqa: PLC0415

    numbers = AnchorCluster("s", "s.md", ("num:412", "num:2019", "num:2021", "num:23", "num:7"), 0, 9, 0, 9)
    with_name = AnchorCluster(
        "s", "s.md", ("num:412", "num:2019", "name:cohen", "num:23", "num:7", "num:2019.0"), 0, 9, 0, 9
    )
    assert numbers.confidence() == "review"
    assert with_name.confidence() == "high"
    assert with_name.named_count == 1


def test_the_reading_does_not_call_a_cluster_proof(tmp_path) -> None:
    report, _ = _run(tmp_path, "tr.md", TURKISH_TRANSLATION, ENGLISH_METHODS)
    reading = report.reading()
    assert "kanıt değil işarettir" in reading
    assert "elle karşılaştır" in reading


def test_the_evidence_and_the_reading_agree(tmp_path) -> None:
    """These two travelled apart once, when the evidence said "proof" and the
    reading said "flag". A test so they cannot again."""
    report, _ = _run(tmp_path, "tr.md", TURKISH_TRANSLATION, ENGLISH_METHODS)
    assert "kanıttır" not in report.evidence
    assert "işarettir" in report.evidence


def test_no_cluster_reads_as_not_evidence_of_absence(tmp_path) -> None:
    # The source must be a *different* language, or there is nothing to search
    # and the report correctly says the pass never ran.
    report, _ = _run(
        tmp_path,
        "ctl.md",
        "Bu çalışmada yalnızca nitel bir anlatı yöntemi kullanılmıştır ve "
        "sayısal hiçbir eşik belirlenmemiştir.\n",
        "This study used a purely qualitative narrative design and set no "
        "numeric thresholds of any kind.\n",
    )
    assert report.clusters_found == 0
    assert "anlamına gelmez" in report.reading()
    assert "%74" in report.reading()


# --------------------------------------------------------------------- anchors
def test_sentence_initial_capitals_are_not_names() -> None:
    """The tokenizer strips punctuation, so the period sits in the gap.

    Checking ``token.text[-1]`` never fires, which is how "Veriler" and "Her"
    became fake name anchors.
    """
    text = "Veriler temizlendi. Her katılımcı onayladı. Rapor Smith tarafından yazıldı."
    segmentation = SentenceSplitter(get_settings().segmentation).split(text)
    labels = [label for _, label in _anchors(segmentation.tokens, text)]
    assert "name:veriler" not in labels
    assert "name:her" not in labels
    # Mid-sentence, so it really is a surname and not a sentence-initial capital.
    assert "name:smith" in labels


def test_a_token_after_a_heading_is_treated_as_sentence_initial() -> None:
    """A heading has no full stop, so the newline has to count as a boundary."""
    text = "# GİRİŞ\n\nVeriler toplandı.\n"
    segmentation = SentenceSplitter(get_settings().segmentation).split(text)
    labels = [label for _, label in _anchors(segmentation.tokens, text)]
    assert "name:veriler" not in labels


def test_decimal_precision_distinguishes_anchors() -> None:
    """0.80 and 0.8 are different anchors; a threshold's precision is a
    fingerprint."""
    text = "Eşik 0.80 idi. Oran 0.8 bulundu."
    segmentation = SentenceSplitter(get_settings().segmentation).split(text)
    labels = [label for _, label in _anchors(segmentation.tokens, text)]
    assert "num:0.80" in labels
    assert "num:0.8" in labels


def test_bare_zero_and_one_carry_nothing() -> None:
    text = "Değer 0 ve 1 olarak verildi."
    segmentation = SentenceSplitter(get_settings().segmentation).split(text)
    labels = [label for _, label in _anchors(segmentation.tokens, text)]
    assert "num:0" not in labels
    assert "num:1" not in labels


def test_a_year_alone_is_not_strong_evidence() -> None:
    from checker_app.services.crosslingual import _is_strong  # noqa: PLC0415

    assert _is_strong("num:412")
    assert _is_strong("doi:10.1234/x")
    assert not _is_strong("year:2019")
    assert not _is_strong("name:smith")
    assert MIN_STRONG_ANCHORS >= 2


def test_order_is_required_so_two_coincidences_do_not_fuse(tmp_path) -> None:
    """The same two numbers in a different order are two unrelated sentences."""
    report, _ = _run(
        tmp_path,
        "tr.md",
        "Bir çalışmada 2019 yılında 412 katılımcı yer aldı ve 0.80 eşiği "
        "kullanıldı. Ayrıca Cohen (1988) tarafından 23 sonuç raporlandı.\n",
        "Cohen (1988) reported 412 records. The 0.80 threshold came from 2019 "
        "work. In total 23 sets were retained.\n",
    )
    # Whatever it finds, it must not claim a long ordered run from numbers that
    # appear in a different order in the source.
    for cluster in report.clusters:
        assert cluster.size <= WINDOW_TOKENS


# ------------------------------------------------------------------- the gate
def test_same_language_sources_are_not_searched(tmp_path) -> None:
    """A Turkish source is the word matcher's job."""
    report, _ = _run(
        tmp_path,
        "tr.md",
        "Katılımcılar 412 öğrenciden oluştu ve 2019 yılında veri toplandı. "
        "Cohen (1988) 0.80 eşiğini önermiştir.\n",
        "Katılımcı sayısı 412 olarak belirlenmiştir.\n",
    )
    assert report.cross_language_sources == 0
    assert report.clusters_found == 0
    assert "çapraz dil taraması yapılmadı" in report.reading()


def test_the_report_carries_the_measured_ceiling(tmp_path) -> None:
    report, _ = _run(tmp_path, "tr.md", TURKISH_TRANSLATION, ENGLISH_METHODS)
    payload = report.to_dict()
    assert "%95.25" in payload["evidence"]
    assert "%74.10" in payload["evidence"]
    assert "KAYIT TÜRÜNE" in payload["evidence"]
    assert "ALT SINIRDIR" in payload["caveat"]


def test_thresholds_are_documented_in_the_payload(tmp_path) -> None:
    report, _ = _run(tmp_path, "tr.md", TURKISH_TRANSLATION, ENGLISH_METHODS)
    payload = report.to_dict()
    assert payload["min_cluster"] == MIN_CLUSTER
    assert payload["min_strong_anchors"] == MIN_STRONG_ANCHORS
    assert payload["window_tokens"] == WINDOW_TOKENS
    assert set(payload["confidence_counts"]) == {"high", "medium", "review"}


def _occurrences(haystack: str, needle: str) -> list[int]:
    found: list[int] = []
    start = haystack.find(needle)
    while start != -1:
        found.append(start)
        start = haystack.find(needle, start + 1)
    return found


def test_match_kind_exists_for_the_pipeline() -> None:
    assert MatchKind.CROSS_LINGUAL.value == "cross_lingual"


def test_the_module_docstring_is_corrected_in_place() -> None:
    """This docstring once overstated its evidence; the numbers must not drift back.

    Two claims were corrected after review. The 80 / 26.7 / 16.7 sequence is one
    Arabic literary translation tested with three weak systems, not a general
    cross-lingual result, and a larger modern EN-FA measurement showed no drop at
    all for paraphrase. And the TR-MTEB figures are EN-TR bitext 99.43 against
    6.78 - 73.07 and 37.02 were the `Mean(Task)` column, misattributed.

    The correction quotes the wrong numbers in order to label them, so the check
    is that they only ever appear next to a word that condemns them - not that
    they are absent.
    """
    from checker_app.services import crosslingual  # noqa: PLC0415

    doc = crosslingual.__doc__ or ""
    assert "6.78" in doc and "99.43" in doc, "doğru bitext rakamları eksik"
    assert "0.531" in doc, "Türkçe için ölçülen leksikal örtüşme kayboldu"
    assert "0.97" in doc and "0.96" in doc, "EN↔FA karşı bulgusu eksik"
    # The ensemble caveat is the part that was previously missing entirely.
    assert "fusion of eight methods" in doc
    assert "34.49" in doc, "bilimsel korpusta en iyi tek yöntemin skoru eksik"

    for wrong in ("37.02", "73.07"):
        for position in _occurrences(doc, wrong):
            window = doc[max(0, position - 260) : position + 260].lower()
            assert any(
                marker in window
                for marker in ("wrong", "not bitext", "mean(task)", "misattribut")
            ), f"{wrong} bir yerde hâlâ doğru olarak sunuluyor: ...{window[200:320]}..."


def test_the_reference_audit_reports_a_positive_predictive_value() -> None:
    """A flag count is not a problem count, and the payload has to say so.

    Measured: at a 1-2% base rate the strongest verifiers reach only 5-18% PPV -
    four to nine false alarms per true catch - and RefChecker's FPR is 50.7%.
    """
    from checker_app.services.references import (  # noqa: PLC0415
        POSITIVE_PREDICTIVE_VALUE,
        expected_true_findings,
    )

    ppv = POSITIVE_PREDICTIVE_VALUE
    assert ppv["academic_paper_base_rate"] == (0.0031, 0.0081)
    assert ppv["verified_pp_range"] == (0.02, 0.18)
    assert ppv["false_alarms_per_true_catch"] == "4-9"
    assert ppv["refchecker_measured_fpr"] == pytest.approx(0.507)

    # A range, not an invented point estimate: with one flag the honest answer
    # is "probably not".
    assert "muhtemelen 0" in expected_true_findings(1)
    assert expected_true_findings(0) == "0"
    assert "40" in expected_true_findings(40)