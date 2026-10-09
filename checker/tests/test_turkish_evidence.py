"""Turkish evidence: what is measured, and the two prior-pass negatives.

One prior claim this file corrects rather than confirms. Earlier passes of
``LITERATURE.md`` asserted "no published AUROC/FPR for any detector on Turkish".
That is **false**: Renklier & Sarıtaş (2026) report AUC-ROC 99.31% with
specificity 94.16%, i.e. FPR 5.84%, on 2,398 held-out Turkish items. The test
below pins the real numbers so the wrong claim cannot come back.

The other negative holds: there is still no published distribution of Turkish
*thesis* similarity percentages.

The finding that matters most for a thesis tool is Altıntop (2026), which ran
eight detectors on genuinely human Turkish academic prose.
"""

from __future__ import annotations

import pytest

from checker_app.services.compliance import TURKISH_SIMILARITY_NORM, YOK_GUIDE
from checker_app.services.reliability import CALIBRATION_EVIDENCE
from checker_app.services.scoring import (
    AI_HUMAN_BASELINE,
    AI_TURKISH_FPR,
    AI_TURKISH_FPR_SOURCE,
)
from checker_app.services.similarity import (
    INSTITUTION_BANDS,
    TurkishAiPresence,
    TurkishBaseline,
)
from checker_app.services.stylometry import (
    LENGTH_TURKISH_HUMAN_WORDS,
    LENGTH_TURKISH_LLM_WORDS,
    LEXICAL_RICHNESS_TURKISH,
)

ALTINTOP = "10.56493/nkusbmyo.1866431"
RENKLIER = "10.28948/ngumuh.1930411"


# ------------------------------------------------------------- the corrected claim
def test_turkish_detector_auroc_exists_and_is_pinned() -> None:
    """The earlier "no published Turkish AUROC" claim was wrong.

    Renklier & Sarıtaş 2026, DOI 10.28948/ngumuh.1930411: AUC-ROC 99.31%,
    specificity 94.16% -> FPR 5.84%, on a 2,398-item held-out Turkish test set.
    """
    assert RENKLIER in AI_TURKISH_FPR
    assert "%99.31" in AI_TURKISH_FPR
    assert "%94.16" in AI_TURKISH_FPR
    assert "%5.84" in AI_TURKISH_FPR, "FPR özgüllükten türetilmeli"


def test_the_two_registers_closest_to_a_thesis_score_lowest() -> None:
    """Academic 95.21% and official/legal 94.99%, against 99%+ elsewhere.

    This is why the published Turkish numbers do not transfer to thesis prose.
    """
    assert "%95.21" in AI_TURKISH_FPR
    assert "%94.99" in AI_TURKISH_FPR
    assert "haber/özet/ödev kayıtlarında" in AI_TURKISH_FPR


# ------------------------------------------------------------------- Altıntop 2026
def test_detectors_on_human_turkish_prose_reach_eighty_nine_percent() -> None:
    """8 detectors, one 5,715-word Turkish academic text written with no AI at all.

    Justdone 89% AI, ZeroGPT ~80%, TruthScan 40%. A tool that trusted a detector
    score on Turkish thesis prose would have accused this text on most services.
    """
    assert "Altıntop" in AI_TURKISH_FPR
    assert "%89 AI" in AI_TURKISH_FPR
    assert "%80 AI" in AI_TURKISH_FPR
    assert "%40 AI" in AI_TURKISH_FPR
    assert "%100 insan" in AI_TURKISH_FPR
    assert AI_TURKISH_FPR_SOURCE.startswith("Altıntop (2026)")
    assert ALTINTOP in AI_TURKISH_FPR_SOURCE


def test_the_same_content_scores_differently_by_language() -> None:
    """ZeroGPT: 0% AI on the French original, 73.25% AI on the Turkish translation.

    A 73-point swing from language alone. This is the measurement that forbids
    calibrating a Turkish tool from English or French performance.
    """
    assert "%73.25" in AI_TURKISH_FPR
    assert "Fransızca" in AI_TURKISH_FPR
    assert "Derrida" in AI_TURKISH_FPR


def test_the_plagiarism_engines_on_that_same_text_stayed_sane() -> None:
    """Turnitin 9% / 4%, i.e. the failure is in the detectors, not the text."""
    assert "%9 (yalnız alıntı filtresi)" in AI_TURKISH_FPR
    assert "%4 (alıntı + kaynakça hariç)" in AI_TURKISH_FPR


# ------------------------------------------------------------------ Turkish TTR
def test_turkish_ttr_does_not_separate_the_classes() -> None:
    """Human 0.756 sits *inside* the AI range [0.709, 0.808].

    Claude Sonnet 4.6 and DeepSeek V3 are both **more** lexically diverse than
    Turkish humans. Any Turkish feature that weights TTR as an AI signal will
    invert - which is why the trio is reported as context and never scored.
    """
    human = LEXICAL_RICHNESS_TURKISH["human"]
    ai_values = [v for k, v in LEXICAL_RICHNESS_TURKISH.items() if k != "human"]
    assert human == pytest.approx(0.756)
    assert min(ai_values) < human < max(ai_values), "insan değeri AI aralığının içinde olmalı"
    assert LEXICAL_RICHNESS_TURKISH["claude-sonnet-4.6"] > human
    assert LEXICAL_RICHNESS_TURKISH["deepseek-v3"] > human


def test_turkish_length_is_a_sharper_confound_than_the_general_measurement() -> None:
    """Human 234.7 words against 141.2-184.8 for every LLM: 21-40% shorter."""
    assert pytest.approx(234.7) == LENGTH_TURKISH_HUMAN_WORDS
    assert LENGTH_TURKISH_LLM_WORDS[1] < LENGTH_TURKISH_HUMAN_WORDS
    shortfall = 1 - LENGTH_TURKISH_LLM_WORDS[1] / LENGTH_TURKISH_HUMAN_WORDS
    assert 0.21 <= shortfall <= 0.40


def test_subword_fragmentation_points_the_other_way() -> None:
    """Human Turkish has the *highest* tokens/word of the set (1.522).

    So tokens-per-word is not a usable AI signal either: it would run backwards.
    """
    stats = LEXICAL_RICHNESS_TURKISH
    assert stats  # the reference table is populated
    from checker_app.services.stylometry import LENGTH_TURKISH  # noqa: PLC0415

    assert LENGTH_TURKISH["human_tokens_per_word"] == pytest.approx(1.522)


# ------------------------------------------------------------------ Turkish baseline
def test_the_turkish_ai_presence_distribution_is_reported() -> None:
    """Akkaya & Beygirci 2026, 204 hand-verified Turkish articles on DergiPark."""
    presence = TurkishAiPresence()
    assert presence.sample_size == 204
    assert presence.mean_percent == pytest.approx(20.0)
    assert presence.below_20_percent == 122
    assert presence.below_20_share == pytest.approx(0.598)
    assert presence.below_20_band_mean_percent == pytest.approx(6.0)
    assert presence.tr_dizin_mean_percent == pytest.approx(12.0)
    assert presence.non_tr_dizin_mean_percent == pytest.approx(28.0)


def test_turkish_ai_concentrates_in_the_opening_not_the_method() -> None:
    """Introduction + literature review 49.0%, method 2.9%.

    A thesis whose literature review is flagged and whose method section is not
    is the ordinary case, not the suspicious one.
    """
    sections = TurkishAiPresence().section_rates
    assert sections["introduction_literature"] == pytest.approx(0.490)
    assert sections["method"] == pytest.approx(0.029)
    assert sections["introduction_literature"] > sections["method"] * 10


def test_the_presence_distribution_carries_its_own_caveat() -> None:
    payload = TurkishAiPresence().to_dict()
    assert "tez değil" in payload["caveat"]
    assert "DOI 10.46452/baksoder.1899625" in payload["source"]
    # It independently lands on the same 20% the tool uses as its reporting floor.
    assert "raporlama tabanı" in payload["reading"]


def test_the_similarity_baseline_is_unchanged_and_still_labelled_as_narrow() -> None:
    """The similarity baseline (Toprak 2014, n=600) is a different measurement."""
    baseline = TurkishBaseline()
    assert baseline.mean_percent == pytest.approx(28.7)
    assert baseline.sample_size == 600
    assert "narrow" in baseline.caveat or "dar" in baseline.caveat


# --------------------------------------------------------------------- YÖK's rule
def test_yok_guide_has_no_numeric_threshold_and_does_not_mention_theses() -> None:
    """Both facts were verified in the 20-page May 2024 guide.

    This is why the tool reports a percentage but never calls it a national rule.
    """
    assert YOK_GUIDE["numeric_threshold"] is None
    assert YOK_GUIDE["mentions_thesis"] is False
    assert YOK_GUIDE["date"] == "Mayıs 2024"
    assert YOK_GUIDE["pages"] == 20
    assert YOK_GUIDE["url"].startswith("https://")


def test_yok_guide_separates_permitted_from_forbidden_uses() -> None:
    """The forbidden side is the part a thesis needs.

    Permitted includes literature review, source organisation, grammar check and
    translation. Forbidden is hypothesis generation, discussion, interpretation
    and application.
    """
    assert "kaynak araştırması" in YOK_GUIDE["permitted"]
    assert "çeviri" in YOK_GUIDE["permitted"]
    assert "hipotez üretimi" in YOK_GUIDE["forbidden"]
    assert "yorum" in YOK_GUIDE["forbidden"]
    assert "tüm hukuki ve etik sorumluluğu" in YOK_GUIDE["conditions"]


def test_turkish_de_facto_similarity_thresholds_are_recorded_as_practice() -> None:
    """15% general, 20% "mostly accepted", 5% from one source.

    Altıntop 2026 citing Toprak 2017 and Güçlüer vd. 2024. Practice, not a rule -
    so they must not be presented as binding.
    """
    assert TURKISH_SIMILARITY_NORM["general_ceiling_percent"] == pytest.approx(15.0)
    assert TURKISH_SIMILARITY_NORM["mostly_accepted_percent"] == pytest.approx(20.0)
    assert TURKISH_SIMILARITY_NORM["single_source_problem_percent"] == pytest.approx(5.0)
    assert "bağlayıcı ulusal eşik değildir" in TURKISH_SIMILARITY_NORM["source"]


def test_the_de_facto_norms_line_up_with_the_published_bands() -> None:
    """The 20% de-facto norm is the same number as the ESU / Çukurova band.

    Convergence is not proof, but a coincidence this exact is worth recording.
    """
    values = {b.total_incl_quotes for b in INSTITUTION_BANDS if b.total_incl_quotes}
    assert 20.0 in values
    assert TURKISH_SIMILARITY_NORM["mostly_accepted_percent"] in values


def test_the_human_baseline_and_the_turkish_fpr_cannot_be_confused() -> None:
    """Two different studies, two different registers, both carried together.

    Fiedler & Döpke measured lecturers on German thesis excerpts; Altıntop
    measured detectors on Turkish academic prose. Merging them would produce a
    number that means nothing.
    """
    assert "Alman tez" in AI_HUMAN_BASELINE
    assert "10.1016/j.iree.2025.100321" in AI_HUMAN_BASELINE
    assert "%57" in AI_HUMAN_BASELINE and "%64" in AI_HUMAN_BASELINE
    assert "%57" not in AI_TURKISH_FPR
    assert "%89" not in AI_HUMAN_BASELINE


def test_the_calibration_caveat_no_longer_claims_turkish_numbers_are_absent() -> None:
    """A stale "no Turkish AUROC exists" sentence would contradict the guide."""
    from checker_app.services.reliability import reliability_report  # noqa: PLC0415

    caveats = " ".join(
        reliability_report([(0.02, 0.0)] * 20 + [(0.98, 1.0)] * 20).to_dict()["caveats"]
    )
    assert "AUROC/FPR değeri yoktur" not in caveats, "eski ve yanlış iddia hâlâ duruyor"
    assert "AUROC %99.31" in caveats, "gerçek Türkçe sayıları verilmeli"
    assert "akademik %95.21" in caveats
    assert "%89 AI" in caveats, "Altıntop'ın %89'u kalibrasyon uyarısında olmalı"
    assert "başka bir çalışmadan alınmaz" in caveats
    assert CALIBRATION_EVIDENCE  # the ECE/agreement evidence still travels