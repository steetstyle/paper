"""English mode: the baselines, the bands, and the correction that matters most.

The headline correction is in :func:`test_english_theses_measure_three_times_lower`.
Clean English dissertations measure **9% ± 6%** (Mayes 2017, n=360) while Turkish
ones measure **28.7% ± 11.58** (Toprak 2014, n=600). Applying the Turkish baseline
to an English thesis would flag roughly three quarters of a normal one, so the
baseline is selected by the document's detected language.

A second trap is the difference between a *measured* threshold and a *policy*
one: 15% is where detection was empirically validated (sensitivity 84.8%,
specificity 80.5%, AUC 0.902), while 25-30% is an institutional ceiling that
normal writing never approaches.
"""

from __future__ import annotations

import pytest

from checker_app.services.english_baselines import (
    LLM_AVOIDED_CONNECTIVES,
    LLM_WORD_SIGNALS,
    PROVENANCE,
    EnglishAiPresence,
    EnglishContext,
    EnglishStyleReference,
)
from checker_app.services.similarity import (
    INSTITUTION_BANDS,
    EnglishBaseline,
    TurkishBaseline,
)


# ------------------------------------------------------------------- baselines
def test_english_theses_measure_three_times_lower_than_turkish_ones() -> None:
    """The correction this whole module exists for."""
    english = EnglishBaseline()
    turkish = TurkishBaseline()
    assert english.mean_percent == pytest.approx(9.0)
    assert english.sd_percent == pytest.approx(6.0)
    assert english.sample_size == 360
    assert turkish.mean_percent == pytest.approx(28.7)
    # Roughly a third, not a marginal difference.
    assert english.mean_percent < turkish.mean_percent / 2.5


def test_only_a_minority_of_english_dissertations_exceed_twenty_five() -> None:
    assert EnglishBaseline().excessive_share == pytest.approx(0.022)


def test_the_measured_cutoff_is_not_the_policy_ceiling() -> None:
    """15% is where detection works; 25-30% is a policy line."""
    baseline = EnglishBaseline()
    assert baseline.optimal_cutoff_percent == pytest.approx(15.0)
    assert baseline.cutoff_sensitivity == pytest.approx(0.848)
    assert baseline.cutoff_specificity == pytest.approx(0.805)
    assert baseline.cutoff_auc == pytest.approx(0.902)
    assert baseline.optimal_cutoff_percent < 25.0


def test_the_distributions_that_separate_plagiarised_from_clean_are_recorded() -> None:
    baseline = EnglishBaseline()
    assert baseline.plagiarised_mean == pytest.approx(25.8)
    assert baseline.clean_mean == pytest.approx(11.5)
    # Restricting to the four narrative sections widens the gap.
    assert baseline.section_only_plagiarised > baseline.plagiarised_mean - 1.0
    assert baseline.section_only_clean < baseline.clean_mean - 5.0


def test_the_reading_reflects_the_measurement_not_a_scolding() -> None:
    """A correctly cited essay scoring 30% with no plagiarism is the school's
    own worked example, so the 25%+ branch must not read as a verdict."""
    baseline = EnglishBaseline()
    assert "normaldir" in baseline.reading(5.0)
    assert "duyarlılık" in baseline.reading(16.0)
    assert "intihal yoktu" in baseline.reading(30.0)


# ----------------------------------------------------------------------- bands
def test_english_bands_carry_their_source_urls() -> None:
    english = [b for b in INSTITUTION_BANDS if b.language == "en"]
    assert len(english) >= 7
    assert all(b.url.startswith("http") for b in english)


def test_bands_are_filtered_to_the_document_language() -> None:
    """A Turkish threshold applied to an English thesis would be worse than
    useless given the 3x difference in measured clean corpora."""
    turkish = [b for b in INSTITUTION_BANDS if b.language == "tr"]
    english = [b for b in INSTITUTION_BANDS if b.language == "en"]
    assert len(turkish) == 5
    assert len(english) >= 7
    assert not ({b.institution for b in turkish} & {b.institution for b in english})


def test_virginia_techs_internal_contradiction_is_recorded() -> None:
    """VT publishes <25% on one page and <15% on another. Citing one number
    would be citing a documentation bug."""
    vt = next(b for b in INSTITUTION_BANDS if "Virginia Tech" in b.institution)
    assert vt.total_incl_quotes == pytest.approx(25.0)
    assert "çelişiyor" in vt.note
    assert "tek sayı" in vt.note


def test_panjab_university_is_the_only_per_section_rule() -> None:
    """Its literature review allows 50%, which is the thesis-relevant point:
    heavy overlap in a literature review is expected, not suspicious."""
    pau = next(b for b in INSTITUTION_BANDS if "Panjab" in b.institution)
    assert "literatür %50" in pau.note
    assert "tam tez %20" in pau.note


def test_institutions_that_refuse_a_threshold_are_rows_not_blanks() -> None:
    """Rhodes and Glasgow Caledonian publish a position *against* thresholds.

    A blank row would read as "we looked and found nothing"; what they actually
    say is stronger and citable, so it is reported as their reason.
    """
    refusals = [b for b in INSTITUTION_BANDS if b.no_threshold_reason]
    assert len(refusals) >= 3
    rhodes = next(b for b in INSTITUTION_BANDS if "Rhodes" in b.institution)
    assert rhodes.total_incl_quotes is None
    assert "reddediyor" in rhodes.no_threshold_reason
    gcu = next(b for b in INSTITUTION_BANDS if "Glasgow" in b.institution)
    assert "şişirdiğini" in gcu.note


# --------------------------------------------------------------- AI prevalence
def test_arxiv_computer_science_carries_twenty_two_percent() -> None:
    presence = EnglishAiPresence()
    assert presence.arxiv_cs_abstracts == pytest.approx(0.225)
    assert presence.arxiv_cs_introductions == pytest.approx(0.196)
    assert presence.sample_size == 1_121_912


def test_mathematics_is_the_outlier_and_needs_its_own_prior() -> None:
    """Abstracts 7.7% and introductions 4.1% - barely above the estimator floor.
    Reading a mathematics thesis against the CS figure inverts the prior."""
    presence = EnglishAiPresence()
    assert presence.arxiv_math_abstracts == pytest.approx(0.077)
    assert presence.arxiv_math_introductions == pytest.approx(0.041)
    assert presence.arxiv_math_introductions < presence.arxiv_math_abstracts
    assert presence.arxiv_math_abstracts < presence.arxiv_cs_abstracts / 2


def test_the_estimator_error_band_is_reported() -> None:
    """Any prevalence figure should carry its method's own error."""
    presence = EnglishAiPresence()
    assert presence.estimator_error_pp == pytest.approx(3.5)
    assert str(presence.estimator_error_pp) in presence.reading(0.225)


def test_presence_reading_refuses_to_call_it_authorship() -> None:
    """It is the share of LLM-*modified* text, which still contains human writing."""
    reading = EnglishAiPresence().reading(0.30)
    assert "LLM ile değiştirilmiş" in reading
    assert "insan metni içerir" in reading
    assert "insan yazımından ayırt edilemiyor" in EnglishAiPresence().reading(0.01)


def test_methods_and_experiments_are_the_least_affected_sections() -> None:
    """Published direction only - the values themselves are not published."""
    presence = EnglishAiPresence()
    assert set(presence.least_affected_sections) == {"methods", "experiments"}
    assert "UNVERIFIED" in presence.caveat


def test_disclosure_almost_never_happens() -> None:
    assert EnglishAiPresence().disclosure_rate == pytest.approx(0.010)


# --------------------------------------------------------------- style reference
def test_results_prose_is_repetitive_by_design() -> None:
    """MTLD 54.6 in Results against 85.2 in Discussion. A low-diversity flag on a
    Results section is weak evidence - and it agrees independently with Liang
    vd.'s finding that methods are the least LLM-modified."""
    style = EnglishStyleReference()
    assert style.section_mtld_median["results"] < style.section_mtld_median["discussion"] / 1.5
    assert "zayıf kanıttır" in style.section_reading("results")


def test_methods_has_the_shortest_sentences() -> None:
    style = EnglishStyleReference()
    lengths = style.section_sentence_length_median
    assert lengths["methods"] == min(lengths.values())
    assert "yazım göstergesi değil" in style.section_reading("methods")


def test_mathematics_is_lexically_different_from_other_fields() -> None:
    """MTLD 60.9 against 94-102. Any English threshold keyed to academic prose
    mis-flags mathematics unless it is field-conditioned."""
    style = EnglishStyleReference()
    mtld = style.by_discipline_mtld
    assert mtld["mathematics"] < 65
    assert all(mtld[name] > 90 for name in mtld if name != "mathematics")


def test_the_l2_row_prevents_a_lexical_false_accusation() -> None:
    """Human L2 writers at CEFR A2 measure MTLD 58 ± 22 against an L1 academic
    median of 93.6. A Turkish student writing English looks lexically poor by L1
    standards and is entirely human."""
    style = EnglishStyleReference()
    assert style.l2_a2_mtld < style.mtld_mean
    note = style.l2_note()
    assert "tamamen insan yazmış olabilir" in note
    assert style.l2_a1_mtld < style.l2_a2_mtld


def test_the_style_reference_declares_it_is_an_original_measurement() -> None:
    """Not a published table. A reader must be able to tell."""
    payload = EnglishStyleReference().to_dict()
    assert "kendi ölçümüdür" in payload["caveat"]
    assert "pre-ChatGPT" in payload["caveat"] or "pre-LLM" in payload["source"]
    assert "pre-LLM" in payload["source"]
    assert payload["sample_size"] == 4_462
    assert payload["section_sample_size"] == 48


# --------------------------------------------------------------- lexical signals
def test_the_llm_word_set_is_measured_not_opinion() -> None:
    assert LLM_WORD_SIGNALS["potential"] == pytest.approx(5.2)
    assert "delve" in LLM_WORD_SIGNALS
    assert "insights" in LLM_WORD_SIGNALS


def test_the_inverse_signal_is_the_more_useful_half() -> None:
    """Thelwall & Kousha measured that LLMs *avoid* thus/moreover, and both are
    high-frequency human academic connectives. So their absence is evidence -
    computable offline with no model."""
    assert "thus" in LLM_AVOIDED_CONNECTIVES
    assert "moreover" in LLM_AVOIDED_CONNECTIVES


# ------------------------------------------------------------------- the context
def test_genre_and_length_change_detector_accuracy() -> None:
    """Turnitin 0.86 humanities vs 0.51 science; 0.87 at 300-330 words vs 0.56 at
    450-550. So a single document-level English score is wrong."""
    context = EnglishContext(presence={}, style={})
    assert "0.86" in context.genre_note and "0.51" in context.genre_note
    assert "0.87" in context.genre_note and "0.56" in context.genre_note
    assert "p=.0149" in context.genre_note


def test_the_l2_bias_is_stated_as_a_calibration_problem_not_a_constant() -> None:
    """Liang 2023 reported 61.4% mean on TOEFL-91; the EACL 2026 replication found
    23.1% on the same set and a *reversed* perplexity direction in Czech. So the
    tool cannot hard-code it."""
    context = EnglishContext(presence={}, style={})
    assert "61.4" in context.replication_note
    assert "23.1" in context.replication_note
    assert "yön bile" in context.replication_note
    assert "sabit bir yanlış pozitif oranı varsaymaz" in context.replication_note


def test_eighty_two_percent_of_confirmed_cases_came_from_non_english_speaking_countries() -> None:
    context = EnglishContext(presence={}, style={})
    assert "%82" in context.l1_l2_note


def test_provenance_names_every_source() -> None:
    for key in ("liang", "kobak", "thelwall", "hadra", "liang2023", "original"):
        assert key in PROVENANCE
        assert PROVENANCE[key]
    assert "10.1038/s41562-025-02273-8" in PROVENANCE["liang"]
    assert "10.1126/sciadv.adt3813" in PROVENANCE["kobak"]