"""Thesis structure: section roles drive scoring, so they must be right."""

from __future__ import annotations

import pytest

from checker_app.domain.enums import BlockType
from checker_app.domain.sections import (
    SectionRole,
    classify_section,
    is_boilerplate,
    role_label,
    role_prior,
)
from checker_app.domain.segmentation import SentenceSplitter

CASES = [
    (["Giriş"], SectionRole.INTRODUCTION),
    (["1. GİRİŞ VE AMAÇ"], SectionRole.INTRODUCTION),
    (["BÖLÜM 3: YÖNTEM"], SectionRole.METHOD),
    (["Yöntem ve Deneyler"], SectionRole.METHOD),
    (["3. İlgili Çalışmalar"], SectionRole.RELATED_WORK),
    (["Literatür Taraması"], SectionRole.RELATED_WORK),
    (["Veri Seti"], SectionRole.DATA),
    (["4. Bulgular ve Tartışma"], SectionRole.RESULTS),
    (["BULGULAR"], SectionRole.RESULTS),
    (["Tartışma"], SectionRole.DISCUSSION),
    (["5. Sonuç ve Öneriler"], SectionRole.CONCLUSION),
    (["SONUÇ"], SectionRole.CONCLUSION),
    (["Özet"], SectionRole.ABSTRACT),
    (["TÜRKÇE ÖZET"], SectionRole.ABSTRACT),
    (["Abstract"], SectionRole.ABSTRACT),
    (["Kaynakça"], SectionRole.REFERENCES),
    (["KAYNAKÇA"], SectionRole.REFERENCES),
    (["Kaynaklar"], SectionRole.REFERENCES),
    (["References"], SectionRole.REFERENCES),
    (["Teşekkür"], SectionRole.ACKNOWLEDGMENTS),
    (["İçindekiler"], SectionRole.FRONT_MATTER),
    (["Kısaltmalar"], SectionRole.FRONT_MATTER),
    (["Önsöz"], SectionRole.FRONT_MATTER),
    (["Ek A"], SectionRole.APPENDIX),
    (["Ekler"], SectionRole.APPENDIX),
    (["2. Yöntem"], SectionRole.METHOD),
    (["3. Bulgular"], SectionRole.RESULTS),
]


@pytest.mark.parametrize("path,expected", CASES)
def test_turkish_and_english_headings(path: list[str], expected: SectionRole) -> None:
    assert classify_section(path) is expected


def test_deepest_heading_wins() -> None:
    assert classify_section(["Bölüm 2", "Yöntem"]) is SectionRole.METHOD


def test_unknown_heading_is_other_but_not_an_error() -> None:
    assert classify_section(["Elektrot Ölçümleri"]) is SectionRole.OTHER
    assert classify_section([]) is SectionRole.OTHER
    # "Deney ..." is a method heading, deliberately.
    assert classify_section(["Deney 7 - Elektrot"]) is SectionRole.METHOD


def test_references_are_not_scored_and_not_matched() -> None:
    assert classify_section(["Kaynakça"]).matched is False
    assert classify_section(["Kaynakça"]).scored is False
    assert is_boilerplate(SectionRole.ACKNOWLEDGMENTS)
    assert not is_boilerplate(SectionRole.METHOD)


def test_abstract_has_the_highest_prior_and_data_the_lowest() -> None:
    assert role_prior(SectionRole.ABSTRACT) > role_prior(SectionRole.METHOD)
    assert role_prior(SectionRole.DATA) < 0
    assert role_prior(SectionRole.OTHER) == 0.0


def test_role_labels_are_turkish() -> None:
    assert role_label(SectionRole.REFERENCES) == "Kaynakça"
    assert role_label(SectionRole.OTHER) == "Diğer"


def test_splitter_attaches_the_role_to_every_sentence() -> None:
    text = (
        "# Özet\n\nBu çalışmada konu şudur.\n\n"
        "# Yöntem\n\nŞu yöntem kullanılmıştır.\n\n"
        "# Kaynakça\n\nYılmaz, A. (2020). Bir çalışma."
    )
    sentences = SentenceSplitter().split(text).sentences
    roles = {s.location.section_role for s in sentences if s.location.block_type is BlockType.PROSE}
    assert SectionRole.ABSTRACT in roles
    assert SectionRole.METHOD in roles
    assert SectionRole.REFERENCES in roles


def test_numbered_heading_is_not_a_sentence_boundary_problem() -> None:
    text = "# 1. GİRİŞ VE AMAÇ\n\nBu çalışmanın amacı şudur ve iki bölümden oluşmaktadır."
    result = SentenceSplitter().split(text)
    assert result.sentences[0].location.section_role is SectionRole.INTRODUCTION
