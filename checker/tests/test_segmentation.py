"""Segmentation: the guarantee is that offsets are exact and structure is kept."""

from __future__ import annotations

from checker_app.domain.enums import BlockType
from checker_app.domain.segmentation import SentenceSplitter
from checker_app.domain.text import language_confidence


def texts_of(result) -> list[str]:
    return [s.text for s in result.sentences]


# --------------------------------------------------------------------- offsets
def test_offsets_are_exact() -> None:
    text = "Merhaba dünya. Bu ikinci cümledir! Üçüncü?\n\nYeni paragraf."
    result = SentenceSplitter().split(text)
    for sentence in result.sentences:
        assert text[sentence.location.char_start : sentence.location.char_end] == sentence.text


def test_line_and_paragraph_locations() -> None:
    text = "Birinci satır burada. İkinci cümle aynı satırda.\n\nYeni paragraf cümlesi."
    result = SentenceSplitter().split(text)
    first, second, third = result.sentences
    assert first.location.line_start == 1
    assert second.location.line_start == 1
    assert third.location.line_start == 3
    assert third.location.paragraph_index == 2


def test_formfeed_is_a_page_break() -> None:
    text = "Sayfa bir cümlesi.\fSayfa iki cümlesi."
    result = SentenceSplitter().split(text)
    assert [s.location.page for s in result.sentences] == [1, 2]


def test_token_stream_is_contiguous(tr_result) -> None:
    tokens = tr_result.tokens
    for previous, current in zip(tokens, tokens[1:], strict=False):
        assert previous.char_end <= current.char_start
    for sentence in tr_result.sentences:
        assert tokens[sentence.token_start].char_start >= sentence.location.char_start
        assert tokens[sentence.token_end - 1].char_end <= sentence.location.char_end


# --------------------------------------------------------------------- guards
def test_abbreviations_do_not_split() -> None:
    text = "Doğan vb. sonuçları tartışmıştır. Prof. Dr. Ahmet Yılmaz (2020) farklı ilerlemiştir."
    result = SentenceSplitter().split(text)
    assert len(result.sentences) == 2


def test_initials_do_not_split() -> None:
    text = "However, J. Smith argued otherwise. The rest of the sentence follows here."
    result = SentenceSplitter().split(text)
    assert len(result.sentences) == 2


def test_decimals_and_urls_do_not_split() -> None:
    text = "Değer 3,14 ve 1.000,50 olarak verildi. Kaynak https://arxiv.org/abs/1706.03762 adresindedir."
    result = SentenceSplitter().split(text)
    assert len(result.sentences) == 2


def test_ellipsis_and_mixed_terminators() -> None:
    text = "Çalışma üç aşamadan oluşmaktadır... Birincisi veri toplamadır! İkincisi eğitimdir?"
    result = SentenceSplitter().split(text)
    assert len(result.sentences) == 3


def test_quote_stays_together_unless_configured() -> None:
    text = 'Bize göre, "bu cümle tırnak içindedir." Sonraki cümle ayrı kalır.'
    kept = SentenceSplitter().split(text)
    assert len(kept.sentences) == 1

    from checker_app.config import SegmentationSettings

    split = SentenceSplitter(SegmentationSettings(split_inside_quotes=True)).split(text)
    assert len(split.sentences) == 2


# ------------------------------------------------------------------- structure
def test_headings_form_a_section_trail() -> None:
    text = "# Giriş\n\nBirinci cümle burada.\n\n## Yöntem\n\nİkinci cümle burada."
    result = SentenceSplitter().split(text)
    heading = result.sentences[0]
    assert heading.location.block_type is BlockType.HEADING
    assert heading.location.section_path == ("Giriş",)
    assert result.sentences[2].location.section_path == ("Giriş", "Yöntem")


def test_list_markers_are_stripped() -> None:
    text = "- Birinci madde burada ve oldukça uzun bir cümledir.\n- İkinci madde daha kısa."
    result = SentenceSplitter().split(text)
    assert all(not s.text.startswith("-") for s in result.sentences)
    assert all(s.location.block_type is BlockType.LIST_ITEM for s in result.sentences)


def test_code_block_is_one_unit() -> None:
    text = (
        "Açıklama cümlesi burada.\n\n```python\ndef f(x):\n    return x + 1\n```\n\nSonraki cümle."
    )
    result = SentenceSplitter().split(text)
    kinds = [s.location.block_type for s in result.sentences]
    assert BlockType.CODE in kinds
    code = next(s for s in result.sentences if s.location.block_type is BlockType.CODE)
    assert code.text.startswith("```python")
    assert "def f(x)" in code.text


def test_short_fragments_merge_into_previous() -> None:
    text = "Bu cümle oldukça uzun ve anlamlı bir açıklama içeriyor. Not: kısa."
    result = SentenceSplitter().split(text)
    assert len(result.sentences) == 1


# -------------------------------------------------------------------- language
def test_language_detection_uses_plain_lowercase() -> None:
    english = "I spent two hours in the lab and the results were not what I expected at all."
    assert language_confidence(english)[0] == "en"

    turkish = "Dün akşam laboratuvarda oturup veriye baktım ve sonuçlar beklediğim gibi çıkmadı."
    assert language_confidence(turkish)[0] == "tr"


def test_empty_document() -> None:
    result = SentenceSplitter().split("   \n\n  ")
    assert result.sentences == ()
    assert result.page_count == 1
