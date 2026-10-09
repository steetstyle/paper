"""Tokenisation, folding and shingles."""

from __future__ import annotations

from checker_app.domain.text import fold_text, shingles, tokenize, word_count


def test_turkish_case_folding() -> None:
    """Dotted/dotless I: "I" is "ı", "İ" is "i" - the opposite of str.lower()."""
    assert fold_text("IŞIK", strip_diacritics=False) == "ışık"
    assert fold_text("İstanbul", strip_diacritics=False) == "istanbul"
    assert fold_text("İZMİR", strip_diacritics=False) == "izmir"


def test_diacritics_and_digit_folding() -> None:
    assert fold_text("Göztepe", strip_diacritics=True) == "goztepe"
    # Note: "ı" survives diacritic stripping (it is not an accent, it is a
    # different letter), while "â" -> "a".
    assert fold_text("2024 yılı", normalize_digits=True) == "0000 yılı"


def test_tokenize_offsets_and_is_number() -> None:
    text = "Değer 3,14 ve 1.000,50 oldukça yüksek."
    tokens = tokenize(text)
    assert [t.text for t in tokens] == ["Değer", "3,14", "ve", "1.000,50", "oldukça", "yüksek"]
    for token in tokens:
        assert text[token.char_start : token.char_end] == token.text
    assert tokens[1].is_number and tokens[3].is_number
    assert not tokens[0].is_number


def test_tokenize_offset_shift_and_index_continuity() -> None:
    tokens = tokenize("ikinci kısım", offset=100, start_index=5)
    assert tokens[0].char_start == 100
    assert tokens[0].index == 5


def test_apostrophes_stay_inside_words() -> None:
    tokens = tokenize("Türkiye'nin başkenti Ankara'dır.")
    assert [t.text for t in tokens] == ["Türkiye'nin", "başkenti", "Ankara'dır"]


def test_shingles_slide_over_the_token_stream() -> None:
    tokens = tokenize("bir iki üç dört beş", strip_diacritics=False)
    grams = list(shingles(tokens, 3))
    assert len(grams) == 3
    assert grams[0][0] == 0
    assert grams[0][1] == ("bir", "iki", "üç")
    assert grams[2][0] == 2
    assert list(shingles(tokens, 99)) == []


def test_word_count() -> None:
    assert word_count("  bir  iki \n üç ") == 3
