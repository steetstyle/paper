"""Turkish text out of a PDF, where the diacritics arrive detached.

This is the tool's primary use case, and the primary input path was broken for
it. Measured on a real Turkish thesis - Marmara University, ModernBERT, 160,040
characters - the extractor returned:

    T.C. MARMARA ¨UN˙IVERS˙ITES˙I M ¨UHEND˙ISL˙IK FAK¨ULTES˙I

Six thousand six hundred and seventy-two *spacing* diacritics, in place of the
combining marks, each on its own before or after its letter. Of the six Turkish
letters that carry a diacritic, the extractor returned **zero** occurrences of
Ğ, İ, Ş, ş and ğ. Every Turkish-specific measurement in this tool - the lexicon
rules, the richness table, the Turkish similarity baseline - was running on text
in which most Turkish letters simply did not exist.

The repair is only allowed to act where Unicode itself vouches for the result:
a spacing mark is turned into a combining one and kept only if NFC spells the
pair as a single code point.
"""

from __future__ import annotations

from checker_app.services.sources import repair_split_diacritics


# ------------------------------------------------------------------ the cases
def test_a_mark_before_its_letter_is_recomposed() -> None:
    """``bas ¸arımı`` -> ``başarımı``: the cedilla trails its base."""
    assert repair_split_diacritics("bas ¸arımı") == "başarımı"


def test_a_mark_after_its_letter_is_recomposed() -> None:
    """``E˘gitim`` -> ``Eğitim``: the breve leads its base. Both orders occur."""
    assert repair_split_diacritics("E˘gitim") == "Eğitim"


def test_the_mark_binds_to_the_letter_the_document_put_it_next_to() -> None:
    """Both sides usually compose, so the side is a decision, not a detail.

    "E" + breve is Ĕ and "g" + breve is ğ, both real letters. Reading the breve as
    belonging to what precedes it turns "Eğitim" into "Ĕgitim"; reading the caron
    that way turns "Škoda" into "Sǩoda". The map records which side each mark
    arrives on, measured from the thesis.
    """
    assert repair_split_diacritics("E˘gitim") == "Eğitim"  # breve follows its g
    assert repair_split_diacritics("Sˇkoda") == "Škoda"  # caron trails its S
    assert repair_split_diacritics("C¸C˙IFT") == "ÇİFT"  # cedilla and dot lead


def test_every_turkish_letter_survives() -> None:
    cases = {
        "C¸": "Ç",
        "¸c": "ç",
        "S¸": "Ş",
        "¸s": "ş",
        "G˘": "Ğ",
        "˘g": "ğ",
        "I˙": "İ",
        "¨U": "Ü",
        "¨u": "ü",
        "¨O": "Ö",
        "¨o": "ö",
    }
    for broken, expected in cases.items():
        assert repair_split_diacritics(broken) == expected, broken


def test_the_base_letter_is_emitted_twice_and_only_once_survives() -> None:
    """``ÇİFT`` arrives as C, cedilla, C, dot, I, F, T."""
    assert repair_split_diacritics("C¸C˙IFT") == "ÇİFT"


def test_the_real_word_is_recovered_not_the_letter_count() -> None:
    before = "T.C. MARMARA ¨UN˙IVERS˙ITES˙I M ¨UHEND˙ISL˙IK FAK¨ULTES˙I"
    assert repair_split_diacritics(before) == "T.C. MARMARAÜNİVERSİTESİ MÜHENDİSLİK FAKÜLTESİ"


# ----------------------------------------------------------------- the guards
def test_real_symbols_are_never_touched() -> None:
    """The middle dot and the bullet look like a cedilla and are not one."""
    for text in (
        "Kaynakça · Şekil 3 · madde",
        "Sonuçlar • Giriş • Tartışma",
        "x = a · b",
        "Bir başlık: Özet — Giriş",
    ):
        assert repair_split_diacritics(text) == text, text


def test_a_mark_with_no_letter_is_left_alone() -> None:
    """A stray accent has nothing to compose with."""
    assert repair_split_diacritics("bir ¨ iki") == "bir ¨ iki"
    assert repair_split_diacritics("sayı 5 ¨ 7") == "sayı 5 ¨ 7"


def test_text_without_any_mark_is_returned_unchanged() -> None:
    text = "ModernBERT is a recent encoder with 8192 context length."
    assert repair_split_diacritics(text) == text
    assert repair_split_diacritics("") == ""


def test_non_turkish_compositions_still_work() -> None:
    """The rule is Unicode's, not a Turkish special case."""
    # Czech and Slovak letters compose exactly as Turkish ones do, and the caron
    # follows the letter it trails - which is the side the document puts it on.
    assert repair_split_diacritics("Sˇkoda") == "Škoda"
    assert repair_split_diacritics("Pˇraha") == "Přaha"


def test_the_whitespace_decision_was_measured() -> None:
    """Both readings look identical in the text; the word count decided it.

    Run over the whole thesis and compared against the word count of its own
    LaTeX sources: keeping the space gave 23,964 words, deleting it 20,715, and
    the sources are about 17,590 before tables and captions. Deleting is closer,
    and the residue - a fused word like "MARMARAÜNİVERSİTESİ" - is visible in
    the extracted text rather than silent.
    """
    # The repaired thesis text lands near the "delete" figure, not the "keep" one.
    fused = repair_split_diacritics("MARMARA ¨UN˙IVERS˙ITES˙I")
    assert fused == "MARMARAÜNİVERSİTESİ"
    assert " " not in fused