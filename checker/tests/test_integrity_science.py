"""Text-integrity checks must not cry wolf on scientific writing.

Three false positives were found by running this tool on a real 154-page
astrophysics thesis (arXiv:1407.6566) and reading what it complained about:

- **Greek letters counted as homoglyphs.** Eleven findings, all of them ordinary
  notation: ``hν`` for photon energy, ``ǫν`` for a velocity parameter. Each one
  declared every AI score on the document unreliable, which is exactly the kind
  of false alarm that trains a reader to ignore the real ones.
- **Digit-adjacent ellipses counted as punctuation manipulation.** All five
  matches were legitimate: the math range ``M^0.1...0.6`` written twice, and two
  ADS bibliographic codes, ``A&A...534A..120T`` and ``A&A...558A..75T``.
- The document as a whole flipped to ``tampered`` on the strength of those, so a
  60,000-word physics thesis was reported as evidence of quote manipulation.

Homoglyph substitution is a Cyrillic technique - the Cyrillic alphabet has a
lookalike for every Latin letter, and no scholarly reason to use one. Greek is
notation. The rule now says so, and says which of the two it is looking at.
"""

from __future__ import annotations

from checker_app.services.integrity import check_integrity

#: A body of ordinary English, so the only findings are the ones under test.
PROSE = (
    "The data cleaning procedure was performed by two independent reviewers. "
    "Both reviewers examined all 412 records that were retrieved. "
) * 40


def _codes(text: str) -> set[str]:
    return {finding.code for finding in check_integrity(text).findings}


# ------------------------------------------------------------------ homoglyph
def test_cyrillic_inside_a_latin_word_is_a_homoglyph() -> None:
    """The actual attack vector: а p p l e with a Cyrillic а."""
    report = check_integrity("The рареment was distributed to сonneсted units.")
    assert "homoglyph" in {f.code for f in report.findings}
    assert report.tampered, "Kiril karışımı manipülasyon işareti olmalı"


def test_physics_notation_is_not_a_homoglyph() -> None:
    """Eleven of these fired on the real thesis: hν, ǫν, M^α.

    Flagging them declared the whole document's AI scores unreliable.
    """
    report = check_integrity(
        PROSE + "The photon energy is hν and the deceleration parameter is ǫν."
    )
    assert "homoglyph" not in {f.code for f in report.findings}
    assert not report.tampered


def test_greek_notation_is_not_a_finding_at_all() -> None:
    """It was tempting to report it as informational, and measurement killed that.

    arXiv:1911.03731, a machine-learning thesis, has 316 such tokens - hν, λ, θ,
    α, β, μ - which is what a thesis about learning rules looks like. A finding
    with no action attached to it is noise, and noise costs the reader the
    attention that the real findings need.
    """
    report = check_integrity(PROSE + "The photon energy is hν and the parameter ǫν.")
    assert not report.findings, f"gösterim bulgu olmamalı: {report.findings}"
    assert not report.tampered


# ------------------------------------------------------- repeated punctuation
def test_a_mathematical_range_is_not_punctuation_manipulation() -> None:
    """``M^0.1...0.6`` - a range, against a digit on both sides."""
    report = check_integrity(PROSE + "The scaling relation scales as M^0.1...0.6.")
    assert "repeat_punctuation" not in {f.code for f in report.findings}
    assert not report.tampered


def test_a_bibliographic_code_is_not_punctuation_manipulation() -> None:
    """ADS codes: ``A&A...534A..120T``. All five matches on the real thesis."""
    report = check_integrity(PROSE + "Astronomy & Astrophysics, 2011A&A...534A..120T")
    assert "repeat_punctuation" not in {f.code for f in report.findings}
    assert not report.tampered


def test_an_elided_author_list_is_still_reported() -> None:
    """The case that is not about a digit, and the reason the rule survives.

    The finding needs three occurrences, not one - measured on the real thesis,
    where exactly one survived the digit filter and it was this shape.
    """
    elided = "de Hoon, A., Lamer, G., ......., Takey, A. 2010. "
    report = check_integrity(PROSE + elided * 3)
    assert "repeat_punctuation" in {f.code for f in report.findings}


def test_quote_run_manipulation_is_still_reported() -> None:
    report = check_integrity(PROSE + 'He wrote "a very important result" and left.')
    assert "repeat_punctuation" not in {f.code for f in report.findings}
    report = check_integrity(PROSE + 'The claim is "..." that nobody checked. ' * 3)
    assert "repeat_punctuation" in {f.code for f in report.findings}


def test_display_math_ellipses_are_not_punctuation_manipulation() -> None:
    """Sixty-four instances on arXiv:1911.03731, every one of them a formula.

    Set-builder notation and the vertical ellipsis of a matrix, which a text
    extractor renders as "... . . . ...".
    """
    report = check_integrity(
        PROSE
        + "A training set z = {(x1, h'(x1)), ..., (xm, h'(xm))} over Z: "
        + "z11 ... z1n ... ... ... zm1 ... zmn and dP1(z1)...dPn(zn) = 1. "
    )
    assert "repeat_punctuation" not in {f.code for f in report.findings}
    assert not report.tampered


def test_repeated_punctuation_cannot_alone_invalidate_the_ai_scores() -> None:
    """A signal with no calibration must not move a verdict.

    Three real theses, two fields, every match legitimate - elided author lists,
    ADS codes, functional-calculus integrals - producing 3, 5 and 10 occurrences.
    Not one real instance of quote manipulation was found to calibrate against, so
    the observation still appears but it no longer declares a thesis tampered.
    """
    elided = "de Hoon, A., Lamer, G., ......., Takey, A. 2010. "
    bibcode = "Astronomy & Astrophysics, 2011A&A...534A..120T "
    integral = "the action S(φ1) +... +iS(φn) and ∫[Dφ1...Dφn] exp(iS) "

    seen = check_integrity(PROSE + elided * 3 + bibcode * 3 + integral * 3)
    assert "repeat_punctuation" in {f.code for f in seen.findings}, "gözlem görünmeli"
    assert not seen.tampered, "ama tek başına hüküm döndürmemeli"


def test_homoglyph_still_can_invalidate() -> None:
    """The other direction: the calibrated signal keeps its authority."""
    assert check_integrity(PROSE + "The рареment was sent. ").tampered


def test_a_truncated_document_says_so() -> None:
    """A truncated document is a different finding, not a smaller one.

    Measured on arXiv:1912.04141, a 191-page thesis of 466,068 characters: at the
    old 400,000 limit it was silently cut, and what was cut was the end - the
    bibliography - so the reference audit reported 19 entries where there are 417.
    """
    from checker_app.config import CheckerSettings
    from checker_app.services.runner import ScanRequest, ScanRunner

    # get_settings() is cached and shared, so this test builds its own settings
    # rather than mutating the shared instance.
    settings = CheckerSettings(max_chars=2_000)
    body = "Birinci cümle burada ve oldukça uzun bir cümledir. " * 200
    report = ScanRunner(settings).scan(
        ScanRequest(path="memory.md", text=body, use_perplexity=False, use_classifier=False)
    )
    assert any("text_truncated" in signal for signal in report.degraded_signals), (
        f"kırpma bildirilmedi: {report.degraded_signals}"
    )


def test_the_default_limit_holds_a_real_thesis() -> None:
    """Five real theses run 240,834 to 466,068 characters."""
    from checker_app.config import CheckerSettings

    assert CheckerSettings.model_fields["max_chars"].default >= 466_068


def test_a_maths_operator_is_not_an_emoji() -> None:
    """U+2700-U+27BF is decoration, which is exactly why TeX put the box there.

    Measured on arXiv:q-alg/9607022, a 1996 habilitation thesis: the Klein-Gordon
    d'Alembertian came through as U+2737 and the check reported **231** emoji,
    one for every occurrence of the symbol in "(□ + m²)φ = 0" - the equation the
    chapter is about.
    """
    box = "✷"  # U+2737, the box operator in a TeX font's Dingbats slot
    report = check_integrity(PROSE + f"(□ +m2)φ = 0 ⇒ ({box} +m2)φ = 0." + PROSE)
    assert not [f for f in report.findings if f.code == "emoji"], report.findings
    assert not report.tampered


def test_a_real_emoji_is_still_noticed() -> None:
    """The narrowed range must not become "never report an emoji"."""
    rocket = "🚀"
    report = check_integrity(f"Bu çalışmada {rocket} sonuçlar elde edildi ve tartışıldı.")
    assert "emoji" in {f.code for f in report.findings}


def test_path_integral_ellipses_are_not_punctuation_manipulation() -> None:
    """Maths, and the rule that had to learn to leave it alone."""
    report = check_integrity(
        PROSE + "G(x1,...,xn) = < 0|T [φ(y1)...φ(xm)]|0> with ∫d4y1...d4ym over R^n."
    )
    assert "repeat_punctuation" not in {f.code for f in report.findings}
    assert not report.tampered


def test_a_signature_dot_line_is_not_punctuation_manipulation() -> None:
    """Measured on arXiv:1702.04123: a declaration page carrying thirty-six dots
    for a date to be signed. Long, but it is a form, not a quotation."""
    report = check_integrity(
        PROSE + "December 12, 2016 .................................... ..........."
    )
    assert not report.tampered


def test_an_entry_spanning_a_page_break_is_not_cut_in_half() -> None:
    """A three-character gap is a page break, not a record boundary.

    Measured on arXiv:q-alg/9607022: PDF text is emitted as "\\n\\f\\n" between
    pages, and "B.W. Lee, in Methods in Field Theory, ed. R." / "Delbourgo, D.
    Kreimer, Phys.Lett.B366 (1996) 421" are one entry. Split there, the year sits
    in the half that gets thrown away.
    """
    from checker_app.services.references import _PAGE_BREAK_GAP

    assert _PAGE_BREAK_GAP == 3


def test_older_pdfs_space_their_digits() -> None:
    """"Phys.Lett.B366 (1 996) 421" is a 1996 paper, not a yearless one."""
    from checker_app.services.references import (
        _IDENTIFIER_RE,
        _SPLIT_DIGITS_RE,
        _YEAR_RE,
    )

    text = "Delbourgo, D. Kreimer, Phys.Lett.B366 (1 996) 421."
    prose = _IDENTIFIER_RE.sub(" ", text)
    assert _YEAR_RE.findall(prose) == []
    joined = _SPLIT_DIGITS_RE.sub("", prose)
    assert _YEAR_RE.findall(joined)[-1] == "1996"
    # Safe for a real page range: rejoined it is eight digits, and no four-digit
    # pattern matches that.
    assert _YEAR_RE.findall(_SPLIT_DIGITS_RE.sub("", "pp. 1415 1443, 1999.")) == ["1999"]


# ------------------------------------------------------------- the whole doc
def test_a_physics_thesis_is_not_declared_tampered() -> None:
    """The end-to-end property: no false alarm on 60,000 words of science."""
    text = (
        PROSE
        + "The photon energy is hν and the deceleration parameter is ǫν. "
        + "The scaling relation scales as M^0.1...0.6. "
        + "Astronomy & Astrophysics, 2011A&A...534A..120T "
    ) * 10
    report = check_integrity(text)
    assert not report.tampered, "bilimsel metin manipüle sayılmamalı"
    assert "homoglyph" not in {f.code for f in report.findings}


def test_the_distinction_is_enforced_not_merely_documented() -> None:
    """Both alphabets in one sentence, and only the Cyrillic one is flagged."""
    report = check_integrity(
        PROSE + "The energy is hν and the рареment was distributed to сonneсted units."
    )
    codes = {f.code for f in report.findings}
    assert "homoglyph" in codes
    assert report.tampered

    homoglyph = next(f for f in report.findings if f.code == "homoglyph")
    assert "Kiril" in homoglyph.detail, "hangi alfabenin arandığı belirtilmeli"
    assert "hν" not in homoglyph.examples, "gösterim homoglif sayılmamalı"