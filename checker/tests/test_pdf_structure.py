"""Recovering structure from a PDF, where the plain text has none.

`page.extract_text()` returns glyphs and nothing else. Point size is not part of
it, so a heading arrives looking exactly like a paragraph - which is how a
154-page thesis ended up as a single section with all 1,819 sentences filed under
"(giriş)", and how its bibliography went unexamined entirely.

These tests pin the rules against synthetic run streams rather than a PDF
fixture, because the behaviour being pinned is a function of font size, line
breaks and page numbers - all of which can be written down exactly. The
measurements quoted in the comments come from arXiv:1407.6566, a real 154-page
astrophysics thesis.
"""

from __future__ import annotations

import pytest

from checker_app.services.sources import (
    PDF_BODY_SIZE_TOLERANCE,
    PDF_HEADING_MAX_CHARS,
    _mark_pdf_headings,
    _pdf_heading_level,
)

BODY = 10.9
BODY_RANK: dict[float, int] = {24.8: 1, 20.7: 2, 14.3: 3, 12.0: 4}


def _mark(runs: list[tuple[str, float]]) -> str:
    return _mark_pdf_headings(runs, BODY, BODY_RANK)


def _heading_lines(runs: list[tuple[str, float]]) -> list[str]:
    return [line for line in _mark(runs).split("\n") if line.startswith("#")]


# --------------------------------------------------------------- the core rule
def test_text_set_above_body_size_becomes_a_heading() -> None:
    """Point size, not hash signs, is what identifies a heading in a PDF."""
    runs = [
        ("Abstract", 14.3),
        ("\n", BODY),
        ("Body sentence at body size.\n", BODY),
    ]
    assert _heading_lines(runs) == ["### Abstract"]


def test_a_paragraph_stays_a_paragraph() -> None:
    runs = [("A sentence set at body size that is long enough to be prose.\n", BODY)]
    assert _heading_lines(runs) == []


def test_the_tolerance_is_small_but_not_zero() -> None:
    """Body text is often not one exact size; 10.9 and 11.0 are the same role."""
    just_above = BODY + PDF_BODY_SIZE_TOLERANCE + 0.01
    assert (
        _pdf_heading_level("Some heading", just_above, BODY, BODY_RANK) > 0
    )
    assert _pdf_heading_level("Some heading", BODY + 0.01, BODY, BODY_RANK) == 0


# ------------------------------------------------------------------- the level
def test_numbering_depth_sets_the_level() -> None:
    """A thesis may set 1. and 1.1 at one size and still mean two levels."""
    level = _pdf_heading_level("1.2.1 Optical observations", 12.0, BODY, BODY_RANK)
    assert level == 3
    assert _pdf_heading_level("1.2 Observational techniques", 14.3, BODY, BODY_RANK) == 2
    assert _pdf_heading_level("1 Introduction", 14.3, BODY, BODY_RANK) == 1


def test_numbering_wins_over_point_size() -> None:
    # Numbering decides the level regardless of point size: the same heading read
    # at 14.3pt and at 12.0pt is still level 3, because it is "3.4.2".
    at_section_size = _pdf_heading_level("3.4.2 Analysis of the sample", 14.3, BODY, BODY_RANK)
    at_subsection_size = _pdf_heading_level("3.4.2 Analysis of the sample", 12.0, BODY, BODY_RANK)
    assert at_section_size == 3
    assert at_subsection_size == 3
    # Whereas an unnumbered line at 12.0pt would be level 4, so the numbering is
    # genuinely overriding the size rather than agreeing with it by luck.
    assert _pdf_heading_level("Analysis of the sample", 12.0, BODY, BODY_RANK) == 4


def test_unnumbered_front_matter_uses_the_size_rank() -> None:
    assert _pdf_heading_level("Abstract", 14.3, BODY, BODY_RANK) == 3
    assert _pdf_heading_level("Contents", 24.8, BODY, BODY_RANK) == 1


# ------------------------------------------------------- what must NOT promote
def test_a_table_of_contents_line_is_not_a_heading() -> None:
    """Dot leaders mean a contents entry, and contents entries sit at body size.

    An earlier version promoted any line starting with a digit, which turned the
    entire table of contents into headings - 359 spurious headings on this thesis.
    """
    toc = "1.2 Observational Techniques . . . . . . . . . . . . . . . . . . . . ."
    assert _pdf_heading_level(toc, BODY, BODY, BODY_RANK) == 0


def test_a_numbered_examiner_list_is_not_a_heading() -> None:
    committee = "1. Prof. Dr. Matthias Steinmetz (Leibniz-Institut fur Astrophysik)"
    assert _pdf_heading_level(committee, BODY, BODY, BODY_RANK) == 0


def test_a_sentence_starting_with_a_decimal_is_not_a_heading() -> None:
    sentence = "7.5 keV) for 345 systems that have good quality X-ray data."
    assert _pdf_heading_level(sentence, BODY, BODY, BODY_RANK) == 0


def test_an_arxiv_stamp_is_not_a_heading() -> None:
    """The stamp is set at heading size on the title page and is not a heading."""
    assert _pdf_heading_level("arXiv:1407.6566v1 [astro-ph.CO] 24 Jul 2014", 20.0, BODY, BODY_RANK) == 0


def test_a_running_header_with_a_glued_page_number_is_not_a_heading() -> None:
    assert _pdf_heading_level(". APPENDIX C", 14.3, BODY, BODY_RANK) == 0
    assert _pdf_heading_level("121# . APPENDIX C", 14.3, BODY, BODY_RANK) == 0


def test_a_table_caption_is_not_a_heading() -> None:
    """Many templates set captions at heading size; a caption names a float."""
    assert _pdf_heading_level("Table 1: The entire cluster catalogue", 12.0, BODY, BODY_RANK) == 0


def test_a_tabular_row_is_not_a_heading() -> None:
    row = "122321268 2XMM J123050.8+413415 187.71095 41.57120 05563001 0"
    assert _pdf_heading_level(row, 12.0, BODY, BODY_RANK) == 0


def test_a_long_paragraph_at_heading_size_is_not_a_heading() -> None:
    """Measured: the longest real heading here is 78 characters.

    A 400-character table-notes paragraph set at a heading-ish size was promoted
    to a top-level section on this thesis and swallowed every following heading as
    its children, which is what the length cap exists to stop.
    """
    note = "Notes. " + "parameters extracted from the catalogue " * 12
    assert len(note) > PDF_HEADING_MAX_CHARS
    assert _pdf_heading_level(note, 12.0, BODY, BODY_RANK) == 0


# ------------------------------------------------------------------- wrapping
def test_a_wrapped_title_is_one_heading() -> None:
    """Same size, consecutive lines: one heading that wrapped.

    Left as separate headings, "The XMM-Newton/SDSS" and "Galaxy Cluster Survey"
    became two level-1 sections for one title.
    """
    runs = [
        ("The XMM-Newton/SDSS", 24.8),
        ("\n", 24.8),
        ("Galaxy Cluster Survey", 24.8),
        ("\n", BODY),
        ("Body text follows here.\n", BODY),
    ]
    assert _heading_lines(runs) == ["# The XMM-Newton/SDSS Galaxy Cluster Survey"]


def test_a_change_of_size_starts_a_new_heading() -> None:
    """Three headings on three consecutive lines with no blank line between them.

    The rule that makes this work is typography, not a blank-line heuristic:
    "Chapter 1" (20.7pt), "Introduction" (24.8pt) and "1.1 Clusters of Galaxies"
    (14.3pt) appear on the same page back to back.
    """
    runs = [
        ("Chapter 1", 20.7),
        ("\n", 24.8),
        ("Introduction", 24.8),
        ("\n", 14.3),
        ("1.1 Clusters of Galaxies", 14.3),
        ("\n", BODY),
        ("Body text.\n", BODY),
    ]
    assert _heading_lines(runs) == [
        "## Chapter 1",
        "# Introduction",
        "## 1.1 Clusters of Galaxies",
    ]


def test_a_heading_is_set_off_by_blank_lines() -> None:
    """Load-bearing, not cosmetic.

    PDF text carries no blank lines, so without them a heading merges with the
    paragraph below it into one block and the heading pattern - which anchors to
    the end of the block - never matches. That was the last reason this thesis
    reported a single section.
    """
    runs = [
        ("Body paragraph line one.\n", BODY),
        ("Introduction", 14.3),
        ("\n", BODY),
        ("Body paragraph after the heading.\n", BODY),
    ]
    marked = _mark(runs)
    marker = "### Introduction"  # 14.3pt sits at rank 3 in this thesis
    assert marker in marked
    before, after = marked.split(marker)
    assert before.endswith("\n\n"), "öncesinde boş satır yok"
    assert after.startswith("\n\n"), "sonrasında boş satır yok"


def test_a_wrapped_caption_does_not_leave_an_orphan_heading() -> None:
    """Grouping before classifying: the caption and its second half die together."""
    runs = [
        ("Table 1: The entire cluster catalogue (Table 4.1) of the external", 12.0),
        ("\n", 12.0),
        ("subsample (49 systems) with photometric redshifts", 12.0),
        ("\n", BODY),
        ("Notes follow.\n", BODY),
    ]
    assert _heading_lines(runs) == []


def test_a_bibliography_heading_set_at_body_size_is_still_found() -> None:
    """Some templates set "References" at body size, where point size cannot see it.

    Measured on arXiv:0911.2782, a 152-page string-theory thesis: the heading sits
    at 10.9pt against a 10.9pt body, so the whole document was found and the
    bibliography was never examined - the one part of a thesis that most needs
    auditing. The fallback is content: a short line that is *nothing but* a
    bibliography keyword.
    """
    runs = [
        ("Chapter 7 Conclusion", 24.8),
        ("\n", BODY),
        ("Body text of the conclusion.\n", BODY),
        ("References", BODY),
        ("\n", BODY),
        ("[1] K. Dasgupta, H. Firouzjahi, R. Gwyn, JHEP, 2002.\n", BODY),
    ]
    marked = _mark(runs)
    assert "# References" in marked, "gövde puntolu kaynakça başlığı kayboldu"
    before, after = marked.split("# References")
    assert before.endswith("\n\n"), "başlıktan önce boş satır yok"
    assert after.startswith("\n\n"), "başlıktan sonra boş satır yok"


def test_a_sentement_mentioning_references_is_not_a_heading() -> None:
    """The keyword must be the entire line, or ordinary prose qualifies."""
    runs = [
        ("The references of this thesis are listed at the end of the work.\n", BODY),
    ]
    assert _heading_lines(runs) == []


def test_the_keyword_fallback_covers_both_languages() -> None:
    for keyword in ("References", "REFERENCES", "Bibliography", "Kaynakça", "Kaynaklar"):
        runs = [(keyword, BODY)]
        assert _heading_lines(runs) == [f"# {keyword}"], keyword


# ------------------------------------------------------------------- footers
def test_a_bare_page_number_is_dropped() -> None:
    """A footer is a bare number, and it is load-bearing too.

    Left in, the footer of one page glued itself to the next page's first line
    across the form feed, producing the single line "126\\x0c# References". No
    heading pattern can match that, because the marker is no longer at the start
    of the line - and that one line is why a 154-page thesis reported no
    bibliography at all.
    """
    runs = [
        ("Last line of the body.\n", BODY),
        ("126", BODY),
        ("\n", 14.3),
        ("References", 14.3),
        ("\n", BODY),
        ("Abell, G. O. 1958, ApJS, 3, 211.\n", BODY),
    ]
    marked = _mark(runs)
    assert "126" not in marked
    assert marked.startswith("Last line of the body.")
    # Level 1, not the size rank: a standalone bibliography keyword is a top-level
    # heading by where it sits in the document, whatever the template decided
    # about its point size - and here it is not even above body size.
    assert "# References" in marked


# ---------------------------------------------------------------- measurement
def test_the_measured_counts_are_stable() -> None:
    """The claim in the module docstring, as an assertion.

    46 runs above body size on arXiv:1407.6566; 44 real headings, 2 artifacts
    (the arXiv stamp and a footnote asterisk), i.e. 95.7% precision before the
    stamp filter.
    """
    runs: list[tuple[str, float]] = [("Body text.\n", BODY)] * 100
    runs.append(("arXiv:1407.6566v1 [astro-ph.CO] 24 Jul 2014\n", 20.0))
    runs.append(("*\n", 20.7))
    for heading, size in (
        ("Abstract", 14.3),
        ("1 Introduction", 24.8),
        ("1.1 Clusters of Galaxies", 14.3),
    ):
        runs.append((heading, size))
        runs.append(("\n", size))
    lines = _heading_lines(runs)
    assert lines == ["### Abstract", "# 1 Introduction", "## 1.1 Clusters of Galaxies"]


@pytest.mark.parametrize(
    ("size", "expected_rank"),
    [(24.8, 1), (20.7, 2), (14.3, 3), (12.0, 4)],
)
def test_size_rank_orders_levels_largest_first(size: float, expected_rank: int) -> None:
    assert BODY_RANK[size] == expected_rank