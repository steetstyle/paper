"""Reading a publication year out of a reference, across five real styles.

Every case here was measured on a real thesis, and each one was a bug first:

- **Astrofizik** (arXiv:1407.6566): `Abell, G. O. 1958, ApJS, 3, 211 1, 4, 10`
  puts the year first. Taking the last four-digit number read the back-reference.
- **Aynı tez**: `Berlind, A. A., et al. 2006, ApJ S, 167, 1` - the volume `167`
  is comma-terminated exactly like the year, so position alone is not enough.
- **String kuramı** (arXiv:0911.2782): `pp. 1415-1443, 1999, hep-th/9811131`
  puts the year last, and both numbers inside the page range end in a comma too.
- **Aynı tez**: `10.1103/PhysRevB.77.220503` contains `1103`. 263 of 417 entries
  took a year out of a DOI.
- **Yoğunlaştırılmış madde** (arXiv:1912.04141): 466,068 characters, past the old
  400,000 limit - the truncation was silent and it cut the bibliography.
"""

from __future__ import annotations

import pytest

from checker_app.services.references import (
    _AUTHOR_YEAR_RE,
    _IDENTIFIER_RE,
    _PAREN_YEAR_RE,
    _YEAR_RE,
    MIN_PLAUSIBLE_YEAR,
)


def _year(text: str) -> int | None:
    """The same resolution order ``_vet`` uses, exposed for testing."""
    prose = _IDENTIFIER_RE.sub(" ", text)
    positioned = _AUTHOR_YEAR_RE.findall(prose)
    if positioned:
        return int(positioned[-1])
    paren = _PAREN_YEAR_RE.search(prose)
    if paren is not None:
        return int(paren.group(1))
    years = [int(y) for y in _YEAR_RE.findall(prose)]
    return years[-1] if years else None


#: Each style, with the year a person would read. All five from real theses.
STYLES = [
    # Astronomy: Author. Year, Journal, Volume, Page [back-references]
    (
        1958,
        "Abell, G. O. 1958, ApJS, 3, 211 1, 4, 10, 71, 74, 92",
    ),
    # Astronomy with a volume that is comma-terminated like a year
    (
        2006,
        "Berlind, A. A., Frieman, J., Weinberg, D. H., et al. 2006, ApJ S, 167, 1 4, 11, 70",
    ),
    (
        1972,
        "Forman, W., Kellogg, E., Gursky, H., Tananbaum, H., & Giacconi, R. 1972, ApJ, 178, 275",
    ),
    # hep-th: venue, volume, page range, year, arXiv id
    (
        1999,
        "Adv. Theor. Math. Phys. , vol. 3, pp. 1415–1443, 1999, hep-th/9811131.",
    ),
    (
        1998,
        "Gwyn, R. 1998, Rev. Mod. Phys. , vol. 70, pp. 1419–1453, hep-th/9702199.",
    ),
    # MLA-ish: year then a DOI
    (
        2019,
        "Smith, J. ve Jones, P. (2019). Deep learning for assessment. doi:10.1234/jla.2019.5678",
    ),
]


@pytest.mark.parametrize(("expected", "text"), STYLES)
def test_the_year_is_read_from_every_style_measured(expected: int, text: str) -> None:
    assert _year(text) == expected, text


def test_an_identifier_is_not_a_date() -> None:
    """"10.1103/PhysRevB.77.220503" contains 1103, and it is not a year."""
    assert _year("10.1103/PhysRevB.77.220503. (Cited on pages 2, 25, 29).") != 1103
    assert _year("abs/10.1143/JPSJ.79.044705. (Cited on pages 20, 53.)") != 1143


def test_identifiers_are_stripped_before_the_year_is_looked_for() -> None:
    for identifier in (
        "https://doi.org/10.1103/PhysRevB.77.220503",
        "doi: 10.1017/CBO9781107050200",
        "ISBN 978-0030839931",
        "arXiv:1706.03762",
        "10.1103/PhysRevB.77.220503",
    ):
        stripped = _IDENTIFIER_RE.sub(" ", f"Smith, J. 2015. A title. {identifier}")
        assert "1103" not in stripped and "2015" in stripped, identifier


def test_a_page_range_is_not_a_year() -> None:
    """Both numbers in "1415-1443" end in a comma, exactly like a real year."""
    assert _year("Theor. Math. Phys. , vol. 3, pp. 1415–1443, 1999, hep-th/9811131.") == 1999


def test_the_four_digit_pattern_is_actually_four_digits() -> None:
    """A three-digit pattern silently skipped every year of the 1950s-1980s.

    ``1[5-9]\\d`` matches "199" and then fails on the fourth digit, so 1972 and
    1978 - the years of Hubble and Zwicky, cited constantly in this thesis - were
    not candidates at all.
    """
    for year in (1958, 1972, 1978, 1989, 1999, 2006, 2013, 2019):
        assert _AUTHOR_YEAR_RE.search(f"Author, {year}, ApJ") or _PAREN_YEAR_RE.search(
            f"Author ({year}). Title."
        ), year


def test_an_impossible_year_is_still_reported() -> None:
    """The rule that reads the year correctly is not the rule that judges it."""
    assert MIN_PLAUSIBLE_YEAR == 1665
    assert _year("Anachronism, P. (1600). Early measurement. Monographs.") == 1600
    assert _year("Doe, A. (2099). Future perspectives on tutoring. Journal.") == 2099


def test_foundational_physics_is_not_impossible() -> None:
    for text in (
        "Hubble, E. P. 1926, ApJ, 64, 321.",
        "Zwicky, F. 1933, Helvetica Physica Acta, 6, 110.",
        "Smith, S. 1936, ApJ, 83, 23.",
    ):
        assert _year(text) and _year(text) >= MIN_PLAUSIBLE_YEAR, text