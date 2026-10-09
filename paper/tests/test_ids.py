"""ArXiv identifier handling."""

from __future__ import annotations

import pytest

from paper_app.domain.ids import (
    normalize_arxiv_id,
    parse_arxiv_id,
    url_for_ar5iv,
    url_for_html,
    url_for_pdf,
    versioned_id,
)


@pytest.mark.parametrize(
    "value,expected_id,expected_version",
    [
        ("2401.01234", "2401.01234", None),
        ("2401.01234v3", "2401.01234", 3),
        ("arXiv:2401.01234v1", "2401.01234", 1),
        ("https://arxiv.org/abs/2401.01234", "2401.01234", None),
        ("https://arxiv.org/abs/2401.01234v2", "2401.01234", 2),
        ("http://arxiv.org/pdf/2401.01234v5.pdf", "2401.01234", 5),
        ("https://arxiv.org/html/2401.01234v1", "2401.01234", 1),
        ("https://ar5iv.labs.arxiv.org/html/hep-th/9901001", "hep-th/9901001", None),
        ("math.GT/0309136", "math.GT/0309136", None),
        ("  cs.CL/0701001  ", "cs.CL/0701001", None),
    ],
)
def test_parse_arxiv_id(value: str, expected_id: str, expected_version: int | None) -> None:
    parsed = parse_arxiv_id(value)
    assert parsed.id == expected_id
    assert parsed.version == expected_version
    assert normalize_arxiv_id(value) == expected_id


def test_parse_arxiv_id_rejects_garbage() -> None:
    for bad in ("", "not-an-id", "2401.01234x", "12345"):
        with pytest.raises(ValueError):
            parse_arxiv_id(bad)


def test_versioned_id_roundtrip() -> None:
    assert versioned_id("2401.01234") == "2401.01234"
    assert versioned_id("2401.01234", 3) == "2401.01234v3"
    assert parse_arxiv_id("2401.01234v3").versioned == "2401.01234v3"


def test_url_builders() -> None:
    assert url_for_pdf("2401.01234", 2) == "https://arxiv.org/pdf/2401.01234v2"
    assert url_for_html("2401.01234", 2) == "https://arxiv.org/html/2401.01234v2"
    assert url_for_ar5iv("2401.01234") == "https://ar5iv.labs.arxiv.org/html/2401.01234"