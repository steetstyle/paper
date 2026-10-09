"""ArXiv identifier helpers.

Handles both modern (``2401.01234``) and legacy (``hep-th/9901001``) ids as well
as versioned forms (``2401.01234v3``) and full URLs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = [
    "ArxivId",
    "normalize_arxiv_id",
    "parse_arxiv_id",
    "versioned_id",
    "strip_version",
    "url_for_abs",
    "url_for_pdf",
    "url_for_html",
    "url_for_ar5iv",
]

_MODERN_ID = r"\d{4}\.\d{4,5}"
_LEGACY_ID = r"[a-zA-Z][a-zA-Z\-]*(?:\.[A-Z]{2})?/\d{7}"
_ANY_ID = rf"(?:{_MODERN_ID}|{_LEGACY_ID})"
# Case-insensitive on the version marker: arXiv only ever emits lowercase `v`,
# but a pasted or hand-typed uppercase `V` is common enough that rejecting it
# with "unrecognised arXiv identifier" is a worse answer than accepting it. Safe
# because an id's numeric tail is always digits, so `V` before digits can only
# be a version marker — never part of the identifier itself.
_ID_RE = re.compile(rf"^(?P<id>{_ANY_ID})(?P<version>[vV](?P<num>\d+))?$")
_VERSION_TAIL_RE = re.compile(r"[vV](?P<num>\d+)$")

_ABS_PREFIXES = (
    "http://arxiv.org/abs/",
    "https://arxiv.org/abs/",
    "http://export.arxiv.org/abs/",
    "https://export.arxiv.org/abs/",
    "arxiv:",
    "arXiv:",
)
_PDF_PREFIXES = (
    "http://arxiv.org/pdf/",
    "https://arxiv.org/pdf/",
    "http://export.arxiv.org/pdf/",
    "https://export.arxiv.org/pdf/",
)
_HTML_PREFIXES = (
    "http://arxiv.org/html/",
    "https://arxiv.org/html/",
)
_AR5IV_PREFIXES = (
    "http://ar5iv.labs.arxiv.org/html/",
    "https://ar5iv.labs.arxiv.org/html/",
)


@dataclass(frozen=True, slots=True)
class ArxivId:
    """A parsed ArXiv identifier."""

    id: str
    version: int | None = None

    @property
    def versioned(self) -> str:
        return versioned_id(self.id, self.version)

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.versioned


def strip_version(value: str) -> tuple[str, int | None]:
    """Split a trailing ``vN`` off an identifier.

    >>> strip_version("2401.01234v2")
    ('2401.01234', 2)
    >>> strip_version("2401.01234V2")
    ('2401.01234', 2)
    """
    text = value.strip()
    match = _VERSION_TAIL_RE.search(text)
    if match and _ID_RE.match(text):
        return text[: match.start()], int(match.group("num"))
    return text, None


def parse_arxiv_id(value: str) -> ArxivId:
    """Parse an id, versioned id, or ArXiv URL into :class:`ArxivId`.

    Raises ``ValueError`` when the value cannot be interpreted.
    """
    text = value.strip()
    if not text:
        raise ValueError("empty arXiv id")

    for prefix in (*_ABS_PREFIXES, *_PDF_PREFIXES, *_HTML_PREFIXES, *_AR5IV_PREFIXES):
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix) :]
            break

    # Drop query strings / fragments some clients append.
    for sep in ("?", "#"):
        if sep in text:
            text = text.split(sep, 1)[0]

    text = text.strip().rstrip("/")
    # `arXiv.org/pdf/2401.01234.pdf` -> `2401.01234`
    if text.lower().endswith(".pdf"):
        text = text[: -len(".pdf")]

    base, version = strip_version(text)
    match = _ID_RE.match(base)
    if not match:
        raise ValueError(f"unrecognised arXiv identifier: {value!r}")
    return ArxivId(id=match.group("id"), version=version)


def normalize_arxiv_id(value: str) -> str:
    """Return the versionless canonical id (used as the DB primary business key)."""
    return parse_arxiv_id(value).id


def versioned_id(arxiv_id: str, version: int | None = None) -> str:
    """Build a versioned id string. ``version=None`` means "latest"."""
    base = normalize_arxiv_id(arxiv_id)
    return f"{base}v{version}" if version is not None else base


def url_for_abs(arxiv_id: str, version: int | None = None) -> str:
    return f"https://arxiv.org/abs/{versioned_id(arxiv_id, version)}"


def url_for_pdf(arxiv_id: str, version: int | None = None) -> str:
    return f"https://arxiv.org/pdf/{versioned_id(arxiv_id, version)}"


def url_for_html(arxiv_id: str, version: int | None = None) -> str:
    return f"https://arxiv.org/html/{versioned_id(arxiv_id, version)}"


def url_for_ar5iv(arxiv_id: str, version: int | None = None) -> str:
    return f"https://ar5iv.labs.arxiv.org/html/{versioned_id(arxiv_id, version)}"