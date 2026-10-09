"""Thesis structure: which section a heading path belongs to.

A thesis is not a homogeneous text, and neither are the signals:

* the **reference list** shares text with every source by design - counting it as
  overlap is a guaranteed false accusation (2025-26 corpus-scale reuse work
  strips reference sections, affiliations and licence boilerplate *before*
  counting, arXiv:2609.32963)
* **acknowledgements** and front matter are boilerplate for scoring purposes
* the **abstract** is the most AI-prone section in practice (it is short,
  formulaic, and generated last) and needs its own baseline
* the **methods** section legitimately uses formulaic, non-personal register, so
  its AI baseline is higher than the introduction's

Section role is inferred from the heading path (``Location.section_path``) using
Turkish and English keywords, which is why the report can say "this finding is in
the Methods chapter" instead of "paragraph 412".
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from enum import StrEnum

__all__ = ["SectionRole", "classify_section", "role_of_path", "is_boilerplate", "role_label"]

# Heading matching folds Turkish letters to ASCII in both cases:
# "İSTANBUL" / "İstanbul" / "i̇stanbul" all become "istanbul".
_FOLD = str.maketrans("İIŞĞÜÖÇışğüöç", "iisguocisguoc")


_ENUM_PREFIX_RE = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*[.)]?|bolum\s*\d+|b[oö]l[uü]m\s*\d+|[ivx]+[.)])\s*", re.IGNORECASE
)


def _norm(text: str) -> str:
    """Fold a heading for matching: Turkish letters to ASCII, then strip the
    enumeration prefix so "1. GİRİŞ VE AMAÇ" matches the "giriş" rule."""
    folded = text.translate(_FOLD).lower().strip()
    return _ENUM_PREFIX_RE.sub("", folded).strip()


class SectionRole(StrEnum):
    """What a heading path denotes in a thesis/article."""

    FRONT_MATTER = "front_matter"
    ABSTRACT = "abstract"
    INTRODUCTION = "introduction"
    RELATED_WORK = "related_work"
    METHOD = "method"
    DATA = "data"
    RESULTS = "results"
    DISCUSSION = "discussion"
    CONCLUSION = "conclusion"
    REFERENCES = "references"
    ACKNOWLEDGMENTS = "acknowledgments"
    APPENDIX = "appendix"
    QUOTATION = "quotation"
    OTHER = "other"

    @property
    def scored(self) -> bool:
        """Should AI-authorship scoring run on this section at all?"""
        return self not in _UNSCORED

    @property
    def matched(self) -> bool:
        """Should plagiarism matching run on this section at all?"""
        return self not in _UNMATCHED


_UNSCORED = frozenset(
    {
        SectionRole.REFERENCES,
        SectionRole.ACKNOWLEDGMENTS,
        SectionRole.FRONT_MATTER,
        SectionRole.QUOTATION,
    }
)
# The reference list *does* get compared against itself in the literature, but for
# an offline checker it is pure noise: it is the source of every "overlap".
_UNMATCHED = frozenset(
    {SectionRole.REFERENCES, SectionRole.ACKNOWLEDGMENTS, SectionRole.FRONT_MATTER}
)

# Order matters: the first match wins, so the specific labels come first.
_RULES: tuple[tuple[SectionRole, tuple[str, ...]], ...] = (
    (
        SectionRole.REFERENCES,
        (
            r"kaynak",  # kaynakça, kaynaklar, kaynak dizini
            r"referans",  # referanslar
            r"bibliyograf",
            r"references?$",
            r"bibliography",
            r"works cited",
            r"literature cited",
        ),
    ),
    (
        SectionRole.ACKNOWLEDGMENTS,
        (r"teşekk", r"acknowledge", r"sag ol", r"sağ ol"),
    ),
    (
        SectionRole.FRONT_MATTER,
        (
            r"içindekiler",
            r"table of contents",
            r"simge",
            r"kısaltma",
            r"abbreviation",
            r"liste of|list of",
            r"tablo dizini",
            r"şekil dizini",
            r"tez künyesi",
            r"onay sayfalar?",
            r"önsöz",
            r"preface",
            r"foreword",
        ),
    ),
    (
        SectionRole.ABSTRACT,
        (
            r"^\S*\bturkce\s+oz(?:e)?t\b",  # "ÖZET" exported without the umlaut
            r"^\S*\bingilizce\s+abstract\b",
            r"^\S*\bozet\b",
            r"^\S*\babstract\b",
            r"^\S*\bsummary\b",
        ),
    ),
    (
        SectionRole.APPENDIX,
        (
            r"^\S*\bek\b",
            r"^\S*\bekler\b",
            r"^\S*\bappendix\b",
            r"^\S*\bannex\b",
            r"^\S*\bsupplementary\b",
        ),
    ),
    (
        SectionRole.CONCLUSION,
        (
            r"^\S*\bsonuç\b",
            r"^\S*\bsonuçlar\b",
            r"^conclusion",
            r"^öneriler",
            r"^suggestion",
            r"^recommendation",
            r"^concluding",
            r"^closing",
        ),
    ),
    (
        SectionRole.RELATED_WORK,
        (
            r"ilgili çalışma",
            r"literatür",
            r"literatur",
            r"kavramsal çerçeve",
            r"kuramsal",
            r"related work",
            r"background",
            r"theoretical framework",
            r" ön bilgi",
            r" önceki çalışma",
        ),
    ),
    (
        SectionRole.METHOD,
        (
            r"yöntem",
            r"metodoloj",
            r"materyal",
            r"^\S*\bdeney",
            r"^\S*\byöntem",
            r"method",
            r"methodolog",
            r"materials?",
            r"approach",
            r"experimental setup",
            r"study design",
            r"veri toplama",
            r"data collection",
            r"kurulum",
            r"implementation",
        ),
    ),
    (
        SectionRole.DATA,
        (r"veri seti", r"veriler", r"dataset", r"^\S*\bdata\b", r"veri kaynağı", r"data source"),
    ),
    (
        SectionRole.RESULTS,
        (
            r"^\S*\bbulgu",
            r"^\S*\bsonuçlar?\b",
            r"^\S*\bresults?\b",
            r"^\S*\bfindings?\b",
            r"^\S*\bgözlem",
        ),
    ),
    (
        SectionRole.DISCUSSION,
        (
            r"tartış",
            r"discussion",
            r"^\S*\bdeğerlendir",
            r"karşılaştırma",
            r"comparison",
            r"sınırlılık",
            r"limitation",
        ),
    ),
    (
        SectionRole.INTRODUCTION,
        (
            r"^\S*\bgiriş\b",
            r"^\S*\bintroduction\b",
            r"problem durum",
            r"problem statement",
            r"amaç ve kapsam",
            r"^amaç$",
            r"^motivation",
            r"^background and aim",
        ),
    ),
    (
        SectionRole.QUOTATION,
        (r"^\S*\balıntı", r"^\S*\bquotation\b", r"^\S*\bquotations\b"),
    ),
)

# Headings are folded before matching, so the patterns are folded too: the code
# below can stay readable Turkish ("^giriş") and still match "GİRİŞ".
_COMPILED: tuple[tuple[SectionRole, tuple[re.Pattern[str], ...]], ...] = tuple(
    (role, tuple(re.compile(_norm(p), re.IGNORECASE) for p in patterns))
    for role, patterns in _RULES
)

# Sections whose AI-assist baseline is materially different from the document
# mean. Values are ordinal nudges, applied by the scorer as a prior adjustment.
_ROLE_PRIOR: dict[SectionRole, float] = {
    SectionRole.ABSTRACT: 0.10,
    SectionRole.RELATED_WORK: 0.06,
    SectionRole.INTRODUCTION: 0.04,
    SectionRole.CONCLUSION: 0.04,
    SectionRole.METHOD: -0.02,
    SectionRole.RESULTS: -0.04,
    SectionRole.DATA: -0.04,
    SectionRole.DISCUSSION: -0.02,
    SectionRole.OTHER: 0.0,
}


def classify_section(titles: Sequence[str]) -> SectionRole:
    """Role from a heading path (deepest heading wins, else shallowest)."""
    if not titles:
        return SectionRole.OTHER
    for title in reversed(list(titles)):
        normalised = _norm(title)
        for role, patterns in _COMPILED:
            if any(pattern.search(normalised) for pattern in patterns):
                return role
    return SectionRole.OTHER


def role_of_path(path: Sequence[str]) -> SectionRole:
    return classify_section(path)


def is_boilerplate(role: SectionRole) -> bool:
    """True for sections that are excluded from authorship scoring."""
    return role in _UNSCORED


def role_label(role: SectionRole) -> str:
    return {
        SectionRole.FRONT_MATTER: "Ön bölüm",
        SectionRole.ABSTRACT: "Özet",
        SectionRole.INTRODUCTION: "Giriş",
        SectionRole.RELATED_WORK: "İlgili çalışmalar",
        SectionRole.METHOD: "Yöntem",
        SectionRole.DATA: "Veri",
        SectionRole.RESULTS: "Bulgular",
        SectionRole.DISCUSSION: "Tartışma",
        SectionRole.CONCLUSION: "Sonuç",
        SectionRole.REFERENCES: "Kaynakça",
        SectionRole.ACKNOWLEDGMENTS: "Teşekkür",
        SectionRole.APPENDIX: "Ekler",
        SectionRole.QUOTATION: "Alıntılar",
        SectionRole.OTHER: "Diğer",
    }[role]


def role_prior(role: SectionRole) -> float:
    """Prior adjustment for the AI score, from how sections actually read."""
    return _ROLE_PRIOR.get(role, 0.0)
