"""Is this document text itself tampered with?

Detectors that rely on token identity fall apart on homoglyph substitutions: on
the RAID benchmark the average accuracy drop across five detectors under
homoglyph attack is **40.6%**, and Binoculars alone loses **41.9% points**
(arXiv:2405.07940). Symonym substitution costs Binoculars another 36.1 points.

An attack is therefore applied *to the document being checked*, which means the
tampering is visible in the bytes. Finding it changes the interpretation of
everything else: a very low perplexity on a document full of Cyrillic homoglyphs
is not evidence of authorship, it is evidence of an obfuscation attempt - in
either direction, and worth reporting on its own.

Scope note: this is a *character-level* diagnostic, not a detector. It reports
facts about the file (``homoglyph_words``, ``zero_width_chars``) and the caller
decides what to do with them.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass

__all__ = ["IntegrityFinding", "IntegrityReport", "check_integrity"]

# Written as escapes on purpose: these characters are invisible in an editor and
# routinely get stripped when a file is edited.
_ZERO_WIDTH = frozenset(
    {
        "\u200b",  # zero width space
        "\u200c",  # zero width non-joiner
        "\u200d",  # zero width joiner
        "\u200e",  # left-to-right mark
        "\u200f",  # right-to-left mark
        "\u2060",  # word joiner
        "\u2061",  # function application
        "\u2062",  # invisible times
        "\u2063",  # invisible separator
        "\u2064",  # invisible plus
        "\ufeff",  # zero width no-break space
    }
)
_SOFT_HYPHEN = "\u00ad"
_INVISIBLE = _ZERO_WIDTH | {_SOFT_HYPHEN}

# Latin letters that a Latin word must not contain.
_CYRILLIC = "абвгдежзийклмнопрстуфхцчшщыэюяАБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЫЭЮЯ"
_GREEK = "αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ"
_NON_LATIN = frozenset(_CYRILLIC + _GREEK)

#: Homoglyph substitution is a *Cyrillic* technique, and that is the vector worth
#: flagging: the Cyrillic alphabet contains lookalikes for every Latin letter an
#: author would type, and no legitimate reason to use them.
#:
#: Greek is different, and including it made this rule fire on physics. Measured on
#: a real 154-page astrophysics thesis (arXiv:1407.6566) it produced eleven
#: "homoglyph" findings - ``hν`` for photon energy and ``ǫν`` for a velocity -
#: every one of them ordinary notation. Flagging those is not a weak signal, it is
#: a wrong one, and it declares every AI score on scientific text unreliable.
#: Greek letters mixed into Latin are therefore reported separately, at low
#: severity, as notation rather than as tampering.
_CYRILLIC_ONLY = frozenset(_CYRILLIC)
_CONFUSABLES = str.maketrans(
    {
        "а": "a",
        "е": "e",
        "о": "o",
        "р": "p",
        "с": "c",
        "х": "x",
        "у": "y",
        "А": "A",
        "В": "B",
        "Е": "E",
        "К": "K",
        "М": "M",
        "Н": "H",
        "О": "O",
        "Р": "P",
        "С": "C",
        "Т": "T",
        "Х": "X",
        "ο": "o",
        "α": "a",
        "ρ": "p",
    }
)

_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
_REPEAT_PUNCT_RE = re.compile(r"([!?.,;:])\1{2,}")
_EMOJI_RE = re.compile("[\U0001f300-\U0001faff☀-➿]")


@dataclass(frozen=True, slots=True)
class IntegrityFinding:
    """One concrete observation about the bytes of the file."""

    code: str
    count: int
    detail: str
    examples: tuple[str, ...] = ()
    severity: str = "medium"


@dataclass(frozen=True, slots=True)
class IntegrityReport:
    """Byte-level health of the document, and what it does to the AI signals."""

    findings: tuple[IntegrityFinding, ...] = ()
    characters: int = 0

    @property
    def tampered(self) -> bool:
        """True when something was done to the text that a detector would trip on."""
        return any(f.code in _TAMPER_CODES for f in self.findings)

    @property
    def severity(self) -> str:
        if any(f.code == "homoglyph" and f.count >= 5 for f in self.findings):
            return "high"
        return "high" if self.tampered else "none"

    def to_dict(self) -> dict[str, object]:
        return {
            "characters": self.characters,
            "tampered": self.tampered,
            "severity": self.severity,
            "findings": [
                {
                    "code": f.code,
                    "count": f.count,
                    "detail": f.detail,
                    "severity": f.severity,
                    "examples": list(f.examples),
                }
                for f in self.findings
            ],
        }

    def as_degradation(self) -> str | None:
        """A one-line reason to distrust the AI scores, or ``None``."""
        if not self.tampered:
            return None
        codes = ", ".join(f.code for f in self.findings if f.code in _TAMPER_CODES)
        return (
            f"metin karakter düzeyinde değiştirilmiş ({codes}); "
            "AI sinyalleri güvenilmez kabul edilmeli "
            "(RAID: homoglyph altında ortalama doğruluk düşüşü %40.6)"
        )


_TAMPER_CODES = frozenset({"homoglyph", "zero_width", "soft_hyphen", "repeat_punctuation"})


def check_integrity(text: str) -> IntegrityReport:
    """Look for the byte-level tricks that break token-identity detectors."""
    findings: list[IntegrityFinding] = []

    zero_width = [(i, ch) for i, ch in enumerate(text) if ch in _ZERO_WIDTH]
    if zero_width:
        findings.append(
            IntegrityFinding(
                code="zero_width",
                count=len(zero_width),
                detail="Görünmez karakter (sıfır genişlik) bulundu.",
                examples=tuple(_context(text, i) for i, _ in zero_width[:3]),
                severity="high",
            )
        )

    soft = text.count(_SOFT_HYPHEN)
    if soft:
        findings.append(
            IntegrityFinding(
                code="soft_hyphen",
                count=soft,
                detail="Yumuşak tire (U+00AD) bulundu.",
                examples=(_context(text, text.index(_SOFT_HYPHEN)),),
                severity="medium",
            )
        )

    homoglyph_words = [
        word
        for match in _WORD_RE.finditer(text)
        if (word := match.group(0))
        and any(ch in _CYRILLIC_ONLY for ch in word)
        and any(ch.isalpha() and ch not in _CYRILLIC_ONLY for ch in word)
    ]
    if homoglyph_words:
        findings.append(
            IntegrityFinding(
                code="homoglyph",
                count=len(homoglyph_words),
                detail="Latin yazıyla karışan Kiril harfleri bulundu (homoglyph).",
                examples=tuple(homoglyph_words[:5]),
                severity="high",
            )
        )

    greek_words = [
        word
        for match in _WORD_RE.finditer(text)
        if (word := match.group(0))
        and any(ch in _GREEK for ch in word)
        and any(ch.isalpha() and ch not in _GREEK for ch in word)
    ]
    if greek_words:
        # Informational only. In a scientific text this is notation ("hν",
        # "ǫν"), which is what it almost always is; a Cyrillic mix is not.
        findings.append(
            IntegrityFinding(
                code="greek_notation",
                count=len(greek_words),
                detail=(
                    "Latin yazıyla karışan Yunan harfleri bulundu. Bilimsel metinde "
                    "bu genellikle gösterimdir (hν, ǫν) ve tek başına sorun değildir; "
                    "homoglif saldırısı için bakılacak harf Kiril'dir."
                ),
                examples=tuple(greek_words[:5]),
                severity="low",
            )
        )

    # Repeated punctuation touching a digit is notation, not manipulation.
    #
    # Measured on a real astrophysics thesis (arXiv:1407.6566), every one of the
    # five matches was legitimate: a math range "M^0.1...0.6" written twice, and
    # two ADS bibliographic codes "A&A...534A..120T" and "A&A...558A..75T". All
    # of them sit against a digit. A run that follows words - "......., Takey"
    # with an elided author list - is a different thing, and is still reported.
    repeats = [
        match.group(0)
        for match in _REPEAT_PUNCT_RE.finditer(text)
        if not (text[match.start() - 1 : match.start()].isdigit())
        and not (text[match.end() : match.end() + 1].isdigit())
    ]
    if len(repeats) >= 3:
        findings.append(
            IntegrityFinding(
                code="repeat_punctuation",
                count=len(repeats),
                detail="Arka arkaya tekrarlanan noktalama (tırnak/ellipsis manipülasyonu).",
                examples=tuple(_context(text, text.find(r)) for r in repeats[:3]),
                severity="low",
            )
        )

    control = sum(
        1
        for ch in text
        if unicodedata.category(ch) in {"Cc", "Cf"}
        and ch not in "\n\r\t\f"
        and ch not in _INVISIBLE
    )
    if control:
        findings.append(
            IntegrityFinding(
                code="control_chars",
                count=control,
                detail="Kontrol karakteri içeriyor.",
                severity="low",
            )
        )

    emoji = len(_EMOJI_RE.findall(text))
    if emoji:
        findings.append(
            IntegrityFinding(
                code="emoji",
                count=emoji,
                detail="Emoji bulundu (akademik metin için stil bulgusu).",
                severity="low",
            )
        )

    return IntegrityReport(findings=tuple(findings), characters=len(text))


def _context(text: str, offset: int, radius: int = 24) -> str:
    left = max(0, offset - radius)
    right = min(len(text), offset + radius)
    return (
        ("…" if left else "")
        + text[left:right].replace("\n", " ")
        + ("…" if right < len(text) else "")
    )


def folding_cost(text: str) -> float:
    """Share of characters that homoglyph folding would change (0..1).

    Diagnostic only; used to warn that diacritic-insensitive comparison is
    unreliable for this file.
    """
    if not text:
        return 0.0
    changed = sum(1 for ch in text if ch.translate(_CONFUSABLES) != ch)
    return changed / len(text)


def words_with_mixed_scripts(text: str) -> Sequence[str]:
    return [
        match.group(0)
        for match in _WORD_RE.finditer(text)
        if any(ch in _NON_LATIN for ch in match.group(0))
    ]
