"""Tokenisation, normalisation and language guessing.

Everything the plagiarism and stylometry layers compare goes through
:func:`tokenize`, so a match means "the same normalised word at the same place",
never "the same string". Offsets are character-exact into the original
document, which is what makes it possible to say *where* a match was found.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

__all__ = [
    "Token",
    "fold_text",
    "tokenize",
    "shingles",
    "detect_language",
    "language_confidence",
    "word_count",
    "split_words",
]

# A word, or a number that may contain . , separators and % (TR: "%12", EN: "12%").
_TOKEN_RE = re.compile(r"\d+(?:[.,]\d+)*\s*%?|[^\W\d_]+(?:['’ʼ][^\W\d_]+)*", re.UNICODE)
_WS_RE = re.compile(r"\s+")

_TR_STOPWORDS = frozenset(
    {
        "ve",
        "bir",
        "bu",
        "için",
        "ile",
        "olarak",
        "daha",
        "ancak",
        "çok",
        "gibi",
        "kadar",
        "sonra",
        "değil",
        "olan",
        "var",
        "ise",
        "ya",
        "her",
        "veya",
        "ki",
        "ne",
        "nasıl",
        "kullanılmıştır",
        "elde",
        "sonuç",
        "çalışma",
        "yöntem",
    }
)
_EN_STOPWORDS = frozenset(
    {
        "the",
        "and",
        "of",
        "to",
        "in",
        "is",
        "that",
        "for",
        "with",
        "this",
        "are",
        "was",
        "we",
        "have",
        "which",
        "using",
        "used",
        "results",
        "method",
        "study",
        "paper",
        "can",
        "these",
        "been",
        "from",
        "our",
        "more",
        "than",
    }
)
_TR_CHARS = frozenset("ığşçöüİ")
_TR_STRONG_CHARS = frozenset("ığşİ")


@dataclass(frozen=True, slots=True)
class Token:
    """One normalised word plus where it came from."""

    index: int
    """Position in the document-wide token stream."""

    text: str
    """Surface form as written (used for display and citation snippets)."""

    norm: str
    """Case/diacritic folded form used for comparison."""

    char_start: int
    char_end: int
    is_number: bool


def _strip_marks(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text)
    return unicodedata.normalize(
        "NFC", "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    )


def fold_text(text: str, *, strip_diacritics: bool = True, normalize_digits: bool = False) -> str:
    """Case-fold for comparison, with Turkish casing rules.

    ``"I" -> "ı"`` and ``"İ" -> "i"``: Python's :meth:`str.lower` gets Turkish
    wrong, and a Turkish corpus is full of dotted/dotless I collisions.
    """
    text = unicodedata.normalize("NFKC", text)
    # "Türkiye'nin" and "Türkiye nin" are the same word: Turkish attaches case
    # and possessive suffixes with an apostrophe, so they must compare equal.
    text = text.replace("'", "").replace("’", "").replace("ʼ", "")
    text = text.replace("I", "ı").replace("İ", "i").replace("î", "i")
    text = text.lower()
    if strip_diacritics:
        text = _strip_marks(text)
    if normalize_digits:
        text = re.sub(r"\d", "0", text)
    return text


def tokenize(
    text: str,
    *,
    strip_diacritics: bool = True,
    normalize_digits: bool = False,
    offset: int = 0,
    start_index: int = 0,
) -> list[Token]:
    """Split ``text`` into tokens whose offsets are shifted by ``offset``.

    ``offset`` lets callers tokenise a slice of a bigger document while still
    reporting absolute character positions.
    """
    tokens: list[Token] = []
    for i, match in enumerate(_TOKEN_RE.finditer(text)):
        surface = match.group(0)
        stripped = surface.strip()
        if not stripped:
            continue
        lead = len(surface) - len(surface.lstrip())
        tokens.append(
            Token(
                index=start_index + i,
                text=stripped,
                norm=fold_text(
                    stripped,
                    strip_diacritics=strip_diacritics,
                    normalize_digits=normalize_digits,
                ),
                char_start=offset + match.start() + lead,
                char_end=offset + match.start() + lead + len(stripped),
                is_number=stripped[0].isdigit(),
            )
        )
    return tokens


def shingles(tokens: Sequence[Token], n: int) -> Iterator[tuple[int, tuple[str, ...]]]:
    """Yield ``(start_token_index, n_gram)`` over a token sequence."""
    if n <= 0 or len(tokens) < n:
        return
    for i in range(len(tokens) - n + 1):
        yield i, tuple(t.norm for t in tokens[i : i + n])


def split_words(text: str) -> list[str]:
    return _WS_RE.split(text.strip())


def word_count(text: str) -> int:
    return len(split_words(text))


def language_confidence(text: str) -> tuple[str, float]:
    """Guess ``tr`` / ``en`` / ``unknown`` from stopwords and alphabet.

    Returns the label with a 0..1 confidence. Turkish-specific letters are
    nearly decisive; otherwise the two stopword lists are counted.

    Note: plain :meth:`str.lower` is used on purpose. Folding with Turkish
    casing rules would turn every English "I" into "ı" and make English text
    look Turkish.
    """
    lowered = unicodedata.normalize("NFKC", text).lower()
    letters = [ch for ch in lowered if ch.isalpha()]
    if len(letters) < 30:
        return "unknown", 0.0

    strong = sum(1 for ch in letters if ch in _TR_STRONG_CHARS)
    turkish_letters = sum(1 for ch in letters if ch in _TR_CHARS)
    words = re.findall(r"[^\W\d_]+", lowered)
    tr_hits = sum(1 for w in words if w in _TR_STOPWORDS)
    en_hits = sum(1 for w in words if w in _EN_STOPWORDS)

    if strong >= 2:
        return "tr", min(0.99, 0.6 + strong / max(1, len(letters)) * 20 + tr_hits / len(words))
    if turkish_letters / len(letters) > 0.05 and tr_hits > en_hits:
        return "tr", min(0.95, 0.55 + turkish_letters / len(letters))

    total = tr_hits + en_hits
    if total == 0:
        return "unknown", 0.0
    if tr_hits > en_hits * 1.5:
        return "tr", min(0.9, 0.5 + (tr_hits - en_hits) / max(1, total))
    if en_hits > tr_hits:
        return "en", min(0.9, 0.5 + (en_hits - tr_hits) / max(1, total))
    return "unknown", 0.35


def detect_language(text: str) -> str:
    return language_confidence(text)[0]
