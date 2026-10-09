"""Text utilities shared by chunking, extraction and indexing."""

from __future__ import annotations

import hashlib
import re
import unicodedata

_WS_RE = re.compile(r"[ \t ]+")
_MULTI_NL_RE = re.compile(r"\n{3,}")
_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])[\"'’”\)\]]*\s+")
# Tokens that end in a period without ending a sentence.
_ABBREVIATIONS = frozenset(
    {
        "al", "e.g", "i.e", "etc", "vs", "cf", "approx", "resp", "ca", "viz",
        "fig", "figs", "fig.", "eq", "eqs", "eq.", "ref", "refs", "ref.",
        "sec", "secs", "sec.", "ch", "chap", "tab", "tabs", "tab.",
        "no", "nos", "no.", "vol", "vols", "pp", "p", "ed", "eds", "dr", "prof",
        "mr", "mrs", "ms", "st", "jr", "sr", "inc", "ltd", "co", "dept",
    }
)
_LOWERCASE_START_RE = re.compile(r"^[a-zà-öø-ÿ]")
_ENUMERATION_RE = re.compile(r"^\(?[0-9a-z]{1,3}[.)]\s")

# Markdown / HTML constructs removed before embedding.
_MD_CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
_MD_INLINE_CODE_RE = re.compile(r"`([^`]*)`")
_MD_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_MD_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_MD_REF_LINK_RE = re.compile(r"\[([^\]]*)\]\[[^\]]*\]")
_MD_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MD_EMPHASIS_RE = re.compile(r"(\*{1,3}|_{1,3})(\S(?:.*?\S)?)\1", re.DOTALL)
_MD_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.*)$", re.MULTILINE)
_MD_LIST_RE = re.compile(r"^\s*(?:[-*+]|\d+\.)\s+", re.MULTILINE)
_MD_QUOTE_RE = re.compile(r"^\s*>\s?", re.MULTILINE)
_MD_RULE_RE = re.compile(r"^\s*([-*_])\1{2,}\s*$", re.MULTILINE)
_MD_ESCAPE_RE = re.compile(r"\\([\\`*_{}\[\]()#+\-.!>~|])")

# Boilerplate dropped from ArXiv HTML renderings.
_ARXIV_NOISE_RE = re.compile(
    r"(arXiv:\d{4}\.\d{4,5}v\d+|Download PDF|Report number|Submitted on .*?\(v\d+\)|"
    r"Skip to main content|\bCopyright\b.*)",
    re.IGNORECASE,
)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalise_unicode(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def collapse_whitespace(text: str) -> str:
    """Collapse intra-line spacing while keeping paragraph structure."""
    text = _WS_RE.sub(" ", text)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return _MULTI_NL_RE.sub("\n\n", text).strip()


def approx_tokens(text: str) -> int:
    """Cheap, provider-agnostic token estimate (~4 chars/token for English)."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def strip_markdown(text: str, *, drop_math: bool = True) -> str:
    """Convert markdown to readable prose suitable for embedding."""
    if not text:
        return ""
    out = text
    out = _MD_CODE_FENCE_RE.sub(" ", out)
    out = _MD_RULE_RE.sub(" ", out)
    out = _MD_IMAGE_RE.sub(" ", out)
    out = _MD_LINK_RE.sub(r"\1", out)
    out = _MD_REF_LINK_RE.sub(r"\1", out)
    out = _MD_INLINE_CODE_RE.sub(r"\1", out)
    out = _MD_HEADING_RE.sub(r"\1", out)
    out = _MD_LIST_RE.sub("", out)
    out = _MD_QUOTE_RE.sub("", out)
    out = _MD_HTML_TAG_RE.sub(" ", out)
    out = _MD_EMPHASIS_RE.sub(r"\2", out)
    out = _MD_ESCAPE_RE.sub(r"\1", out)
    if drop_math:
        out = re.sub(r"\$\$?[^$]*\$\$?", " ", out)
    return collapse_whitespace(out)


def strip_arxiv_noise(text: str) -> str:
    return _ARXIV_NOISE_RE.sub(" ", text).strip()


def split_sentences(text: str) -> list[str]:
    """Lightweight sentence splitter (no spaCy dependency).

    Splits on terminal punctuation, then glues back fragments that are clearly
    not sentence starts: abbreviations (``et al.``), lowercase continuations and
    enumerated/parenthetical items (``(1)``).
    """
    if not text.strip():
        return []
    parts = [part.strip() for part in _SENTENCE_SPLIT_RE.split(text) if part.strip()]
    if not parts:
        return []

    merged = [parts[0]]
    for part in parts[1:]:
        if _is_continuation(merged[-1], part):
            merged[-1] = f"{merged[-1]} {part}"
        else:
            merged.append(part)
    return merged


def _is_continuation(previous: str, candidate: str) -> bool:
    first_token = candidate.split()[0] if candidate.split() else ""
    normalised = first_token.lower().strip("\"'“‘([")
    if normalised in _ABBREVIATIONS:
        return True
    if _ENUMERATION_RE.match(candidate):
        return True
    if _LOWERCASE_START_RE.match(candidate):
        return True
    # "Dr. Smith" style: previous fragment is a bare title.
    return len(previous.split()) == 1 and previous.rstrip(".").lower() in _ABBREVIATIONS


def truncate(text: str, limit: int, suffix: str = "…") -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - len(suffix))].rstrip() + suffix


HEADING_PATTERN = re.compile(r"^(?P<hashes>#{1,6})\s+(?P<title>.+?)\s*#*$")


def parse_heading(line: str) -> tuple[int, str] | None:
    """Return ``(level, title)`` when ``line`` is a markdown ATX heading."""
    match = HEADING_PATTERN.match(line.strip())
    if not match:
        return None
    return len(match.group("hashes")), match.group("title").strip()


def iter_paragraphs(text: str) -> list[str]:
    """Split on blank lines, keeping code blocks intact."""
    paragraphs: list[str] = []
    buffer: list[str] = []
    in_fence = False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        if not in_fence and not line.strip() and buffer:
            paragraphs.append("\n".join(buffer).strip())
            buffer = []
            continue
        buffer.append(line)
    if buffer:
        paragraphs.append("\n".join(buffer).strip())
    return [p for p in paragraphs if p]