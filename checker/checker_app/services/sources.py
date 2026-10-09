"""Loading reference sources and the document under test.

References come from files, directories, URLs, stdin or literal text. Whatever
the origin, they end up as a :class:`~checker_app.domain.models.SourceDocument` with a
token stream built using the same normalisation as the document, so the two can
be compared directly.

Extracted text (PDF/HTML) is cached on disk under ``.checker_cache`` keyed by
path, size and mtime: parsing a 300 page PDF on every scan is the difference
between a two second and a two minute tool.
"""

from __future__ import annotations

import hashlib
import re
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from checker_app.config import CheckerSettings
from checker_app.domain.models import SourceDocument
from checker_app.domain.text import Token, language_confidence, tokenize
from checker_app.logging import get_logger

__all__ = ["SourceLoader", "LoadedDocument", "DEFAULT_EXTENSIONS", "looks_like_own_output"]

# Markers this tool writes into its reports. A `--ref-dir .` that includes them
# compares the thesis with its own annotations, which produces a spectacular and
# completely meaningless 100% overlap.
_OWN_OUTPUT_MARKERS = (
    "Metin risk raporu",
    "Uyum kontrol listesi",
    "Benzerlik raporu",
    "Cümle düzeyi fark",
    "\u27e6#",  # annotated-text badge
    "kalıcı olması için tez",
)

logger = get_logger("checker.sources")

DEFAULT_EXTENSIONS = (".txt", ".md", ".markdown", ".text", ".rst", ".html", ".htm", ".pdf")
_HTML_SUFFIXES = frozenset({".html", ".htm", ".xhtml"})


@dataclass(frozen=True, slots=True)
class LoadedDocument:
    """Raw text plus where it came from."""

    path: str
    text: str
    kind: str = "file"
    meta: dict[str, object] = field(default_factory=dict)

    def source_id(self) -> str:
        digest = hashlib.sha1(self.path.encode("utf-8")).hexdigest()[:12]
        return f"{Path(self.path).stem[:40]}-{digest}"


class SourceLoader:
    """Read documents and references from anywhere, with a text cache."""

    def __init__(self, settings: CheckerSettings) -> None:
        self._settings = settings
        self._cache_dir = Path(settings.plagiarism.cache_dir)
        self._strip = settings.plagiarism.strip_diacritics
        self._fold_digits = settings.plagiarism.normalize_digits

    # ------------------------------------------------------------------ public
    def load_document(self, path: str | Path) -> LoadedDocument:
        """The file being analysed."""
        resolved = Path(path).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"dosya bulunamadı: {resolved}")
        text = self._read(resolved)
        return LoadedDocument(
            path=str(resolved),
            text=text,
            kind="file",
            meta={"suffix": resolved.suffix.lower(), "bytes": resolved.stat().st_size},
        )

    def load_stdin(self) -> LoadedDocument:
        data = sys.stdin.read()
        return LoadedDocument(path="<stdin>", text=data, kind="stdin", meta={})

    def load_reference(self, spec: str) -> LoadedDocument:
        """A reference given as a path, a URL, or literal text."""
        if spec.startswith(("http://", "https://")):
            return self._load_url(spec)
        if spec == "-":
            return self.load_stdin()
        path = Path(spec).expanduser()
        if path.is_dir():
            raise IsADirectoryError(f"{spec} bir dizin; tüm dosyalar için --ref-dir kullanın.")
        resolved = path.resolve()
        if not resolved.exists():
            # Treat as literal text so callers can pass a quote on the CLI.
            return LoadedDocument(path=f"<text:{spec[:40]}>", text=spec, kind="text", meta={})
        return LoadedDocument(
            path=str(resolved),
            text=self._read(resolved),
            kind="file",
            meta={"suffix": resolved.suffix.lower()},
        )

    def load_directory(
        self, directory: str | Path, pattern: str = "*", recursive: bool = True
    ) -> list[LoadedDocument]:
        base = Path(directory).expanduser().resolve()
        if not base.is_dir():
            raise NotADirectoryError(f"dizin bulunamadı: {base}")
        globber = base.rglob if recursive else base.glob
        paths = [
            p
            for p in sorted(globber(pattern))
            if p.is_file() and p.suffix.lower() in DEFAULT_EXTENSIONS
        ]
        if not paths:
            logger.warning("kaynakta eşleşen dosya yok: %s/%s", base, pattern)
        loaded: list[LoadedDocument] = []
        for path in paths:
            try:
                loaded.append(
                    LoadedDocument(
                        path=str(path),
                        text=self._read(path),
                        kind="dir",
                        meta={"suffix": path.suffix.lower()},
                    )
                )
            except OSError as exc:  # pragma: no cover - filesystem dependent
                logger.warning("okunamadı %s: %s", path, exc)
        return loaded

    def to_source(self, loaded: LoadedDocument) -> SourceDocument:
        """Wrap loaded text with a token stream built for comparison."""
        text = loaded.text[: self._settings.plagiarism.max_source_chars]
        tokens: Sequence[Token] = tokenize(
            text,
            strip_diacritics=self._strip,
            normalize_digits=self._fold_digits,
        )
        language, confidence = language_confidence(text)
        return SourceDocument(
            source_id=loaded.source_id(),
            name=_display_name(loaded.path),
            kind=loaded.kind,
            text=text,
            tokens=tuple(tokens),
            page_count=text.count("\f") + 1,
            meta=dict(loaded.meta or {}),
            language=language,
            language_confidence=round(confidence, 3),
        )

    def to_sources(self, loaded: Iterable[LoadedDocument]) -> list[SourceDocument]:
        return [self.to_source(item) for item in loaded]

    # ----------------------------------------------------------------- private
    def _read(self, path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            cached = self._cached_text(path)
            if cached is not None:
                return cached
            text = _extract_pdf(path)
            self._store_cache(path, text)
            return text
        if suffix in _HTML_SUFFIXES:
            return _extract_html(self._decode(path))
        return self._decode(path)

    def _decode(self, path: Path) -> str:
        raw = path.read_bytes()
        for encoding in (self._settings.default_encoding, "utf-8", "latin-1"):
            try:
                return raw.decode(encoding)
            except UnicodeDecodeError:
                continue
        return raw.decode("utf-8", errors="replace")  # pragma: no cover - defensive

    def _cache_key(self, path: Path) -> Path:
        stat = path.stat()
        digest = hashlib.sha1(
            f"{path.resolve()}|{stat.st_size}|{int(stat.st_mtime)}".encode()
        ).hexdigest()[:16]
        return self._cache_dir / f"{path.stem[:40]}-{digest}.txt"

    def _cached_text(self, path: Path) -> str | None:
        cache_file = self._cache_key(path)
        if cache_file.exists():
            try:
                return cache_file.read_text(encoding="utf-8")
            except OSError:  # pragma: no cover - defensive
                return None
        return None

    def _store_cache(self, path: Path, text: str) -> None:
        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            self._cache_key(path).write_text(text, encoding="utf-8")
        except OSError as exc:  # pragma: no cover - filesystem dependent
            logger.debug("önbellek yazılamadı %s: %s", path, exc)

    def _load_url(self, url: str) -> LoadedDocument:
        import httpx  # noqa: PLC0415 - optional dependency

        with httpx.Client(timeout=30, follow_redirects=True) as client:
            response = client.get(url, headers={"User-Agent": "checker-app/0.1"})
            response.raise_for_status()
            body = response.text
        if url.lower().endswith((".html", ".htm")) or "<html" in body[:2000].lower():
            body = _extract_html(body)
        return LoadedDocument(path=url, text=body, kind="url", meta={"url": url})


def looks_like_own_output(text: str, *, sample_chars: int = 4000) -> bool:
    """True when a file appears to be a report produced by this tool."""
    head = text[:sample_chars]
    return any(marker in head for marker in _OWN_OUTPUT_MARKERS)


def _display_name(path: str) -> str:
    if path.startswith("http"):
        return path if len(path) <= 60 else path[:57] + "…"
    return Path(path).name or path


def _extract_html(markup: str) -> str:
    try:
        from bs4 import BeautifulSoup  # noqa: PLC0415 - optional dependency
    except ImportError:
        logger.warning("beautifulsoup4 yok; HTML ham metin olarak karşılaştırılıyor")
        return markup
    soup = BeautifulSoup(markup, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return soup.get_text("\n")


#: A PDF heading is set larger than the body text. The gap is small in points but
#: unambiguous in practice: measured on a real 154-page astrophysics thesis
#: (arXiv:1407.6566), body text sits at 10.9pt for 92,226 characters while every
#: heading sits at 12.0pt or above for 978 characters in total. Body text is found
#: as the character-weighted mode, so this does not assume any particular thesis
#: template.
PDF_BODY_SIZE_TOLERANCE = 0.4

#: Synthetic arXiv stamps are set at heading size on the title page and are not
#: headings. Same for a footnote marker.
_PDF_STAMP_RE = re.compile(r"^\s*(arXiv:|https?://|\*+$|\W+$)", re.IGNORECASE)

#: A table caption is set at heading size in many templates but names a float,
#: not a section. A running header ("121  APPENDIX C") or a row of tabular
#: coordinates is likewise not a section: what identifies both is that they are
#: mostly digits, punctuation or whitespace. Measured on arXiv:1407.6566 this
#: removes the remaining spurious headings left after the size check.
_PDF_CAPTION_RE = re.compile(
    r"^\s*(?:Table|Figure|Şekil|Tablo|Chart|Plate|Listing|Algorithm)\s*[\d.]",
    re.IGNORECASE,
)
_PDF_NON_PROSE_RATIO = 0.45

#: A heading is short. Measured on arXiv:1407.6566 the longest real heading is 78
#: characters; the longest spurious one, a table-notes paragraph set at a
#: heading-ish size, ran past 400. Without this cap that paragraph became a
#: top-level section and swallowed every following heading as its children.
PDF_HEADING_MAX_CHARS = 120

#: A bare number on its own line is a page footer.
_PDF_FOOTER_RE = re.compile(r"^\d{1,4}$")

#: A running header whose page number was glued on in front of it: "121  APPENDIX
#: C", "121# . APPENDIX C". Digits then punctuation then a real title. The digits
#: alone keep the letters-ratio test happy, so this needs its own rule.
_PDF_RUNNING_HEADER_RE = re.compile(r"^\d{1,4}\s*[^\w\s]{1,2}\s")

#: Some templates set the bibliography heading at body size, where no amount of
#: point-size analysis can reach it. Measured on arXiv:0911.2782 (a 152-page
#: string-theory thesis): "References" sits at 10.9pt against a 10.9pt body, so
#: the whole document was found and the bibliography was never examined - the
#: one part of a thesis that most needs auditing.
#:
#: The fallback is content, not style: a short line that is *nothing but* a
#: bibliography keyword. Requiring the entire line to be the keyword keeps a
#: sentence that merely mentions references out of it, and a thesis whose only
#: line reading "References" is a heading, not prose.
_PDF_STANDALONE_KEYWORD_RE = re.compile(
    r"^(references|bibliography|literature\s+cited|works\s+cited|"
    r"kaynakça|kaynaklar|referanslar| bibliography)$",
    re.IGNORECASE,
)

#: Numbering depth gives the level directly and is more reliable than font size
#: when a thesis styles 1. and 1.1. at the same point size.
_PDF_NUMBERED_RE = re.compile(r"^\s*(?:Chapter\s+)?(\d{1,2}(?:\.\d{1,2}){0,3})\.?\s+\S")

#: Lines that end the way prose ends, not the way headings end.


def _pdf_heading_level(
    text: str, font_size: float, body_size: float, size_rank: dict[float, int]
) -> int:
    """Markdown heading level for one PDF line, or ``0`` if it is not a heading.

    Font size decides *whether* something is a heading; numbering only decides
    *which level*. Keeping those separate is what an earlier version of this
    function got wrong, and it cost 359 spurious headings on arXiv:1407.6566: any
    line beginning with a digit became a heading, so the table of contents (with
    its dot leaders), the examination-committee list ("1. Prof. Dr. Steinmetz") and
    a sentence beginning "7.5 keV) for 345 systems" were all promoted. Those lines
    are set at body size.

    Where the two disagree, numbering wins: a thesis that sets ``1.`` and ``1.1``
    at the same point size still means two levels, which a size-only reading would
    collapse.
    """
    stripped = text.strip()
    if not stripped:
        return 0
    if _PDF_STANDALONE_KEYWORD_RE.match(" ".join(stripped.split())):
        # A bibliography heading set at body size. It is a top-level heading by
        # position in the document whatever the template decided about its size.
        return 1
    if _PDF_STAMP_RE.match(stripped):
        return 0
    if not stripped[0].isalnum():
        # A running header with the page number glued in front of it
        # ("121  APPENDIX C") starts with punctuation or a bare number.
        return 0
    if _PDF_RUNNING_HEADER_RE.match(stripped):
        return 0
    if font_size <= body_size + PDF_BODY_SIZE_TOLERANCE:
        return 0
    if len(stripped) > PDF_HEADING_MAX_CHARS:
        return 0
    if _PDF_CAPTION_RE.match(stripped):
        return 0
    letters = sum(1 for ch in stripped if ch.isalpha())
    if letters / max(1, len(stripped)) < _PDF_NON_PROSE_RATIO:
        return 0
    numbered = _PDF_NUMBERED_RE.match(stripped)
    if numbered is not None:
        return len(numbered.group(1).split("."))
    # Front matter (Abstract, Acknowledgements, Contents) is often centred at
    # heading size without a number; the size rank gives it a level.
    return size_rank.get(round(font_size, 1), 1)


def _extract_pdf(path: Path) -> str:
    """Extract PDF text, restoring the heading structure the plain text loses.

    A plain ``extract_text()`` call throws away everything that distinguishes a
    heading from a paragraph, because it is a sequence of glyphs and the size is
    not part of it. The consequence here was severe: a 154-page thesis produced
    **one** section and every one of its 1,819 sentences was filed as
    ``(giriş)`` - the whole section-aware reporting surface, per-section AI share,
    per-section plagiarism, discourse profiles, was dead on the exact input a
    thesis arrives in.

    So the visitor callback is used to recover font size per run, the headings are
    re-expressed as Markdown ``#`` prefixes, and the existing splitter - which
    already understands Markdown - does the rest untouched.

    Measured on arXiv:1407.6566 (154 pages, 4,700 lines): 46 runs set above body
    size, of which **44 are real headings and 2 are artifacts** (an arXiv stamp
    and a footnote asterisk), i.e. **95.7% precision**, with the two misses
    removed by :data:`_PDF_STAMP_RE`.
    """
    try:
        from pypdf import PdfReader  # noqa: PLC0415 - optional dependency
    except ImportError:
        logger.warning("pypdf yok; PDF karşılaştırılamıyor: %s", path)
        return ""

    reader = PdfReader(str(path))
    per_page: list[list[tuple[str, float]]] = []
    for page in reader.pages:
        runs: list[tuple[str, float]] = []

        def visit(text: str, _cm, _tm, _font_dict, font_size, _runs=runs) -> None:
            if text:
                _runs.append((text, float(font_size or 0.0)))

        try:
            page.extract_text(visitor_text=visit)
        except Exception:  # noqa: BLE001 - a broken font must not lose the page
            logger.warning("sayfa okunamadı, metin çıkarımı atlandı: %s", path)
            per_page.append([])
            continue
        per_page.append(runs)

    # The body size is decided once for the whole document, not per page. Deciding
    # it per page let a table page set the mode to the table's own font, after
    # which ordinary prose on that page looked like a heading - a 400-character
    # table-notes paragraph on arXiv:1407.6566 was promoted to a section.
    weights: dict[float, int] = {}
    for runs in per_page:
        for text, size in runs:
            key = round(size, 1)
            weights[key] = weights.get(key, 0) + len(text.strip())
    body_size = max(weights, key=lambda key: weights[key]) if weights else 0.0
    heading_sizes = sorted(
        (size for size in weights if size > body_size + PDF_BODY_SIZE_TOLERANCE),
        reverse=True,
    )
    size_rank = {size: rank for rank, size in enumerate(heading_sizes, start=1)}

    pages: list[str] = []
    for runs in per_page:
        if not runs:
            pages.append("\n\f\n")
            continue
        pages.append("\n\f\n" + _mark_pdf_headings(runs, body_size, size_rank) + "\n")
    return "".join(pages)


def _mark_pdf_headings(
    runs: Sequence[tuple[str, float]], body_size: float, size_rank: dict[float, int]
) -> str:
    """Rebuild page text, prefixing heading lines with Markdown ``#``."""
    # Build the page as one string plus the font size of every character, then
    # split on newlines. Splitting the runs themselves does not work: a newline
    # arrives sometimes inside a run and sometimes as a run consisting of only
    # "\n", and split() consumes it as a separator either way, which silently
    # deleted the blank lines. With the blank lines gone, headings merged
    # ("Introduction 1.1 Clusters of Galaxies") and grouping could not know where
    # one heading ended and the next began.
    text_parts: list[str] = []
    size_parts: list[float] = []
    for run_text, run_size in runs:
        text_parts.append(run_text)
        size_parts.extend([run_size] * len(run_text))
    page_text = "".join(text_parts)

    lines: list[tuple[str, float]] = []
    offset = 0
    for chunk in page_text.split("\n"):
        width = len(chunk)
        lines.append((chunk, max(size_parts[offset : offset + width], default=0.0)))
        offset += width + 1

    # Group first, classify second. A heading set over two lines and a table caption
    # set over three are each one visual block; classifying line by line split the
    # title "The XMM-Newton/SDSS Galaxy Cluster Survey" into two level-1 sections
    # and left the second half of every caption stranded as a heading of its own.
    #
    # The grouping rule is typography, not guesswork: consecutive lines set at the
    # *same* size are one heading that wrapped. A change of size starts a new
    # heading, which is what separates "Chapter 1" (20.7pt) from "Introduction"
    # (24.8pt) from "1.1 Clusters of Galaxies" (14.3pt) on three consecutive lines
    # with no blank line between any of them. A line at body size ends the block.
    blocks: list[tuple[str, float, bool]] = []
    buffer: list[str] = []
    buffer_size = 0.0

    def close() -> None:
        if buffer:
            blocks.append((" ".join(" ".join(buffer).split()), buffer_size, True))
            buffer.clear()

    for line, size in lines:
        above = size > body_size + PDF_BODY_SIZE_TOLERANCE and line.strip()
        if above:
            if buffer and abs(size - buffer_size) > 0.05:
                close()
            buffer.append(line)
            buffer_size = size
            continue
        close()
        blocks.append((line, size, False))
    close()

    out: list[str] = []
    for block, size, candidate in blocks:
        # A running footer is a bare page number. Left in, it glued itself to the
        # next page's first line across the form feed, producing the single line
        # "126\x0c# References" - which no heading pattern can match, because the
        # marker is no longer at the start of the line. That one line is why a
        # 154-page thesis reported no bibliography at all.
        if not candidate and _PDF_FOOTER_RE.match(block.strip()):
            continue
        # Two ways to earn the heading marker, one way to emit it. The keyword
        # fallback is asked of every line, body-size ones included, because that
        # is exactly the case it exists for: on arXiv:0911.2782 the "References"
        # heading is set at 10.9pt against a 10.9pt body, so it never becomes a
        # candidate, and the whole bibliography went unexamined - the one part of
        # a thesis that most needs auditing.
        level = 0
        if _PDF_STANDALONE_KEYWORD_RE.match(" ".join(block.strip().split())):
            level = 1
        elif candidate:
            level = _pdf_heading_level(block, size, body_size, size_rank)
        if level:
            # Blank lines around the heading, and they are load-bearing. PDF text
            # carries no blank lines at all, so without them the heading merges
            # with the paragraph below it into one block and _HEADING_RE - which
            # anchors to the end of the block - never matches. That was the last
            # reason a 154-page thesis produced a single section: the 70 headings
            # were in the text and invisible. It bit the keyword path too: the
            # marker was emitted, the block was still glued to the paragraph, and
            # the section path never advanced past the previous chapter.
            if out and out[-1].strip():
                out.append("")
            out.append(f"{'#' * level} {block.strip()}")
            out.append("")
            continue
        out.append(block)
    return "\n".join(out)
