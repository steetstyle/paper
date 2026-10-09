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
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from checker_app.config import CheckerSettings
from checker_app.domain.models import SourceDocument
from checker_app.domain.text import Token, tokenize
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
        return SourceDocument(
            source_id=loaded.source_id(),
            name=_display_name(loaded.path),
            kind=loaded.kind,
            text=text,
            tokens=tuple(tokens),
            page_count=text.count("\f") + 1,
            meta=dict(loaded.meta or {}),
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


def _extract_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader  # noqa: PLC0415 - optional dependency
    except ImportError:
        logger.warning("pypdf yok; PDF karşılaştırılamıyor: %s", path)
        return ""
    reader = PdfReader(str(path))
    pages: list[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(f"\f{text}")  # form feed keeps the page map
    return "".join(pages)
