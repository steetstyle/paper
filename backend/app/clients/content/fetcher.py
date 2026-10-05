"""Retrieve the best available full text for a paper.

Strategy (configurable via ``prefer_html``):

1. If ArXiv published an HTML5 rendering (``arxiv.org/html/<id>v<n>``) download it.
2. Otherwise fall back to ``ar5iv`` (LaTeXML conversion of older papers).
3. Otherwise download the PDF and leave conversion to the extraction layer.

ArXiv recommends that bulk downloads be spaced out, which the shared
``HttpFetcher`` rate limiter enforces.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

from app.config import ArxivSettings
from app.domain.enums import ContentKind, ContentSource
from app.domain.ids import parse_arxiv_id, url_for_ar5iv, url_for_html, url_for_pdf
from app.domain.models import ContentPayload, PaperMetadata
from app.infra.http import ContentTooLarge, HttpError, HttpFetcher
from app.infra.storage import BlobStore, LocalBlobStore, StoredBlob
from app.logging import get_logger

logger = get_logger(__name__)


class ContentUnavailable(RuntimeError):
    """No HTML rendering and no downloadable PDF."""


@dataclass(slots=True)
class FetchOutcome:
    payload: ContentPayload
    blob: StoredBlob
    temp_path: Path | None = None
    source: ContentSource = ContentSource.ARXIV_HTML

    def cleanup(self) -> None:
        if self.temp_path is not None and self.temp_path.exists():
            self.temp_path.unlink(missing_ok=True)


class ContentFetcher:
    def __init__(
        self,
        settings: ArxivSettings,
        fetcher: HttpFetcher,
        blob_store: BlobStore | None = None,
    ) -> None:
        self._settings = settings
        self._http = fetcher
        self._blobs = blob_store or LocalBlobStore()

    # ------------------------------------------------------------------- HTML
    async def fetch_html(self, paper: PaperMetadata) -> FetchOutcome | None:
        """Try ArXiv HTML, then ar5iv. Returns ``None`` when neither exists."""
        identifier = parse_arxiv_id(paper.versioned_id or paper.arxiv_id)
        candidates: list[tuple[str, ContentSource]] = []

        html_url = paper.html_url or url_for_html(identifier.id, identifier.version)
        candidates.append((html_url, ContentSource.ARXIV_HTML))
        if self._settings.allow_ar5iv_fallback:
            candidates.append((url_for_ar5iv(identifier.id, identifier.version), ContentSource.AR5IV))

        for url, source in candidates:
            try:
                if not await self._http.exists(url):
                    logger.info("html_absent", extra={"url": url})
                    continue
                return await self._download_html(url, source)
            except (HttpError, ContentTooLarge) as exc:
                logger.warning("html_failed", extra={"url": url, "error": str(exc)})
                continue
        return None

    async def _download_html(self, url: str, source: ContentSource) -> FetchOutcome:
        response = await self._http.get_text(url, max_bytes=self._settings.max_html_bytes)
        raw = response.content
        blob = self._blobs.put_bytes(raw, prefix="raw")
        temp_path = Path(tempfile.mkdtemp(prefix="paper-html-")) / "source.html"
        temp_path.write_bytes(raw)
        content_type = response.headers.get("content-type", "text/html").split(";")[0]
        logger.info(
            "html_downloaded", extra={"url": url, "bytes": len(raw), "source": source.value}
        )
        return FetchOutcome(
            payload=ContentPayload(
                kind=ContentKind.HTML,
                uri=blob.uri,
                content_type=content_type,
                size_bytes=len(raw),
                sha256=blob.sha256,
                source_url=url,
                encoding=response.charset_encoding,
                local_path=str(temp_path),
            ),
            blob=blob,
            temp_path=temp_path,
            source=source,
        )

    # -------------------------------------------------------------------- PDF
    async def fetch_pdf(self, paper: PaperMetadata) -> FetchOutcome:
        identifier = parse_arxiv_id(paper.versioned_id or paper.arxiv_id)
        url = paper.pdf_url or url_for_pdf(identifier.id, identifier.version)
        temp_dir = Path(tempfile.mkdtemp(prefix="paper-pdf-"))
        temp_path = temp_dir / f"{identifier.id.replace('/', '_')}.pdf"
        sha256, size = await self._http.download(
            url, temp_path, max_bytes=self._settings.max_pdf_bytes
        )
        blob = self._blobs.put_file(temp_path, prefix="raw")
        logger.info("pdf_downloaded", extra={"url": url, "bytes": size})
        return FetchOutcome(
            payload=ContentPayload(
                kind=ContentKind.PDF,
                uri=blob.uri,
                content_type="application/pdf",
                size_bytes=size,
                sha256=blob.sha256,
                source_url=url,
                local_path=str(temp_path),
            ),
            blob=blob,
            temp_path=temp_path,
            source=ContentSource.PDF_MINERU,
        )

    # -------------------------------------------------------------- combined
    async def fetch_full_text(
        self, paper: PaperMetadata, *, prefer_html: bool = True
    ) -> FetchOutcome:
        """Return HTML when available, else PDF. Raises when neither works."""
        if prefer_html:
            html = await self.fetch_html(paper)
            if html is not None:
                return html
        try:
            return await self.fetch_pdf(paper)
        except ContentTooLarge as exc:
            raise ContentUnavailable(f"PDF too large to ingest: {exc}") from exc
        except HttpError as exc:
            if not prefer_html:
                raise ContentUnavailable(
                    f"no HTML rendering and PDF download failed: {exc}"
                ) from exc
            raise ContentUnavailable(
                f"no HTML rendering and PDF download failed: {exc}"
            ) from exc

    @property
    def blob_store(self) -> BlobStore:
        return self._blobs