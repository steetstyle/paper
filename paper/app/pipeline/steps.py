"""Composable pipeline steps.

Each step is an independent unit with its own failure boundary, duration
record and skip condition, so the runner stays trivial and steps stay
independently testable.

    metadata -> persist -> fetch -> extract -> references -> assets -> chunk
    -> embed -> index
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from app.clients.arxiv.client import ArxivClient
from app.clients.content.assets import (
    Assets,
    extract_assets_from_html,
    extract_assets_from_mineru,
)
from app.clients.content.fetcher import ContentFetcher
from app.clients.content.html_extractor import html_to_markdown
from app.clients.content.mineru import (
    ExtractionError,
    ExtractRequest,
    MineruExtractor,
)
from app.clients.content.references import (
    extract_references,
    extract_references_from_text,
    summarise,
)
from app.db.models import Chunk
from app.db.repositories import EmbeddingRepository
from app.db.spaces import EmbeddingSpace
from app.db.vector_store.base import VectorStore
from app.db.vector_store.schema import embedding_table, ensure_space_table
from app.domain.enums import ContentKind, ContentSource
from app.domain.models import ContentPayload, ExtractedDocument, TextChunk, VectorRecord
from app.embeddings.base import EmbeddingProvider
from app.infra.storage import BlobStore
from app.logging import get_logger
from app.pipeline.context import PipelineContext
from app.services.chunker import ChunkingService
from app.services.page_map import build_page_map
from app.services.sections import build_sections

logger = get_logger(__name__)


@dataclass(slots=True)
class StepResult:
    """What a step produced."""

    ok: bool = True
    skipped: bool = False
    data: dict[str, Any] | None = None
    fatal: bool = True
    """``False`` lets the pipeline continue (e.g. MinerU failed -> abstract only)."""

    @classmethod
    def skip(cls, reason: str, **data: Any) -> StepResult:
        return cls(skipped=True, data={"reason": reason, **data})


class Step(ABC):
    name: str = "step"

    @abstractmethod
    async def run(self, ctx: PipelineContext) -> StepResult: ...

    def should_skip(self, ctx: PipelineContext) -> str | None:
        """Return a reason string when the step can be skipped."""
        return None


# --------------------------------------------------------------------------- 1
class FetchMetadataStep(Step):
    """Resolve the target paper's metadata from ArXiv (or an already-known hit)."""

    name = "fetch_metadata"

    def __init__(self, arxiv_client: ArxivClient) -> None:
        self._arxiv = arxiv_client

    async def run(self, ctx: PipelineContext) -> StepResult:
        if ctx.metadata is not None:
            return StepResult.skip("metadata_provided", arxiv_id=ctx.metadata.arxiv_id)
        if not ctx.arxiv_id:
            return StepResult(ok=False, data={"error": "no arxiv_id or metadata provided"})
        ctx.metadata = await self._arxiv.get_paper(ctx.arxiv_id)
        ctx.arxiv_id = ctx.metadata.arxiv_id
        return StepResult(data={"arxiv_id": ctx.metadata.arxiv_id, "version": ctx.metadata.version})


# --------------------------------------------------------------------------- 2
class PersistMetadataStep(Step):
    """Upsert the paper row so everything downstream can hang off a stable id."""

    name = "persist_metadata"

    async def run(self, ctx: PipelineContext) -> StepResult:
        if ctx.metadata is None:
            return StepResult(ok=False, data={"error": "metadata unavailable"})
        paper = await ctx.papers.upsert(ctx.metadata)
        ctx.paper_id = paper.id
        return StepResult(data={"paper_id": paper.id, "versioned_id": paper.versioned_id})


# --------------------------------------------------------------------------- 3
class FetchContentStep(Step):
    """Download the HTML rendering when present, else the PDF."""

    name = "fetch_content"

    def __init__(self, fetcher: ContentFetcher) -> None:
        self._fetcher = fetcher

    def should_skip(self, ctx: PipelineContext) -> str | None:
        if ctx.force:
            return None
        return None

    async def run(self, ctx: PipelineContext) -> StepResult:
        if ctx.metadata is None:
            return StepResult(ok=False, data={"error": "metadata unavailable"})

        if not ctx.force:
            existing = await self._reuse_existing(ctx)
            if existing is not None:
                ctx.content = existing
                ctx.content_kind = ContentKind(existing.kind)
                return StepResult.skip("already_downloaded", kind=existing.kind)

        if ctx.local_path is not None:
            # A local file has no URL to fetch and no HTML rendering to prefer,
            # so the arXiv strategy does not apply to it at all.
            outcome = await self._fetcher.fetch_local(ctx.local_path)
            page_count = outcome.payload.page_count
        else:
            outcome = await self._fetcher.fetch_full_text(
                ctx.metadata, prefer_html=ctx.prefer_html
            )
            page_count = None

        pages = ctx.page_range
        if pages is not None:
            pages = pages.clamp(page_count)
            if pages is None:
                return StepResult(
                    ok=False,
                    data={
                        "error": (
                            f"pages {ctx.page_range} are past the end of a "
                            f"{page_count}-page document"
                        )
                    },
                )

        ctx.content = outcome.payload
        ctx.content_kind = ContentKind(outcome.payload.kind)
        # Read once here, used once later: the section step needs the bookmark
        # tree and reopening a 721-page book for it would be pure waste.
        ctx.pdf_outline = tuple(outcome.payload.outline or ())
        ctx.note("temp_path", outcome.temp_path)

        meta: dict[str, Any] = {"preferred": ctx.prefer_html}
        if page_count is not None:
            meta["page_count"] = page_count
        document = await ctx.documents.record(
            paper_id=ctx.require_paper_id(),
            kind=ctx.content_kind,
            uri=outcome.payload.uri,
            content_type=outcome.payload.content_type,
            size_bytes=outcome.payload.size_bytes,
            sha256=outcome.payload.sha256,
            source_url=outcome.payload.source_url,
            page_start=pages.start if pages else None,
            page_end=pages.end if pages else None,
            meta=meta,
        )
        ctx.note("raw_document_id", document.id)
        return StepResult(
            data={
                "kind": ctx.content_kind.value,
                "bytes": outcome.payload.size_bytes,
                "url": outcome.payload.source_url,
                "page_count": page_count,
                "pages": str(pages) if pages else None,
            }
        )

    async def _reuse_existing(self, ctx: PipelineContext) -> ContentPayload | None:
        paper_id = ctx.paper_id
        if paper_id is None:
            return None
        wanted = {ContentKind.HTML.value, ContentKind.PDF.value}
        rows = await ctx.documents.list_for_paper(paper_id)
        for row in rows:
            if row.kind in wanted:
                return ContentPayload(
                    kind=row.kind,
                    uri=row.uri,
                    content_type=row.content_type,
                    size_bytes=row.size_bytes,
                    sha256=row.sha256,
                    source_url=row.source_url,
                )
        return None


# --------------------------------------------------------------------------- 4
class ExtractTextStep(Step):
    """HTML -> markdown, or PDF -> MinerU -> markdown."""

    name = "extract_text"

    def __init__(self, extractor: MineruExtractor, blob_store: BlobStore) -> None:
        self._extractor = extractor
        self._blobs = blob_store

    async def run(self, ctx: PipelineContext) -> StepResult:
        content = ctx.content
        if content is None:
            return StepResult(ok=False, data={"error": "no content fetched"})

        # Resume before extracting. The decision lives in `run` rather than in
        # `should_skip` because skipping is not enough here: a resumed run has to
        # *load* the stored markdown and blocks back into the context, and
        # `should_skip` is synchronous and cannot do IO.
        reused = await self._reuse_stored(ctx)
        if reused is not None:
            return reused

        if content.kind == ContentKind.HTML.value:
            return await self._from_html(ctx, content)
        if content.kind == ContentKind.PDF.value:
            return await self._from_pdf(ctx, content)

        return StepResult(ok=False, data={"error": f"unsupported content kind {content.kind}"})

    async def _reuse_stored(
        self, ctx: PipelineContext
    ) -> StepResult | None:
        """Reload a previous extraction instead of running MinerU again.

        The single most expensive step by a wide margin — measured at about 1.7
        pages/s, so a 140-page book is minutes of GPU time and a 721-page one is
        closer to an hour. A run interrupted after extraction and retried was
        paying that cost again for a result already on disk.

        Requires the block list as well as the markdown: the markdown alone cannot
        produce page numbers or sections, so resuming without it would silently
        hand the rest of the pipeline a document with no provenance. A document
        extracted without blocks (the pypdf fallback, HTML) is therefore resumed
        as markdown-only, which is exactly as much as was stored.
        """
        if ctx.force or ctx.paper_id is None:
            return None
        rows = await ctx.documents.list_for_paper(
            ctx.paper_id, kind=ContentKind.MINERU_MARKDOWN.value
        )
        if not rows:
            rows = await ctx.documents.list_for_paper(ctx.paper_id, kind=ContentKind.TEXT.value)
        if not rows:
            return None

        newest = rows[-1]
        markdown = self._read_blob(newest.uri, newest.sha256)
        if markdown is None:
            # The blob store lost it; extracting again is the only option, and
            # silently continuing with nothing would be worse.
            logger.warning(
                "extract_resume_unavailable",
                extra={"paper_id": ctx.paper_id, "sha256": newest.sha256[:12]},
            )
            return None

        paged = await self._source_is_paged(ctx)
        if paged and not (newest.meta or {}).get("blocks_sha256"):
            # The markdown is there but the block list is not, so there are no page
            # numbers and no sections to rebuild. Resuming would hand the rest of
            # the pipeline a document *poorer* than the one already stored, and
            # quietly: the text would be identical. A full extraction costs a few
            # minutes once and self-heals, which is the cheaper trade.
            logger.info(
                "extract_resume_incomplete",
                extra={
                    "paper_id": ctx.paper_id,
                    "reason": "no stored block list for a paged document",
                },
            )
            return None

        blocks: list[dict[str, Any]] = []
        blocks_meta = newest.meta or {}
        blocks_uri = blocks_meta.get("blocks_uri")
        blocks_sha = blocks_meta.get("blocks_sha256")
        raw_blocks = (
            self._read_blob(str(blocks_uri), str(blocks_sha))
            if blocks_uri and blocks_sha
            else None
        )
        if raw_blocks is not None:
            try:
                loaded = json.loads(raw_blocks)
                if isinstance(loaded, list):
                    blocks = [b for b in loaded if isinstance(b, dict)]
            except json.JSONDecodeError as exc:
                logger.warning(
                    "extract_blocks_unreadable",
                    extra={"sha256": str(blocks_sha)[:12], "error": str(exc)},
                )
        source = ContentSource((newest.meta or {}).get("source") or ContentSource.PDF_MINERU.value)
        document = ExtractedDocument(
            markdown=markdown,
            source=source,
            backend=str((newest.meta or {}).get("backend") or "resumed"),
            text=_to_text(markdown),
            meta={**(newest.meta or {}), "resumed": True},
            blocks=blocks,
        )
        if not document.is_usable:
            return None
        ctx.document = document
        ctx.note("markdown_sha256", newest.sha256)
        ctx.note("markdown_uri", newest.uri)
        ctx.note("resumed_extraction", True)
        logger.info(
            "extract_resumed",
            extra={
                "paper_id": ctx.paper_id,
                "chars": document.char_count,
                "blocks": len(blocks),
            },
        )
        return StepResult(
            skipped=True,
            data={
                "source": document.source.value,
                "backend": document.backend,
                "chars": document.char_count,
                "blocks": len(blocks),
                "kind": ContentKind.MINERU_MARKDOWN.value,
                "sha256": newest.sha256,
                "resumed": True,
            },
        )

    # -- HTML ----------------------------------------------------------------
    async def _source_is_paged(self, ctx: PipelineContext) -> bool:
        """Whether this paper's source is a PDF, and therefore has pages.

        The distinction decides whether markdown-only is a complete resume. HTML
        has no page numbers ever, so resuming it from the markdown alone loses
        nothing; a PDF without its block list loses every page and every section.
        """
        if ctx.content is not None and ctx.content.kind == ContentKind.PDF.value:
            return True
        if ctx.paper_id is None:
            return False
        rows = await ctx.documents.list_for_paper(ctx.paper_id, kind=ContentKind.PDF.value)
        return bool(rows)

    def _read_blob(self, uri: str, sha256: str) -> str | None:
        """Read a stored blob as text, or ``None`` if it is gone.

        Through ``open``/``exists`` rather than ``path_for``: the blob store is
        an interface with an in-memory implementation, and reaching for a
        filesystem path would make resuming work only under the local store.
        """
        if not self._blobs.exists(sha256):
            return None
        try:
            with self._blobs.open(uri) as handle:
                return handle.read().decode("utf-8", errors="replace")
        except OSError as exc:
            logger.warning("extract_blob_unreadable", extra={"sha256": sha256[:12], "error": str(exc)})
            return None

    async def _from_html(self, ctx: PipelineContext, content: ContentPayload) -> StepResult:
        path = self._blobs.path_for(content.sha256)
        html = (
            path.read_text(errors="replace")
            if path is not None
            else _read_uri(content.uri)
        )

        markdown = html_to_markdown(html)
        source = (
            ContentSource.AR5IV if "ar5iv" in content.source_url else ContentSource.ARXIV_HTML
        )
        document = ExtractedDocument(
            markdown=markdown,
            source=source,
            backend="html_to_markdown",
            text=_to_text(markdown),
            meta={"html_sha256": content.sha256},
        )
        ctx.document = document
        return await self._persist(ctx, document, ContentKind.TEXT)

    # -- PDF -----------------------------------------------------------------
    async def _from_pdf(self, ctx: PipelineContext, content: ContentPayload) -> StepResult:
        path = self._blobs.path_for(content.sha256)
        if path is None:
            return StepResult(ok=False, data={"error": f"blob missing for {content.sha256}"})
        try:
            document = await self._extractor.extract_pdf(path, self._request(ctx))
        except ExtractionError as exc:
            ctx.warn(f"MinerU extraction failed: {exc}")
            return await self._abstract_fallback(ctx, reason=str(exc))
        ctx.document = document
        return await self._persist(ctx, document, ContentKind.MINERU_MARKDOWN)

    def _request(self, ctx: PipelineContext) -> ExtractRequest | None:
        """Turn the run's per-document intent into an extractor request.

        Returns None for a plain arXiv paper so the common path allocates
        nothing and the extractor's defaults apply exactly as before.
        """
        if ctx.page_range is None and not ctx.mineru_options.has_overrides:
            return None
        return ExtractRequest(options=ctx.mineru_options, page_range=ctx.page_range)

    # -- shared --------------------------------------------------------------
    async def _persist(
        self, ctx: PipelineContext, document: ExtractedDocument, kind: ContentKind
    ) -> StepResult:
        if not document.is_usable:
            ctx.warn("extraction produced no text")
            return await self._abstract_fallback(ctx, reason="empty extraction")

        blob = self._blobs.put_bytes(document.markdown.encode("utf-8"), prefix="derived")
        ctx.note("markdown_sha256", blob.sha256)
        ctx.note("markdown_uri", blob.uri)

        # The block list is stored alongside the markdown, not left in memory.
        # Without it a resumed run has the text but no page numbers and no
        # sections, because MinerU's `page_idx` and `bbox` live only in the
        # blocks — the markdown has neither. That is the difference between
        # resuming and redoing, so it is written as a first-class artefact.
        meta: dict[str, Any] = {
            "source": document.source.value,
            "backend": document.backend,
            "chars": document.char_count,
            **document.meta,
        }
        if document.blocks:
            blocks_blob = self._blobs.put_bytes(
                json.dumps(document.blocks, separators=(",", ":")).encode("utf-8"),
                prefix="derived",
            )
            ctx.note("blocks_sha256", blocks_blob.sha256)
            # The URI, not only the hash: the store's `open` is keyed on the URI,
            # and it has an in-memory implementation with no filesystem path to
            # fall back on. A hash alone would make the blocks unreloadable there.
            meta["blocks_sha256"] = blocks_blob.sha256
            meta["blocks_uri"] = blocks_blob.uri

        derived = await ctx.documents.record(
            paper_id=ctx.require_paper_id(),
            kind=kind,
            uri=blob.uri,
            content_type="text/markdown",
            size_bytes=blob.size_bytes,
            sha256=blob.sha256,
            source_url=ctx.content.source_url if ctx.content is not None else "",
            meta=meta,
        )
        ctx.note("markdown_document_id", derived.id)
        ctx.note("derived_document_id", derived.id)
        return StepResult(
            data={
                "source": document.source.value,
                "backend": document.backend,
                "chars": document.char_count,
                "blocks": len(document.blocks),
                "kind": kind.value,
                "sha256": blob.sha256,
            }
        )

    async def _abstract_fallback(self, ctx: PipelineContext, *, reason: str) -> StepResult:
        """Never lose a paper: index metadata + abstract when extraction fails."""
        if ctx.metadata is None:
            return StepResult(ok=False, data={"error": reason})
        markdown = f"# {ctx.metadata.title}\n\n**Abstract**\n\n{ctx.metadata.abstract}"
        ctx.document = ExtractedDocument(
            markdown=markdown,
            source=ContentSource.ABSTRACT_ONLY,
            backend="abstract",
            text=_to_text(markdown),
            meta={"fallback_reason": reason},
            warnings=(reason,),
        )
        ctx.note("fallback", "abstract_only")
        return StepResult(ok=True, fatal=False, data={"fallback": "abstract_only", "reason": reason})


def _to_text(markdown: str) -> str:
    from app.infra.text import strip_markdown

    return strip_markdown(markdown)


def _read_uri(uri: str) -> str:
    from pathlib import Path

    if uri.startswith("file://"):
        return Path(uri[len("file://") :]).read_text(errors="replace")
    return ""


# --------------------------------------------------------------------------- 5
class ExtractReferencesStep(Step):
    """Read the bibliography out of the paper's HTML and store the edges.

    Runs after text extraction because it needs the same downloaded source. The
    markdown extractor deliberately drops ``.ltx_bibliography`` — a bibliography
    is not body text — so the references come from the HTML directly.

    Never fatal. Papers without a machine-readable bibliography are the norm for
    older PDFs, and losing a citation graph is not a reason to lose the paper.
    """

    name = "extract_references"

    def __init__(self, blob_store: BlobStore) -> None:
        self._blobs = blob_store

    async def run(self, ctx: PipelineContext) -> StepResult:
        paper_id = ctx.paper_id
        if paper_id is None:
            return StepResult.skip("no paper row yet")

        html = self._html_for(ctx)
        references = extract_references(html)
        if not references and ctx.document is not None:
            references = extract_references_from_text(ctx.document.markdown)

        stats = await ctx.references.replace_for_paper(
            paper_id, references, source=ctx.content_kind or ""
        )
        # Also resolve references written by *other* papers that point here: this
        # paper may be the one they were waiting for.
        await ctx.references.link_to_papers()
        ctx.reference_count = stats["total"]
        return StepResult(
            data={
                **stats,
                "source": "html" if html else "markdown",
                **summarise(references),
            }
        )

    def _html_for(self, ctx: PipelineContext) -> str | None:
        content = ctx.content
        if content is None or content.kind != ContentKind.HTML.value:
            return None
        path = self._blobs.path_for(content.sha256)
        if path is None:
            return None
        try:
            html = path.read_text(errors="replace")
        except OSError as exc:  # pragma: no cover - unreadable blob
            logger.warning("references_html_unreadable", extra={"error": str(exc)})
            return None
        return html if "ltx_bibitem" in html else None


# --------------------------------------------------------------------------- 6
class ExtractAssetsStep(Step):
    """Store the paper's figures, tables and equations.

    Runs after text extraction because it reads the same downloaded source, and
    which source that is decides what we can get:

    - HTML gives every equation (139 inline + 3 display on 1706.03762v7) with
      captions and image URLs, but no page numbers and no image bytes.
    - MinerU gives page numbers, bounding boxes and a cropped image for every
      asset, but only display equations (5 on the same paper).

    So both sources are read when available, and the `source` column records
    which one produced a row.

    Never fatal: a rendering without figures is normal, and losing a figure is
    not a reason to lose the paper.
    """

    name = "extract_assets"

    def __init__(self, blob_store: BlobStore) -> None:
        self._blobs = blob_store

    async def run(self, ctx: PipelineContext) -> StepResult:
        paper_id = ctx.paper_id
        if paper_id is None:
            return StepResult.skip("no paper row yet")

        assets = self._from_html(ctx).merge(self._from_mineru(ctx)).deduplicate()
        if assets.is_empty():
            return StepResult(
                ok=True, fatal=False, data={"reason": "no assets in the source"}
            )

        counts = await ctx.assets.replace_for_paper(paper_id, assets)
        ctx.asset_counts = counts
        return StepResult(
            data={
                **counts,
                "display_equations": sum(1 for e in assets.equations if e.is_display),
                "figures_with_image": sum(1 for f in assets.figures if f.has_image()),
                "sources": sorted({a.source for a in _all(assets)}),
            }
        )

    def _from_html(self, ctx: PipelineContext) -> Assets:
        content = ctx.content
        if content is None or content.kind != ContentKind.HTML.value:
            return Assets()
        path = self._blobs.path_for(content.sha256)
        if path is None:
            return Assets()
        try:
            html = path.read_text(errors="replace")
        except OSError as exc:  # pragma: no cover - unreadable blob
            logger.warning("assets_html_unreadable", extra={"error": str(exc)})
            return Assets()
        # page_url, not page_url + "/": see extract_assets_from_html.
        return extract_assets_from_html(html, page_url=content.source_url)

    def _from_mineru(self, ctx: PipelineContext) -> Assets:
        """Assets from the MinerU block list already in memory.

        Read from the document rather than from the output directory, because the
        extractor deletes that directory before this step runs. Falling back to the
        path is kept for callers that kept their artifacts (``keep_artifacts``),
        where the images are still on disk and worth resolving.
        """
        if ctx.document is None:
            return Assets()
        blocks = ctx.document.blocks
        if blocks:
            return extract_assets_from_mineru(blocks=blocks)
        raw = ctx.document.meta.get("mineru_output_dir")
        if not raw:
            return Assets()
        return extract_assets_from_mineru(Path(str(raw)))


def _all(assets: Assets) -> list[Any]:  # noqa: ANN401
    return [*assets.figures, *assets.tables, *assets.equations]


# --------------------------------------------------------------------------- 7
class ChunkTextStep(Step):
    name = "chunk_text"

    def __init__(self, chunker: ChunkingService) -> None:
        self._chunker = chunker

    async def run(self, ctx: PipelineContext) -> StepResult:
        if ctx.document is None or not ctx.document.is_usable:
            if ctx.metadata is None:
                return StepResult(ok=False, data={"error": "nothing to chunk"})
            ctx.chunks = self._chunker.chunk_abstract(ctx.metadata.title, ctx.metadata.abstract)
        else:
            header = ctx.metadata.title if ctx.metadata else None
            ctx.chunks = self._chunker.chunk(
                ctx.document.markdown,
                source=ctx.document.source,
                prefix_header=header,
            )
        if not ctx.chunks:
            return StepResult(ok=False, data={"error": "chunking produced no chunks"})

        located = self._locate_pages(ctx)
        raw_document_id = ctx.facts.get("raw_document_id")
        ctx.chunk_rows = await ctx.chunks_repo.replace_for_paper(
            ctx.require_paper_id(), located, raw_document_id=raw_document_id
        )
        tokens = sum(chunk.token_count for chunk in located)
        with_pages = sum(1 for chunk in located if chunk.page_start is not None)
        return StepResult(
            data={
                "chunks": len(located),
                "tokens": tokens,
                "avg_tokens": round(tokens / len(located), 1),
                "chunks_with_pages": with_pages,
            }
        )

    def _locate_pages(self, ctx: PipelineContext) -> list[TextChunk]:
        """Attach page numbers by matching each chunk back to MinerU's blocks.

        Only possible when the extraction produced blocks: HTML has no pages, and
        the pypdf fallback emits one markdown block per page without a block list
        to match against. In those cases the chunks are returned untouched, with
        ``page_start`` left ``None`` — the same answer an arXiv paper gets today.

        A chunk that cannot be located inherits the previous chunk's page rather
        than staying empty: chunks are consecutive slices of one document, so the
        chunk before it is the best available evidence. Measured on a 25-page
        extraction this converts the misses into near-misses instead of blanks,
        and it can only be wrong about a chunk that contains no locatable text.
        """
        blocks = ctx.document.blocks if ctx.document else []
        if not blocks:
            return ctx.chunks

        page_map = build_page_map(
            blocks,
            page_offset=ctx.page_range.start if ctx.page_range else 1,
            total_pages=ctx.metadata.page_count if ctx.metadata else None,
        )
        if not page_map.usable:
            return ctx.chunks
        ctx.page_map = page_map

        cursor = 0
        previous: int | None = None
        located: list[TextChunk] = []
        for chunk in ctx.chunks:
            page, cursor = page_map.page_for(chunk.text, cursor)
            if page is None:
                page = previous
            else:
                previous = page
            located.append(replace(chunk, page_start=page, page_end=page))
        ctx.note("pages_located", sum(1 for c in located if c.page_start is not None))
        return located


# --------------------------------------------------------------------------- 8
class BuildSectionsStep(Step):
    """Store the document's own structure and point its chunks at it.

    Runs after chunking because a chunk can only be placed in a section once its
    page is known, and the page comes from the same MinerU blocks the chunking
    step used. The two share one page map through the context rather than reading
    ``content_list.json`` twice.

    Never fatal. A document with neither bookmarks nor detectable headings simply
    has no sections — which is a degraded navigation experience, not a reason to
    lose the text that was extracted successfully.
    """

    name = "build_sections"

    async def run(self, ctx: PipelineContext) -> StepResult:
        paper_id = ctx.paper_id
        if paper_id is None:
            return StepResult.skip("no paper row yet")

        # HTML has no pages, so nothing built from it can be navigated to. Skipped
        # here rather than filtered downstream so the run says *why* instead of
        # reporting "no structure found" — which reads like the document has none,
        # when in truth its headings are the renderer's furniture.
        if ctx.content is not None and ctx.content.kind != ContentKind.PDF.value:
            return StepResult(
                ok=True,
                fatal=False,
                data={
                    "sections": 0,
                    "reason": f"{ctx.content.kind} source has no pages to point at",
                },
            )

        markdown = ctx.document.markdown if ctx.document else ""
        sections = build_sections(
            outline=tuple(ctx.pdf_outline),
            markdown=markdown,
            page_map=ctx.page_map,
            document_pages=ctx.metadata.page_count if ctx.metadata else None,
        )
        if not sections:
            return StepResult(
                ok=True,
                fatal=False,
                data={"sections": 0, "reason": "no navigable structure found"},
            )

        rows = await ctx.sections_repo.replace_for_paper(paper_id, sections)
        ctx.sections = list(rows)
        linked = await ctx.chunks_repo.assign_sections(ctx.chunk_rows, rows)
        ctx.note("sections", len(rows))
        return StepResult(
            data={
                "sections": len(rows),
                "from_outline": sum(1 for r in rows if r.source == "outline"),
                "from_markdown": sum(1 for r in rows if r.source != "outline"),
                "chunks_in_sections": linked,
            }
        )


# --------------------------------------------------------------------------- 9
class EmbedChunksStep(Step):
    """Embed chunks and write them into this space's own table."""

    name = "embed_chunks"

    def __init__(
        self,
        provider: EmbeddingProvider,
        space: EmbeddingSpace,
        *,
        batch_size: int = 64,
        ensure_table: bool = True,
    ) -> None:
        self._provider = provider
        self._space = space
        self._batch_size = batch_size
        self._ensure_table = ensure_table
        self._table_ready = False
        self._table = embedding_table(space)

    @property
    def space(self) -> EmbeddingSpace:
        return self._space

    async def run(self, ctx: PipelineContext) -> StepResult:
        rows = list(ctx.chunk_rows) or await self._load_rows(ctx)
        ctx.chunk_rows = rows
        if not rows:
            return StepResult(ok=False, data={"error": "no chunks to embed"})

        # The space owns the width; the provider must agree with it. Catching it
        # here beats a confusing "expected N-dim vectors" further down.
        expected = self._space.dimensions
        if self._provider.dimensions != expected:
            return StepResult(
                ok=False,
                data={
                    "error": (
                        f"space {self._space.name!r} is {expected}d but provider "
                        f"{self._provider.model!r} emits {self._provider.dimensions}d"
                    )
                },
            )

        vectors = await self._embed_all([row.text for row in rows])
        bad = next((v for v in vectors if len(v) != expected), None)
        if bad is not None:
            return StepResult(
                ok=False, data={"error": f"expected {expected}-dim vectors, got {len(bad)}"}
            )

        fingerprint = self._space.fingerprint
        records = [
            (row, self._space.provider, self._space.model, expected, vector, fingerprint)
            for row, vector in zip(rows, vectors, strict=True)
        ]
        await self._make_table_ready(ctx)
        repo = EmbeddingRepository(ctx.session, self._table)
        ctx.embedded_count = await repo.replace_for_paper(ctx.require_paper_id(), records)
        ctx.note("fingerprint", fingerprint)
        ctx.note("space", self._space.name)
        return StepResult(
            data={
                "embeddings": ctx.embedded_count,
                "fingerprint": fingerprint,
                "space": self._space.name,
                "table": self._space.resolved_table,
            }
        )

    async def _make_table_ready(self, ctx: PipelineContext) -> None:
        """Create this space's table on first use, honouring VECTOR_AUTO_CREATE.

        One catalogue query per step instance, after which it is skipped.
        """
        if self._table_ready or not self._ensure_table:
            return
        connection = await ctx.session.connection()
        created = await connection.run_sync(
            lambda conn: ensure_space_table(conn, self._space)
        )
        self._table_ready = True
        if created:
            ctx.note("space_table_created", self._space.resolved_table)

    async def _embed_all(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            out.extend(await self._provider.embed_documents(batch))
        return out

    async def _load_rows(self, ctx: PipelineContext) -> list[Chunk]:
        return list(await ctx.chunks_repo.list_for_paper(ctx.require_paper_id()))


# --------------------------------------------------------------------------- 9
class IndexVectorsStep(Step):
    """Copy vectors from the space's table into an external vector store.

    Skipped for pgvector, where that table *is* the index — writing twice would
    double the cost of the largest table in the system for no benefit.
    """

    name = "index_vectors"

    def __init__(self, vector_store: VectorStore) -> None:
        self._store = vector_store
        self._table = embedding_table(vector_store.space)

    @property
    def space(self) -> EmbeddingSpace:
        return self._store.space

    async def run(self, ctx: PipelineContext) -> StepResult:
        rows = list(ctx.chunk_rows) or list(
            await ctx.chunks_repo.list_for_paper(ctx.require_paper_id())
        )
        if not rows:
            return StepResult(ok=False, data={"error": "no chunks to index"})

        if self._store.persists_relationally:
            return StepResult(
                skipped=True,
                data={
                    "reason": "store_is_relational",
                    "store": self._store.name,
                    "space": self.space.name,
                    "indexed": len(rows),
                },
            )

        fingerprint = self.space.fingerprint
        vectors = await self._load_vectors(ctx, rows)

        # Chunk ids are regenerated on every ingest, so a plain upsert would
        # leave the previous generation's vectors orphaned in the store.
        removed = await self._store.delete_for_papers([ctx.require_paper_id()])
        records = [
            VectorRecord(
                chunk_id=row.id,
                paper_id=row.paper_id,
                vector=vector,
                payload={
                    "provider": self.space.provider,
                    "model": self.space.model,
                    "fingerprint": fingerprint,
                    "paper_id": row.paper_id,
                    "ordinal": row.ordinal,
                    "source": row.source,
                    # Carried so the in-memory store can filter by kind the same
                    # way pgvector filters in SQL. `section_ordinal` and
                    # `page_start` are here for the same reason: a section is a
                    # first-class scope for a search, and a backend that cannot
                    # filter on it can only be told about it after ranking.
                    "content_kind": row.content_kind,
                    "section_ordinal": row.section_ordinal,
                    "page_start": row.page_start,
                    "heading": row.heading,
                    "categories": _categories_for(ctx, row),
                },
            )
            for row, vector in zip(rows, vectors, strict=True)
        ]
        written = await self._store.upsert(records)
        return StepResult(
            data={
                "indexed": written,
                "removed_stale": removed,
                "store": self._store.name,
                "space": self.space.name,
            }
        )

    async def _load_vectors(self, ctx: PipelineContext, rows: list[Chunk]) -> list[list[float]]:
        repo = EmbeddingRepository(ctx.session, self._table)
        by_chunk = await repo.load_vectors([row.id for row in rows])
        missing = [row.id for row in rows if not by_chunk.get(row.id)]
        if missing:
            raise RuntimeError(f"{len(missing)} chunks have no embedding row")
        return [by_chunk[row.id] for row in rows]


def _categories_for(ctx: PipelineContext, row: Chunk) -> list[str]:
    if ctx.metadata is not None:
        return list(ctx.metadata.categories)
    return []


# --------------------------------------------------------------------------- 10
class FinalizeStep(Step):
    """Stamp ingestion metadata on the paper row."""

    name = "finalize"

    async def run(self, ctx: PipelineContext) -> StepResult:
        if ctx.paper_id is None:
            return StepResult(ok=False, data={"error": "paper_id missing"})
        await ctx.papers.mark_ingested(ctx.paper_id)
        return StepResult(
            data={
                "chunks": len(ctx.chunks),
                "embeddings": ctx.embedded_count,
                "source": ctx.source.value,
                "warnings": ctx.warnings,
            }
        )


__all__ = [
    "Step",
    "StepResult",
    "FetchMetadataStep",
    "PersistMetadataStep",
    "FetchContentStep",
    "ExtractTextStep",
    "ChunkTextStep",
    "EmbedChunksStep",
    "IndexVectorsStep",
    "FinalizeStep",
]