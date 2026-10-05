"""Composable pipeline steps.

Each step is an independent unit with its own failure boundary, duration
record and skip condition, so the runner stays trivial and steps stay
independently testable.

    metadata -> persist -> fetch -> extract -> references -> assets -> chunk
    -> embed -> index
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
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
from app.clients.content.mineru import ExtractionError, MineruExtractor
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
from app.domain.models import ContentPayload, ExtractedDocument, VectorRecord
from app.embeddings.base import EmbeddingProvider
from app.infra.storage import BlobStore
from app.logging import get_logger
from app.pipeline.context import PipelineContext
from app.services.chunker import ChunkingService

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

        outcome = await self._fetcher.fetch_full_text(ctx.metadata, prefer_html=ctx.prefer_html)
        ctx.content = outcome.payload
        ctx.content_kind = ContentKind(outcome.payload.kind)
        ctx.note("temp_path", outcome.temp_path)

        document = await ctx.documents.record(
            paper_id=ctx.require_paper_id(),
            kind=ctx.content_kind,
            uri=outcome.payload.uri,
            content_type=outcome.payload.content_type,
            size_bytes=outcome.payload.size_bytes,
            sha256=outcome.payload.sha256,
            source_url=outcome.payload.source_url,
            meta={"preferred": ctx.prefer_html},
        )
        ctx.note("raw_document_id", document.id)
        return StepResult(
            data={"kind": ctx.content_kind.value, "bytes": outcome.payload.size_bytes,
                  "url": outcome.payload.source_url}
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

        if content.kind == ContentKind.HTML.value:
            return await self._from_html(ctx, content)
        if content.kind == ContentKind.PDF.value:
            return await self._from_pdf(ctx, content)

        return StepResult(ok=False, data={"error": f"unsupported content kind {content.kind}"})

    # -- HTML ----------------------------------------------------------------
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
            document = await self._extractor.extract_pdf(path)
        except ExtractionError as exc:
            ctx.warn(f"MinerU extraction failed: {exc}")
            return await self._abstract_fallback(ctx, reason=str(exc))
        ctx.document = document
        return await self._persist(ctx, document, ContentKind.MINERU_MARKDOWN)

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
        derived = await ctx.documents.record(
            paper_id=ctx.require_paper_id(),
            kind=kind,
            uri=blob.uri,
            content_type="text/markdown",
            size_bytes=blob.size_bytes,
            sha256=blob.sha256,
            source_url=ctx.content.source_url if ctx.content is not None else "",
            meta={
                "source": document.source.value,
                "backend": document.backend,
                "chars": document.char_count,
                **document.meta,
            },
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
        if ctx.document is None:
            return Assets()
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

        raw_document_id = ctx.facts.get("raw_document_id")
        ctx.chunk_rows = await ctx.chunks_repo.replace_for_paper(
            ctx.require_paper_id(), ctx.chunks, raw_document_id=raw_document_id
        )
        tokens = sum(chunk.token_count for chunk in ctx.chunks)
        return StepResult(
            data={"chunks": len(ctx.chunks), "tokens": tokens,
                  "avg_tokens": round(tokens / len(ctx.chunks), 1)}
        )


# --------------------------------------------------------------------------- 8
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
                    # way pgvector filters in SQL.
                    "content_kind": row.content_kind,
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