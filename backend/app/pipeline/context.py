"""Pipeline context.

Mutable, explicitly-passed state shared by every step. Steps read what earlier
steps wrote and may declare their outputs as ``skippable`` so a resumed run
does not redo expensive work (downloads, MinerU) it already has.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.asset_repository import AssetRepository
from app.db.models import Chunk
from app.db.reference_repository import ReferenceRepository
from app.db.repositories import (
    ChunkRepository,
    PaperRepository,
    RawDocumentRepository,
    RunRepository,
)
from app.domain.enums import ContentKind, ContentSource
from app.domain.models import (
    ContentPayload,
    ExtractedDocument,
    PaperMetadata,
    SearchQuery,
    TextChunk,
)


@dataclass(slots=True)
class PipelineContext:
    """Everything a step needs, nothing it does not."""

    session: AsyncSession
    run_id: str

    # ------------------------------------------------------------- input
    arxiv_id: str | None = None
    query: SearchQuery | None = None
    prefer_html: bool = True
    force: bool = False

    # ------------------------------------------------- populated as we go
    metadata: PaperMetadata | None = None
    paper_id: str | None = None
    content: ContentPayload | None = None
    content_kind: ContentKind | None = None
    document: ExtractedDocument | None = None
    chunks: list[TextChunk] = field(default_factory=list)
    chunk_rows: list[Chunk] = field(default_factory=list)
    embedded_count: int = 0
    reference_count: int = 0
    asset_counts: dict[str, int] = field(default_factory=dict)

    # ------------------------------------------------------- bookkeeping
    timings: dict[str, float] = field(default_factory=dict)
    facts: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    # ------------------------------------------------------ repositories
    # Always available; constructed from the session so steps never build
    # repositories themselves and never need optional-checks. The embedding
    # repository is absent on purpose: it is scoped to one embedding space, so
    # the embed/index steps own it.
    # Blob store for image hashing; set by the container so the context stays
    # free of infrastructure construction.
    blob_store: Any = field(default=None, repr=False)
    papers: PaperRepository = field(init=False)
    documents: RawDocumentRepository = field(init=False)
    chunks_repo: ChunkRepository = field(init=False)
    runs: RunRepository = field(init=False)
    references: ReferenceRepository = field(init=False)
    assets: AssetRepository = field(init=False)

    def __post_init__(self) -> None:
        self.papers = PaperRepository(self.session)
        self.documents = RawDocumentRepository(self.session)
        self.chunks_repo = ChunkRepository(self.session)
        self.runs = RunRepository(self.session)
        self.references = ReferenceRepository(self.session)
        # Needs the blob store so cropped images can be hashed in.
        self.assets = AssetRepository(self.session, blobs=self.blob_store)

    # ------------------------------------------------------------- helpers
    def require_paper_id(self) -> str:
        if self.paper_id is None:
            raise RuntimeError("paper_id is not set — run the metadata step first")
        return self.paper_id

    def require_content(self) -> ContentPayload:
        if self.content is None:
            raise RuntimeError("content is not set — run the fetch step first")
        return self.content

    def note(self, key: str, value: Any) -> None:
        self.facts[key] = value

    def warn(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)

    @property
    def source(self) -> ContentSource:
        if self.document is not None:
            return self.document.source
        return ContentSource.ABSTRACT_ONLY