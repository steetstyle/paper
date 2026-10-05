"""Pydantic schemas for the HTTP layer (kept separate from domain objects)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.domain.enums import RunStatus, StepStatus


class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class AuthorOut(BaseModel):
    name: str
    affiliation: str | None = None


class PaperOut(ORMModel):
    arxiv_id: str
    versioned_id: str
    title: str
    abstract: str
    authors: list[AuthorOut] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)
    primary_category: str | None = None
    published_at: datetime | None = None
    updated_at: datetime | None = None
    doi: str | None = None
    comment: str | None = None
    journal_ref: str | None = None
    abs_url: str = ""
    pdf_url: str = ""
    html_url: str | None = None
    chunk_count: int = 0
    ingested_at: datetime | None = None


class SearchResponse(BaseModel):
    total_results: int
    start: int
    items_per_page: int
    papers: list[PaperOut]
    filtered_out: int = 0
    next_start: int | None = None
    warnings: list[str] = Field(default_factory=list)
    hint: str | None = None
    """Why an empty or narrowed result set looks the way it does."""
    search_query: str | None = None
    """Exactly what was sent to ArXiv, so callers can audit the filter."""
    post_filter: dict[str, object] = Field(default_factory=dict)


class ContentOut(BaseModel):
    kind: str
    uri: str
    content_type: str
    size_bytes: int
    sha256: str
    source_url: str


class ChunkOut(ORMModel):
    ordinal: int
    text: str
    token_count: int
    heading: str | None = None
    section_path: list[str] = Field(default_factory=list)
    source: str


class PaperDetail(PaperOut):
    contents: list[ContentOut] = Field(default_factory=list)


class IngestRequest(BaseModel):
    arxiv_id: str | None = Field(default=None, description="Single ArXiv id, versioned id or URL.")
    query: str | None = Field(default=None, description="Raw ArXiv search query to harvest ids from.")
    limit: int = Field(default=1, ge=1, le=100)
    prefer_html: bool = True
    force: bool = False
    space: str | None = Field(
        default=None,
        description="Embedding space (model) to write. Defaults to the active space.",
    )


class IngestAccepted(BaseModel):
    run_ids: list[str]
    paper_ids: list[str]
    status: RunStatus
    space: str = "default"


class StepRunOut(ORMModel):
    name: str
    status: StepStatus
    attempt: int = 1
    duration_ms: float | None = None
    error: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None


class RunOut(ORMModel):
    id: str
    paper_id: str | None = None
    arxiv_id: str | None = None
    status: RunStatus
    error: str | None = None
    chunk_count: int = 0
    embedding_count: int = 0
    content_kind: str | None = None
    trigger: str = "manual"
    requested_by: str = "api"
    steps: list[StepRunOut] = Field(default_factory=list)
    timings: dict[str, Any] = Field(default_factory=dict)
    started_at: datetime | None = None
    finished_at: datetime | None = None


class SemanticSearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=10, ge=1, le=100)
    category: str | None = None
    paper_ids: list[str] | None = None
    min_score: float | None = None
    space: str | None = Field(
        default=None,
        description="Embedding space (model) to query. Defaults to the active space.",
    )
    project: str | None = Field(
        default=None,
        description="Restrict to the papers of this project. Intersects paper_ids.",
    )
    content_kinds: list[str] | None = Field(
        default=None,
        description=(
            "Restrict to chunks of these kinds: body, abstract, figure, table, "
            "equation, reference, code. Aliases such as 'equations' or 'figs' "
            "are accepted. An unknown value is a 422, never a silent no-match."
        ),
    )
    sources: list[str] | None = None


class SemanticSearchHit(BaseModel):
    chunk_id: str
    paper_id: str
    score: float
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class SemanticSearchResponse(BaseModel):
    query: str
    provider: str
    model: str
    space: str = "default"
    dimensions: int = 0
    hits: list[SemanticSearchHit]
    scope: dict[str, object] = Field(
        default_factory=dict,
        description="What the search was actually restricted to, as applied.",
    )


class HealthResponse(BaseModel):
    status: str
    version: str
    database: str
    vector_store: str
    embeddings: str
    mineru_backends: list[str] = Field(default_factory=list)
    mineru_detail: dict[str, str] = Field(default_factory=dict)
    mineru_version: str | None = None
    active_space: str = "default"
    space_count: int = 0

# --------------------------------------------------------------------- projects
class ProjectOut(BaseModel):
    id: str
    slug: str
    name: str
    description: str | None = None
    is_archived: bool = False
    paper_count: int = 0
    read_count: int = 0


class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: str | None = None


class ProjectPaperOut(BaseModel):
    arxiv_id: str
    versioned_id: str
    title: str
    categories: list[str] = Field(default_factory=list)
    primary_category: str | None = None
    authors: list[str] = Field(default_factory=list)
    is_read: bool = False
    note: str | None = None
    projects: list[str] = Field(default_factory=list)
    """Every project holding this paper — the reverse view of an import."""


class ImportResult(BaseModel):
    project: str
    requested: int
    resolved: int
    added: int
    missing: list[str] = Field(default_factory=list)
    papers: list[str] = Field(default_factory=list)


# ------------------------------------------------------------------ references
class ReferenceOut(BaseModel):
    ordinal: int
    cited_arxiv_id: str | None = None
    cited_paper_id: str | None = None
    resolved: bool = False
    """True when the cited paper is in this corpus."""
    authors: str | None = None
    title: str | None = None
    year: int | None = None
    venue: str | None = None
    label: str = ""
    raw_text: str | None = None


class ReferencesResponse(BaseModel):
    arxiv_id: str
    direction: str = "references"
    count: int = 0
    references: list[ReferenceOut] = Field(default_factory=list)
    cited_by: list[str] = Field(default_factory=list)
    """arXiv ids of stored papers citing this one, for ``direction=cited-by``."""


# ---------------------------------------------------------------------- assets
class FigureOut(BaseModel):
    ordinal: int
    label: str | None = None
    caption: str | None = None
    image_url: str | None = None
    image_sha256: str | None = None
    page_idx: int | None = None
    width: int | None = None
    height: int | None = None
    source: str = "html"


class TableAssetOut(BaseModel):
    ordinal: int
    label: str | None = None
    caption: str | None = None
    body_html: str | None = None
    row_count: int | None = None
    column_count: int | None = None
    image_url: str | None = None
    image_sha256: str | None = None
    page_idx: int | None = None
    source: str = "html"


class EquationOut(BaseModel):
    ordinal: int
    latex: str | None = None
    is_display: bool = False
    page_idx: int | None = None
    image_sha256: str | None = None
    source: str = "html"


class AssetCounts(BaseModel):
    figures: int = 0
    tables: int = 0
    equations: int = 0
    display_equations: int = 0


class AssetsResponse(BaseModel):
    arxiv_id: str
    title: str
    counts: AssetCounts = Field(default_factory=AssetCounts)
    figures: list[FigureOut] = Field(default_factory=list)
    tables: list[TableAssetOut] = Field(default_factory=list)
    equations: list[EquationOut] = Field(default_factory=list)
