"""Pydantic schemas for the HTTP layer (kept separate from domain objects)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.api.devices import DeviceField
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
    content_kind: str = Field(
        default="body",
        description="What this chunk is: body, abstract, figure, table, equation, "
        "reference, code. The same value `--content` / `content_kinds` filters on.",
    )


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
    device: DeviceField = Field(
        default=None,
        description=(
            "Device for this run's embedding steps: cpu, cuda, cuda:1, mps, npu, "
            "or a device index. Defaults to the configured device (cpu). Asking "
            "for cuda loads a second copy of the model onto the GPU and holds it "
            "there for the duration of the run; anything else is a 422."
        ),
    )
    # No extract_device: MinerU options reach a run only through
    # IngestionService.ingest_file(options=MineruOptions(...)), and the arxiv_id /
    # query paths below have no such parameter. A field accepted and then dropped
    # would be worse than an absent one, so extraction device stays a CLI concern
    # (`paper ingest --extract-device`) until the service grows the argument.


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
    device: DeviceField = Field(
        default=None,
        description=(
            "Device to embed this query on: cpu, cuda, cuda:1, mps, npu, or a "
            "device index. Defaults to the configured device (cpu). Only worth "
            "setting for a local model — a remote provider's latency is not this "
            "process's to move."
        ),
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
    section: str | None = Field(
        default=None,
        description=(
            "Restrict to a named part of a document, written the way a reader "
            "refers to it: the book's own numbering ('2.2') or a word from its "
            "title ('Debye'). Resolved before ranking, so `top_k` counts matches "
            "from that section rather than the section's best matches out of a "
            "corpus-wide top-k. A name that matches nothing is a 422, never an "
            "empty result set: 'this section does not discuss it' and 'you "
            "misnamed it' are otherwise the same answer."
        ),
    )
    doc_key: str | None = Field(
        default=None,
        description=(
            "Which document `section` is a section of: a doc_key, arXiv id or "
            "internal id. Omitted, the name is matched across the whole corpus, "
            "which for a word as ordinary as 'Introduction' is hundreds of "
            "sections in books you did not mean. Needs `section`: to search one "
            "document outright, use `paper_ids`."
        ),
    )


class SemanticSearchHit(BaseModel):
    chunk_id: str
    paper_id: str
    score: float
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class SectionOut(BaseModel):
    """One section a ``section`` name resolved to.

    Field for field what :meth:`app.services.sections_query.SectionMatch.as_dict`
    returns, because that is already the wire shape the other two surfaces answer
    with: one section match spelled three ways is the drift the shared resolver
    exists to prevent.

    Note the two titles: ``title`` is the *document's*, ``section`` is the
    heading's.
    """

    doc_key: str
    title: str
    ordinal: int
    section: str
    level: int
    page_start: int | None = None
    page_end: int | None = None
    pages: str = "-"
    """The page range as a reader would say it: ``24``, ``24-31``, or ``-``."""
    source: str = ""
    """Which structure this row came from: ``outline`` (the document's own
    bookmarks) or ``markdown`` (headings recovered from the extracted text)."""
    matched: str = "contains"
    """How the name matched the title: ``exact``, ``prefix`` or ``contains``."""


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
    sections: list[SectionOut] = Field(
        default_factory=list,
        description=(
            "The sections `section` widened to, best match first. Empty when no "
            "section was asked for. A word often names several sections, and a "
            "caller who cannot see which ones answered has been told nothing "
            "about how narrow the search really was."
        ),
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
