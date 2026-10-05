"""Pure domain objects.

These carry no I/O and no framework imports so they can be produced by the
ArXiv client, the fetcher, the extractor and the pipeline alike.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from app.domain.enums import ChunkKind, ContentSource

__all__ = [
    "Author",
    "PaperMetadata",
    "SearchQuery",
    "SearchResultPage",
    "SearchHit",
    "ContentPayload",
    "ExtractedDocument",
    "TextChunk",
    "EmbeddingVector",
    "VectorRecord",
    "SearchHitWithScore",
]


@dataclass(frozen=True, slots=True)
class Author:
    name: str
    affiliation: str | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("author name must not be empty")


@dataclass(frozen=True, slots=True)
class Reference:
    """One bibliography entry, before it is matched against the corpus.

    Everything here is optional except ``raw_text`` because a bibliography is
    free text: the fields are a lossy best effort, and ``raw_text`` is the only
    part that is always true.
    """

    raw_text: str
    ordinal: int
    title: str | None = None
    authors: str | None = None
    year: int | None = None
    venue: str | None = None
    cited_arxiv_id: str | None = None
    """Present when the entry names an arXiv paper, which is what allows the
    reference to be linked to a real corpus row later."""

    def label(self) -> str:
        who = (self.authors or "").strip()
        who = who.split(",")[0].split(" and ")[0].strip() if who else "?"
        title = (self.title or self.raw_text).strip()
        year = f" ({self.year})" if self.year else ""
        return f"{who}{year} — {title[:90]}"


@dataclass(frozen=True, slots=True)
class ProjectInfo:
    """A project as returned by the repository."""

    id: str
    name: str
    slug: str
    description: str | None = None
    is_archived: bool = False
    paper_count: int = 0
    read_count: int = 0
    created_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class PaperMetadata:
    """Normalised ArXiv metadata for one paper version."""

    arxiv_id: str
    versioned_id: str
    version: int | None
    title: str
    abstract: str
    authors: tuple[Author, ...] = ()
    categories: tuple[str, ...] = ()
    primary_category: str | None = None
    published_at: datetime | None = None
    updated_at: datetime | None = None
    doi: str | None = None
    comment: str | None = None
    journal_ref: str | None = None
    abs_url: str = ""
    pdf_url: str = ""
    html_url: str | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def latest_versioned_id(self) -> str:
        """Versioned id pointing at the most recent version."""
        return f"{self.arxiv_id}v{self.version}" if self.version else self.arxiv_id

    def author_names(self) -> list[str]:
        return [a.name for a in self.authors]


@dataclass(frozen=True, slots=True)
class SearchQuery:
    """Structured ArXiv query.

    Values inside one field are OR-ed, fields are AND-ed, which mirrors how
    users expect ``ti:"a" OR ti:"b" AND cat:cs.LG`` style narrowing to behave.
    """

    all_terms: tuple[str, ...] = ()
    title_terms: tuple[str, ...] = ()
    author_terms: tuple[str, ...] = ()
    abstract_terms: tuple[str, ...] = ()
    category_terms: tuple[str, ...] = ()
    id_list: tuple[str, ...] = ()
    raw: str | None = None
    max_results: int = 10
    start: int = 0
    sort_by: str = "relevance"
    sort_order: str = "descending"

    def is_empty(self) -> bool:
        return not (
            self.all_terms
            or self.title_terms
            or self.author_terms
            or self.abstract_terms
            or self.category_terms
            or self.id_list
            or self.raw
        )


@dataclass(frozen=True, slots=True)
class SearchHit:
    metadata: PaperMetadata
    relevance_score: float | None = None


@dataclass(frozen=True, slots=True)
class SearchResultPage:
    hits: tuple[SearchHit, ...]
    total_results: int
    start: int
    items_per_page: int

    def __len__(self) -> int:
        return len(self.hits)

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self.hits)

    @property
    def has_more(self) -> bool:
        return self.start + len(self.hits) < self.total_results


@dataclass(frozen=True, slots=True)
class ContentPayload:
    """A downloaded source artefact (HTML or PDF)."""

    kind: str  # ContentKind value
    uri: str  # blob store location
    content_type: str
    size_bytes: int
    sha256: str
    source_url: str
    encoding: str | None = None
    local_path: str | None = None


@dataclass(frozen=True, slots=True)
class ExtractedDocument:
    """Normalised extraction result regardless of which backend produced it."""

    markdown: str
    source: ContentSource
    backend: str
    text: str
    meta: dict[str, Any] = field(default_factory=dict)
    blocks: list[dict[str, Any]] = field(default_factory=list, repr=False)
    warnings: tuple[str, ...] = ()

    @property
    def is_usable(self) -> bool:
        return len(self.text.strip()) > 0

    @property
    def char_count(self) -> int:
        return len(self.text)


@dataclass(frozen=True, slots=True)
class TextChunk:
    """A retrievable unit of text."""

    ordinal: int
    text: str
    token_count: int
    char_start: int
    char_end: int
    heading: str | None = None
    section_path: tuple[str, ...] = ()
    content_hash: str = ""
    source: ContentSource = ContentSource.ABSTRACT_ONLY
    kind: ChunkKind = ChunkKind.BODY
    """What this chunk is, decided once by the chunker. See
    :func:`app.services.chunk_kinds.classify_chunk`."""

    def with_hash(self) -> TextChunk:
        import hashlib

        return TextChunk(
            ordinal=self.ordinal,
            text=self.text,
            token_count=self.token_count,
            char_start=self.char_start,
            char_end=self.char_end,
            heading=self.heading,
            section_path=self.section_path,
            content_hash=hashlib.sha256(self.text.encode("utf-8")).hexdigest(),
            source=self.source,
            kind=self.kind,
        )


@dataclass(frozen=True, slots=True)
class EmbeddingVector:
    """An embedding plus the identity of the model that produced it."""

    chunk_id: str
    paper_id: str
    vector: list[float]
    provider: str
    model: str
    dimensions: int

    def fingerprint(self) -> str:
        """Used to invalidate cached vectors when the model/dims change."""
        return f"{self.provider}:{self.model}:{self.dimensions}"


@dataclass(frozen=True, slots=True)
class VectorRecord:
    """Upsert payload for a :class:`~app.db.vector_store.base.VectorStore`."""

    chunk_id: str
    paper_id: str
    vector: list[float]
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SearchHitWithScore:
    chunk_id: str
    paper_id: str
    score: float
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)