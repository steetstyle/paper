"""ORM models.

Relational tables carry everything that benefits from joins/aggregation
(metadata, content provenance, chunk text, run bookkeeping) while the
``embeddings`` table holds the vectors and is indexed by pgvector / Qdrant.

Embeddings store the provider + model fingerprint so a model swap is detectable
and old vectors can be invalidated explicitly rather than silently.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import TIMESTAMP, Base, VectorColumn, utcnow


def _uuid() -> str:
    return uuid.uuid4().hex


class Paper(Base):
    __tablename__ = "papers"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    arxiv_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    versioned_id: Mapped[str] = mapped_column(String(72), index=True)
    version: Mapped[int | None] = mapped_column(Integer)

    title: Mapped[str] = mapped_column(Text)
    abstract: Mapped[str] = mapped_column(Text, default="")

    # Authors and categories live in their own tables (see PaperAuthor /
    # PaperCategory) so they can be joined and aggregated. The properties
    # `categories` and `author_names` below keep the
    # read side looking like the old JSON columns.
    primary_category: Mapped[str | None] = mapped_column(String(32), index=True)
    """Denormalised for the common `WHERE primary_category = ?` lookup; the
    authoritative value is the `is_primary` flag on :class:`PaperCategory`."""

    published_at: Mapped[datetime | None] = mapped_column(TIMESTAMP)
    updated_at_arxiv: Mapped[datetime | None] = mapped_column(TIMESTAMP)
    ingested_at: Mapped[datetime | None] = mapped_column(TIMESTAMP)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(TIMESTAMP, default=utcnow, onupdate=utcnow)

    doi: Mapped[str | None] = mapped_column(String(255))
    comment: Mapped[str | None] = mapped_column(Text)
    journal_ref: Mapped[str | None] = mapped_column(Text)
    abs_url: Mapped[str] = mapped_column(String(512), default="")
    pdf_url: Mapped[str] = mapped_column(String(512), default="")
    html_url: Mapped[str | None] = mapped_column(String(512))

    raw_entry: Mapped[dict[str, Any] | None] = mapped_column(JSON, default=None)

    contents: Mapped[list[RawDocument]] = relationship(
        back_populates="paper", cascade="all, delete-orphan", lazy="selectin"
    )
    chunks: Mapped[list[Chunk]] = relationship(
        back_populates="paper", cascade="all, delete-orphan", lazy="noload"
    )
    embeddings: Mapped[list[Embedding]] = relationship(
        back_populates="paper", cascade="all, delete-orphan", lazy="noload"
    )
    author_links: Mapped[list[PaperAuthor]] = relationship(
        back_populates="paper",
        cascade="all, delete-orphan",
        order_by="PaperAuthor.ordinal",
        # selectin, not the default lazy load: `categories` / `author_names` are
        # read on nearly every response, and a deferred load from async code
        # raises MissingGreenlet. `raiseload` on the writer side is not wanted,
        # so an explicit refresh happens in the repository instead.
        lazy="selectin",
    )
    category_links: Mapped[list[PaperCategory]] = relationship(
        back_populates="paper",
        cascade="all, delete-orphan",
        order_by="PaperCategory.ordinal",
        lazy="selectin",
    )

    @property
    def categories(self) -> list[str]:
        """Category codes, primary first, in ArXiv's declared order."""
        return [link.category for link in self.category_links]

    @property
    def author_names(self) -> list[str]:
        return [link.author.name for link in self.author_links if link.author]

    def authors(self) -> list[dict[str, Any]]:
        """Authors as ``{name, affiliation}`` dicts, matching the old JSON shape."""
        return [
            {"name": link.author.name, "affiliation": link.affiliation}
            for link in self.author_links
            if link.author
        ]

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Paper {self.arxiv_id} {self.title[:40]!r}>"


class Author(Base):
    """A distinct author, shared across every paper they wrote.

    ArXiv gives names as free text with no identifier, so ``normalized_name`` is
    what actually deduplicates: ``"Ashish Vaswani"`` and ``"ashish  vaswani"``
    are the same person, and storing them twice would split their papers.
    """

    __tablename__ = "authors"
    __table_args__ = (Index("ix_authors_normalized_name", "normalized_name"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(Text)
    normalized_name: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP, default=utcnow)

    paper_links: Mapped[list[PaperAuthor]] = relationship(back_populates="author")

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Author {self.name}>"


class PaperAuthor(Base):
    """Ordered many-to-many between :class:`Paper` and :class:`Author`.

    ``ordinal`` matters: ArXiv's author list is a citation-order sequence, not a
    set. ``affiliation`` lives here rather than on the author because it is a
    property of *this* paper, not of the person.
    """

    __tablename__ = "paper_authors"
    __table_args__ = (
        # The pair is the identity, and a surrogate id keeps the ORM and the
        # upsert path from having to invent one.
        UniqueConstraint("paper_id", "author_id", name="uq_paper_authors_paper_author"),
        UniqueConstraint("paper_id", "ordinal", name="uq_paper_authors_paper_ordinal"),
        Index("ix_paper_authors_author", "author_id"),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    paper_id: Mapped[str] = mapped_column(ForeignKey("papers.id", ondelete="CASCADE"))
    author_id: Mapped[str] = mapped_column(ForeignKey("authors.id", ondelete="CASCADE"))
    ordinal: Mapped[int] = mapped_column(Integer)
    affiliation: Mapped[str | None] = mapped_column(Text)

    paper: Mapped[Paper] = relationship(back_populates="author_links")
    author: Mapped[Author] = relationship(back_populates="paper_links", lazy="joined")


class Category(Base):
    """A subject category code such as ``cs.CL``.

    One row per code, referenced by :class:`PaperCategory`. ``parent`` and
    ``depth`` are the taxonomy ArXiv defines (``cs.CL.MS`` -> ``cs.CL`` -> ``cs``),
    kept so a query can walk up the hierarchy in SQL.
    """

    __tablename__ = "categories"

    code: Mapped[str] = mapped_column(String(32), primary_key=True)
    label: Mapped[str | None] = mapped_column(Text)
    parent: Mapped[str | None] = mapped_column(String(32), index=True)
    depth: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP, default=utcnow)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Category {self.code}>"


class PaperCategory(Base):
    """Ordered many-to-many between :class:`Paper` and :class:`Category`.

    Exactly one row per paper carries ``is_primary``, which is what ArXiv's
    ``arxiv:primary_category`` means.
    """

    __tablename__ = "paper_categories"
    __table_args__ = (
        UniqueConstraint("paper_id", "category", name="uq_paper_categories_paper_category"),
        Index("ix_paper_categories_category", "category"),
        Index("ix_paper_categories_paper_primary", "paper_id", "is_primary"),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    paper_id: Mapped[str] = mapped_column(ForeignKey("papers.id", ondelete="CASCADE"))
    category: Mapped[str] = mapped_column(ForeignKey("categories.code", ondelete="CASCADE"))
    ordinal: Mapped[int] = mapped_column(Integer)
    is_primary: Mapped[bool] = mapped_column(Boolean, default=False)

    paper: Mapped[Paper] = relationship(back_populates="category_links")
    category_row: Mapped[Category] = relationship(lazy="noload")

    def __repr__(self) -> str:  # pragma: no cover
        return f"<PaperCategory {self.category}{' primary' if self.is_primary else ''}>"


class RawDocument(Base):
    """A stored source or derived artefact, addressed by content hash."""

    __tablename__ = "raw_documents"
    __table_args__ = (
        UniqueConstraint("paper_id", "kind", "sha256", name="uq_raw_documents_paper_kind_sha"),
        Index("ix_raw_documents_paper_kind", "paper_id", "kind"),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    paper_id: Mapped[str] = mapped_column(ForeignKey("papers.id", ondelete="CASCADE"), index=True)
    kind: Mapped[str] = mapped_column(String(32), index=True)
    source_url: Mapped[str] = mapped_column(String(1024), default="")
    uri: Mapped[str] = mapped_column(String(2048))
    content_type: Mapped[str] = mapped_column(String(128), default="")
    size_bytes: Mapped[int] = mapped_column(BigInteger, default=0)
    sha256: Mapped[str] = mapped_column(String(64), index=True)
    meta: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP, default=utcnow)

    paper: Mapped[Paper] = relationship(back_populates="contents")

    def __repr__(self) -> str:  # pragma: no cover
        return f"<RawDocument {self.kind} {self.sha256[:12]}>"


class Chunk(Base):
    """A retrievable text unit. Vector lives in :class:`Embedding`."""

    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint("paper_id", "ordinal", name="uq_chunks_paper_ordinal"),
        Index("ix_chunks_paper_ordinal", "paper_id", "ordinal"),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    paper_id: Mapped[str] = mapped_column(ForeignKey("papers.id", ondelete="CASCADE"), index=True)
    raw_document_id: Mapped[str | None] = mapped_column(
        ForeignKey("raw_documents.id", ondelete="SET NULL"), nullable=True
    )

    ordinal: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    # Indexed because "only equations" / "only figures" is a first-class filter,
    # not a post-hoc scan.
    content_kind: Mapped[str] = mapped_column(String(16), default="body", index=True)
    token_count: Mapped[int] = mapped_column(Integer, default=0)
    char_start: Mapped[int] = mapped_column(Integer, default=0)
    char_end: Mapped[int] = mapped_column(Integer, default=0)
    heading: Mapped[str | None] = mapped_column(Text)
    section_path: Mapped[list[str]] = mapped_column(JSON, default=list)
    content_hash: Mapped[str] = mapped_column(String(64), index=True)
    source: Mapped[str] = mapped_column(String(32), default="")
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP, default=utcnow)

    paper: Mapped[Paper] = relationship(back_populates="chunks")
    embedding: Mapped[Embedding | None] = relationship(
        back_populates="chunk", cascade="all, delete-orphan", uselist=False, lazy="noload"
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Chunk {self.paper_id[:8]}#{self.ordinal} {self.token_count}t>"


class Embedding(Base):
    """Vector for one chunk plus the fingerprint of the model that made it."""

    __tablename__ = "embeddings"
    __table_args__ = (
        UniqueConstraint("chunk_id", name="uq_embeddings_chunk"),
        Index("ix_embeddings_paper", "paper_id"),
        Index("ix_embeddings_provider_model", "provider", "model"),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    chunk_id: Mapped[str] = mapped_column(ForeignKey("chunks.id", ondelete="CASCADE"), index=True)
    paper_id: Mapped[str] = mapped_column(ForeignKey("papers.id", ondelete="CASCADE"), index=True)

    provider: Mapped[str] = mapped_column(String(64))
    model: Mapped[str] = mapped_column(String(255))
    dimensions: Mapped[int] = mapped_column(Integer)
    fingerprint: Mapped[str] = mapped_column(String(128), index=True)

    vector = mapped_column(VectorColumn(1536), nullable=True)
    meta: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP, default=utcnow)

    chunk: Mapped[Chunk] = relationship(back_populates="embedding")
    paper: Mapped[Paper] = relationship(back_populates="embeddings")

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Embedding {self.model} d={self.dimensions}>"


class IngestionRun(Base):
    """One end-to-end ingestion attempt for a single paper."""

    __tablename__ = "ingestion_runs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    paper_id: Mapped[str | None] = mapped_column(
        ForeignKey("papers.id", ondelete="CASCADE"), nullable=True, index=True
    )
    arxiv_id: Mapped[str | None] = mapped_column(String(64), index=True)

    status: Mapped[str] = mapped_column(String(16), default="pending", index=True)
    requested_by: Mapped[str] = mapped_column(String(64), default="api")
    trigger: Mapped[str] = mapped_column(String(32), default="manual")
    prefer_html: Mapped[bool] = mapped_column(Boolean, default=True)
    force: Mapped[bool] = mapped_column(Boolean, default=False)

    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    embedding_count: Mapped[int] = mapped_column(Integer, default=0)
    content_kind: Mapped[str | None] = mapped_column(String(32))
    error: Mapped[str | None] = mapped_column(Text)
    timings: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    started_at: Mapped[datetime | None] = mapped_column(TIMESTAMP)
    finished_at: Mapped[datetime | None] = mapped_column(TIMESTAMP)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP, default=utcnow, index=True)

    steps: Mapped[list[PipelineStepRun]] = relationship(
        back_populates="run",
        cascade="all, delete-orphan",
        order_by="PipelineStepRun.attempt",
        lazy="selectin",
    )
    paper: Mapped[Paper | None] = relationship(lazy="noload")

    @property
    def duration_ms(self) -> float | None:
        if self.started_at and self.finished_at:
            return (self.finished_at - self.started_at).total_seconds() * 1000
        return None


class PipelineStepRun(Base):
    """Per-step bookkeeping so failures are attributable and observable."""

    __tablename__ = "pipeline_step_runs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    run_id: Mapped[str] = mapped_column(ForeignKey("ingestion_runs.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(64))
    attempt: Mapped[int] = mapped_column(Integer, default=1)
    status: Mapped[str] = mapped_column(String(16), default="pending")
    duration_ms: Mapped[float | None] = mapped_column(Float)
    error: Mapped[str | None] = mapped_column(Text)
    meta: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    started_at: Mapped[datetime] = mapped_column(TIMESTAMP)
    finished_at: Mapped[datetime | None] = mapped_column(TIMESTAMP)

    run: Mapped[IngestionRun] = relationship(back_populates="steps")


class EmbeddingSpaceRecord(Base):
    """Registry of the embedding spaces this deployment knows about.

    One row per model. The physical vector table is derived from the row (see
    :mod:`app.db.spaces`), so adding a model is *data*, not a migration of the
    existing rows.
    """

    __tablename__ = "embedding_spaces"

    name: Mapped[str] = mapped_column(String(64), primary_key=True)
    provider: Mapped[str] = mapped_column(String(64))
    model: Mapped[str] = mapped_column(String(255))
    dimensions: Mapped[int] = mapped_column(Integer)
    distance: Mapped[str] = mapped_column(String(16), default="cosine")
    table_name: Mapped[str | None] = mapped_column(String(128), unique=True)
    description: Mapped[str | None] = mapped_column(Text)

    is_active: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    is_locked: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default="0"
    )
    """Set for spaces derived from settings; managed spaces cannot be dropped."""

    created_at: Mapped[datetime] = mapped_column(TIMESTAMP, default=utcnow)
    last_used_at: Mapped[datetime | None] = mapped_column(TIMESTAMP)
    vector_count: Mapped[int] = mapped_column(Integer, default=0)

    def as_space(self):
        from app.db.spaces import space_from_row  # noqa: PLC0415

        return space_from_row(self)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<EmbeddingSpaceRecord {self.name} {self.model}@{self.dimensions}>"


class PaperReference(Base):
    """One entry of a paper's bibliography.

    Modelled the way citations actually work: the great majority of cited works
    are **not** in the corpus, so ``cited_paper_id`` is nullable and the entry
    carries its own denormalised copy of what the bibliography said. That is what
    lets a reference be stored before — or without — the cited paper ever being
    ingested.

    ``cited_arxiv_id`` is the identity that makes later resolution possible: when
    that paper *is* ingested, :meth:`link_to_papers` points the row at it instead
    of duplicating it.
    """

    __tablename__ = "paper_references"
    __table_args__ = (
        # Re-extracting a paper's bibliography replaces it wholesale; ordinal is
        # unique per citing paper so there are no orphans and no ambiguity.
        UniqueConstraint("citing_paper_id", "ordinal", name="uq_paper_references_citing_ordinal"),
        Index("ix_paper_references_citing", "citing_paper_id"),
        Index("ix_paper_references_cited", "cited_paper_id"),
        Index("ix_paper_references_cited_arxiv", "cited_arxiv_id"),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    citing_paper_id: Mapped[str] = mapped_column(
        ForeignKey("papers.id", ondelete="CASCADE"), index=True
    )
    cited_paper_id: Mapped[str | None] = mapped_column(
        ForeignKey("papers.id", ondelete="SET NULL"), nullable=True, index=True
    )
    cited_arxiv_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)

    title: Mapped[str | None] = mapped_column(Text)
    authors: Mapped[str | None] = mapped_column(Text)
    year: Mapped[int | None] = mapped_column(Integer)
    venue: Mapped[str | None] = mapped_column(Text)
    raw_text: Mapped[str | None] = mapped_column(Text)
    """The bibliography entry as printed, kept because every field above is a
    lossy guess at it."""

    ordinal: Mapped[int] = mapped_column(Integer)
    source: Mapped[str] = mapped_column(String(32), default="")

    citing_paper: Mapped[Paper] = relationship(foreign_keys=[citing_paper_id], lazy="noload")
    cited_paper: Mapped[Paper | None] = relationship(
        foreign_keys=[cited_paper_id], lazy="noload"
    )

    @property
    def is_resolved(self) -> bool:
        """True when the cited paper is in our own corpus."""
        return self.cited_paper_id is not None

    def label(self) -> str:
        """One-line citation, degrading gracefully when fields are missing."""
        who = (self.authors or "").strip()
        who = who.split(",")[0].split(" and ")[0].strip() if who else "?"
        title = (self.title or self.raw_text or "").strip()
        year = f" ({self.year})" if self.year else ""
        return f"{who}{year} — {title[:90]}" if title else f"{who}{year}"

    def __repr__(self) -> str:  # pragma: no cover
        return f"<PaperReference [{self.ordinal}] {self.label()[:50]}>"


class Project(Base):
    """A named collection of papers.

    Papers are **not** owned by a project: they live once in :class:`Paper` and a
    project joins to them through :class:`ProjectPaper`. Importing a paper into a
    second project adds a link, never a copy, so the corpus stays global and no
    paper can drift between two project views of itself.
    """

    __tablename__ = "projects"
    __table_args__ = (Index("ix_projects_name", "name"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(200))
    slug: Mapped[str] = mapped_column(String(200), unique=True)
    description: Mapped[str | None] = mapped_column(Text)
    is_archived: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(TIMESTAMP, default=utcnow, onupdate=utcnow)

    paper_links: Mapped[list[ProjectPaper]] = relationship(
        back_populates="project", cascade="all, delete-orphan", lazy="noload"
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Project {self.slug}>"


class ProjectPaper(Base):
    """Membership of a paper in a project.

    ``note`` and ``is_read`` are per project, which is the point of the join
    table: two projects can hold the same paper with different notes and
    different read state, and neither can see the other's.
    """

    __tablename__ = "project_papers"
    __table_args__ = (
        UniqueConstraint("project_id", "paper_id", name="uq_project_papers_project_paper"),
        # Reverse lookup: which projects hold this paper.
        Index("ix_project_papers_paper", "paper_id"),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"))
    paper_id: Mapped[str] = mapped_column(ForeignKey("papers.id", ondelete="CASCADE"))
    added_by: Mapped[str] = mapped_column(String(64), default="")
    note: Mapped[str | None] = mapped_column(Text)
    is_read: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP, default=utcnow)

    project: Mapped[Project] = relationship(back_populates="paper_links")
    paper: Mapped[Paper] = relationship(lazy="selectin")

    def __repr__(self) -> str:  # pragma: no cover
        return f"<ProjectPaper {self.project_id[:8]}/{self.paper_id[:8]}>"


class EmbeddingJob(Base):

    __tablename__ = "embedding_jobs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    fingerprint: Mapped[str] = mapped_column(String(128), index=True)
    status: Mapped[str] = mapped_column(String(16), default="pending", index=True)
    total: Mapped[int] = mapped_column(Integer, default=0)
    processed: Mapped[int] = mapped_column(Integer, default=0)
    failed: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(TIMESTAMP, default=utcnow, onupdate=utcnow)


__all__ = [
    "Base",
    "Paper",
    "Author",
    "PaperAuthor",
    "Category",
    "PaperCategory",
    "PaperReference",
    "Project",
    "ProjectPaper",
    "RawDocument",
    "Chunk",
    "Embedding",
    "IngestionRun",
    "PipelineStepRun",
    "EmbeddingJob",
]