"""initial schema

Papers, their content and provenance, vectors, run bookkeeping, the author and
category taxonomy, projects, and the citation graph.

There is no JSON-to-table migration here on purpose: authors, categories and
references were designed as tables from the start rather than converted from
columns later, so nothing has to be backfilled.

Reference handling is the reason for two choices worth naming:

- ``papers.cited_paper_id`` is a nullable self-join on ``papers``. Most cited
  works are not in the corpus, so a reference has to be storable before the
  cited paper is ever ingested. ``cited_arxiv_id`` keeps the identity that makes
  a later resolution possible.
- ``ordinal`` is unique per citing paper, so re-extracting a paper's references
  replaces them wholesale without leaving orphans.

Revision ID: 0001_initial
Revises:
Create Date: 2026-01-01 00:00:00
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from app.config import get_settings

revision: str = "0001_initial"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

TIMESTAMP = sa.DateTime(timezone=True)
# pgvector columns are fixed-width, so the migration must agree with
# EMBEDDING_DIMENSIONS. Changing it afterwards needs a new migration, because
# the column width cannot be altered in place.
DIMENSIONS = get_settings().embedding.dimensions


def upgrade() -> None:
    is_postgres = op.get_bind().dialect.name == "postgresql"

    if is_postgres:
        # pgvector must exist before the vector column is created.
        op.execute("CREATE EXTENSION IF NOT EXISTS vector")
        from pgvector.sqlalchemy import Vector

        # A real type, not `sa.text("vector(n)")`: a TextClause is not a
        # SchemaItem, so create_table rejects it.
        vector_type = Vector(DIMENSIONS)
    else:
        vector_type = sa.JSON()

    def boolean() -> sa.types.TypeEngine:
        return sa.Boolean()

    # ------------------------------------------------------------------ papers
    op.create_table(
        "papers",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("arxiv_id", sa.String(64), nullable=False, unique=True),
        sa.Column("versioned_id", sa.String(72), nullable=False),
        sa.Column("version", sa.Integer()),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("abstract", sa.Text(), nullable=False, server_default=""),
        # Denormalised for the common `WHERE primary_category = ?` lookup; the
        # authoritative value is the is_primary flag on paper_categories.
        sa.Column("primary_category", sa.String(32)),
        sa.Column("published_at", TIMESTAMP),
        sa.Column("updated_at_arxiv", TIMESTAMP),
        sa.Column("ingested_at", TIMESTAMP),
        sa.Column("created_at", TIMESTAMP, nullable=False),
        sa.Column("updated_at", TIMESTAMP, nullable=False),
        sa.Column("doi", sa.String(255)),
        sa.Column("comment", sa.Text()),
        sa.Column("journal_ref", sa.Text()),
        sa.Column("abs_url", sa.String(512), nullable=False, server_default=""),
        sa.Column("pdf_url", sa.String(512), nullable=False, server_default=""),
        sa.Column("html_url", sa.String(512)),
        sa.Column("raw_entry", sa.JSON()),
    )
    op.create_index("ix_papers_arxiv_id", "papers", ["arxiv_id"], unique=True)
    op.create_index("ix_papers_versioned_id", "papers", ["versioned_id"])
    op.create_index("ix_papers_primary_category", "papers", ["primary_category"])

    # ----------------------------------------------------------------- authors
    # One row per person. ArXiv gives names as free text with no identifier, so
    # normalized_name is what actually deduplicates.
    op.create_table(
        "authors",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("normalized_name", sa.String(255), nullable=False),
        sa.Column("created_at", TIMESTAMP, nullable=False),
    )
    op.create_index("ix_authors_normalized_name", "authors", ["normalized_name"])

    # The author list is a citation-order sequence, not a set, hence `ordinal`.
    # Affiliation belongs here rather than on the author: it is a fact about
    # this paper, not about the person.
    op.create_table(
        "paper_authors",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column(
            "paper_id",
            sa.String(32),
            sa.ForeignKey("papers.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "author_id",
            sa.String(32),
            sa.ForeignKey("authors.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("affiliation", sa.Text()),
        sa.UniqueConstraint("paper_id", "author_id", name="uq_paper_authors_paper_author"),
        sa.UniqueConstraint("paper_id", "ordinal", name="uq_paper_authors_paper_ordinal"),
    )
    op.create_index("ix_paper_authors_author", "paper_authors", ["author_id"])

    # -------------------------------------------------------------- categories
    # `parent`/`depth` are the taxonomy ArXiv defines (cs.CL.MS -> cs.CL -> cs),
    # so a subtree can be selected in SQL.
    op.create_table(
        "categories",
        sa.Column("code", sa.String(32), primary_key=True),
        sa.Column("label", sa.Text()),
        sa.Column("parent", sa.String(32)),
        sa.Column("depth", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", TIMESTAMP, nullable=False),
    )
    op.create_index("ix_categories_parent", "categories", ["parent"])

    op.create_table(
        "paper_categories",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column(
            "paper_id",
            sa.String(32),
            sa.ForeignKey("papers.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "category",
            sa.String(32),
            sa.ForeignKey("categories.code", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("is_primary", boolean(), nullable=False, server_default=sa.text("false")),
        sa.UniqueConstraint("paper_id", "category", name="uq_paper_categories_paper_category"),
    )
    op.create_index("ix_paper_categories_category", "paper_categories", ["category"])
    op.create_index("ix_paper_categories_paper_primary", "paper_categories", ["paper_id", "is_primary"])

    # ------------------------------------------------------ paper_references
    # The citation graph. `cited_paper_id` is nullable because most cited works
    # are external; `cited_arxiv_id` is what lets a reference be resolved once
    # that paper is ingested.
    op.create_table(
        "paper_references",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column(
            "citing_paper_id",
            sa.String(32),
            sa.ForeignKey("papers.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "cited_paper_id",
            sa.String(32),
            sa.ForeignKey("papers.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("cited_arxiv_id", sa.String(64)),
        sa.Column("title", sa.Text()),
        sa.Column("authors", sa.Text()),
        sa.Column("year", sa.Integer()),
        sa.Column("venue", sa.Text()),
        sa.Column("raw_text", sa.Text()),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("source", sa.String(32), nullable=False, server_default=""),
        sa.UniqueConstraint("citing_paper_id", "ordinal", name="uq_paper_references_citing_ordinal"),
    )
    op.create_index("ix_paper_references_citing", "paper_references", ["citing_paper_id"])
    op.create_index("ix_paper_references_cited", "paper_references", ["cited_paper_id"])
    op.create_index("ix_paper_references_cited_arxiv", "paper_references", ["cited_arxiv_id"])

    # -------------------------------------------------------------- projects
    # A project is a named collection. Papers stay global: one row in `papers`,
    # referenced by however many projects need it. Importing a paper into a
    # project must never copy it.
    op.create_table(
        "projects",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("slug", sa.String(200), nullable=False, unique=True),
        sa.Column("description", sa.Text()),
        sa.Column("is_archived", boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("created_at", TIMESTAMP, nullable=False),
        sa.Column("updated_at", TIMESTAMP, nullable=False),
    )
    op.create_index("ix_projects_name", "projects", ["name"])

    op.create_table(
        "project_papers",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column(
            "project_id",
            sa.String(32),
            sa.ForeignKey("projects.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "paper_id",
            sa.String(32),
            sa.ForeignKey("papers.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("added_by", sa.String(64), nullable=False, server_default=""),
        sa.Column("note", sa.Text()),
        sa.Column("is_read", boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("created_at", TIMESTAMP, nullable=False),
        sa.UniqueConstraint("project_id", "paper_id", name="uq_project_papers_project_paper"),
    )
    # Reverse lookup: "which projects hold this paper".
    op.create_index("ix_project_papers_paper", "project_papers", ["paper_id"])

    # ---------------------------------------------------- figures and tables
    # Three tables rather than one polymorphic asset row, because each carries a
    # payload the others do not have (a table has body_html, an equation has
    # latex). What they genuinely share is the image/page/source block, repeated
    # here once per table for readability.
    #
    # Assets record *where* the picture is, not always its bytes: the PDF path
    # has a cropped image on disk and stores image_sha256, while the HTML path
    # has a URL and stores that. image_sha256/image_url are therefore both
    # nullable, and `ordinal` is unique per paper so re-extraction replaces.
    # Index names are global in PostgreSQL, so each table needs its own; the
    # names match what `index=True` generates for the ORM columns, which is what
    # keeps the migration and `Base.metadata` in agreement.
    def _create_asset_table(table: str, *extra: sa.Column, constraints=()) -> None:  # noqa: ANN202
        op.create_table(
            table,
            sa.Column("id", sa.String(32), primary_key=True),
            sa.Column(
                "paper_id",
                sa.String(32),
                sa.ForeignKey("papers.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column("ordinal", sa.Integer(), nullable=False),
            sa.Column("label", sa.String(64)),
            sa.Column("caption", sa.Text()),
            sa.Column("image_sha256", sa.String(64)),
            sa.Column("image_url", sa.String(1024)),
            sa.Column("page_idx", sa.Integer()),
            sa.Column("bbox", sa.Text()),
            sa.Column("source", sa.String(32), nullable=False, server_default="html"),
            sa.Column("created_at", TIMESTAMP, nullable=False),
            *extra,
            *constraints,
        )
        op.create_index(f"ix_{table}_paper_id", table, ["paper_id"])
        op.create_index(f"ix_{table}_image_sha256", table, ["image_sha256"])

    _create_asset_table(
        "paper_figures",
        sa.Column("width", sa.Integer()),
        sa.Column("height", sa.Integer()),
        constraints=(
            sa.UniqueConstraint("paper_id", "ordinal", name="uq_paper_figures_paper_ordinal"),
        ),
    )
    _create_asset_table(
        "paper_tables",
        sa.Column("body_html", sa.Text()),
        sa.Column("row_count", sa.Integer()),
        sa.Column("column_count", sa.Integer()),
        constraints=(
            sa.UniqueConstraint("paper_id", "ordinal", name="uq_paper_tables_paper_ordinal"),
        ),
    )
    _create_asset_table(
        "paper_equations",
        sa.Column("latex", sa.Text()),
        sa.Column("text_format", sa.String(16)),
        sa.Column("is_display", boolean(), nullable=False, server_default=sa.text("false")),
        constraints=(
            sa.UniqueConstraint("paper_id", "ordinal", name="uq_paper_equations_paper_ordinal"),
        ),
    )
    # "the real equations of this paper" is the common read, so it is indexed.
    op.create_index(
        "ix_paper_equations_paper_display", "paper_equations", ["paper_id", "is_display"]
    )

    # ------------------------------------------------------------ raw_documents
    op.create_table(
        "raw_documents",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column(
            "paper_id",
            sa.String(32),
            sa.ForeignKey("papers.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("kind", sa.String(32), nullable=False),
        sa.Column("source_url", sa.String(1024), nullable=False, server_default=""),
        sa.Column("uri", sa.String(2048), nullable=False),
        sa.Column("content_type", sa.String(128), nullable=False, server_default=""),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("sha256", sa.String(64), nullable=False),
        sa.Column("meta", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("created_at", TIMESTAMP, nullable=False),
        sa.UniqueConstraint("paper_id", "kind", "sha256", name="uq_raw_documents_paper_kind_sha"),
    )
    op.create_index("ix_raw_documents_paper_id", "raw_documents", ["paper_id"])
    op.create_index("ix_raw_documents_kind", "raw_documents", ["kind"])
    op.create_index("ix_raw_documents_sha256", "raw_documents", ["sha256"])
    op.create_index("ix_raw_documents_paper_kind", "raw_documents", ["paper_id", "kind"])

    # ------------------------------------------------------------------ chunks
    op.create_table(
        "chunks",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column(
            "paper_id",
            sa.String(32),
            sa.ForeignKey("papers.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("raw_document_id", sa.String(32), sa.ForeignKey("raw_documents.id", ondelete="SET NULL")),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column(
            "content_kind",
            sa.String(16),
            nullable=False,
            server_default="body",
        ),
        sa.Column("token_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("char_start", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("char_end", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("heading", sa.Text()),
        sa.Column("section_path", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("source", sa.String(32), nullable=False, server_default=""),
        sa.Column("created_at", TIMESTAMP, nullable=False),
        sa.UniqueConstraint("paper_id", "ordinal", name="uq_chunks_paper_ordinal"),
    )
    op.create_index("ix_chunks_paper_id", "chunks", ["paper_id"])
    op.create_index("ix_chunks_content_hash", "chunks", ["content_hash"])
    op.create_index("ix_chunks_paper_ordinal", "chunks", ["paper_id", "ordinal"])
    op.create_index("ix_chunks_content_kind", "chunks", ["content_kind"])

    # -------------------------------------------------------------- embeddings
    op.create_table(
        "embeddings",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("chunk_id", sa.String(32), sa.ForeignKey("chunks.id", ondelete="CASCADE"), nullable=False),
        sa.Column("paper_id", sa.String(32), sa.ForeignKey("papers.id", ondelete="CASCADE"), nullable=False),
        sa.Column("provider", sa.String(64), nullable=False),
        sa.Column("model", sa.String(255), nullable=False),
        sa.Column("dimensions", sa.Integer(), nullable=False),
        sa.Column("fingerprint", sa.String(128), nullable=False),
        sa.Column("vector", vector_type, nullable=True),
        sa.Column("meta", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("created_at", TIMESTAMP, nullable=False),
        sa.UniqueConstraint("chunk_id", name="uq_embeddings_chunk"),
    )
    op.create_index("ix_embeddings_chunk_id", "embeddings", ["chunk_id"], unique=True)
    op.create_index("ix_embeddings_paper_id", "embeddings", ["paper_id"])
    op.create_index("ix_embeddings_fingerprint", "embeddings", ["fingerprint"])
    op.create_index("ix_embeddings_provider_model", "embeddings", ["provider", "model"])

    if is_postgres:
        # HNSW for fast approximate nearest-neighbour search.
        op.execute(
            "CREATE INDEX IF NOT EXISTS ix_embeddings_vector_hnsw "
            "ON embeddings USING hnsw (vector vector_cosine_ops) "
            "WITH (m = 16, ef_construction = 128)"
        )

    # --------------------------------------------------------- ingestion_runs
    op.create_table(
        "ingestion_runs",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column(
            "paper_id",
            sa.String(32),
            sa.ForeignKey("papers.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column("arxiv_id", sa.String(64)),
        sa.Column("status", sa.String(16), nullable=False, server_default="pending"),
        sa.Column("requested_by", sa.String(64), nullable=False, server_default="api"),
        sa.Column("trigger", sa.String(32), nullable=False, server_default="manual"),
        sa.Column("prefer_html", boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("force", boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("chunk_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("embedding_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("content_kind", sa.String(32)),
        sa.Column("error", sa.Text()),
        sa.Column("timings", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("started_at", TIMESTAMP),
        sa.Column("finished_at", TIMESTAMP),
        sa.Column("created_at", TIMESTAMP, nullable=False),
    )
    op.create_index("ix_ingestion_runs_paper_id", "ingestion_runs", ["paper_id"])
    op.create_index("ix_ingestion_runs_arxiv_id", "ingestion_runs", ["arxiv_id"])
    op.create_index("ix_ingestion_runs_status", "ingestion_runs", ["status"])
    op.create_index("ix_ingestion_runs_created_at", "ingestion_runs", ["created_at"])

    # ------------------------------------------------------ pipeline_step_runs
    op.create_table(
        "pipeline_step_runs",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column(
            "run_id",
            sa.String(32),
            sa.ForeignKey("ingestion_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("name", sa.String(64), nullable=False),
        sa.Column("attempt", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("status", sa.String(16), nullable=False, server_default="pending"),
        sa.Column("duration_ms", sa.Float()),
        sa.Column("error", sa.Text()),
        sa.Column("meta", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("started_at", TIMESTAMP, nullable=False),
        sa.Column("finished_at", TIMESTAMP),
    )
    op.create_index("ix_pipeline_step_runs_run_id", "pipeline_step_runs", ["run_id"])

    # ---------------------------------------------------------- embedding_jobs
    op.create_table(
        "embedding_jobs",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("fingerprint", sa.String(128), nullable=False),
        sa.Column("status", sa.String(16), nullable=False, server_default="pending"),
        sa.Column("total", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("processed", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("failed", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("error", sa.Text()),
        sa.Column("created_at", TIMESTAMP, nullable=False),
        sa.Column("updated_at", TIMESTAMP, nullable=False),
    )
    op.create_index("ix_embedding_jobs_fingerprint", "embedding_jobs", ["fingerprint"])
    op.create_index("ix_embedding_jobs_status", "embedding_jobs", ["status"])


def downgrade() -> None:
    op.drop_table("embedding_jobs")
    op.drop_table("pipeline_step_runs")
    op.drop_table("ingestion_runs")
    op.drop_table("embeddings")
    op.drop_table("chunks")
    op.drop_table("raw_documents")
    op.drop_table("project_papers")
    op.drop_table("projects")
    op.drop_table("paper_equations")
    op.drop_table("paper_tables")
    op.drop_table("paper_figures")
    op.drop_table("paper_references")
    op.drop_table("paper_categories")
    op.drop_table("categories")
    op.drop_table("paper_authors")
    op.drop_table("authors")
    op.drop_table("papers")