"""documents: arXiv papers and local files in one corpus

The corpus was arXiv-shaped: ``papers.arxiv_id`` was ``NOT NULL UNIQUE``, and
every lookup path resolved through an arXiv id. That made a textbook, a set of
lecture notes or a report impossible to represent — not awkwardly, but not at
all. Measured on the three books used to build this:

=========================  ======  ========  ==========================
book                       pages   outline   embedded ``/Title``
=========================  ======  ========  ==========================
Girvin, Condensed Matter     721      341     good
Oxford, Solid State Basics   305      174     good
Pethick, Superconductivity   578        0     ``pethick.dvi`` (garbage)
=========================  ======  ========  ==========================

None of them can be an arXiv paper row. So identity becomes source-agnostic:

* ``doc_key`` is the universal handle — the arXiv id for a paper, a slug for a
  local file. One column, one lookup path, no polymorphic union.
* ``kind`` records what it is, which is what a reader cares about ("is this a
  preprint or my lecture notes?") and what the UI groups by.
* ``arxiv_id`` becomes nullable, with a *partial* unique index so papers still
  cannot collide but documents need no arXiv id at all.

Also added, because they are what makes a 700-page book navigable rather than a
pile of text:

* ``chunks.page_start`` / ``page_end`` — MinerU's ``content_list.json`` already
  carries ``page_idx`` per block and was being thrown away by the chunker.
* ``chunks.section_ordinal`` — a pointer into ``document_sections``, not the
  title text, so filtering by section is a column comparison.
* ``document_sections`` — the book's own structure.
* ``raw_documents.page_start`` / ``page_end`` — which slice of a PDF a row
  covers, so a 700-page book can be ingested in ranges that *extend* rather
  than replace each other.

Purely additive: every existing row keeps its id, and ``doc_key`` is backfilled
from ``arxiv_id``, so no re-ingestion is needed.

Revision ID: 0003_documents
Revises: 0002_embedding_spaces
Create Date: 2026-10-06 00:00:00
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003_documents"
down_revision: str | None = "0002_embedding_spaces"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


#: SQLite has no ``ALTER COLUMN``; Alembic's batch mode rewrites the table
#: instead. The app runs on PostgreSQL and the migration tests run on SQLite, so
#: both paths are implemented rather than one being assumed away.


def upgrade() -> None:
    # ------------------------------------------------------------- identity
    # The UNIQUE on `arxiv_id` stays as it is. SQL permits any number of NULLs
    # in a UNIQUE column, so it already reads "no two papers share an arXiv id,
    # and any number of documents have none" — replacing it with a partial index
    # would say the same through machinery SQLite cannot drop and re-add.
    with op.batch_alter_table("papers") as batch:
        batch.alter_column("arxiv_id", existing_type=sa.String(64), nullable=True)
        batch.alter_column("versioned_id", existing_type=sa.String(72), nullable=True)

    op.add_column(
        "papers",
        sa.Column("kind", sa.String(16), nullable=False, server_default="paper"),
    )
    op.add_column("papers", sa.Column("doc_key", sa.String(128), nullable=True))
    op.add_column("papers", sa.Column("file_sha256", sa.String(64), nullable=True))
    op.add_column("papers", sa.Column("page_count", sa.Integer(), nullable=True))

    # Every existing row is a paper, so its handle is its arXiv id.
    op.execute("UPDATE papers SET doc_key = arxiv_id WHERE doc_key IS NULL")
    with op.batch_alter_table("papers") as batch:
        batch.alter_column("doc_key", existing_type=sa.String(128), nullable=False)
    op.create_index("uq_papers_doc_key", "papers", ["doc_key"], unique=True)
    # Deliberately NOT unique: the same file may legitimately be ingested twice
    # (a re-run after a failure, a second copy under another name). Dedupe is a
    # check `ingest_file` performs and reports, not a constraint that would make
    # `--force` fail.
    op.create_index("ix_papers_file_sha256", "papers", ["file_sha256"])

    # A run has to be findable without an arXiv id in the table it records.
    op.add_column("ingestion_runs", sa.Column("doc_key", sa.String(128), nullable=True))
    op.execute(
        "UPDATE ingestion_runs SET doc_key = arxiv_id WHERE doc_key IS NULL AND arxiv_id IS NOT NULL"
    )
    op.create_index("ix_ingestion_runs_doc_key", "ingestion_runs", ["doc_key"])

    # ----------------------------------------------------------- page ranges
    op.add_column("raw_documents", sa.Column("page_start", sa.Integer(), nullable=True))
    op.add_column("raw_documents", sa.Column("page_end", sa.Integer(), nullable=True))
    # The existing UNIQUE (paper_id, kind, sha256) cannot survive page ranges:
    # one file ingested as 1-200, 201-400 would produce two rows differing only
    # in the range, and the second would be rejected.
    with op.batch_alter_table("raw_documents") as batch:
        batch.drop_constraint("uq_raw_documents_paper_kind_sha", type_="unique")
        batch.create_unique_constraint(
            "uq_raw_documents_paper_kind_sha",
            ["paper_id", "kind", "sha256", "page_start"],
        )

    # --------------------------------------------------------------- chunks
    op.add_column("chunks", sa.Column("page_start", sa.Integer(), nullable=True))
    op.add_column("chunks", sa.Column("page_end", sa.Integer(), nullable=True))
    op.add_column("chunks", sa.Column("section_ordinal", sa.Integer(), nullable=True))
    op.create_index("ix_chunks_pages", "chunks", ["paper_id", "page_start"])

    # ------------------------------------------------------------- sections
    op.create_table(
        "document_sections",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("paper_id", sa.String(32), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        # Depth inferred from the heading's own numbering ("2.3.1" -> 3), not
        # from `#` count: MinerU emits every heading as `#`, so ATX depth is
        # flat and useless for hierarchy.
        sa.Column("level", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("page_start", sa.Integer(), nullable=True),
        sa.Column("page_end", sa.Integer(), nullable=True),
        # outline | markdown | merged. Books disagree: 2 of the 3 used to build
        # this have a real outline, one has none at all.
        sa.Column("source", sa.String(16), nullable=False, server_default="outline"),
        sa.ForeignKeyConstraint(
            ["paper_id"], ["papers.id"], ondelete="CASCADE", name="fk_sections_paper"
        ),
        sa.UniqueConstraint("paper_id", "ordinal", name="uq_document_sections_ordinal"),
    )
    op.create_index(
        "ix_document_sections_paper", "document_sections", ["paper_id", "page_start"]
    )


def downgrade() -> None:
    op.drop_index("ix_document_sections_paper", table_name="document_sections")
    op.drop_table("document_sections")

    op.drop_index("ix_chunks_pages", table_name="chunks")
    op.drop_column("chunks", "section_ordinal")
    op.drop_column("chunks", "page_end")
    op.drop_column("chunks", "page_start")

    with op.batch_alter_table("raw_documents") as batch:
        batch.drop_constraint("uq_raw_documents_paper_kind_sha", type_="unique")
        batch.create_unique_constraint(
            "uq_raw_documents_paper_kind_sha", ["paper_id", "kind", "sha256"]
        )
    op.drop_column("raw_documents", "page_end")
    op.drop_column("raw_documents", "page_start")

    op.drop_index("ix_ingestion_runs_doc_key", table_name="ingestion_runs")
    op.drop_column("ingestion_runs", "doc_key")

    op.drop_index("ix_papers_file_sha256", table_name="papers")
    op.drop_index("uq_papers_doc_key", table_name="papers")
    op.drop_column("papers", "page_count")
    op.drop_column("papers", "file_sha256")
    op.drop_column("papers", "doc_key")
    op.drop_column("papers", "kind")

    # Rows that are not papers cannot be represented again, so the two columns
    # cannot go back to NOT NULL; the UNIQUE that was never removed stays.
    with op.batch_alter_table("papers") as batch:
        batch.alter_column("versioned_id", existing_type=sa.String(72), nullable=True)
        batch.alter_column("arxiv_id", existing_type=sa.String(64), nullable=True)
