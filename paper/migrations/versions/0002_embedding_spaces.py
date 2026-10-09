"""embedding space registry

Vector tables are created per model at runtime (see
``paper_app/db/vector_store/schema.py``) because pgvector's ``vector(n)`` column and
its HNSW index are width-locked. This migration only adds the *registry*: one
row per known model.

The existing ``embeddings`` table is kept and registered as the ``default``
space, so this is additive — no rows move and no re-embedding is required.

Revision ID: 0002_embedding_spaces
Revises: 0001_initial
Create Date: 2026-01-01 00:00:00
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002_embedding_spaces"
down_revision: str | None = "0001_initial"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

TIMESTAMP = sa.DateTime(timezone=True)


def upgrade() -> None:
    op.create_table(
        "embedding_spaces",
        sa.Column("name", sa.String(64), primary_key=True),
        sa.Column("provider", sa.String(64), nullable=False),
        sa.Column("model", sa.String(255), nullable=False),
        sa.Column("dimensions", sa.Integer(), nullable=False),
        sa.Column("distance", sa.String(16), nullable=False, server_default="cosine"),
        sa.Column("table_name", sa.String(128), unique=True),
        sa.Column("description", sa.Text()),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("is_locked", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("created_at", TIMESTAMP, nullable=False),
        sa.Column("last_used_at", TIMESTAMP),
        sa.Column("vector_count", sa.Integer(), nullable=False, server_default="0"),
    )
    op.create_index("ix_embedding_spaces_is_active", "embedding_spaces", ["is_active"])

    # Seed the default space from settings so an existing deployment is
    # immediately addressable by name. The table it points at already exists.
    from paper_app.config import get_settings  # noqa: PLC0415

    settings = get_settings()
    space = settings.embedding.space or "default"
    table = "embeddings" if space == "default" else f"embeddings__{space}"
    op.execute(
        sa.text(
            """
            INSERT INTO embedding_spaces
                (name, provider, model, dimensions, distance, table_name,
                 is_active, is_locked, created_at, vector_count)
            VALUES
                (:name, :provider, :model, :dimensions, :distance, :table_name,
                 true, true, CURRENT_TIMESTAMP, 0)
            ON CONFLICT (name) DO NOTHING
            """
        ).bindparams(
            name=space,
            provider=settings.embedding.provider,
            model=settings.embedding.model,
            dimensions=settings.embedding.dimensions,
            distance=settings.vector.distance,
            table_name=table,
        )
    )


def downgrade() -> None:
    op.drop_index("ix_embedding_spaces_is_active", table_name="embedding_spaces")
    op.drop_table("embedding_spaces")