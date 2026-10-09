"""Embedding spaces: one embedding model, one physical table.

pgvector's ``vector(n)`` column and its HNSW index are fixed-width. Two models
with different dimensions therefore *cannot* share a column, so each model gets
its own table. Chunks stay model-independent — only vectors are duplicated.

    chunks ──┬── embeddings                 (default space, 1536d)
             ├── embeddings__small_384     (Matryoshka-truncated experiment)
             └── embeddings__bge_m3        (different model, 1024d)

Adding a model therefore costs disk, not a migration of existing rows. That is
the whole point: model comparisons become possible without ever re-embedding or
dropping the incumbent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from paper_app.config import AppSettings, EmbeddingSettings

__all__ = [
    "EmbeddingSpace",
    "DEFAULT_SPACE_NAME",
    "LEGACY_TABLE_NAME",
    "slugify",
]

DEFAULT_SPACE_NAME = "default"
LEGACY_TABLE_NAME = "embeddings"
TABLE_PREFIX = "embeddings__"

_MAX_NAME = 48
_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(value: str) -> str:
    """Turn a human space name into a safe SQL identifier fragment.

    >>> slugify("BGE-M3 (1024d)")
    'bge_m3_1024d'
    """
    slug = _SLUG_RE.sub("_", value.strip().lower()).strip("_")
    slug = re.sub(r"_{2,}", "_", slug)
    if not slug:
        raise ValueError(f"space name {value!r} produces an empty slug")
    if slug[0].isdigit():
        slug = f"n{slug}"
    if len(slug) > _MAX_NAME:
        slug = slug[:_MAX_NAME].rstrip("_")
    return slug


@dataclass(frozen=True, slots=True)
class EmbeddingSpace:
    """A named, persisted embedding model configuration."""

    name: str
    provider: str
    model: str
    dimensions: int
    distance: str = "cosine"
    table_name: str | None = None
    description: str | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("space name must not be empty")
        if self.dimensions < 1:
            raise ValueError(f"dimensions must be positive, got {self.dimensions}")
        if self.distance not in DISTANCE_OPS:
            raise ValueError(
                f"unknown distance {self.distance!r}; expected one of {sorted(DISTANCE_OPS)}"
            )
        # Validate the name is representable as a table fragment.
        slugify(self.name)

    # ------------------------------------------------------------- identity
    @property
    def slug(self) -> str:
        return slugify(self.name)

    @property
    def resolved_table(self) -> str:
        """Physical table backing this space.

        ``default`` keeps the table created by migration 0001 so existing
        deployments upgrade without a data migration.
        """
        if self.table_name:
            return self.table_name
        if slugify(self.name) == DEFAULT_SPACE_NAME:
            return LEGACY_TABLE_NAME
        return f"{TABLE_PREFIX}{self.slug}"

    @property
    def fingerprint(self) -> str:
        return f"{self.provider}:{self.model}:{self.dimensions}"

    @property
    def distance_operator(self) -> str:
        return DISTANCE_OPS[self.distance]

    @property
    def index_ops(self) -> str:
        return INDEX_OPS[self.distance]

    # --------------------------------------------------------------- helpers
    @classmethod
    def from_settings(
        cls,
        settings: EmbeddingSettings | dict[str, Any],
        *,
        name: str | None = None,
        distance: str = "cosine",
        table_name: str | None = None,
        description: str | None = None,
    ) -> EmbeddingSpace:
        if isinstance(settings, EmbeddingSettings):
            provider, model, dimensions = (
                settings.provider,
                settings.model,
                settings.dimensions,
            )
        else:
            provider = settings["provider"]
            model = settings["model"]
            dimensions = settings["dimensions"]
        return cls(
            name=name or DEFAULT_SPACE_NAME,
            provider=provider,
            model=model,
            dimensions=dimensions,
            distance=distance,
            table_name=table_name,
            description=description,
        )

    @classmethod
    def default_for(cls, settings: AppSettings, name: str | None = None) -> EmbeddingSpace:
        return cls.from_settings(
            settings.embedding,
            name=name or settings.embedding.space or DEFAULT_SPACE_NAME,
            distance=settings.vector.distance,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "provider": self.provider,
            "model": self.model,
            "dimensions": self.dimensions,
            "distance": self.distance,
            "table": self.resolved_table,
            "fingerprint": self.fingerprint,
            "description": self.description,
        }


# pgvector distance operator + matching HNSW opclass
DISTANCE_OPS = {"cosine": "<=>", "euclid": "<->", "dot": "<=>"}
INDEX_OPS = {"cosine": "vector_cosine_ops", "euclid": "vector_l2_ops", "dot": "vector_ip_ops"}


def space_from_row(row: Any) -> EmbeddingSpace:
    """Build a space from an ``EmbeddingSpaceRecord`` ORM row."""
    return EmbeddingSpace(
        name=row.name,
        provider=row.provider,
        model=row.model,
        dimensions=row.dimensions,
        distance=row.distance,
        table_name=row.table_name,
        description=row.description,
    )


def assert_distinct(spaces: list[EmbeddingSpace]) -> None:
    """Fail loudly when two spaces would collide on a physical table."""
    seen: dict[str, str] = {}
    for space in spaces:
        table = space.resolved_table
        if table in seen and seen[table] != space.name:
            raise ValueError(
                f"spaces {seen[table]!r} and {space.name!r} both map to table {table!r}; "
                "give one an explicit table_name"
            )
        seen[table] = space.name