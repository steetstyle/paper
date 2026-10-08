"""ORM models for figures, tables and equations.

Three tables rather than one polymorphic ``paper_assets`` row because the three
carry genuinely different payloads — a table has ``body_html``, an equation has
``latex`` and ``is_display``, a figure has neither — and a single table would be
mostly nullable columns. The fields they genuinely share are expressed once, in
:class:`AssetMixin`, so the shared logic is written once too.

On images: an asset records **where** its picture is, not always its bytes.

- The PDF path gets cropped images from MinerU on disk, so they are hashed into
  the content-addressed blob store and ``image_sha256`` is set.
- The HTML path has a relative ``src``, so only ``image_url`` is set. Fetching
  every figure of a paper during ingest would mean one rate-limited request per
  figure for images nothing has asked for yet, so it is done on demand instead.

Either way a row can have a URL, a hash, both, or neither, and
:attr:`AssetMixin.has_image` says which.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, declared_attr, mapped_column, relationship

from app.db.base import TIMESTAMP, Base, utcnow
from app.db.models import Paper, _uuid


class AssetMixin:
    """Fields common to every kind of extracted asset."""

    @declared_attr
    def id(cls) -> Mapped[str]:  # noqa: N805
        return mapped_column(String(32), primary_key=True, default=_uuid)

    @declared_attr
    def paper_id(cls) -> Mapped[str]:  # noqa: N805
        return mapped_column(ForeignKey("papers.id", ondelete="CASCADE"), index=True)

    @declared_attr
    def ordinal(cls) -> Mapped[int]:  # noqa: N805
        """Position within the paper, 1-based.

        Unique per paper so re-extraction replaces cleanly instead of
        accumulating a second copy of every figure.
        """
        return mapped_column(Integer)

    @declared_attr
    def label(cls) -> Mapped[str | None]:  # noqa: N805
        """The printed number, e.g. ``"Figure 2"`` or ``"Table 1"``."""
        return mapped_column(String(64))

    @declared_attr
    def caption(cls) -> Mapped[str | None]:  # noqa: N805
        return mapped_column(Text)

    @declared_attr
    def image_sha256(cls) -> Mapped[str | None]:  # noqa: N805
        """Blob-store hash, when we have the bytes."""
        return mapped_column(String(64), nullable=True, index=True)

    @declared_attr
    def image_url(cls) -> Mapped[str | None]:  # noqa: N805
        """Remote location, for HTML-sourced figures we have not fetched."""
        return mapped_column(String(1024), nullable=True)

    @declared_attr
    def page_idx(cls) -> Mapped[int | None]:  # noqa: N805
        """Zero-based page. Only the PDF path can know this."""
        return mapped_column(Integer, nullable=True)

    @declared_attr
    def bbox(cls) -> Mapped[str | None]:  # noqa: N805
        """Serialised ``[x0, y0, x1, y1]`` in PDF points, when known."""
        return mapped_column(Text, nullable=True)

    @declared_attr
    def source(cls) -> Mapped[str]:  # noqa: N805
        """``html`` or ``mineru`` — which extractor produced this row."""
        return mapped_column(String(32), default="html")

    @declared_attr
    def created_at(cls) -> Mapped[datetime]:  # noqa: N805
        return mapped_column(TIMESTAMP, default=utcnow)

    def has_image(self) -> bool:
        return bool(self.image_sha256 or self.image_url)

    def bbox_values(self) -> list[int] | None:
        import json  # noqa: PLC0415

        if not self.bbox:
            return None
        try:
            parsed = json.loads(self.bbox)
        except (TypeError, ValueError):
            return None
        return parsed if isinstance(parsed, list) else None

    def summary(self) -> str:
        text = (self.caption or "").strip()
        return f"{self.label}: {text[:90]}" if self.label else text[:100]


class PaperFigure(AssetMixin, Base):
    """A figure: image plus caption."""

    __tablename__ = "paper_figures"
    __table_args__ = (
        UniqueConstraint("paper_id", "ordinal", name="uq_paper_figures_paper_ordinal"),
    )

    width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height: Mapped[int | None] = mapped_column(Integer, nullable=True)

    paper: Mapped[Paper] = relationship(lazy="noload")

    def __repr__(self) -> str:  # pragma: no cover
        return f"<PaperFigure [{self.ordinal}] {self.label}>"


class PaperTable(AssetMixin, Base):
    """A table: ``body_html`` when the source gave us the markup.

    HTML gives a real ``<table>`` subtree; the PDF path only gives a picture of
    the table, so its ``body_html`` is null and ``image_sha256`` is set. That
    asymmetry is the reason the image fields are shared and nullable.
    """

    __tablename__ = "paper_tables"
    __table_args__ = (
        UniqueConstraint("paper_id", "ordinal", name="uq_paper_tables_paper_ordinal"),
    )

    body_html: Mapped[str | None] = mapped_column(Text)
    row_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    column_count: Mapped[int | None] = mapped_column(Integer, nullable=True)

    paper: Mapped[Paper] = relationship(lazy="noload")

    def __repr__(self) -> str:  # pragma: no cover
        return f"<PaperTable [{self.ordinal}] {self.label}>"


class PaperEquation(AssetMixin, Base):
    """A formula.

    ``is_display`` is the distinction that matters: a paper has a handful of
    numbered display equations and ~140 inline fragments like ``h_t``. Both are
    stored — inline math is real content — but callers almost always want
    ``is_display=True``, so it gets its own flag rather than a naming convention.

    No ``caption``/``label``: nothing in either source reliably pairs a formula
    with its printed number. ``ordinal`` is position, which is not the same
    thing, and pretending otherwise would be a lie in a column name.
    """

    __tablename__ = "paper_equations"
    __table_args__ = (
        UniqueConstraint("paper_id", "ordinal", name="uq_paper_equations_paper_ordinal"),
        # Filtering "the real equations of this paper" is the common read.
        Index("ix_paper_equations_paper_display", "paper_id", "is_display"),
    )

    latex: Mapped[str | None] = mapped_column(Text)
    text_format: Mapped[str | None] = mapped_column(String(16))
    is_display: Mapped[bool] = mapped_column(Boolean, default=False)

    paper: Mapped[Paper] = relationship(lazy="noload")

    def preview(self, limit: int = 80) -> str:
        text = " ".join((self.latex or "").split())
        return text[:limit] + ("..." if len(text) > limit else "")

    def __repr__(self) -> str:  # pragma: no cover
        return f"<PaperEquation [{self.ordinal}] display={self.is_display}>"


__all__ = [
    "AssetMixin",
    "PaperEquation",
    "PaperFigure",
    "PaperTable",
]