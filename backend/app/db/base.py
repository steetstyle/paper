"""Declarative base and shared column types."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    ColumnElement,
    DateTime,
    TypeDecorator,
    func,
    literal,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.types import TypeEngine

JSONB_VALUE = JSONB()


def _as_json_value(value: Any) -> Any:  # noqa: ANN401
    """Bind a Python list as a JSON literal, so `@>` gets jsonb on both sides."""
    return literal(json.dumps(value), type_=JSONB_VALUE)


class Base(DeclarativeBase):
    """Base class for all ORM models."""

    type_annotation_map = {  # noqa: RUF012
        dict[str, Any]: JSON,
        list[Any]: JSON,
    }


def utcnow() -> datetime:
    return datetime.now(UTC)


class UtcDateTime(TypeDecorator):
    """Always store and return timezone-aware UTC timestamps."""

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Any) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

    def process_result_value(self, value: datetime | None, dialect: Any) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)


class VectorColumn(TypeDecorator):
    """Stores embeddings as a native ``vector`` on PostgreSQL and JSON elsewhere.

    Lets the whole relational schema (and the tests) run on SQLite while
    production gets real pgvector indexing via the Alembic migration.
    """

    impl = JSON
    cache_ok = True

    def __init__(self, dimensions: int | None = None) -> None:
        self.dimensions = dimensions
        super().__init__()

    def load_dialect_impl(self, dialect: Any) -> Any:
        if dialect.name == "postgresql":
            try:
                from pgvector.sqlalchemy import Vector  # noqa: PLC0415

                return dialect.type_descriptor(Vector(self.dimensions))
            except ImportError:
                logger_msg = "pgvector not installed; falling back to JSON storage"
                import warnings  # noqa: PLC0415

                warnings.warn(logger_msg, stacklevel=2)
        return dialect.type_descriptor(JSON())

    def process_bind_param(self, value: Any, dialect: Any) -> Any:
        if value is None:
            return None
        if dialect.name == "postgresql":
            try:
                import pgvector  # noqa: PLC0415
            except ImportError:
                return list(value)
            # Already a Vector: an expression like `1.0 - distance` re-binds the
            # value after SQLAlchemy has converted it once, so re-wrapping here
            # would hand pgvector a Vector and raise "expected list or ndarray".
            if isinstance(value, pgvector.Vector):
                return value
            return pgvector.Vector(value)
        return [float(v) for v in value]

    def process_result_value(self, value: Any, dialect: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, str):
            import json  # noqa: PLC0415

            return [float(v) for v in json.loads(value)]
        return [float(v) for v in value]


class JSONContains(ColumnElement):
    """``column`` contains the JSON array ``value``.

    A dedicated element rather than a dialect branch inside the comparator,
    because the dialect is not knowable when the expression is *built* — only
    when it is *compiled*. Each backend gets its own SQL below.
    """

    __visit_name__ = "json_contains"
    inherit_cache = True

    def __init__(self, column: ColumnElement[Any], value: Any) -> None:
        self.column = column
        self.value = value
        self.type = Boolean()


@compiles(JSONContains)
def _compile_json_contains_default(element: JSONContains, compiler: Any, **kwargs: Any) -> str:
    """Fallback for a backend with no jsonb: SQLAlchemy's default LIKE form.

    Substring, so weaker than the real operators — named explicitly so an
    unexpected dialect is a visible fallback rather than a crash.
    """
    return (
        compiler.process(element.column, **kwargs)
        + " LIKE '%' || "
        + compiler.process(element.value, **kwargs)
        + " || '%'"
    )


@compiles(JSONContains, "postgresql")
def _compile_json_contains_postgres(
    element: JSONContains, compiler: Any, **kwargs: Any
) -> str:
    """jsonb's ``@>`` — real containment, and what the GIN index accelerates."""
    # `type_=JSONB` on the literal already binds it as jsonb; an explicit CAST
    # here would only render a redundant double cast.
    return (
        compiler.process(element.column, **kwargs)
        + " @> "
        + compiler.process(element.value, **kwargs)
    )


@compiles(JSONContains, "sqlite")
def _compile_json_contains_sqlite(element: JSONContains, compiler: Any, **kwargs: Any) -> str:
    """SQLite has no ``@>``, so go through ``json_each``: an exact element test."""
    column = compiler.process(element.column, **kwargs)
    value = compiler.process(element.value, **kwargs)
    return (
        f"EXISTS (SELECT 1 FROM json_each({column}) "
        f"WHERE json_each({column}).value = {value})"
    )


class _JSONListComparator(TypeEngine.Comparator):
    """Routes ``contains()`` to :class:`JSONContains`."""

    def contains(self, other: Any, **kwargs: Any) -> Any:  # noqa: ANN401, ANN403
        return JSONContains(self.expr, _as_json_value(other))


class JSONList(TypeDecorator):
    """A JSON array that supports *containment* on every backend.

    ``JSON`` alone cannot do this. Two problems, both of which only show up once
    you point the app at a real PostgreSQL:

    - ``json`` has no GIN operator class, so ``CREATE INDEX ... USING gin`` fails
      with *"data type json has no default operator class for access method
      gin"*. ``jsonb`` does, and is indexable.
    - ``Column.contains()`` on plain ``JSON`` renders ``LIKE '%' || :value`` —
      a substring match over the serialized text, so it can produce false
      positives (a category named ``cs.LG`` also matches ``cs.LG.MS``) and
      cannot use the index.

    PostgreSQL therefore gets native ``jsonb`` with the ``@>`` operator; SQLite
    has no ``@>`` at all, so containment goes through ``json_each``, which is a
    real element test rather than a substring one.
    """

    impl = JSON
    cache_ok = True

    def load_dialect_impl(self, dialect: Any) -> Any:
        if dialect.name == "postgresql":
            return dialect.type_descriptor(JSONB())
        return dialect.type_descriptor(JSON())

    # A plain class attribute, which is how SQLAlchemy's own JSON and JSONB
    # declare it. A method or property here collides with TypeDecorator's
    # memoized property of the same name.
    comparator_factory = _JSONListComparator



TIMESTAMP = UtcDateTime()
SERVER_TIMESTAMP = func.now()