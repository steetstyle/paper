"""One filter parser for every surface.

The CLI, the HTTP API and the MCP server all funnel through
:func:`search_query_from`, so a filter written for one is guaranteed to compile
to the same ArXiv ``search_query`` everywhere. Parity is structural, not tested
into existence by hand.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from paper_app.domain.enums import ArxivSortBy, ArxivSortOrder
from paper_app.domain.filters import (
    ArxivField,
    ArxivFilter,
    ArxivOperator,
    FilterProblem,
    PostFilter,
    compile_query,
    parse_submitted_date,
    validate_arxiv_query,
)
from paper_app.domain.models import PaperMetadata, SearchQuery

__all__ = [
    "SearchRequest",
    "search_query_from",
    "build_filter",
    "validate_filter",
    "apply_post_filter",
    "FIELD_ALIASES",
]

# Readable names accepted by every surface, mapped onto ArXiv prefixes.
FIELD_ALIASES: dict[str, str] = {
    "title": ArxivField.TITLE.value,
    "author": ArxivField.AUTHOR.value,
    "abstract": ArxivField.ABSTRACT.value,
    "comment": ArxivField.COMMENT.value,
    "journal": ArxivField.JOURNAL_REF.value,
    "journal_ref": ArxivField.JOURNAL_REF.value,
    "category": ArxivField.CATEGORY.value,
    "report_number": ArxivField.REPORT_NUMBER.value,
    "all": ArxivField.ALL.value,
}


@dataclass(slots=True)
class SearchRequest:
    """A fully-specified ArXiv search, surface-independent.

    Exactly the arguments ArXiv itself accepts, plus the small set of
    client-side filters its API cannot express.
    """

    # ArXiv query language
    query: str | None = None
    """A raw ``search_query`` expression, passed through verbatim."""
    title: tuple[str, ...] = ()
    author: tuple[str, ...] = ()
    abstract: tuple[str, ...] = ()
    comment: tuple[str, ...] = ()
    journal: tuple[str, ...] = ()
    category: tuple[str, ...] = ()
    report_number: tuple[str, ...] = ()
    operator: ArxivOperator = ArxivOperator.AND
    """How to combine values inside one field. ArXiv ORs nothing implicitly."""

    id_list: tuple[str, ...] = ()
    """Look up specific papers. Preferred over `id:` for version handling."""

    # ArXiv request parameters
    max_results: int = 10
    start: int = 0
    sort_by: str = ArxivSortBy.RELEVANCE.value
    sort_order: str = ArxivSortOrder.DESCENDING.value
    submitted_from: datetime | None = None
    submitted_to: datetime | None = None

    # client-side
    has_pdf: bool | None = None
    has_html: bool | None = None
    has_doi: bool | None = None
    has_journal_ref: bool | None = None
    ingested: bool | None = None
    also_categories: tuple[str, ...] = ()
    exclude_categories: tuple[str, ...] = ()

    # provenance for responses
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def to_search_query(self) -> SearchQuery:
        return SearchQuery(
            raw=compile_query(self.filter, id_list=self.id_list)[0] or None,
            id_list=self.id_list,
            max_results=self.max_results,
            start=self.start,
            sort_by=self.sort_by,
            sort_order=self.sort_order,
        )

    @property
    def post_filter(self) -> PostFilter:
        return PostFilter(
            has_pdf=self.has_pdf,
            has_html=self.has_html,
            has_doi=self.has_doi,
            has_journal_ref=self.has_journal_ref,
            ingested=self.ingested,
            categories=self.also_categories,
            exclude_categories=self.exclude_categories,
        )

    @property
    def filter(self) -> ArxivFilter:
        return build_filter(self)

    def describe(self) -> dict[str, Any]:
        """Echoed back to callers so they can see exactly what ArXiv was asked."""
        return {
            "filter": self.filter.to_dict(),
            "id_list": list(self.id_list),
            "max_results": self.max_results,
            "start": self.start,
            "sort_by": self.sort_by,
            "sort_order": self.sort_order,
            "post_filter": self.post_filter.to_dict(),
            "warnings": list(self.warnings),
        }


def build_filter(request: SearchRequest) -> ArxivFilter:
    """Compile a request into the structured filter, then to ArXiv syntax."""
    buckets: dict[str, Sequence[str]] = {
        "ti": request.title,
        "au": request.author,
        "abs": request.abstract,
        "co": request.comment,
        "jr": request.journal,
        "cat": request.category,
        "rn": request.report_number,
    }
    return ArxivFilter.from_values(
        raw=request.query,
        operator=request.operator,
        submitted_from=request.submitted_from,
        submitted_to=request.submitted_to,
        fields={k: v for k, v in buckets.items() if v},
    )


def validate_filter(request: SearchRequest) -> tuple[str, ...]:
    """Warnings worth surfacing: lowercase operators, `id:`, stray fields.

    Errors in the raw expression are raised as :class:`FilterProblem`; softer
    issues come back as warnings so a search is not blocked over a `ti:` typo.
    """
    warnings: list[str] = []
    if request.query:
        validation = validate_arxiv_query(request.query)
        if not validation.ok:
            raise FilterProblem("; ".join(validation.errors))
        warnings.extend(validation.warnings)
    return tuple(warnings)


def search_query_from(
    *,
    raw: str | None = None,
    phrases: Sequence[str] | None = None,
    fields: dict[str, Sequence[str] | str | None] | None = None,
    operator: str | ArxivOperator = ArxivOperator.AND,
    id_list: Sequence[str] | None = None,
    max_results: int = 10,
    start: int = 0,
    sort_by: str = ArxivSortBy.RELEVANCE.value,
    sort_order: str = ArxivSortOrder.DESCENDING.value,
    submitted_from: str | datetime | None = None,
    submitted_to: str | datetime | None = None,
    post: dict[str, Any] | None = None,
) -> SearchRequest:
    """Build a :class:`SearchRequest` from flat, surface-agnostic arguments.

    Every surface calls exactly this, which is what makes the three of them
    behave identically.
    """
    op = operator if isinstance(operator, ArxivOperator) else ArxivOperator.parse(operator)
    if op is None:
        raise FilterProblem(
            f"unknown operator {operator!r}; ArXiv supports AND, OR and ANDNOT"
        )

    buckets = {name: _as_tuple(value) for name, value in (fields or {}).items()}
    unknown = sorted(set(buckets) - set(FIELD_ALIASES) - {f.value for f in ArxivField})
    if unknown:
        raise FilterProblem(
            f"unknown search field(s): {', '.join(unknown)}. Supported: "
            f"{', '.join(sorted(FIELD_ALIASES))}"
        )

    request = SearchRequest(
        query=_phrase_expression(phrases, raw) or None,
        title=buckets.get("title", ()),
        author=buckets.get("author", ()),
        abstract=buckets.get("abstract", ()),
        comment=buckets.get("comment", ()),
        journal=buckets.get("journal", buckets.get("journal_ref", ())),
        category=buckets.get("category", ()),
        report_number=buckets.get("report_number", ()),
        operator=op,
        id_list=_as_tuple(id_list),
        max_results=max(1, min(int(max_results), 100)),
        start=max(0, int(start)),
        sort_by=sort_by if sort_by in {v.value for v in ArxivSortBy} else ArxivSortBy.RELEVANCE.value,
        sort_order=(
            sort_order
            if sort_order in {v.value for v in ArxivSortOrder}
            else ArxivSortOrder.DESCENDING.value
        ),
        submitted_from=parse_submitted_date(submitted_from),
        submitted_to=parse_submitted_date(submitted_to),
    )

    for name, value in (post or {}).items():
        if not hasattr(request, name):
            raise FilterProblem(f"unknown post-filter {name!r}")
        setattr(request, name, _as_tuple(value) if name in _TUPLE_POST else value)

    request.warnings = validate_filter(request)
    return request


_TUPLE_POST = {"also_categories", "exclude_categories"}


def _phrase_expression(phrases: Sequence[str] | None, raw: str | None) -> str:
    """Combine ``--phrase`` values with a raw expression.

    Exists because shell quoting makes the obvious spelling useless:
    ``paper search "sheaf neural network"`` reaches us with no quotes at all, and
    ArXiv then ORs the words. ``--phrase`` sidesteps the shell entirely.

    Multiple phrases are OR-ed, since passing several means "any of these". When
    a raw expression is *also* given it is AND-ed on, never dropped — the two are
    independent pieces of input and discarding one would silently change the
    search. This is reachable by accident: an unquoted
    ``--phrase sheaf neural network`` is split by the shell into a phrase plus two
    positional words.
    """
    cleaned = [p.strip() for p in (phrases or []) if p and p.strip()]
    raw_text = (raw or "").strip()
    if not cleaned:
        return raw_text
    quoted = [
        p if p.startswith('"') and p.endswith('"') else f'"{p}"' for p in cleaned
    ]
    phrase_expr = " OR ".join(quoted)
    if not raw_text:
        return phrase_expr
    return f"{phrase_expr} {ArxivOperator.AND.value} {raw_text}"


def _as_tuple(value: Sequence[str] | str | None) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value.strip() else ()
    return tuple(str(v).strip() for v in value if str(v).strip())


def apply_post_filter(
    hits: Sequence[Any],  # noqa: ANN401 - SearchHit
    request: SearchRequest,
    *,
    ingested_ids: set[str] | None = None,
) -> list[Any]:
    """Drop hits the client-side filters reject. Order is preserved."""
    post = request.post_filter
    if post.is_empty:
        return list(hits)
    kept = []
    for hit in hits:
        metadata: PaperMetadata = hit.metadata
        ingested = (
            metadata.arxiv_id in ingested_ids
            if ingested_ids is not None
            else request.ingested is None
        )
        if post.apply(metadata, ingested=ingested):
            kept.append(hit)
    return kept