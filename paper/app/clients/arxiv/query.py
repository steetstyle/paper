"""Translate :class:`~app.domain.models.SearchQuery` into an ArXiv ``search_query``."""

from __future__ import annotations

from app.domain.models import SearchQuery

FIELD_PREFIX = {
    "all": "all",
    "title": "ti",
    "author": "au",
    "abstract": "abs",
    "comment": "co",
    "journal": "jr",
    "category": "cat",
    "id": "id",
}


def _quote(term: str) -> str:
    term = term.strip()
    if not term:
        return ""
    # Already an operator expression such as `ti:"foo bar"` — pass through.
    if any(op in term for op in (" AND ", " OR ", " NOT ", ":")) and term[0].isalpha():
        return term
    if '"' in term or "(" in term:
        return term
    if " " in term:
        return f'"{term}"'
    return term


def _field_clause(prefix: str, terms: tuple[str, ...]) -> str | None:
    parts = [_quote(term) for term in terms]
    parts = [p for p in parts if p]
    if not parts:
        return None
    if len(parts) == 1:
        return f"{prefix}:{parts[0]}"
    return "(" + " OR ".join(f"{prefix}:{p}" for p in parts) + ")"


def build_search_query(query: SearchQuery) -> str:
    """Render the boolean expression sent to the ArXiv API.

    >>> build_search_query(SearchQuery(title_terms=("transformer",), category_terms=("cs.LG",)))
    'ti:transformer AND cat:cs.LG'
    """
    if query.raw:
        return query.raw.strip()

    clauses = [
        _field_clause(FIELD_PREFIX["all"], query.all_terms),
        _field_clause(FIELD_PREFIX["title"], query.title_terms),
        _field_clause(FIELD_PREFIX["author"], query.author_terms),
        _field_clause(FIELD_PREFIX["abstract"], query.abstract_terms),
        _field_clause(FIELD_PREFIX["category"], query.category_terms),
        _field_clause(FIELD_PREFIX["id"], query.id_list),
    ]
    return " AND ".join(clause for clause in clauses if clause)


def build_search_params(query: SearchQuery) -> dict[str, str | int]:
    """Request parameters for one page.

    An ``id_list`` request sends **only** ``id_list``. Measured against the live
    API, for the old-style id ``cond-mat/0404680v1``:

    =======================================  ======  ======
    request                                  HTTP    entries
    =======================================  ======  ======
    ``search_query=id:cond-mat/0404680v1``    200     0
    ``id_list=...`` with no ``search_query``  200     1
    both together                             200     0
    =======================================  ======  ======

    The old-style form is not matched by the ``id:`` search field at all, and
    sending it alongside ``id_list`` makes ArXiv ignore ``id_list`` entirely.
    Both cases return an empty result indistinguishable from "no such paper",
    which is how ``get_paper("cond-mat/0305062v1")`` came to report a missing
    paper with no explanation. ``sortBy``/``sortOrder`` are dropped as well:
    ArXiv answers that combination with a 500.
    """
    if query.id_list:
        return {
            "id_list": ",".join(query.id_list),
            "start": max(0, query.start),
            "max_results": max(1, query.max_results),
        }
    return {
        "search_query": build_search_query(query),
        "start": max(0, query.start),
        "max_results": max(1, query.max_results),
        "sortBy": query.sort_by,
        "sortOrder": query.sort_order,
    }
