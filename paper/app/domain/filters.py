"""ArXiv search filters.

Implements ArXiv's own query language exactly as documented in the API user
manual, so a query written for `export.arxiv.org` works here unchanged.

Field prefixes (all nine)::

    ti  title          au  author        abs abstract
    co  comment        jr  journal ref   cat subject category
    rn  report number  id  id (use id_list instead)      all  everything

Boolean operators are **uppercase** and there are exactly three::

    AND      OR      ANDNOT

Adjacent bare words are **ORed, not ANDed**. This is the one behaviour the
manual never states (it only says a space "extends a search_query to include
multiple fields"), and it is measured against the live API::

    sheaf neural network           -> 394907 results   (the union)
    sheaf AND neural AND network   ->     56 results
    "sheaf neural network"         ->     33 results

A misspelt term makes it worse rather than narrower: a word matching nothing is
*dropped*, not treated as zero, so ``sheaf neureal network`` returns 338593.
:func:`validate_arxiv_query` warns about exactly this shape.

Dates are filtered server-side by ArXiv with a ``submittedDate`` range::

    submittedDate:[YYYYMMDDTTTT TO YYYYMMDDTTTT]      # TTTT = HHMM, GMT

Anything ArXiv cannot express (does this entry have a PDF? is it already in our
corpus?) lives in :class:`PostFilter` and is applied to the parsed response.

Everything here is pure: no I/O, no framework imports.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

__all__ = [
    "ArxivField",
    "ArxivOperator",
    "ArxivTerm",
    "ArxivFilter",
    "PostFilter",
    "FilterProblem",
    "compile_query",
    "validate_arxiv_query",
    "format_submitted_date",
    "parse_submitted_date",
]


class ArxivField(StrEnum):
    """The nine documented field prefixes."""

    TITLE = "ti"
    AUTHOR = "au"
    ABSTRACT = "abs"
    COMMENT = "co"
    JOURNAL_REF = "jr"
    CATEGORY = "cat"
    REPORT_NUMBER = "rn"
    ID = "id"
    ALL = "all"

    @property
    def aliases(self) -> tuple[str, ...]:
        """Readable spellings accepted anywhere a prefix is.

        Declared here so the CLI, the HTTP API and the MCP server cannot drift
        apart on what ``--journal``, ``?journal=`` and ``journal_ref=`` each mean.
        """
        extra = {
            "title": self.TITLE,
            "author": self.AUTHOR,
            "abstract": self.ABSTRACT,
            "comment": self.COMMENT,
            "journal": self.JOURNAL_REF,
            "journal_ref": self.JOURNAL_REF,
            "category": self.CATEGORY,
            "report_number": self.REPORT_NUMBER,
        }
        spellings = [member.name.lower() for member in type(self) if member is self]
        spellings += [name for name, target in extra.items() if target is self]
        return tuple(dict.fromkeys(spellings))

    @classmethod
    def resolve(cls, name: str) -> ArxivField:
        """Accept either the ArXiv prefix or a readable alias.

        Both ``ti`` and ``title`` work: CLI flags and query params read better as
        words, while the wire format needs the prefix.
        """
        key = str(name).strip().lower().replace("-", "_")
        for member in cls:
            if key == member.value or key in member.aliases:
                return member
        raise FilterProblem(
            f"unknown field {name!r}; expected one of "
            f"{', '.join(f.value for f in cls)} "
            f"(or aliases: {', '.join(a for f in cls for a in f.aliases)})"
        )

    @property
    def is_id(self) -> bool:
        return self is ArxivField.ID

    @property
    def searches_everything(self) -> bool:
        """`all:` searches each field simultaneously (per the manual)."""
        return self is ArxivField.ALL


class ArxivOperator(StrEnum):
    """The three Boolean operators. ArXiv rejects lowercase forms."""

    AND = "AND"
    OR = "OR"
    ANDNOT = "ANDNOT"

    @classmethod
    def parse(cls, token: str) -> ArxivOperator | None:
        """Match an operator case-insensitively, for forgiving input."""
        upper = token.strip().upper()
        try:
            return cls(upper)
        except ValueError:
            return None


class FilterProblem(ValueError):
    """A user-supplied filter could not be understood."""


def _reject_id_prefix(arxiv_field: ArxivField) -> None:
    """``id:`` is documented as the wrong way to look up a paper.

    ArXiv routes ``id:`` through the search index, so it can miss newer
    versions of a paper; ``id_list`` is the documented parameter for lookups.
    """
    if arxiv_field.is_id:
        raise FilterProblem(
            "use the id_list parameter instead of `id:` — ArXiv documents id_list "
            "as the way to look up articles so versions resolve correctly"
        )


@dataclass(frozen=True, slots=True)
class ArxivTerm:
    """One ``field:value`` leaf.

    ``value`` may be a bare word, a quoted phrase, or a parenthesised group the
    caller built by hand. It is emitted verbatim: re-quoting someone else's
    expression is how semantics get silently changed.
    """

    field_: ArxivField | None
    value: str

    def compile(self) -> str:
        value = self.value.strip()
        if not value:
            raise FilterProblem("empty search term")
        if self.field_ is None:
            return value
        _reject_id_prefix(self.field_)
        return f"{self.field_.value}:{value}"

    def __str__(self) -> str:
        return self.compile()


@dataclass(frozen=True, slots=True)
class ArxivFilter:
    """A structured ArXiv query.

    Terms are combined with ``operator``; pass an already-grouped expression in
    :attr:`raw` to hand ArXiv exactly what you typed.
    """

    terms: tuple[tuple[ArxivTerm, ArxivOperator | None], ...] = ()
    raw: str | None = None
    submitted_from: datetime | None = None
    submitted_to: datetime | None = None

    def compile(self) -> str:
        """Render the ``search_query`` value.

        A raw expression is emitted verbatim — it is already valid ArXiv syntax
        and any re-serialisation would risk changing its meaning — and is
        AND-ed with the structured terms and the date range. Nothing is ever
        silently dropped: a caller who passes both means both.
        """
        parts: list[str] = []
        if self.raw and self.raw.strip():
            parts.append(self.raw.strip())

        structured = self._structured_expression()
        if structured:
            separator = f"{ArxivOperator.AND.value} " if parts else ""
            parts.append(f"{separator}{structured}")

        date_clause = self._date_clause()
        if date_clause:
            if parts:
                parts.append(f"{ArxivOperator.AND.value} {date_clause}")
            else:
                parts.append(date_clause)

        return " ".join(parts)

    def _structured_expression(self) -> str:
        """Render the structured terms, grouped by field.

        ``operator`` binds values *inside* one field only; distinct fields are
        always AND-ed. ``--title a --title b --author ho --op OR`` means "a or
        b, written by ho" — letting OR reach across fields would silently widen
        every such query to the entire record set.
        """
        operator = next((op for _, op in self.terms if op is not None), ArxivOperator.AND)
        groups: dict[str, list[str]] = {}
        for term, _ in self.terms:
            key = term.field_.value if term.field_ else ""
            groups.setdefault(key, []).append(term.compile())
        clauses = [(f" {operator.value} ").join(values) for values in groups.values()]
        return f" {ArxivOperator.AND.value} ".join(clauses)

    def _date_clause(self) -> str | None:
        if self.submitted_from is None and self.submitted_to is None:
            return None
        start = format_submitted_date(self.submitted_from) if self.submitted_from else "000101010000"
        end = (
            format_submitted_date(self.submitted_to)
            if self.submitted_to
            else format_submitted_date(datetime.now(UTC))
        )
        # Separator is a space, not a literal '+'. The manual prints '+TO+' but
        # in a query string '+' *is* a space, so any URL encoder turns it into
        # %2B -- which ArXiv rejects with a 500. A space survives encoding as
        # %20 and is what ArXiv actually parses.
        return f"submittedDate:[{start} TO {end}]"

    @property
    def is_empty(self) -> bool:
        if self.raw:
            return False
        if self.submitted_from or self.submitted_to:
            return False
        return not any(term.value.strip() for term, _ in self.terms)

    @classmethod
    def from_values(
        cls,
        *,
        raw: str | None = None,
        operator: ArxivOperator = ArxivOperator.AND,
        submitted_from: datetime | None = None,
        submitted_to: datetime | None = None,
        fields: Mapping[str, Iterable[str] | str | None] | None = None,
    ) -> ArxivFilter:
        """Build from field buckets, keyed by prefix or readable alias.

        ``fields`` is a mapping rather than ``**kwargs`` because a splat cannot
        be typed, and every caller already assembles the buckets as a dict.
        """
        terms: list[tuple[ArxivTerm, ArxivOperator | None]] = []
        for prefix, values in (fields or {}).items():
            arxiv_field = ArxivField.resolve(prefix)
            # Refused here rather than at compile time, so the mistake is
            # reported where it was made rather than several calls later.
            _reject_id_prefix(arxiv_field)
            if values is None:
                continue
            bucket = [values] if isinstance(values, str) else list(values)
            bucket = [v.strip() for v in bucket if v and v.strip()]
            if not bucket:
                continue
            for value in bucket:
                terms.append((ArxivTerm(arxiv_field, quote_if_needed(value)), operator))
        return cls(
            terms=tuple(terms),
            raw=raw.strip() if raw and raw.strip() else None,
            submitted_from=submitted_from,
            submitted_to=submitted_to,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw": self.raw,
            "compiled": self.compile() if not self.is_empty else None,
            "terms": [
                {"field": t.field_.value if t.field_ else None, "value": t.value}
                for t, _ in self.terms
            ],
            "submitted_from": self.submitted_from.isoformat() if self.submitted_from else None,
            "submitted_to": self.submitted_to.isoformat() if self.submitted_to else None,
        }


def quote_if_needed(value: str) -> str:
    """Quote a multi-word term so ArXiv treats it as one phrase.

    Anything that already carries a prefix, a group or quotes is left alone:
    those are hand-written expressions and re-quoting them would break them.
    """
    text = value.strip()
    if not text:
        return text
    if text[0] in "\"'(" or text[-1] in "\")" or text.endswith("]"):
        return text
    if ":" in text:
        return text
    if re.fullmatch(r"[A-Za-z0-9_.\-]+", text):
        return text
    escaped = text.replace('"', '\\"')
    return f'"{escaped}"'


def format_submitted_date(value: datetime) -> str:
    """``YYYYMMDDTTTT`` in GMT, the format ArXiv documents."""
    moment = value.astimezone(UTC)
    return moment.strftime("%Y%m%d%H%M")


def parse_submitted_date(value: str | datetime | None) -> datetime | None:
    """Accept ``YYYYMMDD``, ``YYYYMMDDTTTT``, ISO-8601, or a datetime."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    text = str(value).strip()
    if not text:
        return None
    if re.fullmatch(r"\d{8}", text):
        return datetime.strptime(text, "%Y%m%d").replace(tzinfo=UTC)
    if re.fullmatch(r"\d{12}", text):
        return datetime.strptime(text, "%Y%m%d%H%M").replace(tzinfo=UTC)
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        return datetime.strptime(text, "%Y-%m-%d").replace(tzinfo=UTC)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise FilterProblem(
            f"cannot read {value!r} as a date; use YYYY-MM-DD, YYYYMMDD, "
            "YYYYMMDDTTTT or an ISO-8601 timestamp"
        ) from exc
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


# ------------------------------------------------------------------ validation
# Deliberately wider than any real prefix: a long unknown prefix such as
# `keyword:transformer` is exactly the mistake worth naming.
_FIELD_TOKEN_RE = re.compile(r"\b([a-zA-Z][a-zA-Z0-9_]{1,20})\s*:")
_KNOWN_PREFIXES = {f.value for f in ArxivField}
# The manual documents exactly one range field usable inside search_query.
# Without this, the very example the manual prints would be rejected as an
# unknown prefix.
_KNOWN_RANGE_PREFIXES = {"submitteddate"}
_KNOWN_ALL_PREFIXES = _KNOWN_PREFIXES | _KNOWN_RANGE_PREFIXES
_OPERATOR_RE = r"\b(andnot|and|or)\b"
_OPERATOR_WORDS = {"AND", "OR", "ANDNOT"}


@dataclass(slots=True)
class QueryValidation:
    """What we can tell the user about a raw ArXiv query."""

    ok: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    normalized: str | None = None
    """Uppercased-operators version, when we could safely produce one."""

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "normalized": self.normalized,
        }


def _split_bare_terms(text: str) -> tuple[list[str], list[str]]:
    """Break a raw query into ``(bare_terms, operators)``.

    A "bare term" is a word with no field prefix, not inside quotes and not part
    of a ``submittedDate`` range. ArXiv ORs these, which is the single most
    common reason a search returns tens of thousands of irrelevant papers — see
    :func:`validate_arxiv_query`.
    """
    # Quoted phrases and ranges are single units; their contents are not terms.
    cleaned = re.sub(r'"[^"]*"', " ", text)
    cleaned = re.sub(r"\[[^\]]*\]", " ", cleaned)

    bare: list[str] = []
    operators: list[str] = []
    for token in cleaned.split():
        stripped = token.strip("()")
        if not stripped:
            continue
        if stripped.upper() in _OPERATOR_WORDS:
            operators.append(stripped.upper())
            continue
        if ":" in stripped:
            continue  # a field-prefixed term: `ti:x`, `cat:cs.CL`
        if _is_date_literal(stripped):
            continue
        bare.append(stripped)
    return bare, operators


def _is_date_literal(token: str) -> bool:
    """True for the numeric parts of a submittedDate range (202301010000)."""
    return bool(re.fullmatch(r"\d{8,12}", token))


def validate_arxiv_query(query: str) -> QueryValidation:
    """Check a hand-written ArXiv query and normalise what is safe to fix.

    Two mistakes account for most failed queries: lowercase boolean operators
    (ArXiv requires uppercase) and using `id:` instead of `id_list`. Both are
    reported rather than silently altered.
    """
    errors: list[str] = []
    warnings: list[str] = []
    text = (query or "").strip()
    if not text:
        return QueryValidation(ok=False, errors=("query is empty",))

    depth = 0
    in_quotes = False
    for char in text:
        if char == '"':
            in_quotes = not in_quotes
        elif not in_quotes and char == "(":
            depth += 1
        elif not in_quotes and char == ")":
            depth -= 1
            if depth < 0:
                errors.append("unbalanced ')' — there is no matching '('")
                break
    if depth > 0:
        errors.append(f"unbalanced '(': {depth} group(s) never closed")
    if in_quotes:
        errors.append('unterminated quote — a phrase needs a closing "')

    if re.search(r"\bid\s*:", text):
        warnings.append(
            "ArXiv documents id_list (not `id:`) for lookups, because it resolves "
            "article versions correctly"
        )

    for prefix in _FIELD_TOKEN_RE.findall(text):
        lowered = prefix.lower()
        if lowered not in _KNOWN_ALL_PREFIXES:
            errors.append(
                f"unknown field prefix {prefix!r}; expected one of "
                f"{', '.join(sorted(_KNOWN_PREFIXES))}"
            )

    # Only complain about operators that are actually misspelt: the regex must
    # match the uppercase forms too, but a correct `AND` is not a problem.
    misspelt = sorted(
        {match.lower() for match in re.findall(_OPERATOR_RE, text, flags=re.IGNORECASE)
         if match != match.upper()}
    )
    if misspelt:
        warnings.append(
            "ArXiv requires UPPERCASE boolean operators; found "
            f"{', '.join(misspelt)}. It may be treating these as plain text."
        )

    bare, operators = _split_bare_terms(text)
    if len(bare) > 1 and not operators:
        phrase = " ".join(bare)
        warnings.append(
            f"ArXiv ORs adjacent terms instead of ANDing them, so {len(bare)} bare "
            f"words ({', '.join(bare[:4])}) match almost anything — results are "
            f"dominated by the most common term. Quote the phrase as "
            f"'{phrase}' (single quotes outside, double quotes inside: a shell "
            f'strips "{phrase}" bare), or AND the terms explicitly.'
        )

    candidate = re.sub(
        _OPERATOR_RE,
        lambda m: m.group(1).upper(),
        text,
        flags=re.IGNORECASE,
    )
    # Only offer a rewrite when we actually changed something.
    normalized: str | None = candidate if candidate != text else None

    return QueryValidation(
        ok=not errors,
        errors=tuple(errors),
        warnings=tuple(warnings),
        normalized=normalized,
    )


# ----------------------------------------------------------------- post filter
@dataclass(frozen=True, slots=True)
class PostFilter:
    """Filters ArXiv's API cannot express, applied to the parsed response.

    Important: these run *after* pagination, so a filtered page can be shorter
    than ``max_results`` even when more matching papers exist. Callers that need
    a full result count should page until a page comes back empty.
    """

    has_pdf: bool | None = None
    has_html: bool | None = None
    has_doi: bool | None = None
    has_journal_ref: bool | None = None
    ingested: bool | None = None
    """Only papers already present in the local corpus (or only those not)."""

    categories: tuple[str, ...] = ()
    """Keep entries carrying *all* of these categories."""

    exclude_categories: tuple[str, ...] = ()

    def apply(self, metadata: Any, *, ingested: bool | None = None) -> bool:  # noqa: ANN401
        """True when ``metadata`` passes every clause."""
        if self.has_pdf is not None and bool(metadata.pdf_url) is not self.has_pdf:
            return False
        if self.has_html is not None and bool(metadata.html_url) is not self.has_html:
            return False
        if self.has_doi is not None and bool(metadata.doi) is not self.has_doi:
            return False
        if self.has_journal_ref is not None:
            has_ref = bool(metadata.journal_ref)
            if has_ref is not self.has_journal_ref:
                return False
        if self.ingested is not None and ingested is not None and ingested is not self.ingested:
            return False
        available = set(metadata.categories)
        if self.categories and not set(self.categories).issubset(available):
            return False
        if self.exclude_categories:
            return not available.intersection(self.exclude_categories)
        return True

    @property
    def is_empty(self) -> bool:
        return self == PostFilter()

    def to_dict(self) -> dict[str, Any]:
        return {
            "has_pdf": self.has_pdf,
            "has_html": self.has_html,
            "has_doi": self.has_doi,
            "has_journal_ref": self.has_journal_ref,
            "ingested": self.ingested,
            "categories": list(self.categories),
            "exclude_categories": list(self.exclude_categories),
        }


def compile_query(
    arxiv_filter: ArxivFilter | None,
    *,
    id_list: Sequence[str] | None = None,
) -> tuple[str, list[str]]:
    """Return ``(search_query, id_list)`` for the ArXiv API.

    ``search_query`` is empty when only ids are given, because ArXiv applies
    different logic to each and rejects the combination of an empty
    ``search_query`` with ``id_list`` inconsistently across mirrors.
    """
    search_query = arxiv_filter.compile() if arxiv_filter else ""
    ids = [i.strip() for i in (id_list or []) if i and i.strip()]
    if not search_query and not ids:
        raise FilterProblem(
            "nothing to search: provide a filter expression, some field values, or ids"
        )
    if not search_query:
        search_query = ""  # id_list alone is a valid query per the manual
    return search_query, ids