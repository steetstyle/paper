"""``paper`` command line interface.

    paper search "cat:cs.CL AND ti:transformer"
    paper ingest 2401.01234
    paper ingest 2401.01234 --space small-384      # add vectors for another model
    paper harvest "ti:diffusion model" --limit 10
    paper ask "how does attention work?" --space small-384
    paper show 2401.01234
    paper spaces list|add|activate|rm|sql
    paper doctor
    paper serve
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import json
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any, TypeVar, cast

import typer
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from app import __version__
from app.config import get_settings, reload_settings
from app.db.models import Paper
from app.domain.enums import RunStatus
from app.domain.models import MineruOptions, PageRange
from app.logging import configure_logging, get_logger

app = typer.Typer(
    name="paper",
    help="ArXiv AI assistant: search, ingest, extract and retrieve.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()
logger = get_logger("paper.cli")

F = TypeVar("F", bound=Callable[..., Any])

db_app = typer.Typer(help="Database utilities.", no_args_is_help=True)
spaces_app = typer.Typer(help="Embedding spaces (one model, one table).", no_args_is_help=True)
projects_app = typer.Typer(help="Projects: named collections over the global corpus.", no_args_is_help=True)
app.add_typer(projects_app, name="projects")
app.add_typer(db_app, name="db")
app.add_typer(spaces_app, name="spaces")


@app.callback()
def _global_options(
    ctx: typer.Context,
    read_only: Annotated[
        bool,
        typer.Option(
            "--read-only",
            help=(
                "Refuse every command that writes. For an agent or a shell that "
                "should be able to look but not touch: the corpus, the database "
                "and the blob store all stay as they are, and a refused command "
                "exits 3 having changed nothing."
            ),
        ),
    ] = False,
) -> None:
    """ArXiv AI assistant: search, ingest, extract and retrieve."""
    read_mode(read_only)
    ctx.obj = {"read_only": read_only}


# --------------------------------------------------------------------- helpers
def _bootstrap() -> None:
    settings = get_settings()
    configure_logging(settings.logging.level, settings.logging.json_output)


def _print_papers(papers) -> None:  # noqa: ANN001 - Sequence[PaperMetadata]
    """Render paper metadata. Takes metadata, not search hits, so every
    command that ends up with papers can share one table."""
    table = Table(show_lines=False, box=None, pad_edge=False)
    table.add_column("arXiv id", style="cyan", no_wrap=True)
    table.add_column("Title", style="bold")
    table.add_column("Cats", style="magenta")
    table.add_column("Date", style="green", no_wrap=True)
    for metadata in papers:
        categories = ",".join(metadata.categories[:3])
        published = metadata.published_at.date().isoformat() if metadata.published_at else "-"
        table.add_row(metadata.versioned_id, metadata.title[:78], categories, published)
    console.print(table)


def _paper_dict(paper) -> dict[str, object]:  # noqa: ANN001 - PaperMetadata
    """One paper as JSON. Shared so `search` and `harvest` agree field-for-field."""
    return {
        "arxiv_id": paper.arxiv_id,
        "versioned_id": paper.versioned_id,
        "title": paper.title,
        "authors": list(paper.author_names),
        "categories": list(paper.categories),
        "abstract": paper.abstract,
        "published": paper.published_at.isoformat() if paper.published_at else None,
        "abs_url": paper.abs_url,
        "pdf_url": paper.pdf_url,
        "html_url": paper.html_url,
    }


# ------------------------------------------------------------------------ search
#: Set by ``--read-only``. Process-wide because the guard is process-wide: the
#: point of the mode is that *this invocation of the CLI* cannot change anything,
#: and a flag threaded through every command would be one someone forgets to pass.
_READ_ONLY = False


def read_mode(enabled: bool) -> None:
    """Enable or clear read-only mode. Called by the global ``--read-only`` flag."""
    global _READ_ONLY  # noqa: PLW0603 - one process, one mode, set once at startup
    _READ_ONLY = enabled


def is_read_only() -> bool:
    return _READ_ONLY


def writes_when(command: str, *flags: str) -> Callable[[F], F]:
    """Mark a command as writing only when one of ``flags`` is set.

    For the commands that read by default and change only when asked twice —
    ``paper runs --reap`` reports and ``--apply`` closes, ``paper sections --prune``
    reports and ``--apply`` deletes. Marking the whole command as a writer would
    refuse the report, which is the part that is safe; marking it as a reader
    would let ``--read-only … --apply`` through, which breaks the one promise the
    mode makes.

    The first positional argument is the Typer parameter name, not the flag: the
    function is called with its arguments already bound, so there is no parsing to
    do and no way for the flag and the parameter to disagree.
    """

    def decorate(function: F) -> F:
        @functools.wraps(function)
        def guarded(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
            if _READ_ONLY and any(bool(kwargs.get(flag)) for flag in flags):
                _refuse(f"{command} --{'/--'.join(flags)}")
            return function(*args, **kwargs)

        guarded._writes_command = command  # type: ignore[attr-defined]
        guarded._writes_flags = flags  # type: ignore[attr-defined]
        guarded._read_only_guarded = True  # type: ignore[attr-defined]
        return cast("F", guarded)

    return decorate


def writes(command: str) -> Callable[[F], F]:
    """Mark a command as one that changes something, and refuse it in read mode.

    The marker is what makes the completeness test possible: ``tests/test_read_only``
    walks every command Typer exposes and asserts it is *either* decorated or in
    the read-only allowlist. A new command therefore cannot ship unguarded — it
    fails a test instead of quietly being allowed to write in a mode whose entire
    promise is that it will not.

    Marking by hand rather than inferring from the name is deliberate: ``show``,
    ``search`` and ``doctor`` read, ``runs`` reads unless reaped, and ``kinds``
    reads unless it is asked to relabel. The verbs do not decide; the behaviour
    does.
    """

    def decorate(function: F) -> F:
        if getattr(function, "_writes_command", None) is not None:  # pragma: no cover
            msg = f"{command} is already marked as writing"
            raise ValueError(msg)
        function._writes_command = command  # type: ignore[attr-defined]
        # Empty tuple = "always writes". A non-empty one names the flags that make
        # it write, which is what lets a report-then-apply command live in the
        # read-only allowlist without lying. See :func:`writes_when`.
        function._writes_flags = ()  # type: ignore[attr-defined]
        if not getattr(function, "_read_only_guarded", False):

            @functools.wraps(function)
            def guarded(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
                if _READ_ONLY:
                    _refuse(command)
                return function(*args, **kwargs)

            guarded._writes_command = command  # type: ignore[attr-defined]
            guarded._writes_flags = ()  # type: ignore[attr-defined]
            guarded._read_only_guarded = True  # type: ignore[attr-defined]
            return cast("F", guarded)
        return function

    return decorate


def _refuse(command: str) -> None:
    """Say no, say why, and exit with the code a caller can branch on."""
    console.print(
        f"[red]read-only mode:[/red] [bold]{command}[/bold] changes the corpus and "
        "was not run."
    )
    console.print(
        "[dim]Drop --read-only to allow writes. Nothing was modified.[/dim]"
    )
    raise typer.Exit(code=3)


#: Commands that change nothing. Everything Typer exposes must appear here or be
#: decorated with :func:`writes`; the completeness test enforces it. Kept
#: explicit rather than inferred so that adding a command is a decision.
READ_ONLY_COMMANDS: frozenset[str] = frozenset({
    "search",
    "assets",
    "ask",
    "show",
    "runs",
    "outline",
    "sections",
    "section",
    "doctor",
    "config",
    "version",
    "session",
    "mcp",
    "serve",
    "db:revision",
    "spaces:list",
    "spaces:sql",
    "projects:list",
    "projects:show",
    "projects:refs",
})


@app.command()
def search(
    filter_args: Annotated[
        list[str] | None,
        typer.Argument(
            metavar="[FILTER]",
            help="ArXiv query, verbatim: 'cat:cs.CL AND ti:\"attention\"'.",
        ),
    ] = None,
    phrase: Annotated[
        list[str] | None,
        typer.Option(
            "--phrase",
            help="Search an exact phrase. Immune to shell quoting.",
        ),
    ] = None,
    title: Annotated[list[str] | None, typer.Option("--title", "-t", help="ti: Title")] = None,
    author: Annotated[list[str] | None, typer.Option("--author", help="au: Author")] = None,
    abstract: Annotated[list[str] | None, typer.Option("--abstract", help="abs: Abstract")] = None,
    comment: Annotated[list[str] | None, typer.Option("--comment", help="co: Comment")] = None,
    journal: Annotated[
        list[str] | None, typer.Option("--journal", help="jr: Journal reference")
    ] = None,
    category: Annotated[
        list[str] | None, typer.Option("--category", "-c", help="cat: Subject category")
    ] = None,
    report_number: Annotated[
        list[str] | None, typer.Option("--report-number", help="rn: Report number")
    ] = None,
    every_field: Annotated[
        list[str] | None, typer.Option("--all", help="all: search every field")
    ] = None,
    operator: Annotated[
        str, typer.Option("--op", help="Combine values within a field: AND | OR | ANDNOT")
    ] = "AND",
    arxiv_id: Annotated[
        list[str] | None, typer.Option("--id", help="Look up specific papers.")
    ] = None,
    submitted_from: Annotated[
        str | None, typer.Option("--since", help="Earliest submission (YYYY-MM-DD).")
    ] = None,
    submitted_to: Annotated[
        str | None, typer.Option("--until", help="Latest submission (YYYY-MM-DD).")
    ] = None,
    max_results: Annotated[int, typer.Option("--max", "-n", min=1, max=100)] = 10,
    start: Annotated[int, typer.Option("--offset", min=0)] = 0,
    sort_by: Annotated[str, typer.Option("--sort-by")] = "relevance",
    sort_order: Annotated[str, typer.Option("--sort-order")] = "descending",
    has_pdf: Annotated[bool | None, typer.Option("--has-pdf/--no-pdf")] = None,
    has_html: Annotated[bool | None, typer.Option("--has-html/--no-html")] = None,
    has_doi: Annotated[bool | None, typer.Option("--has-doi/--no-doi")] = None,
    ingested: Annotated[
        bool | None, typer.Option("--ingested/--not-ingested", help="Filter by local corpus.")
    ] = None,
    exclude_categories: Annotated[
        list[str] | None, typer.Option("--exclude-category")
    ] = None,
    show_query: Annotated[
        bool, typer.Option("--show-query", help="Print the compiled search_query and exit.")
    ] = False,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Search ArXiv with ArXiv's own filter language. Ingests nothing.

    Field prefixes: ti title, au author, abs abstract, co comment, jr journal
    reference, cat subject category, rn report number, all everything.
    Operators must be UPPERCASE: AND, OR, ANDNOT. Phrases go in double quotes,
    parentheses group, dates use --since/--until.

        paper search 'cat:cs.CL AND ti:"attention"'
        paper search --title transformer --category cs.LG --since 2024-01-01
        paper search --title a --title b --op OR
        paper search --id 1706.03762
    """

    async def run() -> int:
        from app.clients.arxiv.filters import search_query_from
        from app.container import get_container
        from app.domain.filters import FilterProblem
        from app.services.arxiv_search import ArxivSearchService

        raw = " ".join(filter_args) if filter_args else None
        try:
            request = search_query_from(
                raw=raw,
                phrases=phrase,
                fields={
                    "title": title,
                    "author": author,
                    "abstract": abstract,
                    "comment": comment,
                    "journal": journal,
                    "category": category,
                    "report_number": report_number,
                    "all": every_field,
                },
                operator=operator,
                id_list=arxiv_id,
                max_results=max_results,
                start=start,
                sort_by=sort_by,
                sort_order=sort_order,
                submitted_from=submitted_from,
                submitted_to=submitted_to,
                post={
                    "has_pdf": has_pdf,
                    "has_html": has_html,
                    "has_doi": has_doi,
                    "ingested": ingested,
                    "exclude_categories": exclude_categories,
                },
            )
        except FilterProblem as exc:
            console.print(f"[red]invalid filter:[/red] {exc}")
            return 2

        if request.filter.is_empty and not request.id_list:
            console.print(
                "[red]nothing to search for[/red] — pass a FILTER or one of "
                "--title / --author / --abstract / --category / --id"
            )
            return 2

        for warning in request.warnings:
            console.print(f"[yellow]warning:[/yellow] {warning}")

        compiled = request.filter.compile()
        if show_query:
            # soft_wrap so the query stays on one line and can be copied as-is.
            console.print(compiled or "(id_list only)", soft_wrap=True, style="cyan")
            return 0

        container = get_container()
        try:
            service = ArxivSearchService(
                container.arxiv, session_factory=container.session_factory
            )
            outcome = await service.search(request)
        finally:
            await container.aclose()

        if not outcome.ok:
            console.print(f"[red]search failed:[/red] {outcome.error}")
            return 1

        if json_output:
            console.print_json(
                json.dumps(
                    {
                        **outcome.as_dict(),
                        "search_query": compiled,
                        "papers": [_paper_dict(paper) for paper in outcome.papers],
                    },
                    default=str,
                )
            )
            return 0

        console.print(f"[dim]search_query:[/dim] {compiled or '(id_list)'}")
        if outcome.hint:
            console.print(f"[yellow]hint:[/yellow] {outcome.hint}")
        summary = f"[bold]{outcome.returned}[/bold] shown"
        if outcome.total_results:
            summary += f" of {outcome.total_results} matching"
        if outcome.filtered_out:
            summary += f", [red]{outcome.filtered_out} filtered out[/red]"
        console.print(summary + "\n")
        if outcome.papers:
            _print_papers(outcome.papers)
        if outcome.next_start:
            console.print(
                f"\n[dim]more available:[/dim] re-run with --offset {outcome.next_start}"
            )
        return 0

    raise typer.Exit(asyncio.run(run()))


# ---------------------------------------------------------------------- ingest
@app.command()
@writes("ingest")
def ingest(
    target: Annotated[
        str,
        typer.Argument(
            metavar="ID_OR_PATH",
            help="ArXiv id, versioned id, URL, or a path to a local PDF.",
        ),
    ],
    pdf: Annotated[bool, typer.Option("--pdf", help="Skip HTML and use the PDF path.")] = False,
    force: Annotated[bool, typer.Option("--force", help="Re-download even if cached.")] = False,
    space: Annotated[
        str | None,
        typer.Option("--space", "-s", help="Embedding space to write. Default: active space."),
    ] = None,
    embed_device: Annotated[
        str | None,
        typer.Option(
            "--device",
            help="Device for embedding: cpu, cuda, cuda:1, mps. Default: configured (cpu).",
        ),
    ] = None,
    extract_device: Annotated[
        str | None,
        typer.Option(
            "--extract-device",
            help="Device for MinerU on this document: cpu, cuda, mps.",
        ),
    ] = None,
    kind: Annotated[
        str,
        typer.Option(
            "--kind",
            "-k",
            help="For a local file: paper, book, notes, report or thesis.",
        ),
    ] = "book",
    doc_title: Annotated[
        str | None,
        typer.Option("--title", help="Override the title of a local document."),
    ] = None,
    doc_key: Annotated[
        str | None,
        typer.Option("--doc-key", help="Handle for a local document. Default: filename slug."),
    ] = None,
    pages: Annotated[
        str | None,
        typer.Option(
            "--pages",
            help="Only these 1-based pages, e.g. 40-52, 40, 40-, -52.",
        ),
    ] = None,
    language: Annotated[
        str | None,
        typer.Option("--language", help="MinerU language hint for a local file, e.g. tr, en."),
    ] = None,
    method: Annotated[
        str | None,
        typer.Option("--method", help="MinerU parse method: auto, txt or ocr."),
    ] = None,
    ocr: Annotated[
        str | None,
        typer.Option("--ocr-language", help="PaddleOCR language, when it must differ."),
    ] = None,
) -> None:
    """Run the full pipeline (fetch -> MinerU -> chunk -> embed) for one paper.

    A path to a PDF ingests a local document instead: same pipeline, same
    embedding space, addressed by a ``doc_key`` rather than an arXiv id.

    Adding a paper to a second space does not touch the first one's vectors:
    each space has its own table.
    """
    is_file = Path(target).expanduser().is_file()

    async def run() -> int:
        from app.container import get_container
        from app.services.ingestion import build_ingestion_service

        container = get_container()
        active = await _resolve(container, space)
        await container.vector_store_for(active).ensure_ready()
        service = build_ingestion_service(container, active, device=_device(embed_device))
        try:
            if is_file:
                result = await service.ingest_file(
                    Path(target),
                    kind=kind,
                    title=doc_title,
                    doc_key=doc_key,
                    page_range=_page_range(pages),
                    options=_mineru_options(language, method, ocr, _device(extract_device)),
                    force=force,
                    requested_by="cli",
                    trigger="cli",
                )
            else:
                result = await service.ingest_paper(
                    target, prefer_html=not pdf, force=force,
                    requested_by="cli", trigger="cli",
                )
        finally:
            await container.aclose()

        console.print(f"[dim]space:[/dim] {active.name} ({active.resolved_table})")
        console.print(
            f"[dim]embedding device:[/dim] {service.embedding_device}"
            + (f" [dim]extraction device:[/dim] {extract_device}" if extract_device else "")
        )
        if is_file and result.status is RunStatus.SKIPPED:
            for message in result.errors.values():
                console.print(f"[yellow]{message}[/yellow]")
                return 0
        _render_runs(result)
        return 0 if result.succeeded else 1

    raise typer.Exit(asyncio.run(run()))


def _identity_block(paper: Paper) -> str:
    """The header rows that describe what this row *is*.

    Split by kind because a preprint and a textbook do not share fields: printing
    ``id: None`` and an empty category list for every local document is what a
    reader sees as a broken row rather than as a book.
    """
    if not paper.is_paper:
        pages = f"pages     : {paper.page_count or 'unknown'}"
        # The source is wherever the bytes came from, which for a local document
        # is a path and is recorded on its raw document, not on the paper row.
        source = next(
            (doc.source_url for doc in paper.contents if doc.source_url),
            "-",
        )
        # Long enough to break the panel across two lines, which turns one field
        # into two misaligned rows. The tail is the informative end.
        if len(source) > 58:
            source = "…" + source[-57:]
        return (
            f"doc-key   : {paper.doc_key}\n"
            f"kind      : {paper.kind}\n"
            f"{pages}\n"
            f"source    : {source}\n"
        )
    return (
        f"id        : {paper.versioned_id}\n"
        f"authors   : {', '.join(paper.author_names[:6])}\n"
        f"categories: {', '.join(paper.categories)}\n"
        f"published : {paper.published_at}\n"
        f"html      : {paper.html_url or '-'}\n"
        f"pdf       : {paper.pdf_url}\n"
    )


#: Shown for a hit that has no section. The backslash is not decoration: Rich reads
#: a bare ``[...]`` as markup and renders it as nothing, so the unescaped version
#: printed an empty column exactly where a label was wanted.
NO_SECTION_LABEL = r"\[no section]"


def _print_by_section(hits: list) -> int:  # noqa: ANN001 - list[SemanticSearchHit]
    """Group hits into the sections and page ranges they fall in.

    The question a list of chunks cannot answer. Ten scattered chunks from a
    700-page book are ten places to look; "pages 24-31 of chapter 2" is one. The
    group is the *section*, so the page range shown is the section's own — which is
    wider than the pages that actually matched, and is the point: the reader wants
    where to read, not which sentences scored highest.

    Hits with no section keep their page, because "page 318, unknown section" is
    still more use than a row that says nothing.
    """
    groups: dict[tuple, dict] = {}
    for hit in hits:
        meta = hit.metadata
        # Grouped by section, not by page. Grouping by the page a chunk starts on
        # put three hits from one section on three rows, which is the opposite of
        # the answer this view exists to give.
        key = (
            meta.get("doc_key") or meta.get("display_id") or "",
            meta.get("section_title") or "",
        )
        row = groups.setdefault(
            key,
            {
                "doc": meta.get("doc_key") or meta.get("display_id") or "",
                "title": meta.get("title") or "",
                "section": meta.get("section_title") or "",
                "level": meta.get("section_level") or 0,
                "section_pages": (
                    meta.get("section_page_start"),
                    meta.get("section_page_end"),
                ),
                "pages": [],
                "hits": 0,
                "best": 0.0,
            },
        )
        row["hits"] += 1
        row["best"] = max(row["best"], hit.score)
        start = meta.get("page_start")
        end = meta.get("page_end")
        if start is not None:
            row["pages"].append((start, end if end is not None else start))

    if not groups:
        console.print("[yellow]no matches[/yellow]")
        return 1

    ordered = sorted(
        groups.values(),
        key=lambda r: (-float(r["best"]), r["doc"]),
    )
    table = Table(box=None, pad_edge=False)
    for column, style in (
        ("pages", "cyan"),
        ("section", "bold"),
        ("doc", "dim"),
        ("hits", "dim"),
        ("best", "magenta"),
    ):
        table.add_column(column, style=style)

    for row in ordered:
        section_start, section_end = row["section_pages"]
        # The section's own span when it is known, since that is the range a
        # reader would open the book at. Falls back to the pages that actually
        # matched, which is all there is for a chunk with no section.
        if section_start is not None:
            spans = (
                str(section_start)
                if section_end in (None, section_start)
                else f"{section_start}-{section_end}"
            )
        else:
            spans = _merge_spans(row["pages"])
        indent = "  " * max(0, int(row["level"]) - 1)
        table.add_row(
            spans,
            # `\[`: Rich reads a bare `[...]` as a markup tag and renders it as
            # nothing, so an unsectioned hit printed an empty column.
            f"{indent}{row['section'] or NO_SECTION_LABEL}",
            row["doc"],
            str(row["hits"]),
            f"{float(row['best']):.4f}",
        )
    console.print(table)
    console.print(
        "[dim]pages are the section's own range — where to read, not which "
        "sentences scored[/dim]"
    )
    return 0


def _merge_spans(pages: list[tuple[int, int]]) -> str:
    """``[(24, 24), (26, 29), (31, 31)]`` -> ``24, 26-29, 31``.

    Merging overlapping and adjacent pages matters because a section's chunks are
    contiguous but hits are not: without it one section hit at pages 24, 25, 26
    prints as three rows of noise instead of one range.
    """
    if not pages:
        return "-"
    ordered = sorted(pages)
    merged: list[list[int]] = [list(ordered[0])]
    for start, end in ordered[1:]:
        last = merged[-1]
        if start <= last[1] + 1:
            last[1] = max(last[1], end)
        else:
            merged.append([start, end])
    return ", ".join(str(s) if s == e else f"{s}-{e}" for s, e in merged)


def _locator(metadata: dict[str, object]) -> str:
    """``p.24 § 2.2 Debye's Calculation`` — page and section of a search hit.

    Empty for an arXiv paper, which has neither. Both halves are optional and
    shown independently, because a chunk can know its page without landing in a
    section and the reverse.
    """
    parts: list[str] = []
    page = metadata.get("page_start")
    if page is not None:
        end = metadata.get("page_end")
        if end is not None and end != page:
            parts.append(f"p.{page}-{end}")
        else:
            parts.append(f"p.{page}")
    section = metadata.get("section_title")
    if section:
        parts.append(f"§ {section}")
    return "  ".join(parts)


def _page_range(value: str | None) -> PageRange | None:
    """Parse ``--pages`` once, here, so a typo fails before the file is read."""
    if not value:
        return None
    try:
        return PageRange.parse(value)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _mineru_options(
    language: str | None,
    method: str | None,
    ocr_language: str | None,
    device: str | None = None,
) -> MineruOptions:
    """Build per-document extraction overrides from the CLI flags.

    Validated against what the CLI actually accepts rather than passed through:
    MinerU's own error for an unknown language arrives minutes later, after a
    model download, which is the worst possible time to learn that a flag was
    misspelled.
    """
    if method is not None and method not in {"auto", "txt", "ocr"}:
        raise typer.BadParameter(f"--method must be auto, txt or ocr, not {method!r}")
    if language is not None and not re.fullmatch(r"[a-z]{2}(-[A-Za-z]+)?", language):
        raise typer.BadParameter(
            f"--language must look like 'en' or 'ch', not {language!r}"
        )
    return MineruOptions(
        language=language, method=method, ocr_language=ocr_language, device=device
    )


@app.command()
@writes("harvest")
def harvest(
    filter_args: Annotated[
        list[str] | None,
        typer.Argument(
            metavar="[FILTER]",
            help="ArXiv query, verbatim: 'cat:cs.CL AND ti:\"attention\"'.",
        ),
    ] = None,
    phrase: Annotated[
        list[str] | None,
        typer.Option(
            "--phrase",
            help="Search an exact phrase. Immune to shell quoting.",
        ),
    ] = None,
    title: Annotated[list[str] | None, typer.Option("--title", "-t", help="ti: Title")] = None,
    author: Annotated[list[str] | None, typer.Option("--author", help="au: Author")] = None,
    abstract: Annotated[list[str] | None, typer.Option("--abstract", help="abs: Abstract")] = None,
    comment: Annotated[list[str] | None, typer.Option("--comment", help="co: Comment")] = None,
    journal: Annotated[
        list[str] | None, typer.Option("--journal", help="jr: Journal reference")
    ] = None,
    category: Annotated[
        list[str] | None, typer.Option("--category", "-c", help="cat: Subject category")
    ] = None,
    report_number: Annotated[
        list[str] | None, typer.Option("--report-number", help="rn: Report number")
    ] = None,
    every_field: Annotated[
        list[str] | None, typer.Option("--all", help="all: search every field")
    ] = None,
    operator: Annotated[
        str, typer.Option("--op", help="Combine values within a field: AND | OR | ANDNOT")
    ] = "AND",
    arxiv_id: Annotated[
        list[str] | None, typer.Option("--id", help="Restrict to these arXiv ids.")
    ] = None,
    submitted_from: Annotated[
        str | None, typer.Option("--since", help="Earliest submission (YYYY-MM-DD).")
    ] = None,
    submitted_to: Annotated[
        str | None, typer.Option("--until", help="Latest submission (YYYY-MM-DD).")
    ] = None,
    limit: Annotated[int, typer.Option("--limit", "-l", min=1, max=100)] = 5,
    start: Annotated[int, typer.Option("--offset", min=0)] = 0,
    sort_by: Annotated[str, typer.Option("--sort-by")] = "relevance",
    sort_order: Annotated[str, typer.Option("--sort-order")] = "descending",
    has_pdf: Annotated[bool | None, typer.Option("--has-pdf/--no-pdf")] = None,
    has_html: Annotated[bool | None, typer.Option("--has-html/--no-html")] = None,
    pdf: Annotated[bool, typer.Option("--pdf", help="Skip HTML and use the PDF path.")] = False,
    force: Annotated[bool, typer.Option("--force", help="Re-download even if cached.")] = False,
    space: Annotated[str | None, typer.Option("--space", "-s")] = None,
) -> None:
    """Ingest the top-N papers matching a filter.

    Takes the same filter language as `paper search`, so a filter that finds
    papers there will ingest exactly those here.
    """

    async def run() -> int:
        from app.clients.arxiv.filters import search_query_from
        from app.container import get_container
        from app.domain.filters import FilterProblem
        from app.services.ingestion import build_ingestion_service

        raw = " ".join(filter_args) if filter_args else None
        try:
            request = search_query_from(
                raw=raw,
                phrases=phrase,
                fields={
                    "title": title,
                    "author": author,
                    "abstract": abstract,
                    "comment": comment,
                    "journal": journal,
                    "category": category,
                    "report_number": report_number,
                    "all": every_field,
                },
                operator=operator,
                id_list=arxiv_id,
                max_results=limit,
                start=start,
                sort_by=sort_by,
                sort_order=sort_order,
                submitted_from=submitted_from,
                submitted_to=submitted_to,
                post={"has_pdf": has_pdf, "has_html": has_html},
            )
        except FilterProblem as exc:
            console.print(f"[red]invalid filter:[/red] {exc}")
            return 2

        if request.filter.is_empty and not request.id_list:
            console.print(
                "[red]nothing to harvest[/red] — pass a FILTER or one of "
                "--title / --author / --abstract / --category / --id"
            )
            return 2

        for warning in request.warnings:
            console.print(f"[yellow]warning:[/yellow] {warning}")

        container = get_container()
        target = await _resolve(container, space)
        await container.vector_store_for(target).ensure_ready()
        service = build_ingestion_service(container, target)
        try:
            result = await service.ingest_query(
                request.to_search_query(),
                limit=limit,
                prefer_html=not pdf,
                force=force,
                requested_by="cli",
                trigger="cli",
            )
        finally:
            await container.aclose()
        console.print(f"[dim]space:[/dim] {target.name} ({target.resolved_table})")
        console.print(f"[dim]filter:[/dim] {request.filter.compile() or '(id_list)'}")
        _render_runs(result)
        return 0 if result.succeeded else 1

    raise typer.Exit(asyncio.run(run()))


# ------------------------------------------------------------------- projects
def _print_project_papers(rows, *, title: str) -> None:  # noqa: ANN001
    """Shared table for `projects show` and `projects add --list`."""
    table = Table(box=None, pad_edge=False)
    table.add_column("", style="green")
    table.add_column("arXiv id", style="cyan", no_wrap=True)
    table.add_column("Title", style="bold")
    table.add_column("Cats", style="magenta")
    table.add_column("Note", style="dim")
    for paper, link in rows:
        table.add_row(
            "x" if link.is_read else " ",
            paper.versioned_id,
            paper.title[:60],
            ",".join(paper.categories[:2]),
            (link.note or "")[:24],
        )
    console.print(Panel(table, title=title, border_style="cyan"))


@projects_app.command("new")
@writes("projects:new")
def projects_new(
    name: Annotated[str, typer.Argument(help="Human-readable project name.")],
    description: Annotated[str | None, typer.Option("--description", "-d")] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Create a project. Papers stay global; a project only references them."""

    async def run() -> int:
        from app.container import get_container
        from app.db.project_repository import ProjectConflictError, ProjectRepository

        container = get_container()
        async with container.session_factory() as session:
            try:
                project = await ProjectRepository(session).create(
                    name, description=description
                )
            except ProjectConflictError as exc:
                console.print(f"[red]{exc}[/red]")
                return 2
            await session.commit()
        if json_output:
            console.print_json(
                json.dumps({"id": project.id, "name": project.name, "slug": project.slug})
            )
        else:
            console.print(f"created [bold]{project.name}[/bold] as [cyan]{project.slug}[/cyan]")
        return 0

    raise typer.Exit(asyncio.run(run()))


@projects_app.command("list")
def projects_list(
    include_archived: Annotated[bool, typer.Option("--all", help="Include archived.")] = False,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """List projects with paper and read counts."""

    async def run() -> int:
        from app.container import get_container
        from app.db.project_repository import ProjectRepository

        container = get_container()
        async with container.session_factory() as session:
            projects = await ProjectRepository(session).list_projects(
                include_archived=include_archived
            )

        if json_output:
            console.print_json(
                json.dumps(
                    [
                        {
                            "slug": row.slug,
                            "name": row.name,
                            "description": row.description,
                            "paper_count": row.paper_count,
                            "read_count": row.read_count,
                            "is_archived": row.is_archived,
                        }
                        for row in projects
                    ]
                )
            )
            return 0

        if not projects:
            console.print("[yellow]no projects yet[/yellow] — try: paper projects new \"My topic\"")
            return 0
        table = Table(box=None, pad_edge=False)
        table.add_column("", style="green")
        table.add_column("slug", style="cyan", no_wrap=True)
        table.add_column("name", style="bold")
        table.add_column("papers", justify="right")
        table.add_column("read", justify="right", style="dim")
        table.add_column("description", style="dim")
        for row in projects:
            table.add_row(
                "*" if row.is_archived else " ",
                row.slug,
                row.name,
                str(row.paper_count),
                str(row.read_count),
                (row.description or "")[:40],
            )
        console.print(table)
        return 0

    raise typer.Exit(asyncio.run(run()))


@projects_app.command("add")
@writes("projects:add")
def projects_add(
    project: Annotated[str, typer.Argument(help="Project slug or name.")],
    paper_ids: Annotated[
        list[str],
        typer.Argument(help="arXiv ids, versioned ids or abs URLs to import."),
    ],
    note: Annotated[str | None, typer.Option("--note", help="Per-project note.")] = None,
    ingest: Annotated[
        bool,
        typer.Option("--ingest", help="Ingest any that are not in the corpus yet."),
    ] = False,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Import papers into a project.

    Papers are never copied: an arXiv id that is not stored yet is fetched and
    ingested first, then linked. Importing the same list twice is a no-op.
    """

    async def run() -> int:
        from app.container import get_container
        from app.db.project_repository import PaperNotFoundError, ProjectRepository
        from app.services.ingestion import build_ingestion_service

        container = get_container()
        target = await _resolve(container, None)
        await container.vector_store_for(target).ensure_ready()
        service = build_ingestion_service(container, target)

        repo = ProjectRepository(container.session_factory())
        async with container.session_factory() as session:
            projects = ProjectRepository(session)
            try:
                found = await projects.require(project)
            except PaperNotFoundError as exc:
                console.print(f"[red]{exc}[/red]")
                return 2
            resolved, missing = await projects.resolve_paper_ids(paper_ids)
        del repo

        if missing and not ingest:
            console.print(
                f"[red]{len(missing)} not in the corpus:[/red] {', '.join(missing[:6])}"
                + ("..." if len(missing) > 6 else "")
            )
            console.print("[dim]re-run with --ingest to fetch and ingest them first[/dim]")
            return 2

        for arxiv_id in missing:
            await _progress_console(f"ingesting {arxiv_id}")
            await service.ingest_paper(
                arxiv_id, prefer_html=True, requested_by="cli", trigger="cli"
            )

        async with container.session_factory() as session:
            projects = ProjectRepository(session)
            found = await projects.require(project)
            resolved, still_missing = await projects.resolve_paper_ids(paper_ids)
            if still_missing:
                console.print(
                    f"[yellow]still missing after ingest:[/yellow] {', '.join(still_missing[:6])}"
                )
            added = await projects.add_papers(
                found, resolved, added_by="cli", note=note
            )
            rows = await projects.list_papers(found)
            await session.commit()

        if json_output:
            console.print_json(
                json.dumps(
                    {
                        "project": found.slug,
                        "requested": len(paper_ids),
                        "resolved": len(resolved),
                        "added": added,
                        "papers": [
                            {"arxiv_id": paper.arxiv_id, "title": paper.title}
                            for paper, _ in rows
                        ],
                    }
                )
            )
        else:
            console.print(
                f"[bold]{added}[/bold] imported into [cyan]{found.slug}[/cyan]"
                f" ({len(paper_ids)} requested, {len(resolved)} resolved)"
            )
            _print_project_papers(rows, title=f"{found.name} — {len(rows)} papers")
        return 0

    raise typer.Exit(asyncio.run(run()))


@projects_app.command("show")
def projects_show(
    project: Annotated[str, typer.Argument(help="Project slug or name.")],
    limit: Annotated[int, typer.Option("--limit", "-n", min=1, max=200)] = 50,
    unread_only: Annotated[bool, typer.Option("--unread")] = False,
    category: Annotated[str | None, typer.Option("--category", "-c")] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """List the papers in a project."""

    async def run() -> int:
        from app.container import get_container
        from app.db.project_repository import PaperNotFoundError, ProjectRepository

        container = get_container()
        async with container.session_factory() as session:
            projects = ProjectRepository(session)
            try:
                found = await projects.require(project)
            except PaperNotFoundError as exc:
                console.print(f"[red]{exc}[/red]")
                return 2
            rows = await projects.list_papers(
                found, limit=limit, unread_only=unread_only, category=category
            )

        if json_output:
            console.print_json(
                json.dumps(
                    [
                        {
                            "arxiv_id": paper.arxiv_id,
                            "title": paper.title,
                            "categories": list(paper.categories),
                            "authors": list(paper.author_names),
                            "is_read": link.is_read,
                            "note": link.note,
                        }
                        for paper, link in rows
                    ]
                )
            )
        else:
            _print_project_papers(rows, title=f"{found.name} — {len(rows)} shown")
        return 0

    raise typer.Exit(asyncio.run(run()))


@projects_app.command("rm")
@writes("projects:rm")
def projects_rm(
    project: Annotated[str, typer.Argument(help="Project slug.")],
    paper_id: Annotated[
        str | None,
        typer.Option("--paper", help="Unlink one paper instead of the project."),
    ] = None,
) -> None:
    """Unlink a paper, or delete the project. Papers themselves always survive."""

    async def run() -> int:
        from app.container import get_container
        from app.db.project_repository import PaperNotFoundError, ProjectRepository

        container = get_container()
        async with container.session_factory() as session:
            projects = ProjectRepository(session)
            try:
                found = await projects.require(project)
            except PaperNotFoundError as exc:
                console.print(f"[red]{exc}[/red]")
                return 2
            if paper_id is None:
                name = found.name
                await projects.delete(found)
                await session.commit()
                console.print(f"deleted project [bold]{name}[/bold]; its papers remain in the corpus")
                return 0
            resolved, _ = await projects.resolve_paper_ids([paper_id])
            if not resolved:
                console.print(f"[red]{paper_id} is not in the corpus[/red]")
                return 2
            removed = await projects.remove_paper(found, resolved[0])
            await session.commit()
        if removed:
            console.print(f"unlinked {paper_id} from [cyan]{found.slug}[/cyan]")
            return 0
        console.print(f"[yellow]{paper_id} was not in {found.slug}[/yellow]")
        return 1

    raise typer.Exit(asyncio.run(run()))


@projects_app.command("read")
@writes("projects:read")
def projects_read(
    project: Annotated[str, typer.Argument(help="Project slug.")],
    paper_id: Annotated[str | None, typer.Argument(help="Paper id; omit for all.")] = None,
    unread: Annotated[bool, typer.Option("--unread", help="Mark unread instead.")] = False,
) -> None:
    """Mark a paper (or the whole project) read or unread."""

    async def run() -> int:
        from app.container import get_container
        from app.db.project_repository import PaperNotFoundError, ProjectRepository

        container = get_container()
        async with container.session_factory() as session:
            projects = ProjectRepository(session)
            try:
                found = await projects.require(project)
            except PaperNotFoundError as exc:
                console.print(f"[red]{exc}[/red]")
                return 2
            targets, outside = (
                await projects.link_paper_ids(found, [paper_id])
                if paper_id
                else (await projects.paper_ids(found), [])
            )
            if outside:
                # Membership, not just existence: a corpus paper the project does
                # not hold would update zero rows and still report success.
                console.print(f"[red]{paper_id} is not in project {found.slug!r}[/red]")
                return 2
            changed = 0
            for target in targets:
                if await projects.set_read(found, target, is_read=not unread):
                    changed += 1
            await session.commit()
        console.print(f"{changed} paper(s) marked {'unread' if unread else 'read'}")
        return 0

    raise typer.Exit(asyncio.run(run()))


# --------------------------------------------------------------------- assets
async def _paper_or_none(session, arxiv_id: str):  # noqa: ANN001, ANN202
    """Resolve an arXiv id the way every paper-facing command does.

    Tries the versioned and bare forms, because callers paste whichever they have.
    """
    from app.db.repositories import PaperRepository

    papers = PaperRepository(session)
    # `resolve`, not `get_by_arxiv_id`: the argument may name a local document,
    # which has no arXiv id at all. `resolve` also normalises, so the versioned
    # and bare arXiv forms both still work.
    return await papers.resolve(arxiv_id)


@app.command()
def assets(
    arxiv_id: Annotated[
        str,
        typer.Argument(help="ArXiv id, versioned id, URL, or a document's doc_key."),
    ],
    kind: Annotated[
        str, typer.Option("--kind", "-k", help="figures | tables | equations | all")
    ] = "all",
    project: Annotated[
        str | None,
        typer.Option(
            "--project",
            "-p",
            help="List figures/tables/equations across a whole project instead of one paper.",
        ),
    ] = None,
    limit: Annotated[int, typer.Option("--limit", "-n", min=1, max=500)] = 20,
    display_only: Annotated[
        bool, typer.Option("--display", help="Equations: only numbered display math.")
    ] = False,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Show a paper's figures, tables and equations.

    Figures and tables carry captions and image locations; equations carry LaTeX.
    Inline math is stored but hidden by default — a paper has ~3 real equations
    and ~140 inline fragments, and you almost always want the first kind.
    """
    show_all = kind in {"all", "*"}
    want = {"figures", "tables", "equations"} if show_all else {kind}

    async def run() -> int:
        from app.container import get_container
        from app.db.asset_repository import AssetRepository

        container = get_container()
        if project:
            # Project-wide listing. An asset question is often a corpus question
            # ("show me every figure about the sheaf laplacian"), and per-paper
            # paging through 36 papers to answer it is the wrong tool.
            return await _assets_across_project(container, project, want, limit, display_only)

        async with container.session_factory() as session:
            paper = await _paper_or_none(session, arxiv_id)
            if paper is None:
                console.print(f"[yellow]{arxiv_id} is not in the corpus[/yellow]")
                return 1
            repo = AssetRepository(session, blobs=container.blob_store)
            counts = await repo.count(paper.id)
            figures = await repo.figures(paper.id, limit=limit) if "figures" in want else []
            tables = await repo.tables(paper.id, limit=limit) if "tables" in want else []
            equations = (
                await repo.equations(
                    paper.id, limit=limit, display_only=display_only or not json_output
                )
                if "equations" in want
                else []
            )

        if json_output:
            console.print_json(
                json.dumps(
                    {
                        "arxiv_id": paper.arxiv_id,
                        "title": paper.title,
                        "counts": counts,
                        "figures": [
                            {
                                "ordinal": f.ordinal,
                                "label": f.label,
                                "caption": f.caption,
                                "image_url": f.image_url,
                                "image_sha256": f.image_sha256,
                                "page_idx": f.page_idx,
                                "source": f.source,
                            }
                            for f in figures
                        ],
                        "tables": [
                            {
                                "ordinal": t.ordinal,
                                "label": t.label,
                                "caption": t.caption,
                                "rows": t.row_count,
                                "columns": t.column_count,
                                "has_html": t.body_html is not None,
                                "source": t.source,
                            }
                            for t in tables
                        ],
                        "equations": [
                            {
                                "ordinal": e.ordinal,
                                "latex": e.latex,
                                "is_display": e.is_display,
                                "page_idx": e.page_idx,
                                "source": e.source,
                            }
                            for e in equations
                        ],
                    }
                )
            )
            return 0

        console.print(
            f"[bold]{paper.title}[/bold]\n"
            f"[dim]{counts['figures']} figures · {counts['tables']} tables · "
            f"{counts['display_equations']} display equations · "
            f"{counts['equations'] - counts['display_equations']} inline[/dim]\n"
        )

        if figures:
            table = Table(box=None, pad_edge=False)
            table.add_column("#", style="dim", justify="right")
            table.add_column("label", style="cyan", no_wrap=True)
            table.add_column("image", style="dim", no_wrap=True)
            table.add_column("caption", style="")
            for figure in figures:
                table.add_row(
                    str(figure.ordinal),
                    figure.label or "-",
                    "blob" if figure.image_sha256 else (figure.image_url or "-"),
                    (figure.caption or "")[:70],
                )
            console.print(table)

        if tables:
            table = Table(box=None, pad_edge=False)
            table.add_column("#", style="dim", justify="right")
            table.add_column("label", style="cyan", no_wrap=True)
            table.add_column("shape", style="dim", no_wrap=True)
            table.add_column("caption", style="")
            for item in tables:
                shape = (
                    f"{item.row_count}x{item.column_count}"
                    if item.row_count
                    else ("image" if item.image_sha256 else "-")
                )
                table.add_row(
                    str(item.ordinal),
                    item.label or "-",
                    shape,
                    (item.caption or "")[:70],
                )
            console.print(table)

        if equations:
            console.print("[bold]equations[/bold] [dim](LaTeX)[/dim]")
            for equation in equations:
                console.print(
                    f"  [dim]{equation.ordinal:>3}[/dim] {equation.preview(96)}"
                )

        if not (figures or tables or equations):
            console.print(
                "[yellow]no assets recorded[/yellow] — figures, tables and equations "
                "are extracted during ingest; re-run with --force"
            )
        return 0

    raise typer.Exit(asyncio.run(run()))


# ---------------------------------------------------------------- references
@projects_app.command("refs")
def projects_refs(
    arxiv_id: Annotated[
        str,
        typer.Argument(help="ArXiv id, versioned id, URL, or a document's doc_key."),
    ],
    direction: Annotated[
        str, typer.Option("--direction", help="references (out) | cited-by (in)")
    ] = "references",
    limit: Annotated[int, typer.Option("--limit", "-n", min=1, max=200)] = 40,
    resolved_only: Annotated[bool, typer.Option("--in-corpus")] = False,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Show a paper's citation edges.

    `references` lists what the paper cites; `cited-by` lists which stored papers
    cite it. `--in-corpus` narrows to references resolved to a real paper.
    """
    incoming = direction in {"in", "cited-by", "cited_by", "citations"}

    async def run() -> int:
        from app.container import get_container
        from app.db.reference_repository import ReferenceRepository
        from app.db.repositories import PaperRepository

        container = get_container()
        async with container.session_factory() as session:
            papers = PaperRepository(session)
            paper = await papers.resolve(arxiv_id)
            if paper is None:
                console.print(f"[yellow]{arxiv_id} is not in the corpus[/yellow]")
                return 1
            refs = ReferenceRepository(session)
            if incoming:
                rows = await refs.list_citations(paper.id)
                citing = await refs.citing_papers(paper.id, limit=limit)
                data = [
                    {"cited_by": row.citing_paper_id, "ordinal": row.ordinal}
                    for row in rows
                ]
                # `display_id` so a local document in a project prints its slug
                # rather than `None`.
                # `display_id` so a local document in a project prints its slug
                # rather than `None`. Typed `dict[str, str]` because every field
                # here is display text, not an optional column.
                papers_out: list[dict[str, str]] = [
                    {
                        "display_id": p.display_id,
                        "arxiv_id": p.arxiv_id or "",
                        "title": p.title,
                    }
                    for p in citing
                ]
                del data
            else:
                rows = await refs.list_references(
                    paper.id, limit=limit, resolved_only=resolved_only
                )
                data = [
                    {
                        "ordinal": row.ordinal,
                        "cited_arxiv_id": row.cited_arxiv_id,
                        "cited_paper_id": row.cited_paper_id,
                        "resolved": row.is_resolved,
                        "authors": row.authors,
                        "title": row.title,
                        "year": row.year,
                        "venue": row.venue,
                        "raw_text": row.raw_text,
                    }
                    for row in rows
                ]
                papers_out = []

        if json_output:
            console.print_json(
                json.dumps(
                    {
                        "arxiv_id": arxiv_id,
                        "direction": "cited-by" if incoming else "references",
                        "count": len(rows),
                        "references" if not incoming else "citations": data,
                        **({"papers": papers_out} if incoming else {}),
                    }
                )
            )
            return 0

        header = (
            f"cited by {len(rows)} of our papers" if incoming else f"{len(rows)} references"
        )
        console.print(f"[bold]{paper.title}[/bold]\n[dim]{header}[/dim]\n")
        table = Table(box=None, pad_edge=False)
        if incoming:
            table.add_column("#", style="dim", justify="right")
            table.add_column("citing paper", style="bold")
        else:
            table.add_column("#", style="dim", justify="right")
            table.add_column("in corpus", style="green", no_wrap=True)
            table.add_column("year", style="dim", no_wrap=True)
            table.add_column("reference", style="")
        for row in rows:
            if incoming:
                continue
            table.add_row(
                str(row.ordinal),
                "yes" if row.is_resolved else (row.cited_arxiv_id or "-"),
                str(row.year or "-"),
                row.label(),
            )
        if rows:
            console.print(table)
        if incoming:
            for entry in papers_out:
                console.print(
                    f"  [cyan]{entry['display_id']}[/cyan] {entry['title'][:60]}"
                )
        if not rows:
            console.print(
                "[yellow]none recorded[/yellow] — references are extracted from the "
                "paper's HTML rendering during ingest; re-run with --force"
            )
        return 0

    raise typer.Exit(asyncio.run(run()))


async def _progress_console(message: str) -> None:
    """One-line progress for multi-item imports."""
    console.print(f"[dim]{message}...[/dim]")


# ------------------------------------------------------------------------- ask
@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="Natural-language question.")],
    top_k: Annotated[int, typer.Option("--top-k", "-k", min=1, max=50)] = 8,
    category: Annotated[str | None, typer.Option("--category", "-c")] = None,
    space: Annotated[str | None, typer.Option("--space", "-s")] = None,
    device: Annotated[
        str | None,
        typer.Option(
            "--device",
            help="Embedding device: cpu, cuda, cuda:1, mps. One query is faster "
            "on CPU; this is here for parity, not speed.",
        ),
    ] = None,
    section: Annotated[
        str | None,
        typer.Option(
            "--section",
            "-S",
            help="Only this section: a book's numbering (2.2) or words (Debye). "
            "Repeatable? No — all matches are included.",
        ),
    ] = None,
    within: Annotated[
        str | None,
        typer.Option(
            "--in",
            help="Restrict --section to one document (its doc_key). Without this, "
            "a section name is matched across the whole corpus.",
        ),
    ] = None,
    by_section: Annotated[
        bool,
        typer.Option(
            "--by-section",
            help="Group the hits into the sections and page ranges they fall in, "
            "instead of listing chunks. This is the 'which pages cover this?' view.",
        ),
    ] = False,
    show_text: Annotated[bool, typer.Option("--text")] = False,
    project: Annotated[
        str | None,
        typer.Option("--project", "-p", help="Limit to the papers of this project."),
    ] = None,
    content: Annotated[
        list[str] | None,
        typer.Option(
            "--content",
            "-C",
            help=(
                "Only these kinds of chunk: body, abstract, figure, table, "
                "equation, reference, code. Repeatable."
            ),
        ),
    ] = None,
    paper: Annotated[
        list[str] | None,
        typer.Option("--paper", help="Limit to these arXiv ids. Repeatable."),
    ] = None,
    source: Annotated[
        list[str] | None,
        typer.Option(
            "--source",
            help="Limit to chunks from this source: arxiv_html, ar5iv, pdf_mineru, "
            "pdf_pypdf, abstract_only. Repeatable.",
        ),
    ] = None,
    min_score: Annotated[
        float | None,
        typer.Option("--min-score", min=0.0, max=1.0, help="Drop hits below this cosine score."),
    ] = None,
) -> None:
    """Semantic search, scoped by project, paper, kind, source or section.

    `--section` narrows to a part of a document before ranking, so `top_k` is
    `top_k` matches *from that section* rather than the section's best matches out
    of a corpus-wide top-k. `--by-section` answers the other question: not "which
    chunks match" but "which pages of which sections cover this".
    """

    async def run() -> int:
        from app.container import get_container
        from app.services.sections_query import find_sections
        from app.services.semantic_search import SemanticSearchService

        try:
            kinds = _parse_kinds(content)
            srcs = _parse_sources(source)
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            return 2
        container = get_container()
        target = await _resolve(container, space)
        store = container.vector_store_for(target)
        service = SemanticSearchService(
            vector_store=store,
            embeddings=container.provider_for(target, _device(device)),
            session_factory=container.session_factory,
            space=target,
        )

        section_keys: list[tuple[str, int]] | None = None
        matched: list = []
        if section:
            async with container.session_factory() as session:
                matched = await find_sections(
                    session, section, doc_keys=[within] if within else None
                )
            if not matched:
                where = f" in {within!r}" if within else " in the corpus"
                console.print(
                    f"[yellow]no section matching {section!r}{where}.[/yellow] "
                    f"Try a word from its title, or its number (2.2). "
                    f"`paper outline {within or '<doc_key>'}` lists them."
                )
                return 1
            section_keys = [item.key for item in matched]

        try:
            hits = await service.search(
                question,
                top_k=top_k,
                category=category,
                project=project,
                content_kinds=kinds,
                paper_ids=list(paper) if paper else None,
                sources=srcs,
                min_score=min_score,
                sections=section_keys,
            )
        except LookupError as exc:
            # e.g. an unknown project slug. HTTP answers 404 and MCP a structured
            # error; a raw traceback out of the CLI would be the outlier.
            console.print(f"[red]{exc}[/red]")
            return 2
        console.print(f"[dim]space {target.name} | {target.model} | {target.dimensions}d[/dim]")
        scope = service.last_scope
        if scope.get("project"):
            console.print(
                f"[dim]project {scope['project']!r} | {scope['project_papers']} papers[/dim]"
            )
        if scope.get("content_kinds"):
            console.print(f"[dim]content {'+'.join(scope['content_kinds'])}[/dim]")
        if section_keys:
            console.print(
                "[dim]section:[/dim] "
                + ", ".join(f"{item.title} [dim](p{item.pages})[/dim]" for item in matched[:6])
                + (f" [dim]+{len(matched) - 6} more[/dim]" if len(matched) > 6 else "")
            )
        console.print()
        if not hits:
            console.print(
                f"[yellow]no matches in space {target.name!r}"
                f"{' — ingest some papers into it first' if await store.count() == 0 else ''}"
                "[/yellow]"
            )
            return 1
        if by_section:
            return _print_by_section(hits)
        for rank, hit in enumerate(hits, start=1):
            metadata = hit.metadata
            # `display_id` and not `arxiv_id`: a textbook has no arXiv id, and
            # printing an empty column beside every book result reads as missing
            # data rather than as "this is not a preprint".
            header = (
                f"[bold]{rank}. {metadata.get('title') or metadata.get('display_id', 'unknown')}[/bold] "
                f"[cyan]{metadata.get('display_id') or ''}[/cyan]"
                f"[dim]{'/' + metadata['kind'] if metadata.get('kind') and metadata['kind'] != 'paper' else ''}[/dim] "
                f"[magenta]score={hit.score:.4f}[/magenta]"
            )
            if metadata.get("content_kind"):
                header += f" [green]{metadata['content_kind']}[/green]"
            console.print(header)
            # Where in the source the passage is: the two things a reader needs to
            # open a book and find it, and the reason a document is worth having
            # in the corpus next to the papers.
            locator = _locator(metadata)
            if locator:
                console.print(f"  [dim]{locator}[/dim]")
            elif metadata.get("heading"):
                # Only as a fallback. The chunker's heading comes from the
                # extracted markdown, which in a book carries both a shallower
                # and a contradictory name for the same passage — showing it
                # beside the document's own section would put two different
                # claims on one line.
                console.print(f"  [dim]§ {metadata['heading']}[/dim]")
            if show_text:
                console.print(f"  {hit.text[:400]}…")
            console.print()
        return 0

    raise typer.Exit(asyncio.run(run()))


# ------------------------------------------------------------------------ show
@app.command()
def show(
    arxiv_id: Annotated[str, typer.Argument()],
    chunks: Annotated[int, typer.Option("--chunks", "-c", min=0, max=200)] = 3,
    content: Annotated[
        list[str] | None,
        typer.Option(
            "--content",
            "-C",
            help="Only these chunk kinds: body, abstract, figure, table, equation, "
            "reference, code. Repeatable.",
        ),
    ] = None,
) -> None:
    """Show what is stored locally for a paper."""

    async def run() -> int:
        from app.db.repositories import ChunkRepository, PaperRepository, SectionRepository
        from app.db.session import get_session_factory

        try:
            kinds = _parse_kinds(content)
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            return 2

        factory = get_session_factory()
        async with factory() as session:
            paper = await PaperRepository(session).resolve(arxiv_id)
            if paper is None:
                console.print(f"[yellow]{arxiv_id} is not ingested[/yellow]")
                return 1
            chunks_repo = ChunkRepository(session)
            # `limit=chunks or None`: `-c 0` means "print the header only".
            rows = await chunks_repo.list_for_paper(
                paper.id, limit=chunks or None, content_kinds=kinds
            )
            total = await chunks_repo.count(paper.id, content_kinds=kinds)
            sections = await SectionRepository(session).list_for_paper(paper.id)
            matching = (
                f" of {await chunks_repo.count(paper.id)} matching {'+'.join(kinds)}"
                if kinds
                else ""
            )
            section_row = f"sections  : {len(sections)}\n" if sections else ""
            body = (
                f"[bold]{paper.title}[/bold]\n\n"
                f"{_identity_block(paper)}"
                f"ingested  : {paper.ingested_at}\n"
                f"chunks    : {total}{matching}\n"
                f"{section_row}"
            )
            console.print(
                Panel(body, title=paper.kind if paper.kind != "paper" else "paper",
                      border_style="cyan")
            )
            for row in rows:
                console.print(
                    f"\n[cyan]#{row.ordinal}[/cyan] [dim]{row.heading or ''}[/dim]"
                    f" [green]{row.content_kind}[/green]"
                )
                console.print(row.text[:500])
            if kinds and not rows:
                console.print(f"\n[yellow]no {', '.join(kinds)} chunks in this paper[/yellow]")
        return 0

    raise typer.Exit(asyncio.run(run()))


@app.command("runs")
@writes_when("runs", "apply")
def runs(
    reap: Annotated[
        bool,
        typer.Option(
            "--reap",
            help="Mark runs whose process is gone as abandoned. Reports first.",
        ),
    ] = False,
    hours: Annotated[
        float,
        typer.Option("--hours", help="How long a run may be untouched before it is stale."),
    ] = 6.0,
    apply: Annotated[
        bool,
        typer.Option("--apply", help="With --reap, actually change the rows."),
    ] = False,
) -> None:
    """Find ingestion runs whose process died, and close them out.

    A run is marked `running` when it starts. If the process that owned it dies —
    a closed laptop, a killed shell, a dropped connection — nothing moves it on,
    and the row keeps claiming to be in progress forever. Measured on this corpus
    before the check existed: 38 such runs, the oldest over a day, two of them
    duplicate attempts at the same paper.

    Reports first. `--reap` alone lists what it found; add `--apply` to change
    anything. Reading state should not mutate it, and a reaped run's row is the
    record of what actually happened.

    Re-running a target then resumes it: the downloaded blob and the extracted
    markdown are reused, so only the cheap steps are redone.
    """

    async def run() -> int:
        from app.db.session import get_session_factory
        from app.services.runs import reap_stale_runs

        if not reap:
            console.print(
                "[dim]nothing to do — pass --reap to look for abandoned runs[/dim]"
            )
            return 0

        factory = get_session_factory()
        async with factory() as session:
            out = await reap_stale_runs(
                session, older_than_hours=hours, dry_run=not apply
            )

        # Narrowed once, here, so the formatting below is type-checked rather than
        # asserted away at every use.
        targets = cast("list[dict[str, Any]]", out["targets"])
        found = cast("int", out["found"])
        if not found:
            console.print(
                f"[green]no abandoned runs[/green] — nothing untouched for {hours:g}h"
            )
            return 0

        table = Table(box=None, pad_edge=False)
        for column, style in (
            ("target", "bold"),
            ("stuck", "yellow"),
            ("age", "red"),
            ("steps", "dim"),
        ):
            table.add_column(column, style=style)
        for row in targets:
            table.add_row(
                str(row["target"]),
                str(row["stuck_runs"]),
                f"{row['age_hours'] or 0.0:.1f}h",
                str(row["steps_completed"]),
            )
        console.print(table)

        if not apply:
            console.print(
                f"\n[yellow]{found} run(s) look abandoned.[/yellow] "
                f"Nothing changed. Re-run with [bold]--apply[/bold] to close them, "
                "then [bold]paper ingest <target>[/bold] to resume."
            )
            return 0
        console.print(
            f"[green]closed {cast('int', out['reaped'])} run(s)[/green] as abandoned"
        )
        for row in targets:
            console.print(f"  resume with: [bold]paper ingest {row['target']}[/bold]")
        return 0

    raise typer.Exit(asyncio.run(run()))


@app.command("sections")
@writes_when("sections", "apply")
def sections_cmd(
    prune: Annotated[
        bool,
        typer.Option("--prune", help="Remove sections that have no page to point at."),
    ] = False,
    show: Annotated[
        bool,
        typer.Option("--show", help="List the documents that would lose sections."),
    ] = False,
    apply: Annotated[
        bool, typer.Option("--apply", help="With --prune, actually delete them.")
    ] = False,
) -> None:
    """Remove sections that cannot be navigated to.

    A section exists so a reader can be told where to read. One with no page cannot
    do that, and in practice it is not a section at all: it is a heading the
    *renderer* produced. Measured here — an arXiv paper ingested from ar5iv
    produced fourteen "sections" that were the abs page's furniture (`Submission
    history`, `Access Paper:`, `BibTeX formatted citation`, `Demos`,
    `arXivLabs: experimental projects`), and across 177 papers those outweighed the
    257 genuine ones sixteen to one.

    Reports first: `--prune` alone tells you what would go, `--show` names the
    documents, and `--apply` is what changes anything. Chunks are never deleted —
    only the navigation label they carried is removed.
    """

    async def run() -> int:
        from app.db.session import get_session_factory
        from app.services.section_prune import (
            documents_with_unplaceable_sections,
            prune_unplaceable_sections,
        )

        factory = get_session_factory()
        async with factory() as session:
            if show or not prune:
                affected = await documents_with_unplaceable_sections(session)
                if not affected:
                    console.print("[green]every section has a page[/green]")
                    return 0
                table = Table(box=None, pad_edge=False)
                for column, style in (("document", "bold"), ("kind", "dim"), ("junk", "red")):
                    table.add_column(column, style=style)
                for row in affected[:25]:
                    table.add_row(row["doc_key"], str(row["kind"]), str(row["sections"]))
                console.print(table)
                if len(affected) > 25:
                    console.print(f"[dim]… and {len(affected) - 25} more documents[/dim]")
                console.print(
                    f"[dim]{sum(int(r['sections']) for r in affected)} sections across "
                    f"{len(affected)} documents have no page[/dim]"
                )
                if not prune:
                    return 0
            report = await prune_unplaceable_sections(session, apply=apply)
        if report.applied:
            console.print(
                f"[green]removed {report.sections} unplaceable section(s) across "
                f"{report.papers} document(s);[/green] {report.chunks_unlinked} chunk "
                f"lost a navigation label and {report.kept} section(s) kept."
            )
        else:
            console.print(
                f"[yellow]{report.sections} section(s) across {report.papers} document(s) "
                f"have no page.[/yellow] Nothing changed. Re-run with --apply to remove "
                f"them."
            )
        return 0

    raise typer.Exit(asyncio.run(run()))


@app.command("section")
def section(
    doc_key: Annotated[
        str, typer.Argument(help="Document: an arXiv id or a local file's doc_key.")
    ],
    which: Annotated[
        str,
        typer.Argument(
            metavar="SECTION",
            help="A book's numbering (2.2) or words from its title (Debye).",
        ),
    ],
    level: Annotated[
        int | None, typer.Option("--level", "-L", help="Accept sections this deep or shallower.")
    ] = None,
    include_subsections: Annotated[
        bool,
        typer.Option(
            "--with-subsections",
            help="Also print the sections nested inside the one matched.",
        ),
    ] = False,
) -> None:
    """Print the text of one section of a document.

    The other half of `paper ask --section`: that one finds passages by meaning
    inside a section, this one gives you the section itself, in order, with the
    page each chunk came from.

    Section content is the chunks that point at the section, not its page range.
    A section can span ten pages of which four produced text, and printing the
    range would promise material the corpus does not hold — so the page is shown
    per chunk, where it is a fact.
    """

    async def run() -> int:
        from app.db.repositories import PaperRepository, SectionRepository
        from app.db.session import get_session_factory
        from app.services.sections_query import find_sections

        factory = get_session_factory()
        async with factory() as session:
            paper = await PaperRepository(session).resolve(doc_key)
            if paper is None:
                console.print(f"[yellow]{doc_key} is not ingested[/yellow]")
                return 1
            found = await find_sections(
                session, which, doc_keys=[paper.doc_key], max_level=level
            )
            if not found:
                console.print(
                    f"[yellow]no section matching {which!r} in {paper.doc_key!r}.[/yellow] "
                    f"Run [bold]paper outline {paper.doc_key}[/bold] to list them."
                )
                return 1

            sections = await SectionRepository(session).list_for_paper(paper.id)
            by_ordinal = {row.ordinal: row for row in sections}
            for match in found:
                await _print_one_section(session, paper, match, by_ordinal, include_subsections)
        return 0

    raise typer.Exit(asyncio.run(run()))


async def _print_one_section(session, paper, match, by_ordinal, nested: bool) -> None:  # noqa: ANN001
    """One section's heading, then its text."""
    from app.db.repositories import SectionRepository  # noqa: PLC0415

    indent = "  " * max(0, match.level - 1)
    console.print(
        f"[bold cyan]{indent}{match.title}[/bold cyan] "
        f"[dim]p{match.pages} · section {match.ordinal} · from {match.source}[/dim]"
    )

    chunks = await SectionRepository(session).list_for_section(paper.id, match.ordinal)
    if not chunks:
        console.print(
            f"[dim]  no text stored for this section"
            f"{f' — its pages ({match.pages}) were never indexed' if match.pages != '-' else ''}[/dim]\n"
        )
    for chunk in chunks:
        page = f"p{chunk.page_start}" if chunk.page_start else "p?"
        console.print(f"  [dim]{page}[/dim] [green]{chunk.content_kind}[/green]")
        console.print(chunk.text.rstrip())
        console.print()

    if not nested:
        return
    from app.services.sections_query import sections_within  # noqa: PLC0415

    for row in sections_within(match, by_ordinal.values()):
        console.print(
            f"[dim]{'  ' * row.level}{row.title} · p{row.page_start or '?'}"
            f" · {row.source}[/dim]"
        )


@app.command("outline")
def outline(
    arxiv_id: Annotated[
        str,
        typer.Argument(help="ArXiv id, versioned id, URL, or a document's doc_key."),
    ],
    max_level: Annotated[
        int | None,
        typer.Option("--level", "-L", help="Only sections this deep or shallower."),
    ] = None,
    after: Annotated[
        int,
        typer.Option("--after", help="Start after this page. For a partial ingest."),
    ] = 0,
) -> None:
    """Print a document's table of contents with page ranges.

    Two sources, both kept: the PDF's own bookmarks when the book has them, and
    the headings the extraction found when it does not. Measured on three
    textbooks: 341 bookmark entries, 174, and **0** — so the ``src`` column is
    worth reading, because a book's outline may be half recovered from its text.

    Sections come from the source document, not from what was extracted, so a
    partial ingest still lists the whole book. ``--after`` narrows to the pages
    that were actually indexed.
    """

    async def run() -> int:
        from app.db.repositories import PaperRepository, SectionRepository
        from app.db.session import get_session_factory

        factory = get_session_factory()
        async with factory() as session:
            paper = await PaperRepository(session).resolve(arxiv_id)
            if paper is None:
                console.print(f"[yellow]{arxiv_id} is not ingested[/yellow]")
                return 1
            rows = await SectionRepository(session).list_for_paper(
                paper.id, max_level=max_level
            )
            if not rows:
                # The two reasons need saying separately. "No bookmarks" sent a
                # reader looking for a PDF that was never downloaded, when the real
                # answer is that this document was ingested from HTML and an HTML
                # page has no pages to point at.
                from_html = not any(doc.kind == "pdf" for doc in paper.contents)
                console.print(
                    "[yellow]no structure recorded[/yellow] — "
                    + (
                        "this document was ingested from an HTML rendering, which "
                        "has no pages, so there is nothing to navigate to. "
                        "Ingest the PDF for a table of contents."
                        if from_html
                        else "this document has neither PDF bookmarks nor headings "
                        "the extraction could read."
                    )
                )
                return 1

            table = Table(box=None, pad_edge=False)
            for column, style in (
                ("", ""),
                ("pages", "cyan"),
                ("src", "dim"),
                ("chunks", "dim"),
            ):
                table.add_column(column, style=style)

            # Counting chunks per section needs the chunk rows, not the section
            # rows: the link is on the chunk, and a section with no chunk is
            # either outside what was indexed or genuinely empty.
            counts = await _chunks_per_section(session, paper.id)
            for row in rows:
                if after and (row.page_start is None or row.page_start <= after):
                    continue
                if row.page_start is None and row.page_end is None:
                    pages = "-"
                elif row.page_end is None or row.page_end == row.page_start:
                    pages = str(row.page_start or "-")
                else:
                    pages = f"{row.page_start}-{row.page_end}"
                table.add_row(
                    f"{'  ' * (row.level - 1)}{row.title}",
                    pages,
                    "pdf" if row.source == "outline" else "text",
                    str(counts.get(row.ordinal, 0)) if counts else "",
                )
            console.print(f"[bold]{paper.title}[/bold] [dim]{paper.display_id}[/dim]")
            console.print(table)
            return 0

    raise typer.Exit(asyncio.run(run()))


async def _chunks_per_section(session: object, paper_id: str) -> dict[int, int]:
    """How many chunks each section holds, for the outline's last column."""
    from sqlalchemy import func, select  # noqa: PLC0415

    from app.db.models import Chunk  # noqa: PLC0415

    result = await session.execute(  # type: ignore[attr-defined]
        select(Chunk.section_ordinal, func.count())
        .where(Chunk.paper_id == paper_id, Chunk.section_ordinal.is_not(None))
        .group_by(Chunk.section_ordinal)
    )
    return {ordinal: count for ordinal, count in result.all()}


# ---------------------------------------------------------------------- spaces
@spaces_app.command("list")
def spaces_list() -> None:
    """List embedding spaces and how many vectors each holds."""

    async def run() -> int:
        from app.container import get_container
        from app.db.space_repository import EmbeddingSpaceRepository

        container = get_container()
        table = Table(box=None, pad_edge=False)
        for col, style in (
            ("", "bold"),
            ("model", ""),
            ("dims", "magenta"),
            ("dist", "dim"),
            ("vectors", "green"),
            ("table", "cyan"),
        ):
            table.add_column(col, style=style)

        async with container.session_factory() as session:
            repo = EmbeddingSpaceRepository(session)
            records = await repo.list()
            if not records:
                console.print("[yellow]no spaces registered[/yellow]")
            for record in records:
                space = record.as_space()
                count = 0
                with contextlib.suppress(Exception):
                    count = await container.vector_store_for(space).count()
                marker = "●" if record.is_active else " "
                table.add_row(
                    f"{marker} {record.name}",
                    f"{space.provider}/{space.model}",
                    str(space.dimensions),
                    space.distance,
                    f"{count:,}",
                    space.resolved_table,
                )
        console.print(table)
        console.print("[dim]● = active space (used when --space is omitted)[/dim]")
        return 0

    raise typer.Exit(asyncio.run(run()))


@spaces_app.command("add")
@writes("spaces:add")
def spaces_add(
    name: Annotated[str, typer.Argument(help="Space name, e.g. 'small-384'.")],
    model: Annotated[str, typer.Option("--model", "-m")],
    dimensions: Annotated[int, typer.Option("--dims", "-d", min=1)],
    provider: Annotated[str, typer.Option("--provider", "-p")] = "openai",
    distance: Annotated[str, typer.Option("--distance")] = "cosine",
    activate: Annotated[bool, typer.Option("--activate")] = False,
    description: Annotated[str | None, typer.Option("--description")] = None,
) -> None:
    """Register a new embedding model as its own space and table."""

    async def run() -> int:
        from app.container import get_container
        from app.db.space_repository import EmbeddingSpaceRepository, SpaceConflictError
        from app.db.spaces import EmbeddingSpace

        container = get_container()
        space = EmbeddingSpace(
            name=name,
            provider=provider,
            model=model,
            dimensions=dimensions,
            distance=distance,
            description=description,
        )
        async with container.session_factory() as session:
            repo = EmbeddingSpaceRepository(session)
            try:
                created = await repo.create(space, is_active=activate)
            except (ValueError, SpaceConflictError) as exc:
                console.print(f"[red]{exc}[/red]")
                return 1
            if activate:
                container.use_space(created)
            await session.commit()

        store = container.vector_store_for(created)
        if container.settings.vector.auto_create:
            await store.ensure_ready()
        console.print(
            f"[green]created[/green] space [bold]{created.name}[/bold] -> "
            f"{created.resolved_table} ({created.dimensions}d, {created.fingerprint})"
        )
        console.print(
            f"[dim]write vectors with:[/dim] paper ingest <id> --space {created.name}"
        )
        return 0

    raise typer.Exit(asyncio.run(run()))


@spaces_app.command("activate")
@writes("spaces:activate")
def spaces_activate(
    name: Annotated[str, typer.Argument()],
) -> None:
    """Make a space the default for new ingests and searches."""

    async def run() -> int:
        from app.container import get_container
        from app.db.space_repository import EmbeddingSpaceRepository, SpaceConflictError

        container = get_container()
        async with container.session_factory() as session:
            try:
                space = await EmbeddingSpaceRepository(session).activate(name)
                await session.commit()
            except SpaceConflictError as exc:
                console.print(f"[red]{exc}[/red]")
                return 1
        container.use_space(space)
        console.print(f"[green]active space:[/green] {space.name} ({space.fingerprint})")
        return 0

    raise typer.Exit(asyncio.run(run()))


@spaces_app.command("rm")
@writes("spaces:rm")
def spaces_rm(
    name: Annotated[str, typer.Argument()],
    drop_table: Annotated[
        bool, typer.Option("--drop-table", help="Also DROP the vector table (destructive).")
    ] = False,
) -> None:
    """Remove a space. Vectors are kept unless --drop-table is given."""

    async def run() -> int:
        from app.container import get_container
        from app.db.space_repository import EmbeddingSpaceRepository, SpaceConflictError

        container = get_container()
        async with container.session_factory() as session:
            repo = EmbeddingSpaceRepository(session)
            if await repo.get_record(name) is None:
                console.print(f"[red]unknown space {name!r}[/red]")
                return 1
            if drop_table:
                space = (await repo.get_record(name)).as_space()  # type: ignore[union-attr]
                await _drop_table(session, space)
                console.print(f"[yellow]dropped table {space.resolved_table}[/yellow]")
            try:
                await repo.delete(name)
                await session.commit()
            except SpaceConflictError as exc:
                console.print(f"[red]{exc}[/red]")
                return 1
        console.print(f"[green]removed[/green] space {name}")
        return 0

    raise typer.Exit(asyncio.run(run()))


@spaces_app.command("sql")
def spaces_sql(
    name: Annotated[str, typer.Argument(help="Space name (must already be registered).")],
    dialect: Annotated[str, typer.Option("--dialect")] = "postgresql",
) -> None:
    """Print the CREATE statements for a space.

    Use this when ``VECTOR_AUTO_CREATE=false``: apply the statements through a
    reviewed Alembic migration instead of at runtime.
    """
    from app.container import get_container
    from app.db.space_repository import EmbeddingSpaceRepository
    from app.db.vector_store.schema import ddl_statements

    async def run() -> int:
        container = get_container()
        async with container.session_factory() as session:
            record = await EmbeddingSpaceRepository(session).get_record(name)
        if record is None:
            console.print(
                f"[red]unknown space {name!r} — register it with `paper spaces add`[/red]"
            )
            return 1
        for statement in ddl_statements(record.as_space(), dialect):
            console.print(Syntax(statement, "sql", theme="ansi_dark"))
        return 0

    raise typer.Exit(asyncio.run(run()))


# ---------------------------------------------------------------------- doctor
@app.command()
@writes("reembed")
def reembed(
    space: Annotated[str, typer.Option("--space", "-s", help="Target space.")] = "default",
    project: Annotated[
        str | None, typer.Option("--project", "-p", help="Only this project's papers.")
    ] = None,
    paper: Annotated[
        list[str] | None, typer.Option("--paper", help="Only these arXiv ids. Repeatable.")
    ] = None,
    force: Annotated[
        bool, typer.Option("--force", help="Re-embed papers already present in the space.")
    ] = False,
    limit: Annotated[int | None, typer.Option("--limit", "-n", min=1)] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="List the papers, embed nothing.")
    ] = False,
    device: Annotated[
        str | None,
        typer.Option(
            "--device",
            help="Embedding device: cpu, cuda, cuda:1, mps. Worth setting here: a "
            "whole space is thousands of chunks.",
        ),
    ] = None,
) -> None:
    """Embed already-ingested chunks into another space. No downloads, no parsing."""

    async def run() -> int:
        from app.container import get_container
        from app.db.space_repository import EmbeddingSpaceRepository
        from app.services.reembed import ReembedService

        container = get_container()
        async with container.session_factory() as session:
            target = await EmbeddingSpaceRepository(session).resolve(space)

        service = ReembedService(
            provider=container.provider_for(target, _device(device)),
            space=target,
            session_factory=container.session_factory,
            store=container.vector_store_for(target),
            on_progress=_reembed_progress,
        )
        if dry_run:
            targets = await service.preview(
                project=project, arxiv_ids=list(paper) if paper else None, limit=limit
            )
            console.print(f"[dim]{len(targets)} papers in space {target.name!r}:[/dim]")
            for arxiv_id in targets:
                console.print(f"  {arxiv_id}")
            return 0

        try:
            report = await service.reembed(
                project=project,
                arxiv_ids=list(paper) if paper else None,
                force=force,
                limit=limit,
            )
        except LookupError as exc:
            # An unknown paper or project means the command was asked for the
            # wrong thing, not that it failed: exit 2 like the other validation
            # paths, so a script can tell "bad input" from "nothing to do".
            console.print(f"[red]{exc}[/red]")
            return 2
        console.print(f"[green]{report.summary()}[/green]")
        for failure in report.failures:
            console.print(f"[red]{failure}[/red]")
        return 1 if report.failures else 0

    raise typer.Exit(asyncio.run(run()))


@app.command("kinds")
@writes("kinds")
def kinds_command(
    all_chunks: Annotated[
        bool,
        typer.Option("--all", help="Re-label every chunk, not just unlabelled ones."),
    ] = False,
    report: Annotated[
        bool, typer.Option("--report", help="Only show the distribution; change nothing.")
    ] = False,
) -> None:
    """Classify stored chunks (body/abstract/figure/table/equation/reference/code)."""

    async def run() -> int:
        from app.container import get_container

        container = get_container()
        if report:
            from sqlalchemy import func, select  # noqa: PLC0415

            from app.db.models import Chunk  # noqa: PLC0415

            async with container.session_factory() as session:
                rows = (
                    await session.execute(
                        select(Chunk.content_kind, func.count())
                        .group_by(Chunk.content_kind)
                        .order_by(func.count().desc())
                    )
                ).all()
            table = Table(box=None, pad_edge=False)
            table.add_column("kind", style="bold")
            table.add_column("chunks", justify="right")
            for kind, count in rows:
                table.add_row(kind, str(count))
            console.print(table)
            return 0

        from app.services.kind_backfill import backfill_chunk_kinds  # noqa: PLC0415

        async with container.session_factory() as session:
            counts = await backfill_chunk_kinds(session, only_body=not all_chunks)
            await session.commit()
        total = sum(counts.values())
        console.print(f"[green]labelled {total} chunks[/green]")
        for kind, count in sorted(counts.items(), key=lambda kv: -kv[1]):
            console.print(f"  {kind:<10} {count}")
        return 0

    raise typer.Exit(asyncio.run(run()))


def _reembed_progress(done: int, total: int, arxiv_id: str, chunks: int) -> None:
    """One updating line. `chunks=0` means the paper was already embedded."""
    note = f"{chunks} chunks" if chunks else "present"
    console.print(
        f"[dim]{done}/{total}[/dim] {arxiv_id} [dim]({note})[/dim]",
        highlight=False,
        soft_wrap=True,
    )


@app.command()
def doctor() -> None:
    """Check the environment: DB, vector store, MinerU, embeddings, spaces."""

    async def run() -> int:
        from app.clients.content.mineru_resolver import installed_version
        from app.container import get_container
        from app.db.session import check_connection
        from app.db.space_repository import EmbeddingSpaceRepository

        settings = reload_settings()
        container = get_container(settings)

        table = Table(box=None, pad_edge=False)
        table.add_column("check", style="bold")
        table.add_column("status")
        table.add_column("detail", style="dim")

        db_ok = await check_connection(settings.database)
        table.add_row("database", _mark(db_ok), settings.database.url.split("@")[-1])

        try:
            provider = container.embeddings
            table.add_row(
                "embeddings",
                _mark(True),
                f"{provider.name} / {provider.model} / {provider.dimensions}d",
            )
        except Exception as exc:  # noqa: BLE001
            table.add_row("embeddings", _mark(False), str(exc))

        active = container.active_space
        try:
            store = container.vector_store_for(active)
            await store.ensure_ready()
            table.add_row(
                "vector store",
                _mark(True),
                f"{store.name} ({await store.count():,} vectors) in {active.resolved_table}",
            )
        except Exception as exc:  # noqa: BLE001
            table.add_row("vector store", _mark(False), str(exc))

        async with container.session_factory() as session:
            spaces = await EmbeddingSpaceRepository(session).list()
            table.add_row(
                "spaces",
                _mark(bool(spaces)),
                ", ".join(s.name for s in spaces) or "none registered",
            )

        extractor = container.mineru
        described = extractor.describe_backends()
        has_mineru = any(name in described for name in ("python_api", "cli"))
        table.add_row(
            "mineru",
            _mark(has_mineru),
            ", ".join(described.values())
            or "not installed — pip install 'paper-app-backend[mineru]'",
        )

        version = installed_version()
        if version:
            table.add_row("mineru version", _mark(True), f"{version} (python package)")

        console.print(table)

        if not has_mineru:
            console.print(
                "\n[yellow]MinerU is unavailable: PDFs will fall back to "
                "abstract-only text extraction.[/yellow]\n"
                "  pip install 'paper-app-backend[mineru]'   # in-process API\n"
                "  MINERU_BACKEND_ORDER=cli                 # use a `mineru` binary\n"
                "  pip install 'paper-app-backend[fallback]' # crude pypdf text layer"
            )
        return 0 if db_ok else 1

    raise typer.Exit(asyncio.run(run()))


# ------------------------------------------------------------------------ misc
@app.command()
def config() -> None:
    """Print the effective (secret-free) configuration."""
    settings = reload_settings()
    console.print_json(json.dumps(settings.model_dump(mode="json"), default=str))


@app.command()
def mcp(
    http: Annotated[
        bool, typer.Option("--http", help="Serve streamable HTTP instead of stdio.")
    ] = False,
    host: Annotated[str, typer.Option("--host")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port")] = 8080,
) -> None:
    """Run the Model Context Protocol server.

    stdio (default) for local clients, --http for remote ones.

    Claude Desktop / Claude Code config:

        {"mcpServers": {"arxiv-assistant":
          {"command": "paper", "args": ["mcp"]}}}
    """
    try:
        from app.mcp.server import run as run_mcp  # noqa: PLC0415
    except ImportError as exc:
        console.print(
            f"[red]MCP support is not installed:[/red] {exc}\n"
            "  pip install 'paper-app-backend[mcp]'"
        )
        raise typer.Exit(1) from exc

    _bootstrap()
    run_mcp(transport="http" if http else "stdio", host=host, port=port)


@app.command()
def serve(
    host: Annotated[str | None, typer.Option("--host")] = None,
    port: Annotated[int | None, typer.Option("--port")] = None,
    reload: Annotated[bool, typer.Option("--reload")] = False,
) -> None:
    """Run the HTTP API."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=host or settings.api.host,
        port=port or settings.api.port,
        reload=reload or settings.api.reload,
        log_config=None,
    )


@app.command()
def version() -> None:
    console.print(f"paper-app-backend {__version__}")


@db_app.command("upgrade")
@writes("db:upgrade")
def db_upgrade(revision: Annotated[str, typer.Argument()] = "head") -> None:
    """Apply Alembic migrations."""
    from alembic import command
    from alembic.config import Config

    root = Path(__file__).resolve().parent.parent
    config = Config(str(root / "alembic.ini"))
    config.set_main_option("script_location", str(root / "migrations"))
    config.set_main_option("sqlalchemy.url", get_settings().database.sync_url or "")
    command.upgrade(config, revision)


@db_app.command("downgrade")
@writes("db:downgrade")
def db_downgrade(revision: Annotated[str, typer.Argument()] = "-1") -> None:
    from alembic import command
    from alembic.config import Config

    root = Path(__file__).resolve().parent.parent
    config = Config(str(root / "alembic.ini"))
    config.set_main_option("script_location", str(root / "migrations"))
    config.set_main_option("sqlalchemy.url", get_settings().database.sync_url or "")
    command.downgrade(config, revision)


@db_app.command("revision")
def db_revision(message: Annotated[str, typer.Option("-m")] = "auto") -> None:
    from alembic import command
    from alembic.config import Config

    root = Path(__file__).resolve().parent.parent
    config = Config(str(root / "alembic.ini"))
    config.set_main_option("script_location", str(root / "migrations"))
    command.revision(config, message=message, autogenerate=True)


# ---------------------------------------------------------------------- private
def _parse_kinds(values: list[str] | None) -> list[str] | None:
    """Validate ``--content`` values. Raises ValueError on an unknown kind."""
    from app.services.chunk_kinds import parse_kinds

    return parse_kinds(values)


async def _assets_across_project(
    container,  # noqa: ANN001
    project: str,
    want: set[str],
    limit: int,
    display_only: bool,
) -> int:
    """List assets across every paper a project holds."""
    from app.db.asset_repository import AssetRepository
    from app.db.project_repository import PaperNotFoundError, ProjectRepository

    async with container.session_factory() as session:
        repo = ProjectRepository(session)
        try:
            found = await repo.require(project)
        except PaperNotFoundError:
            console.print(f"[red]no project matching {project!r}[/red]")
            return 2
        papers = await repo.list_papers(found, limit=limit)
        asset_repo = AssetRepository(session, blobs=container.blob_store)
        rows: list[tuple[str, dict[str, int]]] = []
        for paper, _link in papers:
            counts = await asset_repo.count(paper.id)
            if counts["figures"] or counts["tables"] or counts["display_equations"]:
                rows.append((paper.display_id, counts))

    console.print(
        f"[bold]{project}[/bold] [dim]· {len(rows)} papers with assets[/dim]\n"
    )
    if not rows:
        console.print("[yellow]no figures, tables or equations in this project[/yellow]")
        return 1
    table = Table(box=None, pad_edge=False)
    table.add_column("arxiv_id", style="cyan", no_wrap=True)
    table.add_column("figures", justify="right")
    table.add_column("tables", justify="right")
    table.add_column("display eq", justify="right")
    table.add_column("inline", justify="right")
    for arxiv_id, counts in rows:
        table.add_row(
            arxiv_id,
            str(counts["figures"]),
            str(counts["tables"]),
            str(counts["display_equations"]),
            str(counts["equations"] - counts["display_equations"]),
        )
    console.print(table)
    console.print(
        f"\n[dim]showing {len(rows)} of the project's papers; "
        "pass a paper id to `paper assets` for captions and image URLs.[/dim]"
    )
    return 0


def _parse_sources(values: list[str] | None) -> list[str] | None:
    """Validate ``--source`` values. Raises ValueError on an unknown source."""
    from app.services.semantic_search import parse_sources

    return parse_sources(values)


def _device(value: str | None) -> str | None:
    """Validate ``--device``, or fail on the spot.

    The rule is :func:`app.domain.devices.parse_device`, the same one the HTTP API
    and the MCP tools use, wrapped only so the error arrives as a usage message
    rather than a traceback. Kept in one place because a CLI that accepts a device
    string the API rejects is a difference a caller only finds out about by
    watching one of them work.
    """
    from app.domain.devices import DeviceError, parse_device  # noqa: PLC0415

    try:
        return parse_device(value)
    except DeviceError as exc:
        raise typer.BadParameter(str(exc)) from exc


async def _resolve(container, name: str | None):  # noqa: ANN001
    from app.db.space_repository import EmbeddingSpaceRepository

    if name is None:
        return container.active_space
    async with container.session_factory() as session:
        return await EmbeddingSpaceRepository(session).resolve(name)


async def _drop_table(session, space) -> None:  # noqa: ANN001
    from sqlalchemy import text as sa_text

    table = space.resolved_table
    # Identifier is validated by slugify() in EmbeddingSpace.
    await session.execute(sa_text(f'DROP TABLE IF EXISTS "{table}"'))
    await session.execute(sa_text(f'DROP INDEX IF EXISTS "ix_{table}_hnsw"'))
    await session.execute(sa_text(f'DROP INDEX IF EXISTS "uq_{table}_chunk"'))


def _mark(ok: bool) -> str:
    return "[green]ok[/green]" if ok else "[red]fail[/red]"


def _render_runs(result) -> None:  # noqa: ANN001
    if result.run_ids:
        table = Table(title="Ingestion", box=None, pad_edge=False)
        table.add_column("run", style="dim")
        table.add_column("status")
        table.add_column("error", style="red", overflow="fold")
        for run_id in result.run_ids:
            status_value = result.statuses.get(run_id)
            status_text = status_value.value if status_value else "pending"
            table.add_row(run_id[:8], status_text, result.errors.get(run_id, ""))
        console.print(table)
    else:
        console.print("[yellow]no runs created[/yellow]")


def main() -> None:  # pragma: no cover
    app()


# Imported last, and for its side effect: `session` is a Typer command on `app`,
# and it needs names from this module. Registering it here rather than above
# keeps the circular import one-directional.
from app import cli_session  # noqa: E402, F401  # isort: skip

if __name__ == "__main__":  # pragma: no cover
    sys.exit(app())