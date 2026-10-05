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
import json
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from app import __version__
from app.config import get_settings, reload_settings
from app.logging import configure_logging, get_logger

app = typer.Typer(
    name="paper",
    help="ArXiv AI assistant: search, ingest, extract and retrieve.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()
logger = get_logger("paper.cli")

arxiv_app = typer.Typer(help="Search ArXiv metadata.", no_args_is_help=True)
db_app = typer.Typer(help="Database utilities.", no_args_is_help=True)
spaces_app = typer.Typer(help="Embedding spaces (one model, one table).", no_args_is_help=True)
projects_app = typer.Typer(help="Projects: named collections over the global corpus.", no_args_is_help=True)
app.add_typer(projects_app, name="projects")
app.add_typer(arxiv_app, name="arxiv")
app.add_typer(db_app, name="db")
app.add_typer(spaces_app, name="spaces")


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
def ingest(
    arxiv_id: Annotated[str, typer.Argument(help="ArXiv id, versioned id or URL.")],
    pdf: Annotated[bool, typer.Option("--pdf", help="Skip HTML and use the PDF path.")] = False,
    force: Annotated[bool, typer.Option("--force", help="Re-download even if cached.")] = False,
    space: Annotated[
        str | None,
        typer.Option("--space", "-s", help="Embedding space to write. Default: active space."),
    ] = None,
) -> None:
    """Run the full pipeline (fetch -> MinerU -> chunk -> embed) for one paper.

    Adding a paper to a second space does not touch the first one's vectors:
    each space has its own table.
    """

    async def run() -> int:
        from app.container import get_container
        from app.services.ingestion import build_ingestion_service

        container = get_container()
        target = await _resolve(container, space)
        await container.vector_store_for(target).ensure_ready()
        service = build_ingestion_service(container, target)
        try:
            result = await service.ingest_paper(
                arxiv_id, prefer_html=not pdf, force=force, requested_by="cli", trigger="cli"
            )
        finally:
            await container.aclose()

        console.print(f"[dim]space:[/dim] {target.name} ({target.resolved_table})")
        _render_runs(result)
        return 0 if result.succeeded else 1

    raise typer.Exit(asyncio.run(run()))


@app.command()
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
            targets = (
                (await projects.resolve_paper_ids([paper_id]))[0]
                if paper_id
                else await projects.paper_ids(found)
            )
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
    # get_by_arxiv_id normalises, so the versioned and bare forms both resolve.
    return await papers.get_by_arxiv_id(arxiv_id)


@app.command()
def assets(
    arxiv_id: Annotated[str, typer.Argument(help="Paper to inspect.")],
    kind: Annotated[
        str, typer.Option("--kind", "-k", help="figures | tables | equations | all")
    ] = "all",
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
    arxiv_id: Annotated[str, typer.Argument(help="Paper to inspect.")],
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
            paper = await papers.get_by_arxiv_id(arxiv_id)
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
                papers_out = [
                    {"arxiv_id": p.arxiv_id, "title": p.title} for p in citing
                ]
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
                console.print(f"  [cyan]{entry['arxiv_id']}[/cyan] {entry['title'][:60]}")
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
) -> None:
    """Semantic search, scoped by project, by paper and by content kind."""

    async def run() -> int:
        from app.container import get_container
        from app.services.semantic_search import SemanticSearchService

        try:
            kinds = _parse_kinds(content)
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            return 2
        container = get_container()
        target = await _resolve(container, space)
        store = container.vector_store_for(target)
        service = SemanticSearchService(
            vector_store=store,
            embeddings=container.provider_for(target),
            session_factory=container.session_factory,
            space=target,
        )
        hits = await service.search(
            question,
            top_k=top_k,
            category=category,
            project=project,
            content_kinds=kinds,
            paper_ids=list(paper) if paper else None,
        )
        console.print(f"[dim]space {target.name} | {target.model} | {target.dimensions}d[/dim]")
        scope = service.last_scope
        if scope.get("project"):
            console.print(
                f"[dim]project {scope['project']!r} | {scope['project_papers']} papers[/dim]"
            )
        if scope.get("content_kinds"):
            console.print(f"[dim]content {'+'.join(scope['content_kinds'])}[/dim]\n")
        else:
            console.print()
        if not hits:
            console.print(
                f"[yellow]no matches in space {target.name!r}"
                f"{' — ingest some papers into it first' if await store.count() == 0 else ''}"
                "[/yellow]"
            )
            return 1
        for rank, hit in enumerate(hits, start=1):
            metadata = hit.metadata
            header = (
                f"[bold]{rank}. {metadata.get('title', 'unknown')}[/bold] "
                f"[cyan]{metadata.get('arxiv_id', '')}[/cyan] "
                f"[magenta]score={hit.score:.4f}[/magenta]"
            )
            if metadata.get("content_kind"):
                header += f" [green]{metadata['content_kind']}[/green]"
            console.print(header)
            if metadata.get("heading"):
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
) -> None:
    """Show what is stored locally for a paper."""

    async def run() -> int:
        from app.db.repositories import ChunkRepository, PaperRepository
        from app.db.session import get_session_factory

        factory = get_session_factory()
        async with factory() as session:
            paper = await PaperRepository(session).get_by_arxiv_id(arxiv_id)
            if paper is None:
                console.print(f"[yellow]{arxiv_id} is not ingested[/yellow]")
                return 1
            rows = await ChunkRepository(session).list_for_paper(paper.id, limit=chunks or None)
            body = (
                f"[bold]{paper.title}[/bold]\n\n"
                f"id         : {paper.versioned_id}\n"
                f"authors    : {', '.join(paper.author_names[:6])}\n"
                f"categories : {', '.join(paper.categories)}\n"
                f"published  : {paper.published_at}\n"
                f"ingested   : {paper.ingested_at}\n"
                f"html       : {paper.html_url or '-'}\n"
                f"pdf        : {paper.pdf_url}"
            )
            console.print(Panel(body, title="paper", border_style="cyan"))
            for row in rows:
                console.print(f"\n[cyan]#{row.ordinal}[/cyan] [dim]{row.heading or ''}[/dim]")
                console.print(row.text[:500])
        return 0

    raise typer.Exit(asyncio.run(run()))


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
            provider=container.provider_for(target),
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
            console.print(f"[red]{exc}[/red]")
            return 1
        console.print(f"[green]{report.summary()}[/green]")
        for failure in report.failures:
            console.print(f"[red]{failure}[/red]")
        return 1 if report.failures else 0

    raise typer.Exit(asyncio.run(run()))


@app.command("kinds")
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


if __name__ == "__main__":  # pragma: no cover
    sys.exit(app())