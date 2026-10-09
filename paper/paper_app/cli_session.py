"""An interactive session bound to one project.

`paper ask --project X` answers a question but throws the context away
afterwards: the next question needs the flag again, and there is no way to see
which papers are in scope or which are unread. A reading session is mostly
*sequencing* — read a bit, ask, read a bit — so the project is opened once and
stays in scope for everything that follows.

    paper session sheaf-neural-networks

Inside, the project is implicit:

    papers                    list the project, unread first
    ask <question>            search this project (same engine as `paper ask`)
    read <arxiv_id>           mark read / unread
    show <arxiv_id>           that paper's chunks
    assets                    figures/tables/equations across the project
    set <k=v>                 content filter for ask, e.g. set content=equations
    exit

Deliberately not a REPL over the corpus. Scoping is the whole reason this
exists, so nothing here silently widens to all papers — `paper ask` without
``--project`` is one command away for that.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Annotated, Any

import typer
from rich.panel import Panel
from rich.table import Table

from paper_app.cli import _print_project_papers, app, console
from paper_app.logging import get_logger

logger = get_logger(__name__)

BANNER_COMMANDS = (
    ("papers", "list the project's papers, unread first"),
    ("ask <question>", "semantic search within this project"),
    ("show <arxiv_id>", "chunks of one paper"),
    ("assets", "figures / tables / equations across the project"),
    ("read <arxiv_id>", "toggle read state"),
    ("set content=equations", "restrict ask to a content kind"),
    ("set -c equations", "same, short form"),
    ("help", "this list"),
    ("exit", "leave the session"),
)


@dataclass(slots=True)
class SessionState:
    """What stays in scope for the whole session."""

    project: str
    slug: str
    space_name: str
    content_kinds: list[str] | None = None
    top_k: int = 5
    show_text: bool = True
    asked: int = 0
    papers_seen: set[str] = field(default_factory=set)

    def scope_line(self) -> str:
        parts = [f"project {self.project!r}", self.space_name]
        if self.content_kinds:
            parts.append("content " + "+".join(self.content_kinds))
        parts.append(f"asked {self.asked}")
        return "  |  ".join(parts)


# ----------------------------------------------------------------- resolution
async def _open(project: str, space: str | None) -> SessionState | None:
    """Resolve the project and space, or print why not."""
    from paper_app.container import get_container
    from paper_app.db.project_repository import PaperNotFoundError, ProjectRepository

    container = get_container()
    async with container.session_factory() as session:
        try:
            found = await ProjectRepository(session).require(project)
        except PaperNotFoundError:
            console.print(f"[red]no project matching {project!r}[/red]")
            console.print("[dim]try `paper projects list`[/dim]")
            return None
        slug, name = found.slug, found.name
    try:
        target = await _resolve_space(container, space)
    except Exception as exc:  # noqa: BLE001 - SpaceConflictError
        console.print(f"[red]{exc}[/red]")
        return None
    return SessionState(
        project=name or slug,
        slug=slug,
        space_name=f"{target.name} ({target.model})",
    )


async def _resolve_space(container, name: str | None):  # noqa: ANN001, ANN202
    if name is None:
        return container.active_space
    from paper_app.db.space_repository import EmbeddingSpaceRepository

    async with container.session_factory() as session:
        return await EmbeddingSpaceRepository(session).resolve(name)


# ------------------------------------------------------------------- commands
async def _cmd_papers(state: SessionState, args: str) -> None:
    from paper_app.container import get_container
    from paper_app.db.project_repository import ProjectRepository

    unread_only = "--unread" in args
    limit = 30
    for token in args.split():
        if token.startswith("--limit="):
            limit = max(1, int(token.split("=", 1)[1]))

    container = get_container()
    async with container.session_factory() as session:
        rows = await ProjectRepository(session).list_papers(
            await ProjectRepository(session).require(state.slug),
            limit=limit,
            unread_only=unread_only,
        )
    if not rows:
        console.print(f"[yellow]no {'unread ' if unread_only else ''}papers in this project[/yellow]")
        return
    _print_project_papers(rows, title=f"{state.project} ({len(rows)} shown)")
    for paper, _link in rows:
        state.papers_seen.add(paper.display_id)


async def _cmd_ask(state: SessionState, question: str) -> None:
    from paper_app.container import get_container
    from paper_app.db.space_repository import EmbeddingSpaceRepository
    from paper_app.services.semantic_search import SemanticSearchService

    if not question.strip():
        console.print("[yellow]ask what?[/yellow]")
        return

    container = get_container()
    async with container.session_factory() as session:
        target = await EmbeddingSpaceRepository(session).resolve(state.space_name.split()[0])
    service = SemanticSearchService(
        vector_store=container.vector_store_for(target),
        embeddings=container.provider_for(target),
        session_factory=container.session_factory,
        space=target,
    )
    try:
        hits = await service.search(
            question,
            top_k=state.top_k,
            project=state.slug,
            content_kinds=state.content_kinds,
        )
    except LookupError as exc:
        console.print(f"[red]{exc}[/red]")
        return
    state.asked += 1

    if not hits:
        console.print("[yellow]no matches in this project[/yellow]")
        if state.content_kinds:
            console.print(
                f"[dim]content filter is {state.content_kinds}; "
                "`set content=` clears it[/dim]"
            )
        return
    for rank, hit in enumerate(hits, start=1):
        meta = hit.metadata
        console.print(
            f"[bold]{rank}. {meta.get('title', 'unknown')}[/bold] "
            f"[cyan]{meta.get('arxiv_id', '')}[/cyan] "
            f"[magenta]score={hit.score:.4f}[/magenta]"
            + (f" [green]{meta['content_kind']}[/green]" if meta.get("content_kind") else "")
        )
        if meta.get("heading"):
            console.print(f"  [dim]§ {meta['heading']}[/dim]")
        if state.show_text:
            console.print(f"  {hit.text[:600]}")
        state.papers_seen.add(str(meta.get("arxiv_id") or ""))
        console.print()


async def _cmd_show(state: SessionState, args: str) -> None:
    from paper_app.db.repositories import ChunkRepository, PaperRepository
    from paper_app.db.session import get_session_factory

    target = args.split()[0] if args.split() else ""
    if not target:
        console.print("[yellow]show which paper? try `papers`[/yellow]")
        return
    async with get_session_factory()() as session:
        paper = await PaperRepository(session).resolve(target)
        if paper is None:
            console.print(f"[red]{target} is not ingested[/red]")
            return
        rows = await ChunkRepository(session).list_for_paper(
            paper.id, limit=3, content_kinds=state.content_kinds
        )
        total = await ChunkRepository(session).count(paper.id)
    body = (
        f"[bold]{paper.title}[/bold]\n\n"
        f"id      : {paper.versioned_id}\n"
        f"chunks  : {total}"
        + (f" matching {'+'.join(state.content_kinds)}" if state.content_kinds else "")
        + f"\nauthors : {', '.join(paper.author_names[:5])}"
    )
    console.print(Panel(body, title="paper", border_style="cyan"))
    for row in rows:
        console.print(f"\n[cyan]#{row.ordinal}[/cyan] [dim]{row.heading or ''}[/dim] [green]{row.content_kind}[/green]")
        console.print(row.text[:400])


async def _cmd_assets(state: SessionState, args: str) -> None:
    from paper_app.cli import _assets_across_project

    want = {"figures", "tables", "equations"}
    for token in args.split():
        if token.startswith("--kind="):
            want = {"figures", "tables", "equations"} if token.split("=", 1)[1] == "all" else {token.split("=", 1)[1]}
    from paper_app.container import get_container

    code = await _assets_across_project(get_container(), state.slug, want, 200, False)
    if code == 1:
        return


async def _cmd_read(state: SessionState, args: str) -> None:
    """Toggle read state: one paper, or the whole project when no id is given."""
    from paper_app.db.project_repository import PaperNotFoundError, ProjectRepository
    from paper_app.db.session import get_session_factory

    async with get_session_factory()() as session:
        repo = ProjectRepository(session)
        try:
            project = await repo.require(state.slug)
        except PaperNotFoundError:
            console.print(f"[red]no project {state.slug!r}[/red]")
            return
        if args.split():
            resolved, unresolved = await repo.resolve_paper_ids(args.split())
            if unresolved:
                console.print(f"[red]not in this project: {', '.join(unresolved)}[/red]")
                return
            targets = resolved
        else:
            targets = await repo.paper_ids(project)
        if not targets:
            console.print("[yellow]nothing to mark[/yellow]")
            return

        # Read per row rather than blanket-setting: a mixed project toggles each
        # paper to its opposite instead of flattening to one value, which is
        # what "read" means when you are working through a list.
        current = {
            paper.id: link.is_read
            for paper, link in await repo.list_papers(project, limit=1000)
        }
        changed = 0
        for paper_id in targets:
            if await repo.set_read(project, paper_id, is_read=not current.get(paper_id, False)):
                changed += 1
        await session.commit()
    console.print(f"[green]{changed} paper(s) toggled[/green]")


async def _cmd_set(state: SessionState, args: str) -> None:
    """Change what later `ask` calls are scoped to."""
    from paper_app.services.chunk_kinds import parse_kinds

    if not args.strip():
        console.print(f"[dim]{state.scope_line()}[/dim]")
        return
    # Both `set content=equations` and `set -c equations` are accepted, so the
    # separator is not required. Splitting on "=" first and only then on
    # whitespace is what makes that work; doing it the other way round turns
    # `-c equations` into a key called "c equations".
    key, sep, value = args.partition("=")
    if not sep:
        parts = args.split()
        key, value = parts[0], " ".join(parts[1:])
    key = key.strip().lstrip("-").lower()

    if key in {"c", "content", "kind"}:
        if not value or value in {"-", "none", "all"}:
            state.content_kinds = None
            console.print("[green]content filter cleared[/green]")
            return
        try:
            state.content_kinds = parse_kinds(value.replace(",", " ").split())
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            return
        console.print(f"[green]content = {'+'.join(state.content_kinds or [])}[/green]")
        return
    if key in {"k", "top_k", "top-k"}:
        try:
            state.top_k = max(1, min(50, int(value)))
        except ValueError:
            console.print(f"[red]top_k must be a number, got {value!r}[/red]")
            return
        console.print(f"[green]top_k = {state.top_k}[/green]")
        return
    if key in {"t", "text", "full"}:
        state.show_text = value.lower() not in {"0", "off", "false", "no"}
        console.print(f"[green]text = {state.show_text}[/green]")
        return
    console.print(f"[red]unknown setting {key!r}; try: set content=, set top_k=, set text=[/red]")


async def _cmd_help(_state: SessionState, _args: str) -> None:
    _cmd_help_table()


def _cmd_help_table() -> None:
    table = Table(box=None, pad_edge=False)
    table.add_column("command", style="bold")
    table.add_column("does", style="dim")
    for name, does in BANNER_COMMANDS:
        table.add_row(name, does)
    console.print(Panel(table, title="inside a session", border_style="cyan"))


COMMANDS: dict[str, Any] = {
    "papers": _cmd_papers,
    "ls": _cmd_papers,
    "ask": _cmd_ask,
    "show": _cmd_show,
    "assets": _cmd_assets,
    "read": _cmd_read,
    "set": _cmd_set,
    "help": _cmd_help,
    "?": _cmd_help,
}


# --------------------------------------------------------------------- command
@app.command()
def session(
    project: Annotated[
        str | None,
        typer.Argument(help="Project slug or name. Omit to pick from a list."),
    ] = None,
    space: Annotated[
        str | None, typer.Option("--space", "-s", help="Embedding space to search.")
    ] = None,
) -> None:
    """Open an interactive session scoped to one project."""

    async def run() -> int:
        from paper_app.container import get_container
        from paper_app.db.project_repository import ProjectRepository

        if project is None:
            container = get_container()
            async with container.session_factory() as session:
                rows = await ProjectRepository(session).list_projects()
            if not rows:
                console.print("[yellow]no projects yet — `paper projects new <name>`[/yellow]")
                return 1
            console.print("[bold]which project?[/bold]")
            for row in rows:
                console.print(f"  [cyan]{row.slug}[/cyan]  {row.name} [dim]({row.paper_count} papers)[/dim]")
            console.print("[dim]re-run as `paper session <slug>`[/dim]")
            return 1

        state = await _open(project, space)
        if state is None:
            return 2

        console.print(
            Panel(
                f"[bold]{state.project}[/bold]  [dim]{state.space_name}[/dim]\n"
                "[dim]ask questions without repeating --project. "
                "`help` lists commands, `exit` leaves.[/dim]",
                title="session",
                border_style="green",
            )
        )
        await _cmd_papers(state, "--limit=15")

        while True:
            try:
                line = console.input(f"\n[bold green]({state.project})[/] ").strip()
            except (EOFError, KeyboardInterrupt):
                console.print()
                return 0
            if not line:
                continue
            name, _, rest = line.partition(" ")
            if name in {"exit", "quit", "q"}:
                console.print(f"[dim]{state.asked} question(s) asked. bye.[/dim]")
                return 0
            handler = COMMANDS.get(name)
            if handler is None:
                console.print(f"[red]unknown command {name!r}[/red] — try [green]help[/green]")
                continue
            try:
                await handler(state, rest.strip())
            except Exception as exc:  # noqa: BLE001 - a session must not die
                # One bad command must not end a reading session. Print the
                # message only: Rich renders a full traceback inside the live
                # console, which buries the prompt and is unreadable anyway.
                # The detail is logged for anyone who wants it.
                logger.warning(
                    "session_command_failed",
                    extra={"command": name, "error": f"{type(exc).__name__}: {exc}"},
                )
                console.print(f"[red]{type(exc).__name__}:[/red] {exc}")

    raise typer.Exit(asyncio.run(run()))


__all__ = ["SessionState", "session"]
