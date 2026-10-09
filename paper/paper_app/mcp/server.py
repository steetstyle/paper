"""MCP (Model Context Protocol) server exposing the ArXiv assistant.

Runs **in-process** against the same :class:`~paper_app.container.Container` the CLI
and HTTP API use, so a desktop assistant needs no second process and shares one
database with them.

    paper mcp                       # stdio (Claude Desktop, Claude Code, …)
    paper mcp --http --port 8080     # streamable HTTP for remote clients

Design notes
------------
* Read-only tools are annotated as such (``read_only_hint``) so a client can
  auto-approve the cheap calls and prompt for ingestion.
* ``ingest_paper`` is marked ``idempotent_hint``: re-ingesting replaces that
  paper's vectors instead of duplicating them, so a client may retry it.
* Long operations report progress through ``ctx`` when a request is present and
  stay silent otherwise, so the same function works over the wire and in tests.
* Results are compact and truncated. Dumping a whole paper into a context window
  is the fastest way to waste a model's attention, so every payload points at a
  follow-up tool for more.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

# The context injection slot is `mcpserver.context.Context`. The
# `mcp.server.context` re-export is a *different* class: importing that one makes
# the SDK treat the parameter as an ordinary tool argument and schema generation
# fails.
from mcp.server.mcpserver.context import Context

from paper_app import __version__
from paper_app.clients.arxiv.exceptions import ArxivError
from paper_app.config import get_settings
from paper_app.container import Container, get_container
from paper_app.db.models import Chunk, RawDocument
from paper_app.db.space_repository import SpaceConflictError
from paper_app.db.spaces import EmbeddingSpace
from paper_app.domain.models import SearchQuery
from paper_app.logging import get_logger
from paper_app.mcp.devices import DEVICE_HELP, parse_device
from paper_app.services.chunk_kinds import parse_kinds
from paper_app.services.ingestion import build_ingestion_service
from paper_app.services.semantic_search import SemanticSearchService, parse_sources

logger = get_logger(__name__)

INSTRUCTIONS = """\
ArXiv research assistant.

Typical flow:
  1. `search_arxiv` to find papers (metadata only, fast, no download).
  2. `ingest_paper` on the promising ids - downloads the PDF or HTML, runs
     MinerU, chunks and embeds. Slow (minutes on CPU for a PDF) and the only
     write operation.
  3. `ask_paper_corpus` to search everything already ingested, by meaning.
  4. `read_chunks` / `read_markdown` to pull the text that actually matters.

Prefer `ask_paper_corpus` over re-ingesting: ingested papers are searchable
instantly, while ingestion re-downloads and re-embeds.

Corpus search uses an *embedding space* (one model = one table). Call
`list_embedding_spaces` to see what is available; omit `space` for the active one.
"""

# Tool annotations: cheap calls are read-only, ingestion is a write but safe to
# retry because re-ingesting replaces rather than duplicates.
#
# `open_world_hint` is the one that matters most and the easiest to get wrong. MCP
# clients use it to decide whether a call can run without asking the person: false
# means "touches nothing outside this process", which is a licence to auto-approve.
# Every tool that reaches arxiv.org therefore says true, whether it writes
# (`ingest_paper`) or only reads (`search_arxiv`). Claiming otherwise is not a
# cosmetic slip — it is a promise to the client about what will happen on the
# network, made on the user's behalf without asking them.
_READ_ONLY_REMOTE = {"read_only_hint": True, "open_world_hint": True, "destructive_hint": False}
_READS_LOCAL = {"read_only_hint": True, "open_world_hint": False, "destructive_hint": False}
_WRITES_IDEMPOTENT = {
    "read_only_hint": False,
    "open_world_hint": False,
    "destructive_hint": False,
    "idempotent_hint": True,
}
#: Writes that also download from arxiv.org.
_WRITES_IDEMPOTENT_REMOTE = {
    "read_only_hint": False,
    "open_world_hint": True,
    "destructive_hint": False,
    "idempotent_hint": True,
}


# --------------------------------------------------------------------------- setup
@asynccontextmanager
async def _lifespan(server: Any) -> AsyncIterator[Container]:  # noqa: ANN401
    """Own the container's async resources for the server's lifetime."""
    container = get_container(get_settings())
    await container.startup()
    try:
        yield container
    finally:
        await container.aclose()


def _build_server() -> Any:  # noqa: ANN401
    from mcp.server.mcpserver import MCPServer  # noqa: PLC0415

    return MCPServer(
        name="arxiv-assistant",
        title="ArXiv Assistant",
        version=__version__,
        instructions=INSTRUCTIONS,
        lifespan=_lifespan,
    )


server = _build_server()


def _ann(**kwargs: Any):  # noqa: ANN202
    from mcp.types import ToolAnnotations  # noqa: PLC0415

    return ToolAnnotations(**kwargs)


# --------------------------------------------------------------------------- utils
def _container() -> Container:
    """The process container — the same object the lifespan yields.

    Tools deliberately avoid ``ctx.request_context``: that raises outside a
    request, and a tool should behave identically over the wire and when called
    directly (tests, resource handlers).
    """
    return get_container()


async def _progress(ctx: Context | None, fraction: float, message: str) -> None:
    """Report progress when a request is present; stay silent otherwise.

    ``ctx.log`` is deliberately not used: the SDK deprecated that capability
    (SEP-2577). Progress notifications still reach the client, and the server's
    own logger carries the detail.
    """
    if ctx is None:
        return
    with contextlib.suppress(Exception):
        await ctx.report_progress(fraction, 1.0, message)


def _fail(message: str) -> dict[str, Any]:
    """A structured error.

    A model cannot reason about a protocol-level failure, so the message has to
    be self-explanatory and say what to do next.
    """
    return {"ok": False, "error": message}


def _ok(**payload: Any) -> dict[str, Any]:
    return {"ok": True, **payload}


async def _resolve_space(container: Container, name: str | None) -> EmbeddingSpace:
    if name is None:
        return container.active_space
    from paper_app.db.space_repository import EmbeddingSpaceRepository  # noqa: PLC0415

    async with container.session_factory() as session:
        return await EmbeddingSpaceRepository(session).resolve(name)


def _blob_path(uri: str) -> Path | None:
    """Resolve a blob-store URI to a readable file, if it still exists."""
    candidate = Path(uri[len("file://") :] if uri.startswith("file://") else uri)
    return candidate if candidate.exists() else None


def _section_length(chunks: list[Chunk]) -> int:
    """Characters one section's text is once joined, separators included."""
    return sum(len(chunk.text) for chunk in chunks) + 2 * max(0, len(chunks) - 1)


def _section_parts(chunks: list[Chunk], budget: int) -> list[dict[str, Any]]:
    """A section's chunks as the payload reports them, cut to ``budget``.

    Cut per chunk rather than on the joined string so ``text`` and the pages
    beside it always describe the same words — a chunk trimmed short still
    reports the page it was cut on, which is true of the text that remains.
    """
    parts: list[dict[str, Any]] = []
    used = 0
    for chunk in chunks:
        room = budget - used - 2 * len(parts)
        if room <= 0:
            break
        text = chunk.text[:room]
        parts.append(
            {
                "ordinal": chunk.ordinal,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "kind": chunk.content_kind,
                "text": text,
            }
        )
        used += len(text)
    return parts


def _paper_dict(metadata: Any, *, with_abstract: bool = True) -> dict[str, Any]:  # noqa: ANN401
    out: dict[str, Any] = {
        "arxiv_id": metadata.arxiv_id,
        "versioned_id": metadata.versioned_id,
        "title": metadata.title,
        "authors": metadata.author_names(),
        "categories": list(metadata.categories),
        "primary_category": metadata.primary_category,
        "published": (
            metadata.published_at.date().isoformat() if metadata.published_at else None
        ),
        "abs_url": metadata.abs_url,
        "pdf_url": metadata.pdf_url,
        "html_url": metadata.html_url,
    }
    if with_abstract:
        out["abstract"] = metadata.abstract
    return out


async def _space_rows(container: Container) -> list[dict[str, Any]]:
    """Shared by the list tool and the corpus://spaces resource."""
    from paper_app.db.space_repository import EmbeddingSpaceRepository  # noqa: PLC0415

    active = container.active_space
    rows: list[dict[str, Any]] = []
    async with container.session_factory() as session:
        for record in await EmbeddingSpaceRepository(session).list():
            space = record.as_space()
            count = 0
            with contextlib.suppress(Exception):
                count = await container.vector_store_for(space).count()
            rows.append(
                {
                    "name": space.name,
                    "model": f"{space.provider}/{space.model}",
                    "dimensions": space.dimensions,
                    "vectors": count,
                    "is_active": space.name == active.name,
                }
            )
    return rows


# --------------------------------------------------------------------------- tools
@server.tool(
    name="search_arxiv",
    title="Search ArXiv",
    description=(
        "Search ArXiv metadata. No download, no embedding - so this is fast. "
        "Accepts ArXiv's own query language in `query`, passed through verbatim: "
        "field prefixes ti (title), au (author), abs (abstract), co (comment), "
        "jr (journal ref), cat (category), rn (report number), all (everything); "
        "UPPERCASE operators AND / OR / ANDNOT; parentheses to group; double "
        "quotes for phrases. The individual fields below compile to exactly the "
        "same query as writing the prefix by hand. Use this to discover papers, "
        "then ingest the promising ones."
    ),
    annotations=_ann(**_READ_ONLY_REMOTE),
)
async def search_arxiv(
    query: str | None = None,
    title: str | None = None,
    author: str | None = None,
    abstract: str | None = None,
    comment: str | None = None,
    journal: str | None = None,
    category: str | None = None,
    report_number: str | None = None,
    operator: str = "AND",
    arxiv_id: str | None = None,
    submitted_from: str | None = None,
    submitted_to: str | None = None,
    max_results: int = 10,
    start: int = 0,
    sort_by: str = "relevance",
    sort_order: str = "descending",
    has_pdf: bool | None = None,
    has_html: bool | None = None,
    ingested: bool | None = None,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Search ArXiv metadata using ArXiv's own filter language."""
    from paper_app.clients.arxiv.filters import search_query_from  # noqa: PLC0415
    from paper_app.domain.filters import FilterProblem  # noqa: PLC0415
    from paper_app.services.arxiv_search import ArxivSearchService  # noqa: PLC0415

    try:
        request = search_query_from(
            raw=query,
            fields={
                "title": title,
                "author": author,
                "abstract": abstract,
                "comment": comment,
                "journal": journal,
                "category": category,
                "report_number": report_number,
            },
            operator=operator,
            id_list=arxiv_id,
            max_results=max_results,
            start=start,
            sort_by=sort_by,
            sort_order=sort_order,
            submitted_from=submitted_from,
            submitted_to=submitted_to,
            post={"has_pdf": has_pdf, "has_html": has_html, "ingested": ingested},
        )
    except FilterProblem as exc:
        return _fail(str(exc))

    if request.filter.is_empty and not request.id_list:
        return _fail(
            "Provide at least one of: query, title, author, abstract, comment, "
            "journal, category, report_number, arxiv_id."
        )

    container = _container()
    service = ArxivSearchService(container.arxiv, session_factory=container.session_factory)
    outcome = await service.search(request)
    if not outcome.ok:
        return _fail(f"ArXiv search failed: {outcome.error}")

    await _progress(ctx, 1.0, f"{outcome.returned} of {outcome.total_results} results")
    return _ok(
        total_results=outcome.total_results,
        returned=outcome.returned,
        filtered_out=outcome.filtered_out,
        next_start=outcome.next_start,
        warnings=list(outcome.warnings),
        hint=outcome.hint,
        search_query=request.filter.compile() or None,
        papers=[_paper_dict(paper) for paper in outcome.papers],
    )


@server.tool(
    name="list_projects",
    title="List projects",
    description=(
        "Named collections of papers. Papers are global: a paper can sit in any "
        "number of projects, and is never copied. Use this to see what reading "
        "lists exist before creating or importing into one."
    ),
    annotations=_ann(**_READS_LOCAL),
)
async def list_projects(
    include_archived: bool = False,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """List projects with their paper and read counts."""
    from paper_app.db.project_repository import ProjectRepository  # noqa: PLC0415

    container = _container()
    async with container.session_factory() as session:
        rows = await ProjectRepository(session).list_projects(include_archived=include_archived)
    return _ok(
        count=len(rows),
        projects=[
            {
                "slug": row.slug,
                "name": row.name,
                "description": row.description,
                "paper_count": row.paper_count,
                "read_count": row.read_count,
            }
            for row in rows
        ],
    )


@server.tool(
    name="create_project",
    title="Create a project",
    description="Create a named reading list. Empty until papers are imported into it.",
    annotations=_ann(**_WRITES_IDEMPOTENT),
)
async def create_project(
    name: str,
    description: str | None = None,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Create a project. Idempotent only in the sense that a duplicate slug is rejected."""
    from paper_app.db.project_repository import (  # noqa: PLC0415
        ProjectConflictError,
        ProjectRepository,
    )

    container = _container()
    async with container.session_factory() as session:
        try:
            project = await ProjectRepository(session).create(name, description=description)
        except ProjectConflictError as exc:
            return _fail(str(exc))
        await session.commit()
    return _ok(slug=project.slug, name=project.name, description=project.description)


@server.tool(
    name="import_papers_to_project",
    title="Import papers into a project",
    description=(
        "Add existing papers to a project by arXiv id, versioned id or abs URL. "
        "Papers are linked, never copied, and re-importing is a no-op. Set "
        "`ingest=true` to fetch and ingest anything not already in the corpus — "
        "that is slow (minutes per PDF) and is the only write this does."
    ),
    annotations=_ann(**_WRITES_IDEMPOTENT_REMOTE),
)
async def import_papers_to_project(
    project: str,
    arxiv_ids: list[str],
    note: str | None = None,
    ingest: bool = False,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Link papers into a project, optionally ingesting the missing ones first."""
    from paper_app.db.project_repository import (  # noqa: PLC0415
        PaperNotFoundError,
        ProjectRepository,
    )

    container = _container()
    async with container.session_factory() as session:
        projects = ProjectRepository(session)
        try:
            found = await projects.require(project)
        except PaperNotFoundError as exc:
            return _fail(f"{exc}. Use create_project first.")
        resolved, missing = await projects.resolve_paper_ids(arxiv_ids)

    if missing and ingest:
        from paper_app.services.ingestion import build_ingestion_service  # noqa: PLC0415

        service = build_ingestion_service(container, container.active_space)
        for index, arxiv_id in enumerate(missing, start=1):
            await _progress(ctx, index / len(missing), f"ingesting {arxiv_id}")
            await service.ingest_paper(
                arxiv_id, prefer_html=True, requested_by="mcp", trigger="mcp"
            )
        async with container.session_factory() as session:
            projects = ProjectRepository(session)
            found = await projects.require(project)
            resolved, missing = await projects.resolve_paper_ids(arxiv_ids)

    async with container.session_factory() as session:
        projects = ProjectRepository(session)
        found = await projects.require(project)
        added = await projects.add_papers(found, resolved, added_by="mcp", note=note)
        rows = await projects.list_papers(found, limit=100)
        await session.commit()

    if missing:
        return _ok(
            project=found.slug,
            requested=len(arxiv_ids),
            resolved=len(resolved),
            added=added,
            missing=missing,
            hint=(
                f"{len(missing)} id(s) are not in the corpus. Re-run with "
                "ingest=true to fetch them, or search_arxiv to find valid ids."
            ),
            papers=[{"arxiv_id": paper.arxiv_id, "title": paper.title} for paper, _ in rows],
        )
    return _ok(
        project=found.slug,
        requested=len(arxiv_ids),
        resolved=len(resolved),
        added=added,
        missing=[],
        papers=[{"arxiv_id": paper.arxiv_id, "title": paper.title} for paper, _ in rows],
    )


@server.tool(
    name="list_project_papers",
    title="Papers in a project",
    description=(
        "The papers of one project, with their per-project note and read state. "
        "Pass `unread_only=true` to work through a reading list."
    ),
    annotations=_ann(**_READS_LOCAL),
)
async def list_project_papers(
    project: str,
    limit: int = 50,
    unread_only: bool = False,
    category: str | None = None,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """List a project's papers."""
    from paper_app.db.project_repository import (  # noqa: PLC0415
        PaperNotFoundError,
        ProjectRepository,
    )

    container = _container()
    async with container.session_factory() as session:
        projects = ProjectRepository(session)
        try:
            found = await projects.require(project)
        except PaperNotFoundError as exc:
            return _fail(f"{exc}. Use list_projects to see what exists.")
        rows = await projects.list_papers(
            found, limit=max(1, min(limit, 200)), unread_only=unread_only, category=category
        )
    return _ok(
        project=found.slug,
        count=len(rows),
        papers=[
            {
                "arxiv_id": paper.arxiv_id,
                "title": paper.title,
                "categories": list(paper.categories),
                "authors": list(paper.author_names),
                "is_read": link.is_read,
                "note": link.note,
                "abs_url": paper.abs_url,
            }
            for paper, link in rows
        ],
    )


@server.tool(
    name="list_assets",
    title="A paper's figures, tables and equations",
    description=(
        "Extracted non-text content. Figures and tables carry captions and "
        "where their image lives; equations carry LaTeX. Inline math is stored "
        "but omitted by default — a paper has ~3 real equations and ~140 inline "
        "fragments. Set `include_inline_equations` when hunting for a symbol.\n\n"
        "Pass `arxiv_id` for one paper, or `project` for the whole reading "
        "list — the project form answers 'which papers here have figures at "
        "all', which is one query instead of paging through every paper.\n\n"
        "`kind` is an *asset* type (figures/tables/equations), which is a "
        "different axis from the `content` filter on ask_paper_corpus."
    ),
    annotations=_ann(**_READS_LOCAL),
)
async def list_assets(
    arxiv_id: str | None = None,
    project: str | None = None,
    kind: str = "all",
    limit: int = 25,
    include_inline_equations: bool = False,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Show a paper's figures, tables and equations."""
    from paper_app.db.asset_repository import AssetRepository  # noqa: PLC0415
    from paper_app.db.project_repository import (  # noqa: PLC0415
        PaperNotFoundError,
        ProjectRepository,
    )
    from paper_app.db.repositories import PaperRepository  # noqa: PLC0415

    if project and arxiv_id:
        return _fail("Pass either arxiv_id or project, not both.")

    container = _container()
    wanted = {"figures", "tables", "equations"} if kind in {"all", "*"} else {kind}

    if project:
        async with container.session_factory() as session:
            projects = ProjectRepository(session)
            try:
                found = await projects.require(project)
            except PaperNotFoundError as exc:
                return _fail(str(exc))
            linked = await projects.list_papers(found, limit=max(1, min(limit, 200)))
            asset_repo = AssetRepository(session)
            summary = []
            for linked_paper, _link in linked:
                counts = await asset_repo.count(linked_paper.id)
                if counts["figures"] or counts["tables"] or counts["display_equations"]:
                    summary.append(
                        {
                            "arxiv_id": linked_paper.arxiv_id,
                            "title": linked_paper.title,
                            **counts,
                        }
                    )
        return _ok(
            project=found.slug,
            name=found.name,
            count=len(summary),
            papers=summary,
            hint="Per-paper counts. Call list_assets with arxiv_id for captions and image URLs.",
        )

    if not arxiv_id:
        return _fail("Pass either arxiv_id or project.")

    async with container.session_factory() as session:
        papers = PaperRepository(session)
        paper = await papers.get_by_arxiv_id(arxiv_id)
        if paper is None:
            return _fail(
                f"{arxiv_id} is not ingested. Use ingest_paper first, "
                "or search_arxiv to find valid ids."
            )
        repo = AssetRepository(session)
        counts = await repo.count(paper.id)
        figures = await repo.figures(paper.id, limit=limit) if "figures" in wanted else []
        tables = await repo.tables(paper.id, limit=limit) if "tables" in wanted else []
        equations = (
            await repo.equations(
                paper.id, limit=limit, display_only=not include_inline_equations
            )
            if "equations" in wanted
            else []
        )

    payload: dict[str, Any] = {
        "arxiv_id": paper.arxiv_id,
        "title": paper.title,
        "counts": counts,
    }
    if figures:
        payload["figures"] = [
            {
                "ordinal": row.ordinal,
                "label": row.label,
                "caption": row.caption,
                # Where the picture is, not the bytes: fetch it only if needed.
                "image_url": row.image_url,
                "image_sha256": row.image_sha256,
                "page_idx": row.page_idx,
            }
            for row in figures
        ]
    if tables:
        payload["tables"] = [
            {
                "ordinal": row.ordinal,
                "label": row.label,
                "caption": row.caption,
                "rows": row.row_count,
                "columns": row.column_count,
                "has_markup": row.body_html is not None,
            }
            for row in tables
        ]
    if equations:
        payload["equations"] = [
            {"ordinal": row.ordinal, "latex": row.latex, "page_idx": row.page_idx}
            for row in equations
        ]
    if not (figures or tables or equations):
        payload["hint"] = (
            "No assets recorded. Figures, tables and equations are extracted "
            "during ingest; re-ingest with force to retry."
        )
    return _ok(**payload)


@server.tool(
    name="list_references",
    title="A paper's citations",
    description=(
        "Citation edges of one paper. `direction=\"references\"` lists what it "
        "cites; `direction=\"cited-by\"` lists which stored papers cite it. "
        "References are extracted from the paper's HTML rendering during "
        "ingest, so a paper ingested from a PDF-only source may have none."
    ),
    annotations=_ann(**_READS_LOCAL),
)
async def list_references(
    arxiv_id: str,
    direction: str = "references",
    limit: int = 40,
    resolved_only: bool = False,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Show a paper's outgoing or incoming citation edges."""
    from paper_app.db.reference_repository import ReferenceRepository  # noqa: PLC0415
    from paper_app.db.repositories import PaperRepository  # noqa: PLC0415

    container = _container()
    incoming = direction in {"in", "cited-by", "cited_by", "citations"}
    async with container.session_factory() as session:
        papers = PaperRepository(session)
        paper = await papers.get_by_arxiv_id(arxiv_id)
        if paper is None:
            return _fail(
                f"{arxiv_id} is not ingested. Use ingest_paper first, "
                "or search_arxiv to find valid ids."
            )
        refs = ReferenceRepository(session)
        if incoming:
            citing = await refs.citing_papers(paper.id, limit=max(1, min(limit, 200)))
            return _ok(
                arxiv_id=paper.arxiv_id,
                direction="cited-by",
                count=len(citing),
                cited_by=[
                    {"arxiv_id": row.arxiv_id, "title": row.title} for row in citing
                ],
            )
        rows = await refs.list_references(
            paper.id, limit=max(1, min(limit, 200)), resolved_only=resolved_only
        )
    return _ok(
        arxiv_id=paper.arxiv_id,
        direction="references",
        count=len(rows),
        references=[
            {
                "ordinal": row.ordinal,
                "label": row.label(),
                "cited_arxiv_id": row.cited_arxiv_id,
                "resolved": row.is_resolved,
                "year": row.year,
            }
            for row in rows
        ],
        hint=(
            None
            if rows
            else "No references recorded. They are extracted from the paper's HTML "
            "rendering during ingest; re-ingest with force to retry."
        ),
    )


@server.tool(
    name="get_paper",
    title="Get paper metadata",
    description=(
        "Canonical ArXiv metadata for one paper, by id, versioned id or URL "
        "(1706.03762, 1706.03762v5, arXiv:1706.03762). Metadata only: does not "
        "download or ingest."
    ),
    annotations=_ann(**_READ_ONLY_REMOTE),
)
async def get_paper(
    arxiv_id: str,
    ctx: Context | None = None,
) -> dict[str, Any]:
    container = _container()
    try:
        metadata = await container.arxiv.get_paper(arxiv_id)
    except ValueError as exc:
        return _fail(f"{exc}")
    except ArxivError as exc:
        return _fail(f"Could not fetch {arxiv_id!r}: {exc}")
    return _ok(paper=_paper_dict(metadata))


@server.tool(
    name="ingest_paper",
    title="Ingest a paper",
    description=(
        "Download a paper and run the full pipeline: fetch HTML (or PDF), extract "
        "text with MinerU, chunk, embed and index. Slow - minutes for a PDF on "
        "CPU - and it writes to the corpus. Safe to call twice: re-ingesting "
        "replaces that paper's vectors rather than duplicating them.\n\n"
        "Pass `query` instead of `arxiv_id` to ingest the top N hits of a search.\n\n"
        f"`device` applies to this ingest only: {DEVICE_HELP}"
    ),
    annotations=_ann(**_WRITES_IDEMPOTENT_REMOTE),
)
async def ingest_paper(
    arxiv_id: str | None = None,
    query: str | None = None,
    limit: int = 1,
    prefer_html: bool = True,
    force: bool = False,
    space: str | None = None,
    device: str | None = None,
    ctx: Context | None = None,
) -> dict[str, Any]:
    if not arxiv_id and not query:
        return _fail("Provide either arxiv_id or query.")

    try:
        chosen_device = parse_device(device)
    except ValueError as exc:
        return _fail(str(exc))

    container = _container()
    try:
        target = await _resolve_space(container, space)
    except SpaceConflictError as exc:
        return _fail(str(exc))

    service = build_ingestion_service(container, target, device=chosen_device)
    try:
        await container.vector_store_for(target).ensure_ready()
    except Exception as exc:  # noqa: BLE001
        return _fail(f"Vector store unavailable for space {target.name!r}: {exc}")

    await _progress(ctx, 0.05, f"starting ingestion into space {target.name}")

    if arxiv_id:
        await _progress(ctx, 0.15, f"fetching {arxiv_id}")
        result = await service.ingest_paper(
            arxiv_id,
            prefer_html=prefer_html,
            force=force,
            requested_by="mcp",
            trigger="mcp",
        )
    else:
        await _progress(ctx, 0.1, f"searching for {query!r}")
        result = await service.ingest_query(
            SearchQuery(raw=query, max_results=limit),
            limit=limit,
            prefer_html=prefer_html,
            force=force,
            requested_by="mcp",
            trigger="mcp",
        )

    await _progress(ctx, 1.0, "done")
    return _ok(
        space=target.name,
        # Echoed rather than assumed: a caller who asked for cuda should be able
        # to confirm it got cuda, not the configured default.
        device=service.embedding_device,
        status=result.status.value,
        run_ids=result.run_ids,
        paper_ids=result.paper_ids,
        errors=result.errors or None,
        hint=(
            "Now use ask_paper_corpus to search the ingested text."
            if result.succeeded
            else "Ingestion failed; see errors. Nothing was written."
        ),
    )


@server.tool(
    name="ask_paper_corpus",
    title="Search the ingested corpus",
    description=(
        "Semantic (vector) search across ingested papers. Returns the most "
        "relevant chunks with their paper title, section heading and content "
        "kind.\n\n"
        "Three independent scopes that compose:\n"
        "  project  - only the papers of that project\n"
        "  arxiv_id - only those papers (repeatable)\n"
        "  content  - only these chunk kinds: body, abstract, figure, table, "
        "equation, reference, code (repeatable; plurals like 'equations' work)\n\n"
        "Prefer this over re-ingesting: it is instant. Results come from chunk "
        "text, not abstracts, so the wording differs from ArXiv metadata.\n\n"
        f"`device`: {DEVICE_HELP}\n\n"
        "`section` narrows to one part of a document before ranking: a book's own "
        "numbering (\"2.2\") or words from its title (\"Debye\"). Add `doc_key` so the "
        "name is matched in that book only rather than across the corpus. Every "
        "matching section is included, ranked best-first; each hit reports the "
        "section it came from and the page."
    ),
    annotations=_ann(**_READ_ONLY_REMOTE),
)
async def ask_paper_corpus(
    query: str,
    top_k: int = 8,
    category: str | None = None,
    space: str | None = None,
    max_chars: int = 900,
    project: str | None = None,
    arxiv_id: list[str] | None = None,
    content: list[str] | None = None,
    source: list[str] | None = None,
    min_score: float | None = None,
    device: str | None = None,
    section: str | None = None,
    doc_key: str | None = None,
    ctx: Context | None = None,
) -> dict[str, Any]:
    if not query.strip():
        return _fail("query must not be empty")

    try:
        chosen_device = parse_device(device)
    except ValueError as exc:
        return _fail(str(exc))

    container = _container()
    try:
        target = await _resolve_space(container, space)
        # Parsed before the search so a typo is a clear error, not a quiet
        # zero-result answer that reads like "nothing matched".
        kinds = parse_kinds(content)
        sources = parse_sources(source)
    except SpaceConflictError as exc:
        return _fail(str(exc))
    except ValueError as exc:
        return _fail(str(exc))

    # Resolved here rather than inside the search service: turning "Debye" into an
    # ordinal needs the section table, and a vector store knows nothing about
    # headings. Failing on an unmatched name is deliberate — a section filter that
    # silently matched nothing would look identical to "this topic is not covered".
    section_keys: list[tuple[str, int]] | None = None
    sections_meta: list[dict[str, Any]] = []
    if section:
        from paper_app.services.sections_query import find_sections  # noqa: PLC0415

        async with container.session_factory() as session:
            matched = await find_sections(
                session, section, doc_keys=[doc_key] if doc_key else None
            )
        if not matched:
            where = f" in {doc_key!r}" if doc_key else " in the corpus"
            return _fail(
                f"no section matching {section!r}{where}; try a word from its "
                "title or its number (2.2)"
            )
        section_keys = [item.key for item in matched]
        sections_meta = [item.as_dict() for item in matched]

    service = SemanticSearchService(
        vector_store=container.vector_store_for(target),
        embeddings=container.provider_for(target, chosen_device),
        session_factory=container.session_factory,
        space=target,
    )
    try:
        hits = await service.search(
            query,
            top_k=max(1, min(top_k, 50)),
            category=category,
            paper_ids=list(arxiv_id) if arxiv_id else None,
            content_kinds=kinds,
            project=project,
            sources=sources,
            min_score=min_score,
            sections=section_keys,
        )
    except ValueError as exc:
        return _fail(str(exc))
    except LookupError as exc:
        return _fail(str(exc))

    await _progress(ctx, 1.0, f"{len(hits)} hits")

    if not hits:
        empty = False
        with contextlib.suppress(Exception):
            empty = await container.vector_store_for(target).count() == 0
        if empty:
            return _ok(
                space=target.name,
                hits=[],
                hint="This space is empty - ingest a paper first.",
            )

    return _ok(
        space=target.name,
        model=target.model,
        dimensions=target.dimensions,
        count=len(hits),
        scope=service.last_scope,
        # Which sections the name resolved to. Echoed because "Debye" matching
        # eleven sections is a fact the caller needs in order to read the hits:
        # they cannot tell a widened scope from the one they asked for.
        sections=sections_meta or None,
        hits=[
            {
                # `display_id`, not `arxiv_id`: a textbook has none, and an
                # assistant reading a corpus of papers *and* books needs a handle
                # it can pass to `read_section` or `read_chunks`.
                "arxiv_id": hit.metadata.get("arxiv_id"),
                "display_id": hit.metadata.get("display_id"),
                "doc_key": hit.metadata.get("doc_key"),
                # `content_kind`, not `kind`: the vector payload's `kind` is the
                # *paper's* kind (paper/book/notes), so reading it here answered
                # "what is this chunk" with "paper" for every single chunk.
                "kind": hit.metadata.get("content_kind"),
                "title": hit.metadata.get("title"),
                # The document's own section, preferred over the chunker's
                # markdown heading: in a book the two can name the same passage
                # differently, and only one of them is the book's structure.
                "section": (
                    hit.metadata.get("section_title") or hit.metadata.get("heading")
                ),
                "section_level": hit.metadata.get("section_level"),
                "page_start": hit.metadata.get("page_start"),
                "page_end": hit.metadata.get("page_end"),
                "score": round(hit.score, 4),
                "text": hit.text[:max_chars],
                "truncated": len(hit.text) > max_chars,
            }
            for hit in hits
        ],
    )


# `read_sections` — the map of a document — lives in `paper_app.mcp.tools_corpus`, next to
# `list_documents`, which resolves the same `doc_key`. It was registered here too,
# and the second registration was silently shadowed by the first.

@server.tool(
    name="read_section",
    title="Read one section's text",
    description=(
        "The text of one section of a document, in order, with the page each part "
        "came from. This is the other half of a `section=` search: that one finds "
        "passages by meaning inside a section, this one returns the section.\n\n"
        "`name` accepts the book's own numbering (\"2.2\") or words from its title "
        "(\"Debye\"), and every match is returned ranked best-first — a word often "
        "names several sections. `include_subsections` lists what is nested inside "
        "without their text. `max_chars` is the whole call's budget, not each "
        "section's, so a word naming a dozen sections returns the first few whole "
        "rather than a dozen stubs; `omitted` counts the matches it never "
        "reached.\n\n"
        "The text is the chunks that point at the section, not its page range: a "
        "section can span ten pages of which four produced text, and the page is "
        "reported per part because there it is a fact.\n\n"
        "To see what a document holds before naming anything, read_sections lists "
        "its sections."
    ),
    annotations=_ann(**_READS_LOCAL),
)
async def read_section(
    doc_key: str,
    name: str,
    include_subsections: bool = False,
    max_chars: int = 20000,
    ctx: Context | None = None,
) -> dict[str, Any]:
    from paper_app.db.repositories import PaperRepository, SectionRepository  # noqa: PLC0415
    from paper_app.services.sections_query import find_sections, sections_within  # noqa: PLC0415

    container = _container()
    async with container.session_factory() as session:
        paper = await PaperRepository(session).resolve(doc_key)
        if paper is None:
            return _fail(f"no document matching {doc_key!r}")
        matched = await find_sections(session, name, doc_keys=[paper.doc_key])
        if not matched:
            # Failing rather than returning an empty list: an unmatched name and
            # a section that produced no text look identical otherwise, and the
            # first is a typo the caller can fix.
            return _fail(
                f"no section matching {name!r} in {paper.doc_key!r}; "
                f"use read_sections('{paper.doc_key}') to list them"
            )
        repo = SectionRepository(session)
        all_sections = await repo.list_for_paper(paper.id)
        by_ordinal = {row.ordinal: row for row in all_sections}

        out: list[dict[str, Any]] = []
        # One budget for the call, spent in document order: the matches are ranked
        # best-first, so a name naming a dozen sections spends it on the one the
        # caller most likely meant instead of returning a dozen empty stubs.
        budget = max(1, max_chars)
        for match in matched:
            chunks = await repo.list_for_section(paper.id, match.ordinal)
            parts = _section_parts(chunks, budget)
            text = "\n\n".join(part["text"] for part in parts)
            entry = match.as_dict()
            entry.update(
                {
                    "chunks": parts,
                    "text": text,
                    "truncated": len(text) < _section_length(chunks),
                    "subsections": [
                        {"ordinal": row.ordinal, "title": row.title, "level": row.level}
                        for row in sections_within(match, by_ordinal.values())
                    ]
                    if include_subsections
                    else None,
                }
            )
            out.append(entry)
            budget -= len(text)
            if budget <= 0:
                break
        return _ok(
            doc_key=paper.doc_key,
            name=name,
            count=len(out),
            max_chars=max_chars,
            # How many ranked matches the budget never reached. Named rather than
            # called `truncated`, which each section uses for its own cut text.
            omitted=len(matched) - len(out),
            sections=out,
        )


@server.tool(
    name="read_chunks",
    title="Read a paper's chunks",
    description=(
        "The stored chunks of an ingested paper, in order, optionally from an "
        "ordinal offset. Use after ask_paper_corpus to read more of a section.\n\n"
        "Pass `content` to read only one kind of chunk (same values as "
        "ask_paper_corpus: body, abstract, figure, table, equation, reference, "
        "code; repeat the argument for several). `total_chunks` then counts only "
        "the filtered set, and every chunk carries its `kind`."
    ),
    annotations=_ann(**_READS_LOCAL),
)
async def read_chunks(
    arxiv_id: str,
    offset: int = 0,
    limit: int = 10,
    max_chars: int = 1200,
    content: list[str] | None = None,
    ctx: Context | None = None,
) -> dict[str, Any]:
    from paper_app.db.repositories import ChunkRepository, PaperRepository  # noqa: PLC0415

    try:
        kinds = parse_kinds(content)
    except ValueError as exc:
        return _fail(str(exc))

    container = _container()
    async with container.session_factory() as session:
        # get_by_arxiv_id normalises, so `2401.00001v2` and an abs URL both work.
        paper = await PaperRepository(session).get_by_arxiv_id(arxiv_id)
        if paper is None:
            return _fail(
                f"{arxiv_id} is not ingested. Use ingest_paper first, or "
                "search_arxiv to find it."
            )
        chunks = ChunkRepository(session)
        total = await chunks.count(paper.id, content_kinds=kinds)
        rows = await chunks.list_for_paper(
            paper.id,
            limit=max(1, min(limit, 100)),
            offset=max(0, offset),
            content_kinds=kinds,
        )
        return _ok(
            arxiv_id=paper.arxiv_id,
            title=paper.title,
            content_kinds=kinds or [],
            total_chunks=total,
            paper_total_chunks=await chunks.count(paper.id),
            returned=len(rows),
            chunks=[
                {
                    "ordinal": row.ordinal,
                    "section": row.heading,
                    "section_path": list(row.section_path),
                    "kind": row.content_kind,
                    "tokens": row.token_count,
                    "text": row.text[:max_chars],
                    "truncated": len(row.text) > max_chars,
                }
                for row in rows
            ],
        )


@server.tool(
    name="read_markdown",
    title="Read extracted markdown",
    description=(
        "The extracted markdown for an ingested paper (from its HTML rendering, "
        "or from MinerU when the PDF path was used). For a whole-paper read; "
        "prefer read_chunks when you only need part of it."
    ),
    annotations=_ann(**_READS_LOCAL),
)
async def read_markdown(
    arxiv_id: str,
    max_chars: int = 20000,
    ctx: Context | None = None,
) -> dict[str, Any]:
    from sqlalchemy import select  # noqa: PLC0415

    from paper_app.db.repositories import PaperRepository  # noqa: PLC0415
    from paper_app.domain.ids import normalize_arxiv_id  # noqa: PLC0415

    try:
        target_id = normalize_arxiv_id(arxiv_id)
    except ValueError as exc:
        return _fail(str(exc))

    container = _container()
    async with container.session_factory() as session:
        paper = await PaperRepository(session).get_by_arxiv_id(target_id)
        if paper is None:
            return _fail(f"{target_id} is not ingested.")

        document = (
            await session.execute(
                select(RawDocument)
                .where(
                    RawDocument.paper_id == paper.id,
                    RawDocument.kind.in_(["mineru_markdown", "text"]),
                )
                .order_by(RawDocument.created_at.desc())
                .limit(1)
            )
        ).scalar_one_or_none()
        if document is None:
            return _fail(f"{target_id} has no extracted text yet.")
        path = _blob_path(document.uri)
        if path is None:
            return _fail(
                f"The extracted markdown for {target_id} is no longer on disk "
                f"({document.uri}). Re-ingest to regenerate it."
            )
        title = paper.title

    text = path.read_text(errors="replace")
    truncated = len(text) > max_chars
    return _ok(
        arxiv_id=target_id,
        title=title,
        chars=len(text),
        markdown=text[:max_chars],
        truncated=truncated,
        hint="Use read_chunks for the rest." if truncated else None,
    )


@server.tool(
    name="list_papers",
    title="List ingested papers",
    description="List the papers already in the local corpus, newest first.",
    annotations=_ann(**_READS_LOCAL),
)
async def list_papers(
    limit: int = 20,
    offset: int = 0,
    ingested_only: bool = True,
    ctx: Context | None = None,
) -> dict[str, Any]:
    from paper_app.db.repositories import PaperRepository  # noqa: PLC0415

    container = _container()
    async with container.session_factory() as session:
        papers, total = await PaperRepository(session).list_papers(
            limit=max(1, min(limit, 100)),
            offset=max(0, offset),
            ingested_only=ingested_only,
        )
        return _ok(
            total=total,
            papers=[
                {
                    "arxiv_id": paper.arxiv_id,
                    "title": paper.title,
                    "categories": list(paper.categories),
                    "ingested": paper.ingested_at is not None,
                }
                for paper in papers
            ],
        )


@server.tool(
    name="list_embedding_spaces",
    title="List embedding models",
    description=(
        "The embedding spaces in this corpus (one model, one table), with their "
        "dimensions and vector counts. Pass `space` to the other tools to pick "
        "one; omit it to use the active space."
    ),
    annotations=_ann(**_READS_LOCAL),
)
async def list_embedding_spaces(ctx: Context | None = None) -> dict[str, Any]:
    container = _container()
    return _ok(active=container.active_space.name, spaces=await _space_rows(container))


@server.tool(
    name="status",
    title="Corpus status",
    description=(
        "Health and configuration of the assistant: database, vector store, active "
        "embedding model and available PDF extraction backends."
    ),
    annotations=_ann(**_READS_LOCAL),
)
async def status(ctx: Context | None = None) -> dict[str, Any]:
    from paper_app.clients.content.mineru_resolver import installed_version  # noqa: PLC0415
    from paper_app.db.session import check_connection  # noqa: PLC0415

    container = _container()
    space = container.active_space
    store = container.vector_store_for(space)
    vectors = 0
    with contextlib.suppress(Exception):
        vectors = await store.count()

    return _ok(
        database="up" if await check_connection(container.settings.database) else "down",
        vector_store=store.name,
        active_space=space.name,
        model=f"{space.provider}/{space.model}",
        dimensions=space.dimensions,
        vectors=vectors,
        extraction_backends=container.mineru.describe_backends() or "none",
        mineru_version=installed_version(),
    )


# ------------------------------------------------------------------------ resources
@server.resource(
    "corpus://spaces",
    name="Embedding spaces",
    title="Embedding spaces",
    description="The embedding models registered in this corpus.",
    mime_type="application/json",
)
async def spaces_resource() -> str:
    container = _container()
    rows = await _space_rows(container)
    return json.dumps({"active": container.active_space.name, "spaces": rows}, indent=2)


@server.resource(
    "paper://{arxiv_id}",
    name="Paper metadata",
    title="ArXiv paper metadata",
    description="Canonical ArXiv metadata for a paper.",
    mime_type="application/json",
)
async def paper_resource(arxiv_id: str) -> str:
    container = _container()
    metadata = await container.arxiv.get_paper(arxiv_id)
    return json.dumps(_paper_dict(metadata), indent=2)


# Imported last, for its side effect: `tools_corpus` registers its tools onto
# `server` and reaches back into this module's helpers at call time. Registering
# here rather than at the top keeps that cycle one-directional.
from paper_app.mcp import tools_corpus  # noqa: E402  # isort: skip

tools_corpus.register(server)


# ---------------------------------------------------------------------------- entry
def run(transport: str = "stdio", host: str = "127.0.0.1", port: int = 8080) -> None:
    """Entry point for ``paper mcp``."""
    if transport == "stdio":
        server.run(transport="stdio")
    elif transport == "http":
        server.run(transport="streamable-http", host=host, port=port)
    else:  # pragma: no cover - the CLI validates this
        raise ValueError(f"unknown transport {transport!r}")


__all__ = ["server", "run", "INSTRUCTIONS"]
