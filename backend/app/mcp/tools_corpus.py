"""MCP tools for corpus maintenance and discovery.

Split out of :mod:`app.mcp.server` so the main surface stays navigable as it
grows. Everything here registers onto the same ``server`` object via
:func:`register`, called at the bottom of that module — importing this file has no
effect on its own, which keeps the import cycle one-directional.

Two groups:

* **Discovery** — categories, authors, citation ranking, run history, formula
  search. These answer "what is in the corpus?" without a semantic query.
* **Maintenance** — re-embedding into another space, re-labelling chunk kinds,
  read state, project deletion. Every one of these writes, so each is annotated
  as such; a client that auto-approves read-only tools will never trigger them by
  accident.
"""

from __future__ import annotations

from typing import Any, Protocol

from mcp.server.mcpserver.context import Context

from app.logging import get_logger
from app.mcp.devices import DEVICE_HELP, parse_device

logger = get_logger(__name__)


class _Server(Protocol):
    """Just enough of ``MCPServer`` to register a tool.

    Structural rather than the concrete class: this module is imported *by*
    ``app.mcp.server``, so naming that module's types at runtime would be a
    cycle, and importing the SDK type would tie this file to an SDK version for
    one method's sake.
    """

    def tool(self, **kwargs: Any) -> Any: ...


def register(server: _Server) -> None:
    """Attach every tool in this module to ``server``."""
    _add_discovery_tools(server)
    _add_maintenance_tools(server)


# ------------------------------------------------------------------ helpers
def _fail(message: str) -> dict[str, Any]:
    """A structured error a model can act on.

    Copied rather than imported: ``app.mcp.server`` imports this module, so
    reaching back into it at call time is what keeps the cycle harmless.
    """
    from app.mcp.server import _fail as _server_fail  # noqa: PLC0415

    return _server_fail(message)


def _ok(**payload: Any) -> dict[str, Any]:
    from app.mcp.server import _ok as _server_ok  # noqa: PLC0415

    return _server_ok(**payload)


def _container():  # noqa: ANN201
    from app.mcp.server import _container as _server_container  # noqa: PLC0415

    return _server_container()


def _ann(**kwargs: Any):  # noqa: ANN202
    from app.mcp.server import _ann as _server_ann  # noqa: PLC0415

    return _server_ann(**kwargs)


_READS_LOCAL = {"read_only_hint": True, "open_world_hint": False, "destructive_hint": False}
_WRITES_IDEMPOTENT = {
    "read_only_hint": False,
    "open_world_hint": False,
    "destructive_hint": False,
    "idempotent_hint": True,
}
_WRITES_DESTRUCTIVE = {
    "read_only_hint": False,
    "open_world_hint": False,
    "destructive_hint": True,
    "idempotent_hint": True,
}


# ------------------------------------------------------------------ discovery
def _add_discovery_tools(server: _Server) -> None:
    @server.tool(
        name="search_equations",
        title="Search stored formulas",
        description=(
            "Find stored equations whose LaTeX contains a substring, across the "
            "corpus or scoped to a project and/or specific papers.\\n\\n"
            "This is a substring match, not a semantic one: `q` is LaTeX text "
            "(`softmax`, `\\\\frac{1}{2}`, `sheaf`). For meaning-based search over "
            "formulas use ask_paper_corpus with `content=[\"equation\"]`.\\n\\n"
            "Inline fragments are hidden unless `include_inline_equations` is "
            "true: a paper has ~3 real display equations and ~140 inline ones."
        ),
        annotations=_ann(**_READS_LOCAL),
    )
    async def search_equations(
        q: str,
        project: str | None = None,
        arxiv_id: list[str] | None = None,
        limit: int = 20,
        display_only: bool = True,
        include_inline_equations: bool = False,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        from app.db.asset_repository import AssetRepository  # noqa: PLC0415
        from app.db.project_repository import (  # noqa: PLC0415
            PaperNotFoundError,
            ProjectRepository,
        )
        from app.db.repositories import PaperRepository  # noqa: PLC0415

        if not q.strip():
            return _fail("q must not be empty")

        container = _container()
        async with container.session_factory() as session:
            projects = ProjectRepository(session)
            paper_ids: list[str] | None = None
            if project:
                try:
                    paper_ids = await projects.paper_ids_for(project)
                except PaperNotFoundError as exc:
                    return _fail(str(exc))
            if arxiv_id:
                papers = PaperRepository(session)
                wanted = [await papers.get_by_arxiv_id(one) for one in arxiv_id]
                missing = [one for one, p in zip(arxiv_id, wanted, strict=True) if p is None]
                if missing:
                    return _fail(f"not ingested: {', '.join(missing)}")
                resolved = [p.id for p in wanted if p is not None]
                paper_ids = resolved if paper_ids is None else sorted(set(paper_ids) & set(resolved))

            rows = await AssetRepository(session).search_equations(
                q,
                limit=max(1, min(limit, 200)),
                display_only=display_only and not include_inline_equations,
                paper_ids=paper_ids,
            )
            # Name the papers: an equation without its source is a formula to
            # copy, not a finding to use.
            names = {}
            for row in rows:
                if row.paper_id not in names:
                    paper = await PaperRepository(session).get(row.paper_id)
                    names[row.paper_id] = paper.arxiv_id if paper else None
        return _ok(
            query=q,
            project=project,
            scope_papers=len(paper_ids) if paper_ids is not None else None,
            count=len(rows),
            equations=[
                {
                    "arxiv_id": names.get(row.paper_id),
                    "ordinal": row.ordinal,
                    "latex": row.latex,
                    "is_display": row.is_display,
                    "page_idx": row.page_idx,
                    "source": row.source,
                }
                for row in rows
            ],
        )

    @server.tool(
        name="list_categories",
        title="List subject categories",
        description=(
            "Every ArXiv subject category in the corpus with its paper count, for "
            "use as a `category` filter in ask_paper_corpus or search_arxiv."
        ),
        annotations=_ann(**_READS_LOCAL),
    )
    async def list_categories(ctx: Context | None = None) -> dict[str, Any]:
        from app.db.repositories import PaperRepository  # noqa: PLC0415

        container = _container()
        async with container.session_factory() as session:
            rows = await PaperRepository(session).list_categories()
        rows = [row for row in rows if row["paper_count"]]
        return _ok(count=len(rows), categories=rows)

    @server.tool(
        name="list_authors",
        title="List corpus authors",
        description=(
            "The most prolific authors in the corpus, with paper counts. Pass "
            "`name` to resolve one author and get their papers instead."
        ),
        annotations=_ann(**_READS_LOCAL),
    )
    async def list_authors(
        name: str | None = None,
        limit: int = 50,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        from app.db.repositories import PaperRepository  # noqa: PLC0415

        container = _container()
        async with container.session_factory() as session:
            repo = PaperRepository(session)
            if name:
                found = await repo.find_author(name)
                if found is None:
                    return _fail(f"no author matching {name!r} in the corpus")
                papers = await repo.papers_by_author(name)
                return _ok(
                    author=found,
                    count=len(papers),
                    papers=[
                        {
                            "arxiv_id": paper.arxiv_id,
                            "title": paper.title,
                            "categories": paper.categories,
                        }
                        for paper in papers
                    ],
                )
            rows = await repo.list_authors(limit=max(1, min(limit, 500)))
        return _ok(count=len(rows), authors=rows)

    @server.tool(
        name="list_documents",
        title="List local documents",
        description=(
            "Textbooks, lecture notes, reports and theses in the corpus, with "
            "their page count and the number of chunks each holds. The corpus "
            "holds these alongside the arXiv papers and searches them together, "
            "so this is how you find out what is in it. Filter by `kind` "
            "(book, notes, report, thesis) or pass `doc_key` to resolve one."
        ),
        annotations=_ann(**_READS_LOCAL),
    )
    async def list_documents(
        kind: str | None = None,
        doc_key: str | None = None,
        limit: int = 100,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        from app.db.repositories import PaperRepository  # noqa: PLC0415

        container = _container()
        async with container.session_factory() as session:
            repo = PaperRepository(session)
            if doc_key:
                found = await repo.resolve(doc_key)
                if found is None:
                    return _fail(f"no document matching {doc_key!r}")
                return _ok(documents=[await _describe_document(session, found)])
            rows = await repo.list_documents(kind=kind, limit=max(1, min(limit, 1000)))
            return _ok(
                count=len(rows),
                documents=[await _describe_document(session, row) for row in rows],
            )

    @server.tool(
        name="read_sections",
        title="Read a document's table of contents",
        description=(
            "The map a book is navigated with: its chapters and sections, each "
            "with the pages it spans and how many chunks of it are stored. A "
            "paper does not need this — a 700-page textbook does, because nobody "
            "cites page 400 of one without first asking what is on it.\n\n"
            "`read_section` is the other half: this says which sections exist, "
            "that returns one section's text. Two sources are merged and each "
            "row says which it came from: `pdf` for the document's own bookmarks, "
            "`text` for headings recovered from the extracted text, which is the "
            "only source for a document that ships without bookmarks.\n\n"
            "`after_page` restricts the map to the part that was actually indexed, "
            "which is narrower than `page_count` after a partial ingest, and "
            "`max_level` to chapters or to chapters and sections."
        ),
        annotations=_ann(**_READS_LOCAL),
    )
    async def read_sections(
        doc_key: str,
        max_level: int | None = None,
        after_page: int | None = None,
        limit: int = 300,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        from app.db.repositories import PaperRepository, SectionRepository  # noqa: PLC0415

        container = _container()
        async with container.session_factory() as session:
            found = await PaperRepository(session).resolve(doc_key)
            if found is None:
                return _fail(
                    f"no document matching {doc_key!r}. Use list_documents to see "
                    "what is in the corpus."
                )
            sections = SectionRepository(session)
            rows = await sections.list_for_paper(found.id, max_level=max_level)
            # One query for the whole column, not one per row: a book holds
            # hundreds of sections and the number is printed on every one.
            counts = await sections.count_chunks_in_sections(
                found.id, [row.ordinal for row in rows]
            )
            out = [
                {
                    "ordinal": row.ordinal,
                    "title": row.title,
                    "level": row.level,
                    "page_start": row.page_start,
                    "page_end": row.page_end,
                    "source": row.source,
                    "chunks": counts.get(row.ordinal, 0),
                }
                for row in rows
                if after_page is None
                or (row.page_start is not None and row.page_start >= after_page)
            ]
            return _ok(
                doc_key=found.doc_key,
                title=found.title,
                kind=found.kind,
                page_count=found.page_count,
                shown=len(out),
                total=len(rows),
                sections=out[: max(1, min(limit, 2000))],
            )

    @server.tool(
        name="most_cited_references",
        title="Most-cited corpus papers",
        description=(
            "Papers in the corpus ranked by how many other corpus papers cite "
            "them. Use it to find the field's centre of gravity, or "
            "`direction=cited-by` on a paper's references to see its influence."
        ),
        annotations=_ann(**_READS_LOCAL),
    )
    async def most_cited_references(
        limit: int = 20,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        from app.db.reference_repository import ReferenceRepository  # noqa: PLC0415

        container = _container()
        async with container.session_factory() as session:
            rows = await ReferenceRepository(session).most_cited(limit=max(1, min(limit, 200)))
        return _ok(count=len(rows), papers=rows)

    @server.tool(
        name="list_ingest_runs",
        title="Ingestion history",
        description=(
            "Recent ingestion runs with their status and error, newest first. "
            "Use it when a paper was not searchable after ingest_paper, or to "
            "find which of a batch actually succeeded. `stale=true` lists only "
            "the runs whose owning process is gone — a run stuck in `running` for "
            "hours is not work in progress, it is work that will never finish, and "
            "re-running its target resumes it from the stored artefacts."
        ),
        annotations=_ann(**_READS_LOCAL),
    )
    async def list_ingest_runs(
        status: str | None = None,
        stale: bool = False,
        limit: int = 20,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        from app.db.repositories import RunRepository  # noqa: PLC0415
        from app.services.runs import find_stale_runs  # noqa: PLC0415

        container = _container()
        hours = container.settings.ingestion.stale_run_hours
        async with container.session_factory() as session:
            if stale:
                # `stale` reports from the staleness check rather than filtering
                # the recent list, so a run stuck a week ago is not excluded by a
                # limit that was reached by recent ones.
                found = await find_stale_runs(session, older_than_hours=hours)
                return _ok(
                    stale=True,
                    stale_after_hours=hours,
                    count=len(found),
                    runs=[item.as_dict() for item in found],
                )
            rows = await RunRepository(session).recent_runs(limit=max(1, min(limit, 200)))
        if status:
            wanted = status.strip().lower()
            rows = [row for row in rows if row.status.lower() == wanted]
        return _ok(
            stale=False,
            stale_after_hours=hours,
            count=len(rows),
            runs=[
                {
                    "id": row.id,
                    "arxiv_id": row.arxiv_id,
                    "doc_key": row.doc_key,
                    "status": row.status,
                    "trigger": row.trigger,
                    "steps": len(row.steps),
                    "failed_steps": [
                        step.name for step in row.steps if step.status != "succeeded"
                    ],
                    "error": row.error,
                    "created_at": row.created_at.isoformat() if row.created_at else None,
                }
                for row in rows
            ],
        )

    @server.tool(
        name="reap_ingest_runs",
        title="Close abandoned ingestion runs",
        description=(
            "Mark ingestion runs whose owning process is gone as `abandoned`, so a "
            "list of runs stops reporting work that will never finish. A run is "
            "stale once nothing has touched it for `hours` — measured from its "
            "newest step, not from when it was created, so a long run that is still "
            "making progress is never touched. Default is a dry run: pass "
            "`apply=true` to change anything, because a reaped run's row is the "
            "record of what happened. Then re-run the target with `ingest_paper` to "
            "resume it from the stored blob and extracted markdown."
        ),
        # Idempotent: reaping an already-abandoned run changes nothing, so a
        # client may auto-approve it the way it does the other writes.
        annotations=_ann(**_WRITES_IDEMPOTENT),
    )
    async def reap_ingest_runs(
        hours: float | None = None,
        apply: bool = False,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        from app.services.runs import reap_stale_runs  # noqa: PLC0415

        container = _container()
        bound = hours if hours and hours > 0 else container.settings.ingestion.stale_run_hours
        async with container.session_factory() as session:
            out = await reap_stale_runs(session, older_than_hours=bound, dry_run=not apply)
        out["stale_after_hours"] = bound
        if out["found"] and not apply:
            out["next"] = (
                "Nothing changed. Call again with apply=true to close these runs, "
                "then re-run each target with ingest_paper to resume it."
            )
        return out


async def _describe_document(session: Any, paper: Any) -> dict[str, Any]:  # noqa: ANN401
    """One corpus row as a document: what it is and how much of it is indexed."""
    from sqlalchemy import func, select  # noqa: PLC0415

    from app.db.models import Chunk, DocumentSection  # noqa: PLC0415

    chunks = await session.scalar(
        select(func.count()).select_from(Chunk).where(Chunk.paper_id == paper.id)
    )
    sections = await session.scalar(
        select(func.count()).select_from(DocumentSection).where(
            DocumentSection.paper_id == paper.id
        )
    )
    pages = [
        (row.page_start, row.page_end)
        for row in (
            await session.execute(
                select(Chunk.page_start, Chunk.page_end).where(
                    Chunk.paper_id == paper.id, Chunk.page_start.is_not(None)
                )
            )
        ).all()
        if row.page_start is not None
    ]
    return {
        "doc_key": paper.doc_key,
        "title": paper.title,
        "kind": paper.kind,
        "page_count": paper.page_count,
        "chunks": chunks or 0,
        "sections": sections or 0,
        # The pages actually indexed, which is narrower than `page_count` after
        # a partial ingest and is the honest answer to "can I cite this page?".
        "indexed_pages": [min(p[0] for p in pages), max(p[1] for p in pages)]
        if pages
        else None,
    }


# ---------------------------------------------------------------- maintenance
def _add_maintenance_tools(server: _Server) -> None:
    @server.tool(
        name="reembed_space",
        title="Re-embed into another space",
        description=(
            "Embed chunks that are already ingested into a different embedding "
            "space. No downloads, no re-parsing, **no network**: chunks live in "
            "the `chunks` table and each space has its own vector table, so a "
            "model change costs minutes rather than hours.\\n\\n"
            "Use it after `list_embedding_spaces` shows another model is "
            "registered, instead of re-ingesting every paper.\\n\\n"
            "Scope with `project` or `arxiv_id` (any id spelling resolves). "
            "`dry_run` lists the targets and creates nothing. `force` re-embeds "
            "papers the space already has.\n\n"
            f"`device`: {DEVICE_HELP} This is the one tool where it is usually "
            "worth setting — re-embedding a whole space is thousands of chunks, "
            "which is exactly the case where it measured several times faster."
        ),
        annotations=_ann(**_WRITES_IDEMPOTENT),
    )
    async def reembed_space(
        space: str,
        project: str | None = None,
        arxiv_id: list[str] | None = None,
        force: bool = False,
        limit: int | None = None,
        dry_run: bool = False,
        device: str | None = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        from app.db.space_repository import (  # noqa: PLC0415
            EmbeddingSpaceRepository,
            SpaceConflictError,
        )
        from app.services.reembed import ReembedService  # noqa: PLC0415

        try:
            chosen_device = parse_device(device)
        except ValueError as exc:
            return _fail(str(exc))

        container = _container()
        try:
            async with container.session_factory() as session:
                target = await EmbeddingSpaceRepository(session).resolve(space)
        except SpaceConflictError as exc:
            return _fail(str(exc))

        def on_progress(done: int, total: int, arxiv: str, chunks: int) -> None:
            logger.info("mcp_reembed_progress", extra={"done": done, "total": total})

        service = ReembedService(
            provider=container.provider_for(target, chosen_device),
            space=target,
            session_factory=container.session_factory,
            store=container.vector_store_for(target),
            on_progress=on_progress,
        )
        try:
            if dry_run:
                listed = await service.preview(
                    project=project,
                    arxiv_ids=list(arxiv_id) if arxiv_id else None,
                    limit=limit,
                )
                return _ok(
                    space=target.name,
                    model=target.model,
                    dimensions=target.dimensions,
                    dry_run=True,
                    count=len(listed),
                    papers=listed,
                )
            report = await service.reembed(
                project=project,
                arxiv_ids=list(arxiv_id) if arxiv_id else None,
                force=force,
                limit=limit,
            )
        except LookupError as exc:
            return _fail(str(exc))
        return _ok(
            space=report.space,
            model=report.model,
            dimensions=report.dimensions,
            papers_seen=report.papers_seen,
            papers_embedded=report.papers_embedded,
            papers_skipped=report.papers_skipped,
            chunks_embedded=report.chunks_embedded,
            seconds=round(report.seconds, 1),
            failures=report.failures,
        )

    @server.tool(
        name="chunk_kinds",
        title="Inspect or relabel chunk kinds",
        description=(
            "Every stored chunk carries a `content_kind` — body, abstract, "
            "figure, table, equation, reference, code — and that is what "
            "ask_paper_corpus and read_chunks filter on.\\n\\n"
            "`report` (default) shows the distribution and changes nothing. "
            "`relabel` re-classifies stored chunks, which is how a corpus "
            "ingested before a classifier fix picks up the correction without "
            "re-ingesting.\\n\\n"
            "Relabelling can only re-read text that was extracted; it cannot "
            "invent content that was never extracted. For that, re-ingest the "
            "paper with `force`."
        ),
        annotations=_ann(**_WRITES_IDEMPOTENT),
    )
    async def chunk_kinds(
        relabel: bool = False,
        scope: str | None = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        from sqlalchemy import func, select  # noqa: PLC0415

        from app.db.models import Chunk  # noqa: PLC0415
        from app.services.kind_backfill import backfill_chunk_kinds  # noqa: PLC0415

        container = _container()
        if not relabel:
            async with container.session_factory() as session:
                rows = (
                    await session.execute(
                        select(Chunk.content_kind, func.count())
                        .group_by(Chunk.content_kind)
                        .order_by(func.count().desc())
                    )
                ).all()
            return _ok(
                total=sum(count for _kind, count in rows),
                kinds=[{"kind": kind, "chunks": int(count)} for kind, count in rows],
            )

        async with container.session_factory() as session:
            paper_id = None
            if scope:
                from app.db.project_repository import ProjectRepository  # noqa: PLC0415
                from app.db.repositories import PaperRepository  # noqa: PLC0415

                projects = ProjectRepository(session)
                try:
                    found = await projects.get(scope)
                except Exception:  # noqa: BLE001
                    found = None
                if found is not None:
                    ids = await projects.paper_ids(found)
                    paper_id = ids[0] if ids else None
                else:
                    paper = await PaperRepository(session).get_by_arxiv_id(scope)
                    if paper is None:
                        return _fail(
                            f"{scope!r} is neither an ingested paper nor a project"
                        )
                    paper_id = paper.id
            counts = await backfill_chunk_kinds(session, only_body=False, paper_id=paper_id)
            await session.commit()
        return _ok(
            relabelled=True,
            scope=scope,
            scanned=sum(counts.values()),
            counts=counts,
        )

    @server.tool(
        name="set_paper_read",
        title="Mark project papers read",
        description=(
            "Toggle read state for one paper in a project, or for the whole "
            "project when `arxiv_id` is omitted. Reading a list is mostly "
            "sequencing, and this is how progress is kept."
        ),
        annotations=_ann(**_WRITES_IDEMPOTENT),
    )
    async def set_paper_read(
        project: str,
        arxiv_id: str | None = None,
        is_read: bool | None = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        from app.db.project_repository import PaperNotFoundError, ProjectRepository  # noqa: PLC0415

        container = _container()
        async with container.session_factory() as session:
            repo = ProjectRepository(session)
            try:
                found = await repo.require(project)
            except PaperNotFoundError as exc:
                return _fail(str(exc))
            if arxiv_id:
                # Membership, not just existence: a corpus paper the project does
                # not hold would update zero rows and still report success.
                targets, outside = await repo.link_paper_ids(found, [arxiv_id])
                if not targets:
                    return _fail(
                        f"{arxiv_id} is not in project {found.slug!r}. "
                        "Use import_papers_to_project first."
                    )
            else:
                targets = await repo.paper_ids(found)
            if not targets:
                return _ok(project=found.slug, changed=0, note="no papers in scope")
            current = {
                paper.id: link.is_read
                for paper, link in await repo.list_papers(found, limit=1000)
            }
            changed = 0
            for paper_id in targets:
                # Explicit `is_read` wins; otherwise toggle per row so a mixed
                # project flips each paper rather than flattening the list.
                target_state = (
                    not current.get(paper_id, False) if is_read is None else is_read
                )
                if await repo.set_read(found, paper_id, is_read=target_state):
                    changed += 1
            await session.commit()
        return _ok(project=found.slug, changed=changed)

    @server.tool(
        name="delete_project",
        title="Delete a project",
        description=(
            "Delete a project. **The papers survive** — a project is only a "
            "label over the global corpus, so this removes the reading list and "
            "nothing else. Use it to clean up a mistyped name."
        ),
        annotations=_ann(**_WRITES_DESTRUCTIVE),
    )
    async def delete_project(
        project: str,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        from app.db.project_repository import PaperNotFoundError, ProjectRepository  # noqa: PLC0415

        container = _container()
        async with container.session_factory() as session:
            repo = ProjectRepository(session)
            try:
                found = await repo.require(project)
            except PaperNotFoundError as exc:
                return _fail(str(exc))
            slug, name = found.slug, found.name
            papers = len(await repo.paper_ids(found))
            await repo.delete(found)
            await session.commit()
        return _ok(
            deleted=slug,
            name=name,
            papers_retained=papers,
            note="papers are global and were kept",
        )


__all__ = ["register"]
