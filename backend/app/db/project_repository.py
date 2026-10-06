"""Projects: named collections over the global paper corpus.

The central rule is that a project *references* papers, it does not own them.
Adding a paper to a second project inserts one junction row; the ``papers`` row,
its chunks and its vectors are shared, so the corpus stays single-copy and two
project views of a paper can never disagree.
"""

from __future__ import annotations

import re
import unicodedata
import uuid
from collections.abc import Sequence

from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.base import utcnow
from app.db.models import Paper, PaperCategory, Project, ProjectPaper
from app.domain.ids import normalize_arxiv_id as domain_normalize_arxiv_id
from app.domain.models import ProjectInfo
from app.logging import get_logger

logger = get_logger(__name__)


class ProjectConflictError(RuntimeError):
    """A project with that name or slug already exists."""


class PaperNotFoundError(LookupError):
    """No stored paper for that arXiv id."""


def slugify(name: str) -> str:
    """A URL- and shell-friendly handle from a human name.

    ``"Graph Neural Networks (2024)"`` -> ``"graph-neural-networks-2024"``.
    Accents are folded, because ``ş`` in a slug is a portability trap rather than
    a feature.
    """
    folded = unicodedata.normalize("NFKD", name)
    ascii_only = folded.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", ascii_only).strip("-").lower()
    return slug or "project"


class ProjectRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    # ------------------------------------------------------------------ write
    async def create(
        self,
        name: str,
        *,
        description: str | None = None,
        slug: str | None = None,
    ) -> Project:
        handle = slugify(slug or name)
        clash = await self._session.scalar(
            select(Project).where(Project.slug == handle).limit(1)
        )
        if clash is not None:
            raise ProjectConflictError(
                f"project slug {handle!r} is taken by {clash.name!r}; pass a different name"
            )
        project = Project(
            id=uuid.uuid4().hex,
            name=name.strip(),
            slug=handle,
            description=(description or "").strip() or None,
        )
        self._session.add(project)
        await self._session.flush()
        logger.info("project_created", extra={"project": project.slug})
        return project

    async def add_papers(
        self,
        project: Project,
        paper_ids: Sequence[str],
        *,
        added_by: str = "",
        note: str | None = None,
    ) -> int:
        """Link papers into a project, skipping ones already linked.

        Inserted with ``ON CONFLICT DO NOTHING`` semantics so importing the same
        list twice is a no-op rather than an error — re-running an import is the
        normal case.
        """
        if not paper_ids:
            return 0
        existing = set(
            (
                await self._session.scalars(
                    select(ProjectPaper.paper_id).where(
                        ProjectPaper.project_id == project.id,
                        ProjectPaper.paper_id.in_(paper_ids),
                    )
                )
            ).all()
        )
        rows = [
            ProjectPaper(
                id=uuid.uuid4().hex,
                project_id=project.id,
                paper_id=paper_id,
                added_by=added_by,
                note=note,
            )
            for paper_id in paper_ids
            if paper_id not in existing
        ]
        if rows:
            self._session.add_all(rows)
            await self._session.flush()
            project.updated_at = utcnow()
        logger.info(
            "project_papers_added",
            extra={"project": project.slug, "added": len(rows), "skipped": len(existing)},
        )
        return len(rows)

    async def remove_paper(self, project: Project, paper_id: str) -> bool:
        result = await self._session.execute(
            delete(ProjectPaper).where(
                ProjectPaper.project_id == project.id,
                ProjectPaper.paper_id == paper_id,
            )
        )
        await self._session.flush()
        removed = bool(getattr(result, "rowcount", 0))
        if removed:
            # The paper itself stays: it is global, and another project (or the
            # corpus at large) may still reference it.
            logger.info("project_paper_removed", extra={"project": project.slug})
        return removed

    async def set_note(self, project: Project, paper_id: str, note: str | None) -> bool:
        result = await self._session.execute(
            update(ProjectPaper)
            .where(
                ProjectPaper.project_id == project.id,
                ProjectPaper.paper_id == paper_id,
            )
            .values(note=note)
            .execution_options(synchronize_session=False)
        )
        await self._session.flush()
        return bool(getattr(result, "rowcount", 0))

    async def set_read(self, project: Project, paper_id: str, *, is_read: bool) -> bool:
        result = await self._session.execute(
            update(ProjectPaper)
            .where(
                ProjectPaper.project_id == project.id,
                ProjectPaper.paper_id == paper_id,
            )
            .values(is_read=is_read)
            .execution_options(synchronize_session=False)
        )
        await self._session.flush()
        return bool(getattr(result, "rowcount", 0))

    async def archive(self, project: Project, *, archived: bool = True) -> None:
        project.is_archived = archived
        project.updated_at = utcnow()
        await self._session.flush()

    async def delete(self, project: Project) -> None:
        """Remove the project. Its papers survive — they are global."""
        await self._session.delete(project)
        await self._session.flush()
        logger.info("project_deleted", extra={"project": project.slug})

    # ------------------------------------------------------------------- read
    async def get(self, identifier: str) -> Project | None:
        """Look up by slug first, then by exact name, then by internal id.

        The name fallback is what ``paper projects add "My Reading List"``
        promises, and what every project-scoped search inherits. Case-insensitive
        on slug *and* name, because people retype names with different casing;
        the internal id stays exact since it is machine-generated.
        """
        wanted = identifier.strip()
        result = await self._session.execute(
            select(Project).where(Project.slug == wanted.lower()).limit(1)
        )
        found = result.scalar_one_or_none()
        if found is not None:
            return found
        result = await self._session.execute(
            select(Project).where(func.lower(Project.name) == wanted.lower()).limit(1)
        )
        found = result.scalar_one_or_none()
        if found is not None:
            return found
        return await self._session.scalar(
            select(Project).where(Project.id == wanted).limit(1)
        )

    async def require(self, identifier: str) -> Project:
        project = await self.get(identifier)
        if project is None:
            raise PaperNotFoundError(f"no project matching {identifier!r}")
        return project

    async def paper_ids_for(self, identifier: str) -> list[str]:
        """Paper ids held by the named project. Raises when it does not exist."""
        project = await self.require(identifier)
        return await self.paper_ids(project)

    async def link_paper_ids(
        self, project: Project, identifiers: Sequence[str]
    ) -> tuple[list[str], list[str]]:
        """Split identifiers into those the project holds and those it does not.

        ``(in_project, not_in_project)``, each de-duplicated and in the order
        given. Needed because resolving an arXiv id says only that the *corpus*
        has it: without the membership check, "mark this paper read" on a paper
        outside the project updates zero rows and reports success, which reads
        as "marked" when nothing happened.
        """
        resolved, unresolved = await self.resolve_paper_ids(identifiers)
        if not resolved:
            return [], list(dict.fromkeys(unresolved))
        held = set(await self.paper_ids(project))
        in_project = [pid for pid in resolved if pid in held]
        outside = [pid for pid in resolved if pid not in held]
        # An unresolved id is also "not in this project"; report it rather than
        # swallowing it.
        outside.extend(unresolved)
        return in_project, list(dict.fromkeys(outside))

    async def list_projects(self, *, include_archived: bool = False) -> list[ProjectInfo]:
        """Every project, newest first, with paper and read counts."""
        paper_count = (
            select(func.count())
            .select_from(ProjectPaper)
            .where(ProjectPaper.project_id == Project.id)
            .correlate(Project)
            .scalar_subquery()
        )
        read_count = (
            select(func.count())
            .select_from(ProjectPaper)
            .where(
                ProjectPaper.project_id == Project.id,
                ProjectPaper.is_read.is_(True),
            )
            .correlate(Project)
            .scalar_subquery()
        )
        stmt = select(
            Project,
            paper_count.label("paper_count"),
            read_count.label("read_count"),
        )
        if not include_archived:
            stmt = stmt.where(Project.is_archived.is_(False))
        stmt = stmt.order_by(Project.created_at.desc())
        rows = (await self._session.execute(stmt)).all()
        return [
            ProjectInfo(
                id=project.id,
                name=project.name,
                slug=project.slug,
                description=project.description,
                is_archived=project.is_archived,
                paper_count=int(count or 0),
                read_count=int(read or 0),
                created_at=project.created_at,
            )
            for project, count, read in rows
        ]

    async def list_papers(
        self,
        project: Project,
        *,
        limit: int = 100,
        offset: int = 0,
        unread_only: bool = False,
        category: str | None = None,
    ) -> list[tuple[Paper, ProjectPaper]]:
        """Papers in a project, joined with their per-project link row."""
        stmt = (
            select(Paper, ProjectPaper)
            .join(ProjectPaper, ProjectPaper.paper_id == Paper.id)
            .where(ProjectPaper.project_id == project.id)
        )
        if unread_only:
            stmt = stmt.where(ProjectPaper.is_read.is_(False))
        if category:
            # EXISTS, not a join: a paper with several categories would otherwise
            # be duplicated.
            stmt = stmt.where(
                select(1)
                .select_from(PaperCategory)
                .where(
                    PaperCategory.paper_id == Paper.id,
                    PaperCategory.category == category,
                )
                .exists()
            )
        stmt = stmt.order_by(ProjectPaper.created_at.desc()).limit(limit).offset(offset)
        rows = (await self._session.execute(stmt)).all()
        return [(paper, link) for paper, link in rows]

    async def paper_ids(self, project: Project) -> list[str]:
        return list(
            (
                await self._session.scalars(
                    select(ProjectPaper.paper_id).where(ProjectPaper.project_id == project.id)
                )
            ).all()
        )

    async def projects_for_paper(self, paper_id: str) -> list[str]:
        """Which projects hold a paper. The reverse of the import direction."""
        stmt = (
            select(Project.slug)
            .join(ProjectPaper, ProjectPaper.project_id == Project.id)
            .where(ProjectPaper.paper_id == paper_id)
            .order_by(Project.slug)
        )
        return list((await self._session.scalars(stmt)).all())

    async def resolve_paper_ids(
        self, identifiers: Sequence[str]
    ) -> tuple[list[str], list[str]]:
        """Map user input onto stored paper ids: ``(resolved, unresolved)``.

        Import takes whatever the user has — ``1706.03762``, ``1706.03762v5``,
        an abs URL — so resolution belongs here rather than in every caller.

        Both lists come back because a caller almost always needs to say what it
        could not find: importing half a list silently is worse than refusing.
        Unresolved entries are returned in the order they were given, de-
        duplicated, and de-duplicated resolved ids keep their first position.
        """
        ordered = [i for i in dict.fromkeys(identifiers) if i and i.strip()]
        if not ordered:
            return [], []

        # Matching folds case on both sides; the stored value is what is kept.
        lowered = {i.strip().lower() for i in ordered}
        bare = {normalize_arxiv_reference(i).lower() for i in ordered}
        rows = (
            await self._session.execute(
                select(Paper.id, Paper.arxiv_id, Paper.versioned_id).where(
                    func.lower(Paper.arxiv_id).in_({b for b in bare if b})
                    | func.lower(Paper.versioned_id).in_(lowered)
                    | Paper.id.in_(ordered)
                )
            )
        ).all()

        # One lookup table for every accepted spelling of a paper.
        by_key: dict[str, str] = {}
        for paper_id, arxiv_id, versioned_id in rows:
            for key in (paper_id, arxiv_id, versioned_id):
                by_key[key] = paper_id
                by_key[key.lower()] = paper_id
            by_key[normalize_arxiv_reference(arxiv_id).lower()] = paper_id
            by_key[normalize_arxiv_reference(versioned_id).lower()] = paper_id

        resolved: list[str] = []
        unresolved: list[str] = []
        for original in ordered:
            match: str | None = by_key.get(original) or by_key.get(
                normalize_arxiv_reference(original).lower()
            )
            if match is None:
                unresolved.append(original)
            elif match not in resolved:
                resolved.append(match)
        return resolved, unresolved


def normalize_arxiv_reference(value: str) -> str:
    """``arXiv:1706.03762v5``, an abs URL, or a bare id -> ``1706.03762``.

    Case is preserved: old-style ids are ``math.CO/0309136``, so lowercasing
    would make them unmatchable against the stored ``arxiv_id``. Callers that
    need a case-insensitive comparison fold both sides themselves.
    """
    try:
        return domain_normalize_arxiv_id(value)
    except ValueError:
        return value.strip().lower()


__all__ = [
    "ProjectRepository",
    "ProjectConflictError",
    "PaperNotFoundError",
    "slugify",
    "normalize_arxiv_reference",
]