"""Data access. Repositories hide SQL from services and pipeline steps."""

from __future__ import annotations

import uuid
from collections.abc import Iterable, Sequence
from datetime import datetime
from typing import Any

from sqlalchemy import Table, and_, delete, exists, func, or_, select, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from sqlalchemy.sql import Select

from paper_app.db.base import utcnow
from paper_app.db.models import (
    Author as AuthorRow,
)
from paper_app.db.models import (
    Category,
    Chunk,
    DocumentSection,
    IngestionRun,
    Paper,
    PaperAuthor,
    PaperCategory,
    PipelineStepRun,
    RawDocument,
)
from paper_app.domain.enums import ContentKind, RunStatus
from paper_app.domain.models import (
    Author,
    PaperMetadata,
    SearchHitWithScore,
    TextChunk,
)
from paper_app.logging import get_logger

logger = get_logger(__name__)


def _new_id() -> str:
    return uuid.uuid4().hex


def _bibliographic_to_natural(normalized: str) -> str:
    """``"vaswani, ashish"`` -> ``"ashish vaswani"``.

    ArXiv itself writes ``family, given``, so both orders have to resolve to the
    same stored row.
    """
    if "," not in normalized:
        return normalized
    family, _, given = normalized.partition(",")
    given = given.strip()
    if not given:
        return normalized
    return f"{given} {family.strip()}"


def normalize_author_name(name: str) -> str:
    """Fold a display name to a comparable key.

    ArXiv has no author identifier, so this is what keeps one person's papers
    together across spelling and spacing differences. Case-folded and
    whitespace-collapsed; accents are left alone because folding them would
    need a Unicode table and would merge genuinely different spellings.
    """
    return " ".join(name.split()).casefold()


class PaperRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    @property
    def session(self) -> AsyncSession:
        """The underlying session.

        Exposed because callers occasionally need the same transaction for
        their own statement (a bulk write, or a count across the junction
        tables) and opening a second session would not see uncommitted work.
        """
        return self._session

    async def upsert(self, metadata: PaperMetadata) -> Paper:
        """Insert or refresh a row from source metadata.

        Keyed on ``doc_key`` rather than on ``arxiv_id``: a paper's key is its
        arXiv id, a local file's is a slug, and both go through here. Authors and
        categories are written to their own tables via :meth:`_sync_authors` /
        :meth:`_sync_categories`, which is why they are absent from the column
        dict below.
        """
        existing = await self._find_existing(metadata)
        values: dict[str, Any] = {
            "doc_key": metadata.doc_key,
            "kind": metadata.kind,
            "arxiv_id": metadata.arxiv_id,
            "versioned_id": metadata.versioned_id,
            "version": metadata.version,
            "title": metadata.title,
            "abstract": metadata.abstract,
            "primary_category": metadata.primary_category,
            "published_at": metadata.published_at,
            "updated_at_arxiv": metadata.updated_at,
            "doi": metadata.doi,
            "comment": metadata.comment,
            "journal_ref": metadata.journal_ref,
            "abs_url": metadata.abs_url,
            "pdf_url": metadata.pdf_url,
            "html_url": metadata.html_url,
            "raw_entry": metadata.raw or None,
            "file_sha256": metadata.file_sha256,
            "page_count": metadata.page_count,
            "updated_at": utcnow(),
        }
        if existing is None:
            # Ids are assigned up front so the whole graph can be built while the
            # row is still pending. Appending to a relationship of a *flushed*
            # object makes SQLAlchemy load the existing collection first, and
            # under `lazy="selectin"` that load happens outside `await` and
            # raises MissingGreenlet.
            paper = Paper(id=_new_id(), **values)
            self._session.add(paper)
            await self._write_authors(paper, metadata, replace=False)
            await self._write_categories(paper, metadata, replace=False)
            await self._session.flush()
            # The caller reads `categories` / `author_names` on the returned row
            # straight away, so load them here under `await` rather than letting
            # the property trigger IO from sync context.
            await self._session.refresh(paper, ["author_links", "category_links"])
            logger.info("paper_created", extra={"arxiv_id": metadata.arxiv_id})
            return paper

        # A newer version on ArXiv supersedes the local copy.
        if (
            existing.version is not None
            and metadata.version is not None
            and metadata.version < existing.version
        ):
            logger.info(
                "paper_version_ignored",
                extra={
                    "arxiv_id": metadata.arxiv_id,
                    "local": existing.version,
                    "remote": metadata.version,
                },
            )
            return existing
        for key, value in values.items():
            setattr(existing, key, value)
        await self._write_authors(existing, metadata, replace=True)
        await self._write_categories(existing, metadata, replace=True)
        await self._session.flush()
        await self._session.refresh(existing, ["author_links", "category_links"])
        logger.info("paper_updated", extra={"arxiv_id": metadata.arxiv_id, "version": metadata.version})
        return existing

    async def _write_authors(
        self, paper: Paper, metadata: PaperMetadata, *, replace: bool
    ) -> None:
        """Write this paper's author rows, reusing existing author records.

        A version bump can reorder or add authors, so an update replaces the
        whole list; ``ordinal`` makes it a sequence rather than a set.

        Rows are added by ``paper_id`` rather than appended to the relationship.
        Appending would make SQLAlchemy load the collection first, and under
        ``lazy="selectin"`` that load runs outside ``await`` and raises
        MissingGreenlet.
        """
        if replace:
            await self._session.refresh(paper, ["author_links"])
            paper.author_links.clear()
            await self._session.flush()

        rows: list[PaperAuthor] = []
        for ordinal, author in enumerate(metadata.authors):
            author_row = await self._find_or_create_author(author.name)
            rows.append(
                PaperAuthor(
                    id=_new_id(),
                    paper_id=paper.id,
                    author_id=author_row.id,
                    ordinal=ordinal,
                    affiliation=author.affiliation,
                )
            )
        self._session.add_all(rows)

    async def _find_or_create_author(self, name: str) -> AuthorRow:
        """One row per person, matched on a normalized form of the name."""
        normalized = normalize_author_name(name)
        existing = await self._session.scalar(
            select(AuthorRow).where(AuthorRow.normalized_name == normalized).limit(1)
        )
        if existing is not None:
            # Keep the first-seen spelling; ArXiv's own capitalisation is what
            # readers expect to see back.
            return existing
        # Id assigned here rather than by the column default, so the caller can
        # build its link rows before any flush is needed.
        row = AuthorRow(id=_new_id(), name=name.strip(), normalized_name=normalized)
        self._session.add(row)
        await self._session.flush()
        return row

    async def _ensure_category(self, code: str) -> Category:
        """Register a category code if it is new, deriving its taxonomy place.

        ``cs.CL.MS`` belongs to ``cs.CL`` which belongs to ``cs``; storing the
        chain means a query can select a whole subtree without string surgery in
        Python.
        """
        existing = await self._session.get(Category, code)
        if existing is not None:
            return existing
        parts = code.split(".")
        row = Category(
            code=code,
            parent=".".join(parts[:-1]) or None,
            depth=len(parts) - 1,
        )
        self._session.add(row)
        # Register the ancestors too, so parent is always resolvable.
        for level in range(1, len(parts)):
            ancestor = ".".join(parts[:level])
            if await self._session.get(Category, ancestor) is None:
                self._session.add(
                    Category(code=ancestor, parent=".".join(parts[:level - 1]) or None, depth=level - 1)
                )
        await self._session.flush()
        return row

    async def _write_categories(
        self, paper: Paper, metadata: PaperMetadata, *, replace: bool
    ) -> None:
        """Write this paper's category rows, registering each code first."""
        if replace:
            await self._session.refresh(paper, ["category_links"])
            paper.category_links.clear()
            await self._session.flush()

        primary = metadata.primary_category
        rows: list[PaperCategory] = []
        for ordinal, code in enumerate(metadata.categories):
            await self._ensure_category(code)
            rows.append(
                PaperCategory(
                    id=_new_id(),
                    paper_id=paper.id,
                    category=code,
                    ordinal=ordinal,
                    is_primary=code == primary,
                )
            )
        self._session.add_all(rows)

    async def list_categories(self) -> list[dict[str, Any]]:
        """Every registered category with its paper count.

        Counted by a join rather than a stored column, so the number cannot drift
        when a paper is re-ingested or deleted.
        """
        stmt = (
            select(
                Category.code,
                Category.parent,
                Category.depth,
                func.count(PaperCategory.paper_id).label("paper_count"),
            )
            .outerjoin(PaperCategory, PaperCategory.category == Category.code)
            .group_by(Category.code)
            .order_by(Category.code)
        )
        result = await self._session.execute(stmt)
        return [
            {"code": code, "parent": parent, "depth": depth, "paper_count": int(count)}
            for code, parent, depth, count in result.all()
        ]

    async def list_authors(self, *, limit: int = 50) -> list[dict[str, Any]]:
        """Most prolific authors, by paper count.

        The join is what makes the table worth having: an author's papers are
        reachable by one id, not a substring search over stored JSON.
        """
        stmt = (
            select(
                AuthorRow.id,
                AuthorRow.name,
                func.count(PaperAuthor.paper_id).label("paper_count"),
            )
            .join(PaperAuthor, PaperAuthor.author_id == AuthorRow.id)
            .group_by(AuthorRow.id)
            .order_by(func.count(PaperAuthor.paper_id).desc(), AuthorRow.name)
            .limit(limit)
        )
        result = await self._session.execute(stmt)
        return [
            {"id": row_id, "name": name, "paper_count": int(count)}
            for row_id, name, count in result.all()
        ]

    async def find_author(self, name: str) -> dict[str, Any] | None:
        """Resolve a loose author string to a stored row, or None.

        A partial match is allowed and tried in decreasing order of confidence:
        the whole string first (so ``"Ashish Vaswani"`` cannot be confused with
        a different ``Ashish``), then any ``last, first`` form, then a unique
        surname alone. Surname-only is only accepted when it identifies exactly
        one person — guessing between several would silently return the wrong
        author's papers.
        """
        normalized = normalize_author_name(name)
        if not normalized:
            return None

        # Stored names are normalized, so "Vaswani, Ashish" becomes
        # "ashish vaswani" after separators are folded — otherwise the
        # bibliographic form would never match the plain one.
        candidates = [normalized, _bibliographic_to_natural(normalized)]
        for candidate in dict.fromkeys(candidates):
            match = await self._author_row_by_normalized(candidate)
            if match is not None:
                return match
        if " " not in normalized:
            rows = await self._session.execute(
                select(
                    AuthorRow.id,
                    AuthorRow.name,
                    func.count(PaperAuthor.paper_id).label("paper_count"),
                )
                .join(PaperAuthor, PaperAuthor.author_id == AuthorRow.id)
                .where(AuthorRow.normalized_name.like(f"% {normalized}"))
                .group_by(AuthorRow.id)
                .having(func.count(PaperAuthor.paper_id) > 0)
            )
            unique = rows.all()
            if len(unique) == 1:
                return {
                    "id": unique[0][0],
                    "name": unique[0][1],
                    "paper_count": int(unique[0][2]),
                }
        return None

    async def _author_row_by_normalized(self, normalized: str) -> dict[str, Any] | None:
        row = (
            await self._session.execute(
                select(
                    AuthorRow.id,
                    AuthorRow.name,
                    func.count(PaperAuthor.paper_id).label("paper_count"),
                )
                .join(PaperAuthor, PaperAuthor.author_id == AuthorRow.id)
                .where(AuthorRow.normalized_name == normalized)
                .group_by(AuthorRow.id)
            )
        ).first()
        return (
            {"id": row[0], "name": row[1], "paper_count": int(row[2])} if row else None
        )

    async def papers_by_author(self, name: str) -> list[Paper]:
        """Every paper by one author.

        Resolves the name with :meth:`find_author` first, so a surname or a
        ``last, first`` spelling works here too, then joins on the resolved id.
        """
        match = await self.find_author(name)
        if match is None:
            return []
        stmt = (
            select(Paper)
            .join(PaperAuthor, PaperAuthor.paper_id == Paper.id)
            .where(PaperAuthor.author_id == match["id"])
            .order_by(Paper.published_at.desc().nullslast())
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().unique().all())

    async def descendant_categories(self, code: str) -> list[str]:
        """``code`` plus every category beneath it in the taxonomy.

        Walks the whole subtree with a recursive CTE rather than one level of
        ``parent``. ArXiv's taxonomy is three deep (``cs.CL.MS`` -> ``cs.CL`` ->
        ``cs``), and a single-level lookup would miss the deepest papers when
        someone asks for ``cs``.
        """
        # `::VARCHAR(32)` pins the seed's type. Without it PostgreSQL infers
        # `text` for the parameter and then refuses the query outright:
        # "recursive query column 1 has type text in non-recursive term but type
        # character varying overall".
        stmt = text(
            """
            WITH RECURSIVE subtree(code) AS (
                SELECT CAST(:root AS VARCHAR(32))
                UNION
                SELECT c.code FROM categories c JOIN subtree s ON c.parent = s.code
            )
            SELECT code FROM subtree
            """
        )
        result = await self._session.execute(stmt, {"root": code})
        return list(result.scalars().all())

    async def papers_in_category(
        self,
        code: str,
        *,
        include_subcategories: bool = False,
    ) -> list[Paper]:
        """Papers filed under ``code``.

        With ``include_subcategories`` the match covers the taxonomy subtree, so
        ``cs`` also returns every ``cs.CL.MS`` paper. That works because
        :class:`Category` stores the parent chain, not because the code happens
        to share a string prefix.
        """
        codes = await self.descendant_categories(code) if include_subcategories else [code]
        stmt = (
            select(Paper)
            .join(PaperCategory, PaperCategory.paper_id == Paper.id)
            .where(PaperCategory.category.in_(codes))
            .order_by(Paper.published_at.desc().nullslast())
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().unique().all())

    async def get(self, paper_id: str) -> Paper | None:
        return await self._session.get(Paper, paper_id)

    async def _find_existing(self, metadata: PaperMetadata) -> Paper | None:
        """The row this metadata belongs to, if it is already in the corpus.

        Looked up by ``doc_key`` first, because that is the universal handle and
        the one column every row has. Looking it up by ``arxiv_id`` alone looks
        equivalent and is not: a local document has no arXiv id, so the lookup
        returned nothing and the upsert fell through to an INSERT — which failed
        on the ``doc_key`` unique constraint the moment the same file was
        ingested twice with ``--force``. Found the hard way, on the second ingest
        of a book, as a ``PendingRollbackError`` wrapping the real violation.
        """
        found = await self.get_by_doc_key(metadata.doc_key)
        if found is not None:
            return found
        if metadata.arxiv_id:
            return await self.get_by_arxiv_id(metadata.arxiv_id)
        return None

    async def get_by_doc_key(self, doc_key: str) -> Paper | None:
        """Find a row by its universal handle, exactly as stored."""
        result = await self._session.execute(
            select(Paper).where(Paper.doc_key == doc_key).limit(1)
        )
        return result.scalar_one_or_none()

    async def resolve(self, identifier: str) -> Paper | None:
        """Find a row by anything that names it: a doc_key, an arXiv id or an id.

        The single lookup callers should use, because an ingest argument, a
        project membership or a URL path is any of the three. ``get`` stays the
        strict internal-id read and ``get_by_arxiv_id`` the strict arXiv read, so
        code that means something specific can still say so.
        """
        wanted = identifier.strip()
        if not wanted:
            return None
        result = await self._session.execute(
            select(Paper).where(Paper.doc_key == wanted).limit(1)
        )
        found = result.scalar_one_or_none()
        if found is not None:
            return found
        found = await self.get_by_arxiv_id(wanted)
        if found is not None:
            return found
        return await self.get(wanted)

    async def list_documents(
        self, *, kind: str | None = None, limit: int = 100
    ) -> list[Paper]:
        """The local files in the corpus, newest first.

        Separate from the arXiv listing because a caller asking "what books do I
        have" should not have to know that papers and documents share a table.
        """
        stmt = select(Paper).where(Paper.arxiv_id.is_(None))
        if kind:
            stmt = stmt.where(Paper.kind == kind)
        result = await self._session.execute(
            stmt.order_by(Paper.ingested_at.desc().nullslast()).limit(limit)
        )
        return list(result.scalars().unique().all())

    async def get_by_file_sha256(self, sha256: str) -> Paper | None:
        """Find a document by the hash of its source file.

        The dedupe check behind "this book is already in the corpus". Deliberately
        a method rather than a unique index: the same file may be ingested twice
        on purpose (after a failed run, or as a second copy under another name),
        and a constraint would make ``--force`` impossible.
        """
        result = await self._session.execute(
            select(Paper).where(Paper.file_sha256 == sha256).limit(1)
        )
        return result.scalar_one_or_none()

    async def get_by_arxiv_id(self, arxiv_id: str) -> Paper | None:
        """Find a *paper* by any spelling of its arXiv id.

        Returns None for anything that is not an arXiv paper — a local document
        has no arXiv id, and reporting it here would be a lie.

        Normalisation lives here rather than at each call site because the
        column stores the *versionless* id while people type every other form.
        `paper ingest 2408.05245v1` stores ``2408.05245``, and asking for
        `paper show 2408.05245v1` used to compare the raw string against that
        column, miss, and report "not ingested" for a paper that was ingested
        seconds earlier.

        Also matches the exact ``versioned_id``, which is what lets a paper
        stored under one version be found by another spelling of itself.
        Unparseable input returns None instead of raising: this is a lookup,
        and a lookup miss is not an error.
        """
        from paper_app.domain.ids import normalize_arxiv_id  # noqa: PLC0415

        try:
            canonical = normalize_arxiv_id(arxiv_id)
        except ValueError:
            return None
        result = await self._session.execute(
            select(Paper)
            .where(
                or_(
                    Paper.arxiv_id == canonical,
                    Paper.versioned_id == arxiv_id.strip(),
                )
            )
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def list_papers(
        self,
        *,
        limit: int = 20,
        offset: int = 0,
        category: str | None = None,
        ingested_only: bool = False,
    ) -> tuple[Sequence[Paper], int]:
        stmt = select(Paper)
        if category:
            # EXISTS rather than a join: the relationship is one-to-many, so a
            # join would multiply paper rows and break the count below.
            stmt = stmt.where(
                exists().where(
                    and_(
                        PaperCategory.paper_id == Paper.id,
                        PaperCategory.category == category,
                    )
                )
            )
        if ingested_only:
            stmt = stmt.where(Paper.ingested_at.is_not(None))
        total = await self._session.scalar(
            select(func.count()).select_from(stmt.subquery())
        )
        stmt = stmt.order_by(Paper.ingested_at.desc().nullslast(), Paper.created_at.desc())
        stmt = stmt.limit(limit).offset(offset)
        result = await self._session.execute(stmt)
        return result.scalars().unique().all(), int(total or 0)

    async def all_arxiv_ids(self) -> Select:
        """Every arXiv id we hold. Used by the `ingested` search post-filter."""
        return select(Paper.arxiv_id)

    async def mark_ingested(self, paper_id: str) -> None:
        paper = await self.get(paper_id)
        if paper is not None:
            paper.ingested_at = utcnow()
            await self._session.flush()

    async def chunk_count(self, paper_id: str) -> int:
        return int(
            await self._session.scalar(
                select(func.count()).select_from(Chunk).where(Chunk.paper_id == paper_id)
            )
            or 0
        )


class RawDocumentRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def record(
        self,
        *,
        paper_id: str,
        kind: ContentKind | str,
        uri: str,
        content_type: str,
        size_bytes: int,
        sha256: str,
        source_url: str = "",
        page_start: int | None = None,
        page_end: int | None = None,
        meta: dict[str, Any] | None = None,
    ) -> RawDocument:
        kind_value = str(kind)
        # `page_start` is part of the identity, matching the table's unique key:
        # the same book ingested as 1-200 and then 201-400 is two sources, not
        # one source seen twice, and returning the first for the second would
        # make the second range a silent no-op.
        existing = await self._session.scalar(
            select(RawDocument).where(
                RawDocument.paper_id == paper_id,
                RawDocument.kind == kind_value,
                RawDocument.sha256 == sha256,
                RawDocument.page_start.is_(page_start)
                if page_start is None
                else RawDocument.page_start == page_start,
            )
        )
        if existing is not None:
            # The row is returned, but its metadata is refreshed. Skipping the
            # update loses facts that are derived rather than content-addressed:
            # the block list written alongside the markdown was being dropped on
            # every re-ingest of an identical document, because the markdown hash
            # matched the existing row and the early return discarded the new
            # meta. Content-addressed fields need no rewrite — they cannot differ
            # for the same hash — but `meta` can gain keys between runs.
            if meta and (existing.meta or {}) != meta:
                existing.meta = {**(existing.meta or {}), **meta}
                await self._session.flush()
            return existing
        document = RawDocument(
            paper_id=paper_id,
            kind=kind_value,
            uri=uri,
            content_type=content_type,
            size_bytes=size_bytes,
            sha256=sha256,
            source_url=source_url,
            page_start=page_start,
            page_end=page_end,
            meta=meta or {},
        )
        self._session.add(document)
        await self._session.flush()
        return document

    async def list_for_paper(self, paper_id: str, kind: str | None = None) -> Sequence[RawDocument]:
        stmt = select(RawDocument).where(RawDocument.paper_id == paper_id)
        if kind:
            stmt = stmt.where(RawDocument.kind == str(kind))
        result = await self._session.execute(stmt.order_by(RawDocument.created_at))
        return result.scalars().all()

    async def has_source(self, paper_id: str, kinds: set[str]) -> bool:
        count = await self._session.scalar(
            select(func.count())
            .select_from(RawDocument)
            .where(
                RawDocument.paper_id == paper_id,
                RawDocument.kind.in_([str(k) for k in kinds]),
            )
        )
        return bool(count)


class SectionRepository:
    """A document's own structure: chapters, sections, and the pages they span."""

    def __init__(self, session: Any) -> None:
        self._session = session

    async def replace_for_paper(self, paper_id: str, sections: Sequence[Any]) -> list[DocumentSection]:
        """Delete-then-insert so ordinals stay dense and deterministic."""
        await self._session.execute(
            delete(DocumentSection).where(DocumentSection.paper_id == paper_id)
        )
        rows = [
            DocumentSection(
                paper_id=paper_id,
                ordinal=section.ordinal,
                title=section.title,
                level=section.level,
                page_start=section.page_start,
                page_end=section.page_end,
                source=section.source,
            )
            for section in sections
        ]
        self._session.add_all(rows)
        await self._session.flush()
        return rows

    async def list_for_section(
        self, paper_id: str, ordinal: int
    ) -> list[Chunk]:
        """The chunks of one section, in document order.

        Section content is the chunks that point at it, not the pages it spans.
        The two differ and the difference matters: a section can span ten pages of
        which only four produced text, and printing the page range would promise
        material the corpus does not hold.
        """
        result = await self._session.execute(
            select(Chunk)
            .where(Chunk.paper_id == paper_id, Chunk.section_ordinal == ordinal)
            .order_by(Chunk.ordinal)
        )
        return list(result.scalars().all())

    async def count_chunks_in_sections(
        self, paper_id: str, ordinals: Sequence[int]
    ) -> dict[int, int]:
        """How many chunks each of several sections holds, in one query.

        For a table of contents with a chunk column: N sections means N queries
        otherwise, and the number is on every row.
        """
        if not ordinals:
            return {}
        result = await self._session.execute(
            select(Chunk.section_ordinal, func.count())
            .where(Chunk.paper_id == paper_id, Chunk.section_ordinal.in_(list(ordinals)))
            .group_by(Chunk.section_ordinal)
        )
        return {ordinal: int(count) for ordinal, count in result.all()}

    async def list_for_paper(
        self, paper_id: str, *, max_level: int | None = None
    ) -> Sequence[DocumentSection]:
        stmt = select(DocumentSection).where(DocumentSection.paper_id == paper_id)
        if max_level is not None:
            stmt = stmt.where(DocumentSection.level <= max_level)
        result = await self._session.execute(stmt.order_by(DocumentSection.ordinal))
        return result.scalars().all()


class ChunkRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def replace_for_paper(
        self,
        paper_id: str,
        chunks: Sequence[TextChunk],
        raw_document_id: str | None = None,
    ) -> list[Chunk]:
        """Delete-then-insert so ordinals stay dense and deterministic."""
        await self._session.execute(delete(Chunk).where(Chunk.paper_id == paper_id))
        rows = [
            Chunk(
                paper_id=paper_id,
                raw_document_id=raw_document_id,
                ordinal=chunk.ordinal,
                text=chunk.text,
                token_count=chunk.token_count,
                char_start=chunk.char_start,
                char_end=chunk.char_end,
                heading=chunk.heading,
                section_path=list(chunk.section_path),
                content_hash=chunk.content_hash,
                source=str(chunk.source),
                content_kind=str(chunk.kind),
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                section_ordinal=chunk.section_ordinal,
            )
            for chunk in chunks
        ]
        self._session.add_all(rows)
        await self._session.flush()
        return rows

    async def assign_sections(
        self, chunks: Sequence[Chunk], sections: Sequence[DocumentSection]
    ) -> int:
        """Point each chunk at the section its page falls inside.

        Matched on page, not on the chunk's text: a section is a page range, and
        the text of a chunk rarely starts where the section does. The *tightest*
        containing section wins, so a subsection is preferred over its chapter —
        which is what makes "show me only this subsection" mean what it says.

        Chunks with no page, and pages covered by no section, are left unset
        rather than given a default: a wrong section is worse than none, because
        a filter that quietly excludes things looks like a filter that found
        nothing.
        """
        if not sections or not chunks:
            return 0
        usable = [s for s in sections if s.page_start is not None]
        if not usable:
            return 0
        # Sorted widest-first so the first containing match is the tightest.
        usable.sort(key=lambda s: (s.page_start or 0, -(s.level or 1)))
        starts = [s.page_start or 0 for s in usable]

        import bisect  # noqa: PLC0415 - only needed here

        assigned = 0
        for chunk in chunks:
            if chunk.page_start is None:
                continue
            index = bisect.bisect_right(starts, chunk.page_start) - 1
            for candidate in reversed(usable[: index + 1]):
                end = candidate.page_end
                if end is None or chunk.page_start <= end:
                    chunk.section_ordinal = candidate.ordinal
                    assigned += 1
                    break
        await self._session.flush()
        return assigned

    async def list_for_paper(
        self,
        paper_id: str,
        *,
        limit: int | None = None,
        offset: int = 0,
        content_kinds: Sequence[str] | None = None,
    ) -> Sequence[Chunk]:
        """One paper's chunks in reading order.

        Args:
            content_kinds: restrict to these ``content_kind`` values. An empty
                sequence matches nothing, matching
                :class:`~paper_app.db.vector_store.base.VectorFilter`.
        """
        stmt = select(Chunk).where(Chunk.paper_id == paper_id)
        if content_kinds is not None:
            stmt = stmt.where(Chunk.content_kind.in_(list(content_kinds)))
        stmt = stmt.order_by(Chunk.ordinal)
        # `is not None`, not truthiness: `limit=0` means "return nothing", and
        # treating it as "no limit" hands back the whole paper.
        if limit is not None:
            stmt = stmt.limit(limit).offset(offset)
        result = await self._session.execute(stmt)
        return result.scalars().all()

    async def count(
        self, paper_id: str | None = None, *, content_kinds: Sequence[str] | None = None
    ) -> int:
        stmt = select(func.count()).select_from(Chunk)
        if paper_id:
            stmt = stmt.where(Chunk.paper_id == paper_id)
        if content_kinds is not None:
            stmt = stmt.where(Chunk.content_kind.in_(list(content_kinds)))
        return int(await self._session.scalar(stmt) or 0)

    async def get_many(self, chunk_ids: Sequence[str]) -> Sequence[Chunk]:
        if not chunk_ids:
            return []
        result = await self._session.execute(
            select(Chunk).where(Chunk.id.in_(chunk_ids))
        )
        return result.scalars().all()


class EmbeddingRepository:
    """Vectors for one embedding space.

    Core-level rather than ORM-level because space tables are built at runtime
    (one table per model) and therefore have no mapped class.
    """

    def __init__(self, session: AsyncSession, table: Table) -> None:
        self._session = session
        self._table = table

    @property
    def table(self) -> Table:
        return self._table

    async def replace_for_paper(
        self,
        paper_id: str,
        records: Sequence[tuple[Chunk, str, str, int, list[float], str]],
    ) -> int:
        """``records`` is ``(chunk, provider, model, dims, vector, fingerprint)``.

        Replaces only this space's rows: sibling spaces keep their vectors.
        """
        await self._session.execute(
            delete(self._table).where(self._table.c.paper_id == paper_id)
        )
        rows = [
            {
                "id": uuid.uuid4().hex,
                "chunk_id": chunk.id,
                "paper_id": paper_id,
                "provider": provider,
                "model": model,
                "dimensions": dimensions,
                "fingerprint": fingerprint,
                "vector": vector,
                "meta": {},
                "created_at": utcnow(),
            }
            for chunk, provider, model, dimensions, vector, fingerprint in records
        ]
        if rows:
            await self._session.execute(self._table.insert(), rows)
        return len(rows)

    async def count(self, fingerprint: str | None = None) -> int:
        stmt = select(func.count()).select_from(self._table)
        if fingerprint:
            stmt = stmt.where(self._table.c.fingerprint == fingerprint)
        return int(await self._session.scalar(stmt) or 0)

    async def stale_count(self, fingerprint: str) -> int:
        """Embeddings recorded under a different model fingerprint."""
        return int(
            await self._session.scalar(
                select(func.count())
                .select_from(self._table)
                .where(self._table.c.fingerprint != fingerprint)
            )
            or 0
        )

    async def count_for_paper(self, paper_id: str) -> int:
        """Rows this space holds for one paper; 0 means "never embedded here"."""
        return int(
            await self._session.scalar(
                select(func.count())
                .select_from(self._table)
                .where(self._table.c.paper_id == paper_id)
            )
            or 0
        )

    async def embedded_paper_ids(self) -> set[str]:
        """Papers that already have rows in this space's table.

        Used by re-embedding to skip work already done, which is the difference
        between a five-minute backfill and a five-hour one.
        """
        result = await self._session.execute(select(self._table.c.paper_id).distinct())
        return set(result.scalars().all())

    async def unembedded_chunk_ids(self, paper_id: str) -> Sequence[str]:
        result = await self._session.execute(
            select(Chunk.id)
            .select_from(Chunk)
            .outerjoin(self._table, self._table.c.chunk_id == Chunk.id)
            .where(Chunk.paper_id == paper_id, self._table.c.chunk_id.is_(None))
        )
        return result.scalars().all()

    async def delete_stale(self, fingerprint: str) -> int:
        result = await self._session.execute(
            delete(self._table).where(self._table.c.fingerprint != fingerprint)
        )
        return int(getattr(result, "rowcount", 0) or 0)

    async def load_vectors(self, chunk_ids: Sequence[str]) -> dict[str, list[float]]:
        if not chunk_ids:
            return {}
        result = await self._session.execute(
            select(self._table.c.chunk_id, self._table.c.vector).where(
                self._table.c.chunk_id.in_(list(chunk_ids))
            )
        )
        return {chunk_id: list(vector or []) for chunk_id, vector in result.all()}


class RunRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self,
        *,
        arxiv_id: str | None = None,
        doc_key: str | None = None,
        requested_by: str = "api",
        trigger: str = "manual",
        prefer_html: bool = True,
        force: bool = False,
    ) -> IngestionRun:
        run = IngestionRun(
            arxiv_id=arxiv_id,
            # A run of a local file has no arXiv id, and `doc_key` is what makes
            # it findable by the same argument an ingest takes.
            doc_key=doc_key,
            requested_by=requested_by,
            trigger=trigger,
            prefer_html=prefer_html,
            force=force,
            status=RunStatus.PENDING,
        )
        self._session.add(run)
        await self._session.flush()
        return run

    async def get(self, run_id: str) -> IngestionRun | None:
        result = await self._session.execute(
            select(IngestionRun)
            .where(IngestionRun.id == run_id)
            .options(selectinload(IngestionRun.steps))
        )
        return result.scalar_one_or_none()

    async def mark_running(self, run_id: str, paper_id: str | None = None) -> None:
        run = await self.get(run_id)
        if run is None:
            return
        run.status = RunStatus.RUNNING
        run.started_at = utcnow()
        if paper_id:
            run.paper_id = paper_id
        await self._session.flush()

    async def finish(
        self,
        run_id: str,
        status: RunStatus,
        *,
        error: str | None = None,
        timings: dict[str, Any] | None = None,
        chunk_count: int | None = None,
        embedding_count: int | None = None,
        content_kind: str | None = None,
    ) -> None:
        run = await self.get(run_id)
        if run is None:
            return
        run.status = status
        run.finished_at = utcnow()
        run.error = error
        if timings:
            run.timings = {**(run.timings or {}), **timings}
        if chunk_count is not None:
            run.chunk_count = chunk_count
        if embedding_count is not None:
            run.embedding_count = embedding_count
        if content_kind:
            run.content_kind = content_kind
        await self._session.flush()

    async def add_step(
        self,
        run_id: str,
        *,
        name: str,
        status: str,
        attempt: int = 1,
        duration_ms: float | None = None,
        error: str | None = None,
        meta: dict[str, Any] | None = None,
        started_at: datetime | None = None,
    ) -> PipelineStepRun:
        step = PipelineStepRun(
            run_id=run_id,
            name=name,
            status=status,
            attempt=attempt,
            duration_ms=duration_ms,
            error=error,
            meta=meta or {},
            started_at=started_at or utcnow(),
            finished_at=utcnow() if status in {"succeeded", "failed", "skipped", "cached"} else None,
        )
        self._session.add(step)
        await self._session.flush()
        return step

    async def latest_for_target(self, target: str) -> IngestionRun | None:
        """The most recent run for an arXiv id or a document handle.

        Answers "was the last attempt at this finished?" — which is what decides
        between skipping an already-ingested document and resuming it.
        """
        result = await self._session.execute(
            select(IngestionRun)
            .where((IngestionRun.arxiv_id == target) | (IngestionRun.doc_key == target))
            .order_by(IngestionRun.created_at.desc())
            .limit(1)
        )
        return result.scalars().first()

    async def recent_runs(self, limit: int = 20) -> Sequence[IngestionRun]:
        result = await self._session.execute(
            select(IngestionRun)
            .options(selectinload(IngestionRun.steps))
            .order_by(IngestionRun.created_at.desc())
            .limit(limit)
        )
        return result.scalars().unique().all()


def _section_field(
    titles: dict[tuple[str, int], tuple[str, int, int | None, int | None]],
    chunk: Chunk | None,
    *,
    index: int,
) -> Any:  # noqa: ANN401 - str | int | None, depending on the field asked for
    """One component of a chunk's section reference, or ``None``.

    Split out because the title and the level travel together and indexing into the
    same lookup twice at the call site would be a way to lose the guard that a
    chunk with no section yields ``None`` rather than raising.
    """
    if chunk is None or chunk.section_ordinal is None:
        return None
    found = titles.get((chunk.paper_id, chunk.section_ordinal))
    return None if found is None else found[index]


class SemanticSearchRepository:
    """Relational hydration for vector hits returned by a VectorStore."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def _section_titles(
        self, chunks: Iterable[Chunk]
    ) -> dict[tuple[str, int], tuple[str, int, int | None, int | None]]:
        """Title of the section each chunk points at, in one query.

        Resolved here rather than stored on the chunk because the pointer is the
        point: one heading stored once, and a heading retitled in the source
        reaches every chunk that cites it without a rewrite. Scoped to the papers
        in the result rather than read whole — a corpus with a book's 196
        sections should not pull every other document's on every keystroke.
        """
        wanted = {
            (chunk.paper_id, chunk.section_ordinal)
            for chunk in chunks
            if chunk.section_ordinal is not None
        }
        if not wanted:
            return {}
        paper_ids = {paper_id for paper_id, _ in wanted}
        result = await self._session.execute(
            select(DocumentSection).where(DocumentSection.paper_id.in_(paper_ids))
        )
        # The whole reference, not just the title: a grouped result needs to say
        # "pages 52-58", which is the section's span, not the span of whichever
        # chunk happened to score highest.
        return {
            (row.paper_id, row.ordinal): (row.title, row.level, row.page_start, row.page_end)
            for row in result.scalars().all()
            if (row.paper_id, row.ordinal) in wanted
        }

    async def hydrate(self, hits: Sequence[SearchHitWithScore]) -> list[dict[str, Any]]:
        if not hits:
            return []
        chunk_ids = [hit.chunk_id for hit in hits]
        result = await self._session.execute(
            select(Chunk)
            .where(Chunk.id.in_(chunk_ids))
            .options(selectinload(Chunk.paper))
        )
        chunks = {chunk.id: chunk for chunk in result.scalars().unique().all()}
        titles = await self._section_titles(chunks.values())
        out: list[dict[str, Any]] = []
        for hit in hits:
            chunk = chunks.get(hit.chunk_id)
            paper = chunk.paper if chunk else None
            out.append(
                {
                    "chunk_id": hit.chunk_id,
                    # The chunk row is authoritative: the vector store's payload
                    # may not carry the paper id.
                    "paper_id": chunk.paper_id if chunk else hit.paper_id,
                    "score": hit.score,
                    "text": chunk.text if chunk else hit.text,
                    "metadata": {
                        "arxiv_id": paper.arxiv_id if paper else None,
                        # `display_id`, because a local document has no arXiv id
                        # and printing None for it helps nobody.
                        "display_id": paper.display_id if paper else None,
                        "doc_key": paper.doc_key if paper else None,
                        "kind": paper.kind if paper else None,
                        "page_count": paper.page_count if paper else None,
                        "page_start": chunk.page_start if chunk else None,
                        "page_end": chunk.page_end if chunk else None,
                        # Title *and* level: the level is what lets a caller
                        # indent a hit under the section that contains it, which
                        # is the difference between a flat list of passages and a
                        # table of contents with the relevant parts marked.
                        "section_title": _section_field(titles, chunk, index=0),
                        "section_level": _section_field(titles, chunk, index=1),
                        "section_page_start": _section_field(titles, chunk, index=2),
                        "section_page_end": _section_field(titles, chunk, index=3),
                        "title": paper.title if paper else None,
                        "authors": list(paper.author_names) if paper else [],
                        "categories": list(paper.categories) if paper else [],
                        "primary_category": paper.primary_category if paper else None,
                        "abs_url": paper.abs_url if paper else None,
                        "pdf_url": paper.pdf_url if paper else None,
                        "published_at": paper.published_at.isoformat()
                        if paper and paper.published_at
                        else None,
                        # The store's own payload first: it is the less
                        # authoritative of the two. Splatting it after the chunk
                        # fields would let a stale index-time `content_kind`
                        # overwrite what the `chunks` row says — which is
                        # exactly what a re-classification has to be able to fix.
                        **hit.metadata,
                        "heading": chunk.heading if chunk else None,
                        "section_path": chunk.section_path if chunk else [],
                        "ordinal": chunk.ordinal if chunk else None,
                        "source": chunk.source if chunk else None,
                        # So a caller can group results by what they are
                        # without re-reading the text.
                        "content_kind": chunk.content_kind if chunk else None,
                    },
                }
            )
        return out


def metadata_to_dict(metadata: PaperMetadata) -> dict[str, Any]:  # pragma: no cover - helper
    return {
        "arxiv_id": metadata.arxiv_id,
        "title": metadata.title,
        "authors": [Author(name=a.name).name for a in metadata.authors],
        "categories": list(metadata.categories),
    }