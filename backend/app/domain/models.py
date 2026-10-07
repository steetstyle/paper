"""Pure domain objects.

These carry no I/O and no framework imports so they can be produced by the
ArXiv client, the fetcher, the extractor and the pipeline alike.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from app.domain.enums import ChunkKind, ContentSource

__all__ = [
    "Author",
    "PaperMetadata",
    "SearchQuery",
    "SearchResultPage",
    "SearchHit",
    "ContentPayload",
    "ExtractedDocument",
    "TextChunk",
    "EmbeddingVector",
    "VectorRecord",
    "SearchHitWithScore",
    "PageRange",
    "MineruOptions",
    "doc_slug",
]


@dataclass(frozen=True, slots=True)
class Author:
    name: str
    affiliation: str | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("author name must not be empty")


@dataclass(frozen=True, slots=True)
class Reference:
    """One bibliography entry, before it is matched against the corpus.

    Everything here is optional except ``raw_text`` because a bibliography is
    free text: the fields are a lossy best effort, and ``raw_text`` is the only
    part that is always true.
    """

    raw_text: str
    ordinal: int
    title: str | None = None
    authors: str | None = None
    year: int | None = None
    venue: str | None = None
    cited_arxiv_id: str | None = None
    """Present when the entry names an arXiv paper, which is what allows the
    reference to be linked to a real corpus row later."""

    def label(self) -> str:
        who = (self.authors or "").strip()
        who = who.split(",")[0].split(" and ")[0].strip() if who else "?"
        title = (self.title or self.raw_text).strip()
        year = f" ({self.year})" if self.year else ""
        return f"{who}{year} — {title[:90]}"


@dataclass(frozen=True, slots=True)
class ProjectInfo:
    """A project as returned by the repository."""

    id: str
    name: str
    slug: str
    description: str | None = None
    is_archived: bool = False
    paper_count: int = 0
    read_count: int = 0
    created_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class PaperMetadata:
    """Normalised metadata for one ingestable thing: an arXiv paper or a file.

    One type, two constructors (:meth:`from_arxiv`, :meth:`from_file`), because
    the pipeline takes a single ``ctx.metadata`` and a second metadata class
    would force the persist step to branch on which one it got.

    The arXiv fields are optional rather than a separate class because the
    corpus genuinely holds both: measured on the three books this was built for,
    ``/Title`` was garbage in one of them (``pethick.dvi``) and missing in
    another, so a local file cannot rely on embedded metadata and must be able to
    carry none.
    """

    arxiv_id: str | None
    versioned_id: str | None
    version: int | None
    title: str
    abstract: str
    doc_key: str = ""
    """Universal handle. Defaults to the arXiv id; a local file supplies a slug."""
    kind: str = "paper"
    file_sha256: str | None = None
    page_count: int | None = None
    authors: tuple[Author, ...] = ()
    categories: tuple[str, ...] = ()
    primary_category: str | None = None
    published_at: datetime | None = None
    updated_at: datetime | None = None
    doi: str | None = None
    comment: str | None = None
    journal_ref: str | None = None
    abs_url: str = ""
    pdf_url: str = ""
    html_url: str | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        """Fill ``doc_key`` for an arXiv paper.

        The invariant, enforced once so no caller can get it wrong: a paper's
        handle *is* its arXiv id. A local file supplies its own and leaves
        ``arxiv_id`` empty. Derived rather than required because every existing
        caller builds ``PaperMetadata`` from an ArXiv entry and has no reason to
        know that documents exist.
        """
        if not self.doc_key:
            if not self.arxiv_id:
                raise ValueError(
                    "PaperMetadata needs either arxiv_id or doc_key; "
                    "nothing here identifies the thing being ingested"
                )
            object.__setattr__(self, "doc_key", self.arxiv_id)

    @property
    def latest_versioned_id(self) -> str:
        """Versioned id pointing at the most recent version."""
        if not self.arxiv_id:
            return self.doc_key
        return f"{self.arxiv_id}v{self.version}" if self.version else self.arxiv_id

    @property
    def display_id(self) -> str:
        return self.arxiv_id or self.doc_key

    # ------------------------------------------------------------ constructors
    @classmethod
    def from_file(
        cls,
        *,
        path: Path,
        title: str | None = None,
        doc_key: str | None = None,
        kind: str = "book",
        page_count: int | None = None,
        file_sha256: str | None = None,
        authors: tuple[Author, ...] = (),
        abstract: str = "",
        published_at: datetime | None = None,
        raw: Mapping[str, Any] | None = None,
    ) -> PaperMetadata:
        """Build from a local file: a book, lecture notes, a report.

        There is no ``from_arxiv`` counterpart because there is nothing to share:
        the ArXiv client already constructs this directly from its parsed Atom
        entry, and the two paths share only the fields, not the source.

        Embedded PDF metadata is used **only when it looks like a title**.
        Measured on the three books this was written against: ``/Title`` was
        ``'Modern Condensed Matter Physics'`` for one and ``'pethick.dvi'`` for
        another — a LaTeX build artefact, not a title. A value that ends in a
        source extension or restates the filename is a build artefact by
        definition, so the filename wins over it.
        """
        embedded = _embedded_title(raw or {})
        chosen = title or (_looks_like_title(embedded, path) and embedded) or ""
        stem = doc_slug(path.stem)
        return cls(
            arxiv_id=None,
            versioned_id=None,
            version=None,
            title=(chosen or _title_from_path(path)).strip(),
            abstract=abstract.strip(),
            doc_key=doc_key or stem,
            kind=kind,
            file_sha256=file_sha256,
            page_count=page_count,
            authors=authors,
            published_at=published_at,
            raw=dict(raw or {}),
        )

    def author_names(self) -> list[str]:
        return [a.name for a in self.authors]


_SOURCE_SUFFIXES = frozenset(
    {".tex", ".dvi", ".aux", ".log", ".out", ".ltx", ".cls", ".sty", ".doc", ".docx"}
)


def _embedded_title(meta: Mapping[str, Any]) -> str | None:
    """The document's own ``/Title``, whichever way the caller spelled the key.

    A PDF's ``/Info`` dictionary spells it ``/Title``, with the slash, and that is
    what ``read_pdf_info`` stores verbatim. A plain ``title`` is also accepted
    because that is what a caller assembling metadata by hand would pass.

    Getting this wrong was silent: the lookup asked for ``title``, never found it,
    and every PDF fell through to a title made from its filename — so a book with
    ``/Title = 'Superconductivity, superfluids, and condensates'`` was stored as
    ``'Superconductivity Superfluids And Condensates 685m5mne1x'``.
    """
    for key in ("/Title", "title"):
        value = meta.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _looks_like_title(value: object, path: Path) -> bool:
    """Whether an embedded ``/Title`` can be trusted as a human title.

    ``'pethick.dvi'`` is a TeX job name that survived the build; ``'Lecture 3'``
    is a title. The two are told apart by shape, not by hope: a source-file
    suffix, or a value that merely repeats the filename, is a build artefact.
    """
    if not isinstance(value, str):
        return False
    text = value.strip()
    if not text or "\n" in text:
        return False
    suffix = Path(text).suffix.lower()
    if suffix in _SOURCE_SUFFIXES:
        return False
    # Separators are normalised on both sides: a build tool that writes the job
    # name writes the stem, and ``solid_state_basics`` is that stem just as much
    # as ``solid-state-basics`` is. Comparing them literally let it through.
    flat = lambda value: re.sub(r"[\s_-]+", " ", value).strip().lower()  # noqa: E731
    return flat(text.removesuffix(suffix)) != flat(path.stem)


def _title_from_path(path: Path) -> str:
    """``"solid-state-basics"`` -> ``"Solid state basics"``.

    A filename is the one title a local file always has, so it is the floor the
    other sources fall back to.
    """
    words = path.stem.replace("_", " ").replace("-", " ").split()
    if not words:
        return path.stem or "document"
    return " ".join(w if w.isupper() else w.capitalize() for w in words)


def doc_slug(value: str) -> str:
    """A stable, URL-safe handle from a filename stem.

    Public because two layers need it and neither should own it: the domain builds
    the handle when constructing metadata, and the ingestion service needs the
    same handle before it has any metadata to build it from.
    """
    folded = unicodedata.normalize("NFKD", value)
    ascii_only = folded.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", ascii_only).strip("-").lower()
    return slug[:120].strip("-") or "document"


@dataclass(frozen=True, slots=True)
class SearchQuery:
    """Structured ArXiv query.

    Values inside one field are OR-ed, fields are AND-ed, which mirrors how
    users expect ``ti:"a" OR ti:"b" AND cat:cs.LG`` style narrowing to behave.
    """

    all_terms: tuple[str, ...] = ()
    title_terms: tuple[str, ...] = ()
    author_terms: tuple[str, ...] = ()
    abstract_terms: tuple[str, ...] = ()
    category_terms: tuple[str, ...] = ()
    id_list: tuple[str, ...] = ()
    raw: str | None = None
    max_results: int = 10
    start: int = 0
    sort_by: str = "relevance"
    sort_order: str = "descending"

    def is_empty(self) -> bool:
        return not (
            self.all_terms
            or self.title_terms
            or self.author_terms
            or self.abstract_terms
            or self.category_terms
            or self.id_list
            or self.raw
        )


@dataclass(frozen=True, slots=True)
class SearchHit:
    metadata: PaperMetadata
    relevance_score: float | None = None


@dataclass(frozen=True, slots=True)
class SearchResultPage:
    hits: tuple[SearchHit, ...]
    total_results: int
    start: int
    items_per_page: int

    def __len__(self) -> int:
        return len(self.hits)

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self.hits)

    @property
    def has_more(self) -> bool:
        return self.start + len(self.hits) < self.total_results


@dataclass(frozen=True, slots=True)
class MineruOptions:
    """Per-document extraction overrides.

    Every field is optional and ``None`` means "use the configured default",
    because the right setting differs per document and the corpus holds both
    ends of the range: a Turkish lecture-notes PDF that needs OCR and an
    English textbook with a text layer both belong in the same corpus, and a
    process-wide setting can only be right for one of them.

    Kept on the run rather than in settings so ``paper ingest notes.pdf
    --language tr --method ocr`` affects that file and nothing else.
    """

    language: str | None = None
    """Hint for layout model and OCR, e.g. ``"tr"``, ``"en"``, ``"ch"``."""

    method: str | None = None
    """``auto`` | ``txt`` | ``ocr``. ``auto`` is MinerU's own choice and is the
    default because it measured correctly on every document tried so far."""

    ocr_language: str | None = None
    """PaddleOCR language, when it must differ from the layout-model hint."""

    device: str | None = None
    """``cpu`` | ``cuda`` | ``cuda:0`` | ``mps``, for MinerU's layout models.

    Per document because the corpus holds both kinds of file and one process-wide
    setting cannot be right for both: a scanned textbook wants the GPU for an hour
    of OCR, a note that is three pages does not want to pay the model load."""

    formula: bool | None = None
    table: bool | None = None
    """Both on by default. Turned off for a document that is prose only, where
    the layout model spends most of its time on things that are not there."""

    @property
    def has_overrides(self) -> bool:
        """Whether anything here differs from the configured defaults.

        Read by the pipeline to decide whether a request object is worth building
        at all, so an arXiv paper takes exactly the path it took before any of
        this existed.
        """
        return any(
            value is not None
            for value in (
                self.language,
                self.method,
                self.ocr_language,
                self.device,
                self.formula,
                self.table,
            )
        )

    def merged_with(self, defaults: MineruOptions) -> MineruOptions:
        """Overlay self on top of defaults: self wins field by field."""
        return MineruOptions(
            language=self.language or defaults.language,
            method=self.method or defaults.method,
            ocr_language=self.ocr_language or defaults.ocr_language,
            device=self.device or defaults.device,
            formula=defaults.formula if self.formula is None else self.formula,
            table=defaults.table if self.table is None else self.table,
        )


@dataclass(frozen=True, slots=True)
class PageRange:
    """A 1-based inclusive slice of a paged document.

    Validated on construction rather than at the point of use, because a reversed
    or zero-based range surfaces much later — inside MinerU, as an empty output
    directory and an unclear error — while the mistake is always in the argument
    the user typed.
    """

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 1:
            raise ValueError(f"page range starts at {self.start}; pages are 1-based")
        if self.end < self.start:
            raise ValueError(f"page range {self.start}-{self.end} ends before it starts")

    @property
    def count(self) -> int:
        return self.end - self.start + 1

    def __str__(self) -> str:
        return f"{self.start}-{self.end}"

    @classmethod
    def parse(cls, value: str) -> PageRange | None:
        """Read ``40-52``, ``40``, ``40-`` or ``-52``. ``None`` for an empty value.

        An open upper bound is read as "to the end of the document" and clamped
        against the real page count by whoever has it; this class deliberately
        does not guess a page count of its own.
        """
        text = value.strip()
        if not text:
            return None
        if text.isdigit():
            number = int(text)
            return cls(number, number)
        if text.count("-") != 1:
            raise ValueError(f"cannot read {value!r} as a page range; use 40-52")
        low, _, high = text.partition("-")
        low, high = low.strip(), high.strip()
        # `is None`, not `or`: page 0 is not a page, and `0 or 1` would quietly
        # turn a typo into a valid range starting at the first page.
        start = int(low) if low else None
        end = int(high) if high else None
        if start is None and end is None:
            raise ValueError(f"cannot read {value!r} as a page range")
        # The open upper bound is a sentinel, not a page number: `UNBOUNDED` is
        # spelled out so nobody mistakes it for a real one.
        return cls(1 if start is None else start, cls.UNBOUNDED if end is None else end)

    #: Stand-in for "to the end of the document". Larger than any real book and
    #: never stored, always replaced by the real page count by :meth:`clamp`.
    UNBOUNDED = 10**9

    def clamp(self, page_count: int | None) -> PageRange | None:
        """Resolve an open bound against the document's real page count.

        Returns ``None`` when the slice falls entirely outside the document,
        which is the honest answer for ``--pages 900-950`` on a 721-page book
        and better than an empty run reported as success.
        """
        if page_count is None:
            return self
        if self.start > page_count:
            return None
        return PageRange(self.start, min(self.end, page_count))


@dataclass(frozen=True, slots=True)
class ContentPayload:
    """A downloaded source artefact (HTML or PDF), or a local file's."""

    kind: str  # ContentKind value
    uri: str  # blob store location
    content_type: str
    size_bytes: int
    sha256: str
    source_url: str
    encoding: str | None = None
    local_path: str | None = None
    page_count: int | None = None
    """Pages in the source. Null for HTML, which has none — nullable because
    "page 12" is meaningless for an arXiv HTML rendering and essential in a
    700-page book."""
    outline: tuple[Any, ...] = ()
    """The PDF's bookmark tree, already resolved to 1-based page numbers.

    Carried on the payload rather than re-read by the section step: opening a
    721-page book twice to learn its table of contents is work for no new
    information. Empty for HTML, and empty for a third of real books — measured:
    341 entries for Girvin, 174 for Oxford, **0** for Pethick."""

    page_meta: dict[str, Any] = field(default_factory=dict)
    """Whatever the PDF's own ``/Info`` dictionary claimed, unfiltered.

    Kept raw rather than cleaned because deciding what a title *is* needs the
    original string, and the same dictionary holds fields that turn out to be
    build artefacts."""


@dataclass(frozen=True, slots=True)
class ExtractedDocument:
    """Normalised extraction result regardless of which backend produced it."""

    markdown: str
    source: ContentSource
    backend: str
    text: str
    meta: dict[str, Any] = field(default_factory=dict)
    blocks: list[dict[str, Any]] = field(default_factory=list, repr=False)
    warnings: tuple[str, ...] = ()

    @property
    def is_usable(self) -> bool:
        return len(self.text.strip()) > 0

    @property
    def char_count(self) -> int:
        return len(self.text)


@dataclass(frozen=True, slots=True)
class TextChunk:
    """A retrievable unit of text."""

    ordinal: int
    text: str
    token_count: int
    char_start: int
    char_end: int
    heading: str | None = None
    section_path: tuple[str, ...] = ()
    content_hash: str = ""
    source: ContentSource = ContentSource.ABSTRACT_ONLY
    kind: ChunkKind = ChunkKind.BODY
    """What this chunk is, decided once by the chunker. See
    :func:`app.services.chunk_kinds.classify_chunk`."""

    page_start: int | None = None
    """First page of the source this chunk covers, 1-based in the book's own
    numbering. ``None`` for HTML, which has no pages, and for a chunk whose text
    could not be located — a guessed page would be worse than an absent one."""

    page_end: int | None = None

    section_ordinal: int | None = None
    """Points into ``document_sections`` rather than repeating the title, so
    filtering by section is a column comparison and one heading is stored once."""

    def with_hash(self) -> TextChunk:
        """Copy with the content hash filled in.

        Built with ``dataclasses.replace`` rather than by listing every field:
        a hand-written copy silently drops any field added later, which is how a
        page number would vanish between chunking and storage without any test
        failing.
        """
        import hashlib

        return replace(
            self,
            content_hash=hashlib.sha256(self.text.encode("utf-8")).hexdigest(),
        )


@dataclass(frozen=True, slots=True)
class EmbeddingVector:
    """An embedding plus the identity of the model that produced it."""

    chunk_id: str
    paper_id: str
    vector: list[float]
    provider: str
    model: str
    dimensions: int

    def fingerprint(self) -> str:
        """Used to invalidate cached vectors when the model/dims change."""
        return f"{self.provider}:{self.model}:{self.dimensions}"


@dataclass(frozen=True, slots=True)
class VectorRecord:
    """Upsert payload for a :class:`~app.db.vector_store.base.VectorStore`."""

    chunk_id: str
    paper_id: str
    vector: list[float]
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SearchHitWithScore:
    chunk_id: str
    paper_id: str
    score: float
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)