"""Shared enumerations for the domain layer."""

from __future__ import annotations

from enum import StrEnum


class ContentKind(StrEnum):
    """Physical artefacts we persist for a paper."""

    HTML = "html"
    PDF = "pdf"
    MINERU_MARKDOWN = "mineru_markdown"
    MINERU_BLOCKS = "mineru_blocks"
    TEXT = "text"

    @property
    def is_derived(self) -> bool:
        """Derived artefacts are reproducible from a source blob."""
        return self in {ContentKind.MINERU_MARKDOWN, ContentKind.MINERU_BLOCKS, ContentKind.TEXT}


class ChunkKind(StrEnum):
    """What a chunk *is*, which is what a reader usually wants to filter by.

    Asked for directly ("show me only the equations"), so it has to be a first
    class stored column rather than something inferred at query time from the
    text. Assigned once by the chunker, when the surrounding markdown is still
    available; guessing it later from the chunk alone is what
    :func:`paper_app.services.chunker.classify_chunk` avoids.
    """

    BODY = "body"
    ABSTRACT = "abstract"
    FIGURE = "figure"
    TABLE = "table"
    EQUATION = "equation"
    REFERENCE = "reference"
    CODE = "code"

    @property
    def is_structured(self) -> bool:
        """True for the kinds extracted into their own tables."""
        return self in {
            ChunkKind.FIGURE,
            ChunkKind.TABLE,
            ChunkKind.EQUATION,
            ChunkKind.REFERENCE,
        }


class ContentSource(StrEnum):
    """Where the extracted text came from."""

    ARXIV_HTML = "arxiv_html"
    AR5IV = "ar5iv"
    PDF_MINERU = "pdf_mineru"
    PDF_PYPDF = "pdf_pypdf"
    ABSTRACT_ONLY = "abstract_only"
    LOCAL_FILE = "local_file"
    """The bytes came from a path on this machine, not from a URL."""


class DocumentKind(StrEnum):
    """What a corpus row *is*, for a reader who has to tell them apart.

    A flat list rather than a hierarchy: a lecture-notes PDF and a textbook go
    through the identical pipeline, so a base class would add a branch to every
    dispatch for no behaviour change. Stored on ``papers.kind`` and used to
    group and filter in the CLI and the API.
    """

    PAPER = "paper"
    """An arXiv preprint — the only kind that has an arXiv id."""

    BOOK = "book"
    NOTES = "notes"
    REPORT = "report"
    THESIS = "thesis"

    @property
    def has_arxiv_id(self) -> bool:
        return self is DocumentKind.PAPER


class RunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    PARTIAL = "partial"
    FAILED = "failed"
    SKIPPED = "skipped"
    ABANDONED = "abandoned"
    """The process that owned this run is gone.

    Not the same as ``FAILED``: nothing went wrong with the paper, the work
    simply stopped. Kept distinct because the two need different responses — a
    failed run is worth reading the error for, an abandoned one only needs
    re-running. Measured on the live corpus before this existed: 38 runs stuck in
    ``running``, the oldest for over a day, indistinguishable from work in
    progress.
    """

    @property
    def is_terminal(self) -> bool:
        return self in {
            RunStatus.SUCCEEDED,
            RunStatus.PARTIAL,
            RunStatus.FAILED,
            RunStatus.SKIPPED,
            RunStatus.ABANDONED,
        }

    @property
    def is_resumable(self) -> bool:
        """Whether re-running this target can pick up from what it left behind.

        Anything incomplete can be resumed: the expensive artefacts — the
        downloaded blob and the extracted markdown — are stored as they are
        produced, so a retry reuses them instead of paying for them again.
        """
        return not self.is_terminal


class StepStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"
    CACHED = "cached"


class VectorBackend(StrEnum):
    PGVECTOR = "pgvector"
    QDRANT = "qdrant"
    MEMORY = "memory"


class EmbeddingProviderName(StrEnum):
    OPENAI = "openai"
    SENTENCE_TRANSFORMERS = "sentence-transformers"
    HASHING = "hashing"


class MineruBackend(StrEnum):
    PYTHON_API = "python_api"
    CLI = "cli"
    PYPDF = "pypdf"


class ArxivSortBy(StrEnum):
    RELEVANCE = "relevance"
    LAST_UPDATED_DATE = "lastUpdatedDate"
    SUBMITTED_DATE = "submittedDate"


class ArxivSortOrder(StrEnum):
    ASCENDING = "ascending"
    DESCENDING = "descending"