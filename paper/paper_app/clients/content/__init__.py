"""Source retrieval + PDF text extraction clients."""

from paper_app.clients.content.fetcher import ContentFetcher, ContentUnavailable, FetchOutcome
from paper_app.clients.content.html_extractor import html_to_markdown
from paper_app.clients.content.mineru import (
    ExtractionBackend,
    ExtractionError,
    MineruBackendUnavailable,
    MineruCliBackend,
    MineruExtractor,
    MineruPythonApiBackend,
    PyPdfBackend,
)
from paper_app.clients.content.mineru_resolver import (
    MineruApi,
    MineruCli,
    installed_version,
    resolve_api,
    resolve_cli,
)

__all__ = [
    "ContentFetcher",
    "ContentUnavailable",
    "FetchOutcome",
    "MineruExtractor",
    "MineruPythonApiBackend",
    "MineruCliBackend",
    "PyPdfBackend",
    "ExtractionBackend",
    "ExtractionError",
    "MineruBackendUnavailable",
    "MineruApi",
    "MineruCli",
    "installed_version",
    "resolve_api",
    "resolve_cli",
    "html_to_markdown",
]