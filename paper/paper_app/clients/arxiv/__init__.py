"""ArXiv Atom API client."""

from paper_app.clients.arxiv.client import ArxivClient
from paper_app.clients.arxiv.exceptions import (
    ArxivEmptyResponse,
    ArxivError,
    ArxivNotFound,
    ArxivParseError,
    ArxivRateLimited,
)
from paper_app.clients.arxiv.query import build_search_params, build_search_query

__all__ = [
    "ArxivClient",
    "ArxivError",
    "ArxivEmptyResponse",
    "ArxivNotFound",
    "ArxivParseError",
    "ArxivRateLimited",
    "build_search_params",
    "build_search_query",
]