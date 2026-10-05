"""Async ArXiv API client.

Talks to the official ``export.arxiv.org/api/query`` Atom endpoint and hides
pagination, rate limiting and retry policy behind a small, testable surface.
"""

from __future__ import annotations

import dataclasses
from collections.abc import AsyncIterator
from typing import Any

import httpx

from app.clients.arxiv.exceptions import ArxivEmptyResponse, ArxivNotFound, ArxivRateLimited
from app.clients.arxiv.parser import parse_feed
from app.clients.arxiv.query import build_search_params
from app.config import ArxivSettings
from app.domain.enums import ArxivSortBy, ArxivSortOrder
from app.domain.ids import normalize_arxiv_id, parse_arxiv_id
from app.domain.models import PaperMetadata, SearchQuery, SearchResultPage
from app.infra.http import HttpError, HttpFetcher
from app.logging import get_logger

logger = get_logger(__name__)


class ArxivClient:
    """Search and metadata lookup for ArXiv papers."""

    def __init__(self, settings: ArxivSettings, fetcher: HttpFetcher) -> None:
        self._settings = settings
        self._fetcher = fetcher

    # ------------------------------------------------------------------ search
    async def search(self, query: SearchQuery) -> SearchResultPage:
        """Execute a single query page."""
        if query.is_empty():
            raise ValueError("search query is empty")
        params = build_search_params(query)
        payload = await self._fetch(params)
        page = parse_feed(
            payload, query=query, html_base_url=self._settings.html_base_url
        )
        logger.info(
            "arxiv_search",
            extra={
                "search_query": params["search_query"],
                "hits": len(page),
                "total": page.total_results,
            },
        )
        return page

    async def search_all(
        self, query: SearchQuery, *, limit: int | None = None
    ) -> list[PaperMetadata]:
        """Paginate through results until ``limit`` unique papers are collected."""
        collected: dict[str, PaperMetadata] = {}
        start = query.start
        page_size = min(self._settings.max_results_per_request, max(1, query.max_results))
        target = limit if limit is not None else query.max_results

        while len(collected) < target:
            page_query = SearchQuery(
                all_terms=query.all_terms,
                title_terms=query.title_terms,
                author_terms=query.author_terms,
                abstract_terms=query.abstract_terms,
                category_terms=query.category_terms,
                id_list=query.id_list,
                raw=query.raw,
                max_results=page_size,
                start=start,
                sort_by=query.sort_by,
                sort_order=query.sort_order,
            )
            page = await self.search(page_query)
            if not page:
                break
            for hit in page:
                collected.setdefault(hit.metadata.arxiv_id, hit.metadata)
            start += len(page)
            if not page.has_more:
                break

        results = list(collected.values())
        return results[:target] if limit is not None else results

    async def iter_search(self, query: SearchQuery) -> AsyncIterator[PaperMetadata]:
        """Stream metadata page by page (memory friendly for large harvests)."""
        page_size = self._settings.max_results_per_request
        start = query.start
        seen: set[str] = set()
        while True:
            page = await self.search(dataclasses.replace(query, max_results=page_size, start=start))
            if not page:
                return
            for hit in page:
                if hit.metadata.arxiv_id not in seen:
                    seen.add(hit.metadata.arxiv_id)
                    yield hit.metadata
            if not page.has_more:
                return
            start += len(page)

    # ----------------------------------------------------------------- by id
    async def get_paper(self, arxiv_id: str) -> PaperMetadata:
        """Fetch canonical metadata for one paper (any id/version form accepted)."""
        identifier = normalize_arxiv_id(arxiv_id)
        query = SearchQuery(
            id_list=(arxiv_id if parse_arxiv_id(arxiv_id).version else identifier,),
            max_results=1,
            sort_by=ArxivSortBy.LAST_UPDATED_DATE,
            sort_order=ArxivSortOrder.DESCENDING,
        )
        page = await self.search(query)
        if not page:
            raise ArxivNotFound(f"no ArXiv entry for {arxiv_id!r}")
        return page.hits[0].metadata

    async def get_papers(self, arxiv_ids: list[str]) -> list[PaperMetadata]:
        """Batch metadata lookup using the ``id_list`` parameter (chunks of 50)."""
        out: list[PaperMetadata] = []
        for chunk in _chunks(arxiv_ids, 50):
            query = SearchQuery(
                id_list=tuple(normalize_arxiv_id(a) for a in chunk),
                max_results=len(chunk),
                sort_by=ArxivSortBy.LAST_UPDATED_DATE,
                sort_order=ArxivSortOrder.DESCENDING,
            )
            page = await self.search(query)
            out.extend(hit.metadata for hit in page)
        return out

    # ----------------------------------------------------------------- helpers
    async def _fetch(self, params: dict[str, Any]) -> bytes:
        try:
            response: httpx.Response = await self._fetcher.get_text(
                self._settings.api_base_url, params
            )
        except HttpError as exc:
            if exc.status_code == 429:
                raise ArxivRateLimited("ArXiv rate limit hit (429)") from exc
            raise
        payload = response.content
        if not payload.strip():
            raise ArxivEmptyResponse("ArXiv returned an empty body")
        return payload


def _chunks(items: list[str], size: int) -> list[list[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]