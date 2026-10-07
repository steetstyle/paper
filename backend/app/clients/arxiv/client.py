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
                # An id_list request has no `search_query` param — see
                # build_search_params — so log whichever was actually sent.
                # `params["search_query"]` raised KeyError here before.
                "search_query": params.get("search_query", f"id_list={params.get('id_list')}"),
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
        """Fetch canonical metadata for one paper (any id/version form accepted).

        Falls back to the versionless id when the versioned form does not
        resolve. Measured cases, all old-style ids where ArXiv is inconsistent:

        - ``cond-mat/0305062v1`` answers HTTP 500; ``cond-mat/0305062`` returns
          the paper (as v4). A 500 on a document that demonstrably exists is
          worth one retry, not an error.
        - ``solv-int/9712001v2`` returns nothing because that version was never
          published; the versionless id returns v1. Falling back gives the paper,
          and ``versioned_id`` in the result still reports the real version.

        The retry is only ever *looser*, so it can turn "not found" into "found"
        and never the reverse.
        """
        identifier = normalize_arxiv_id(arxiv_id)
        wanted = arxiv_id if parse_arxiv_id(arxiv_id).version else identifier
        page = await self._id_page(wanted)
        if not page and wanted != identifier:
            logger.info(
                "arxiv_id_version_fallback",
                extra={"requested": arxiv_id, "tried": wanted, "fallback": identifier},
            )
            page = await self._id_page(identifier)
        if not page:
            raise ArxivNotFound(f"no ArXiv entry for {arxiv_id!r}")
        return page.hits[0].metadata

    async def _id_page(self, one_id: str):  # noqa: ANN202
        """One `id_list` page, tolerating ArXiv's 500 on some old-style ids."""
        query = SearchQuery(
            id_list=(one_id,),
            max_results=1,
            sort_by=ArxivSortBy.LAST_UPDATED_DATE,
            sort_order=ArxivSortOrder.DESCENDING,
        )
        try:
            return await self.search(query)
        except HttpError as exc:
            if exc.status_code != 500:
                raise
            # ArXiv serves 500 for `cond-mat/0305062v1` while the versionless
            # form of the same paper returns 200. Let the caller's fallback
            # decide rather than reporting a server error as a missing paper.
            logger.warning(
                "arxiv_id_500",
                extra={"requested": one_id, "status": exc.status_code},
            )
            return SearchResultPage(hits=(), total_results=0, start=0, items_per_page=0)

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
                # Actionable because it is terminal: the client does not wait and
                # retry on a rate limit, so this is the caller's cue rather than a
                # step on the way to an answer. arXiv sends no Retry-After, so it
                # cannot say how long — only that now is too soon.
                raise ArxivRateLimited(
                    "arXiv is rate limiting this client (HTTP 429). It sends no "
                    "Retry-After, so there is no correct time to retry — wait a "
                    "little and ask again. Requests are already spaced 3s apart."
                ) from exc
            raise
        payload = response.content
        if not payload.strip():
            raise ArxivEmptyResponse("ArXiv returned an empty body")
        return payload


def _chunks(items: list[str], size: int) -> list[list[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]