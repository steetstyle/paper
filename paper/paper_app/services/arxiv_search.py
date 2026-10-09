"""ArXiv search as a use case.

The CLI, the HTTP API and the MCP server all call :meth:`ArxivSearchService.search`
with a :class:`~paper_app.clients.arxiv.filters.SearchRequest`, so all three get the
same validation, the same ``search_query`` and the same client-side filtering.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from paper_app.clients.arxiv.client import ArxivClient
from paper_app.clients.arxiv.exceptions import ArxivError
from paper_app.clients.arxiv.filters import SearchRequest
from paper_app.db.repositories import PaperRepository
from paper_app.domain.filters import FilterProblem
from paper_app.domain.models import PaperMetadata
from paper_app.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class SearchOutcome:
    papers: list[PaperMetadata] = field(default_factory=list)
    total_results: int = 0
    returned: int = 0
    filtered_out: int = 0
    warnings: tuple[str, ...] = ()
    hint: str | None = None
    """Why the result set looks the way it does, when that is not obvious."""
    next_start: int | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "error": self.error,
            "total_results": self.total_results,
            "returned": self.returned,
            "filtered_out": self.filtered_out,
            "next_start": self.next_start,
            "warnings": list(self.warnings),
            "hint": self.hint,
        }


def _explain_empty(
    request: SearchRequest,
    page: Any,  # noqa: ANN401 - SearchPage
    papers: list[PaperMetadata],
    filtered_out: int,
) -> str | None:
    """Say *why* a search came back empty, when the reason is not the query.

    An exact phrase containing a misspelling returns nothing at all, which is
    correct but indistinguishable from "this paper does not exist". Naming the
    likely cause is the difference between a user fixing the query and a user
    assuming the tool is broken.
    """
    if papers:
        return None
    if page.total_results > 0:
        if filtered_out:
            return (
                f"ArXiv matched {page.total_results} papers but the client-side "
                "filter removed all of them — loosen it (--no-has-pdf, "
                "--no-has-journal-ref, drop --exclude-category)."
            )
        return (
            "ArXiv reported matches but returned none for this page; try a "
            "larger --max or a different --offset."
        )

    quoted = re.findall(r'"[^"]+"', request.filter.compile())
    if quoted:
        return (
            "No paper matches this exact phrase. ArXiv phrases are literal, so a "
            f"misspelling returns nothing: check the spelling of {quoted[0]}, or "
            "search the words with AND, or drop the quotes."
        )
    if any(len(word) > 2 and not word.isalpha() for word in request.filter.compile().split()):
        return None
    return (
        "ArXiv matched nothing. A term that matches no paper is silently dropped "
        "rather than counted, so a misspelling can still leave a broad query — "
        "try quoting a phrase or ANDing fewer words."
    )


class ArxivSearchService:
    def __init__(
        self,
        arxiv_client: ArxivClient,
        *,
        session_factory: Any | None = None,
    ) -> None:
        self._client = arxiv_client
        self._session_factory = session_factory

    async def search(self, request: SearchRequest) -> SearchOutcome:
        """Run a search and apply the filters ArXiv cannot do server-side.

        Pagination caveat: post-filters run *after* paging, so a page can come
        back shorter than ``max_results`` while more matches exist. The outcome
        reports ``next_start`` so callers keep going.
        """
        if request.filter.is_empty and not request.id_list:
            return SearchOutcome(error="nothing to search for")

        try:
            page = await self._client.search(request.to_search_query())
        except FilterProblem as exc:
            return SearchOutcome(error=str(exc))
        except ArxivError as exc:
            logger.error("arxiv_search_failed", extra={"error": str(exc)})
            return SearchOutcome(error=f"ArXiv request failed: {exc}")

        papers = [hit.metadata for hit in page]
        filtered_out = 0

        if not request.post_filter.is_empty:
            ingested = (
                await self._ingested_ids() if request.ingested is not None else None
            )
            kept = [
                paper
                for paper in papers
                if request.post_filter.apply(
                    paper, ingested=paper.arxiv_id in ingested if ingested else None
                )
            ]
            filtered_out = len(papers) - len(kept)
            papers = kept

        return SearchOutcome(
            papers=papers,
            total_results=page.total_results,
            returned=len(papers),
            filtered_out=filtered_out,
            warnings=request.warnings,
            hint=_explain_empty(request, page, papers, filtered_out),
            next_start=page.start + len(page) if page.has_more else None,
        )

    async def _ingested_ids(self) -> set[str]:
        """Local corpus ids, needed only for the ``ingested`` post-filter."""
        if self._session_factory is None:
            return set()
        async with self._session_factory() as session:
            result = await session.execute(PaperRepository(session).all_arxiv_ids())
            return set(result.scalars().all())