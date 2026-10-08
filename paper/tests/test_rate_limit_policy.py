"""A rate limit is an answer, not a hiccup.

``429`` used to sit in ``RETRYABLE_STATUS`` next to the 5xx codes, so a limited
client retried it four times on an exponential backoff — 1.5s, 3s, 6.75s, about
eleven and a quarter seconds during which arXiv kept saying *slow down*. Measured
on one session of ten id lookups: the first two answered, and each of the next eight
took 12.3s to arrive at the same 429 it could have reported in milliseconds.

arXiv sends no ``Retry-After``, so there is no correct time to retry, and guessing
one is what made the limit last. Failing fast hands the decision to the only party
that knows when it is ready.

The 5xx codes stay retryable, and that is what these tests mostly protect: a fix
for "stop retrying a rate limit" that also stopped retrying a flaky CDN would look
identical in a demo and fail in an ingestion.
"""

from __future__ import annotations

import httpx
import pytest

from app.infra.http import RETRYABLE_STATUS, HttpError, HttpFetcher, RateLimiter


def _fetcher(statuses: list[int], *, max_retries: int = 4) -> tuple[HttpFetcher, list[str]]:
    """A fetcher answering the given statuses in order, recording every attempt."""
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        index = min(len(seen) - 1, len(statuses) - 1)
        return httpx.Response(statuses[index], text="body")

    from app.config import get_settings

    base = get_settings().arxiv
    settings = base.model_copy(
        update={"max_retries": max_retries, "backoff_base_seconds": 0.001}
    )
    fetcher = HttpFetcher(
        settings,
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        # No real 3s waits: the pacing is arithmetic, and a test should not spend
        # it proving that.
        rate_limiter=RateLimiter(0.0),
    )
    return fetcher, seen


class TestRateLimitsAreNotRetried:
    @pytest.mark.asyncio
    async def test_a_429_is_answered_once_and_then_raised(self) -> None:
        fetcher, seen = _fetcher([429])
        with pytest.raises(HttpError) as caught:
            await fetcher.get_text("https://export.arxiv.org/api/query", {"id_list": "x"})
        assert caught.value.status_code == 429
        assert len(seen) == 1, f"retried a rate limit {len(seen) - 1} times"

    @pytest.mark.asyncio
    async def test_it_is_not_a_matter_of_how_long_the_caller_is_faster(self) -> None:
        """A short rate limit is not a shorter one. Still one attempt."""
        fetcher, seen = _fetcher([429])
        with pytest.raises(HttpError):
            await fetcher.get_text("https://export.arxiv.org/api/query")
        assert len(seen) == 1

    def test_429_is_not_in_the_retryable_set(self) -> None:
        """Stated directly as well as by behaviour, because the set is the policy
        and a future edit to it should break something here."""
        assert 429 not in RETRYABLE_STATUS


class TestRealFlakinessIsStillRetried:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", [500, 502, 503, 504, 408, 425])
    async def test_transient_statuses_are_retried(self, status: int) -> None:
        fetcher, seen = _fetcher([status])
        with pytest.raises(HttpError):
            await fetcher.get_text("https://export.arxiv.org/api/query")
        assert len(seen) > 1, f"{status} stopped being retried"

    @pytest.mark.asyncio
    async def test_a_transient_failure_that_clears_is_transparent(self) -> None:
        """The point of retrying: the caller gets the answer and never learns there
        was a blip."""
        fetcher, seen = _fetcher([503, 503, 200])
        response = await fetcher.get_text("https://export.arxiv.org/api/query")
        assert response.status_code == 200
        assert len(seen) == 3

    @pytest.mark.asyncio
    async def test_a_client_error_is_not_retried(self) -> None:
        """404 is an answer too. Retrying it only wastes the caller's time."""
        fetcher, seen = _fetcher([404])
        with pytest.raises(HttpError):
            await fetcher.get_text("https://export.arxiv.org/api/query")
        assert len(seen) == 1

    @pytest.mark.asyncio
    async def test_retries_stay_within_the_configured_budget(self) -> None:
        fetcher, seen = _fetcher([503], max_retries=3)
        with pytest.raises(HttpError):
            await fetcher.get_text("https://export.arxiv.org/api/query")
        assert len(seen) == 3


class TestTheMessageTellsTheCallerWhatToDo:
    """A terminal error is a decision the caller has to make, so it needs enough
    to make it with."""

    @pytest.mark.asyncio
    async def test_a_429_becomes_a_rate_limit_error_saying_so(self) -> None:
        from app.clients.arxiv.client import ArxivClient
        from app.clients.arxiv.exceptions import ArxivRateLimited
        from app.config import get_settings

        fetcher, _ = _fetcher([429])
        client = ArxivClient(get_settings().arxiv, fetcher)
        with pytest.raises(ArxivRateLimited) as caught:
            await client.get_paper("1511.04823")

        message = str(caught.value)
        assert "429" in message
        # Says what to do, and admits there is no correct time to retry — which is
        # why this is not phrased as "please try again shortly".
        assert "wait" in message.lower()
        assert "retry-after" in message.lower()

    @pytest.mark.asyncio
    async def test_the_spacing_is_mentioned_because_it_already_is_applied(self) -> None:
        """Otherwise the message reads as an accusation the client could fix by
        slowing down, which it already does."""
        from app.clients.arxiv.client import ArxivClient
        from app.clients.arxiv.exceptions import ArxivRateLimited
        from app.config import get_settings

        fetcher, _ = _fetcher([429])
        client = ArxivClient(get_settings().arxiv, fetcher)
        with pytest.raises(ArxivRateLimited) as caught:
            await client.get_paper("1511.04823")
        assert "3s" in str(caught.value)