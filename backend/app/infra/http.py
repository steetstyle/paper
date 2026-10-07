"""Shared HTTP transport: retries, timeouts, polite rate limiting."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from types import TracebackType

import httpx
from httpx._types import QueryParamTypes
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from app.config import ArxivSettings
from app.logging import get_logger

logger = get_logger(__name__)

# Transient failures worth retrying; everything else surfaces immediately.
#: Statuses worth trying again on a short exponential backoff.
#:
#: **429 is deliberately not here.** A 429 is not a hiccup, it is the far end
#: saying it wants fewer requests, and arXiv sends no ``Retry-After`` to say how
#: long for — so a retry has to guess. The guess this client used to make cost four
#: attempts over ~11s of waiting (``wait_exponential(multiplier=1.5)`` →
# 1.5 + 3 + 6.75) during which arXiv was still saying *slow down*, which is the
#: one thing that makes a rate limit last longer. Measured on a session that
#: fetched ten ids: the first two answered, the next eight each burned 12.3s to
#: arrive at the same 429.
#:
#: Failing fast and saying so leaves the caller to choose, which is the only party
#: that knows when it is ready.
RETRYABLE_STATUS = frozenset({408, 425, 500, 502, 503, 504, 520, 522, 524})


class HttpError(RuntimeError):
    """An HTTP failure.

    Retryability is a property of the status, not of this class: see
    :data:`RETRYABLE_STATUS` and :meth:`is_retryable`. This docstring used to say
    "Non-retryable HTTP failure" outright, which contradicted the 429 in that set
    — the docstring was the one that was wrong, and it is worth saying so here
    because the two are easy to mistake for each other again.
    """

    def __init__(self, message: str, *, status_code: int | None = None, url: str | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.url = url

    @property
    def is_retryable(self) -> bool:
        return self.status_code in RETRYABLE_STATUS


class ContentTooLarge(HttpError):
    """The response exceeded the configured byte ceiling."""


def should_retry(exc: BaseException) -> bool:
    """Retry predicate: transport hiccups and transient HTTP statuses."""
    if isinstance(exc, ContentTooLarge):
        return False
    if isinstance(exc, HttpError):
        return exc.is_retryable
    return isinstance(exc, httpx.TransportError)


class RateLimiter:
    """Async minimum-interval limiter.

    ArXiv's API terms require >=3s spacing between requests; downloads are
    spaced by a smaller delay so bulk ingestion stays a good citizen.
    """

    def __init__(self, min_interval: float = 0.0) -> None:
        self._min_interval = max(0.0, min_interval)
        self._lock = asyncio.Lock()
        self._last = 0.0

    async def acquire(self) -> None:
        if self._min_interval <= 0:
            return
        async with self._lock:
            loop = asyncio.get_running_loop()
            wait_for = self._min_interval - (loop.time() - self._last)
            if wait_for > 0:
                logger.debug("rate_limit_wait", extra={"seconds": round(wait_for, 3)})
                await asyncio.sleep(wait_for)
            self._last = loop.time()


class HttpFetcher:
    """Thin, opinionated wrapper around ``httpx.AsyncClient``.

    * one shared connection pool per instance
    * retries with exponential backoff on transient failures
    * rate limiting before every request
    * streaming downloads with a hard byte ceiling
    """

    def __init__(
        self,
        settings: ArxivSettings,
        *,
        client: httpx.AsyncClient | None = None,
        rate_limiter: RateLimiter | None = None,
    ) -> None:
        self._settings = settings
        self._client = client
        self._owns_client = client is None
        self._api_limiter = rate_limiter or RateLimiter(settings.request_interval_seconds)
        self._download_limiter = RateLimiter(settings.download_delay_seconds)

    # ------------------------------------------------------------- lifecycle
    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    connect=self._settings.connect_timeout_seconds,
                    read=self._settings.read_timeout_seconds,
                    write=self._settings.read_timeout_seconds,
                    pool=self._settings.connect_timeout_seconds,
                ),
                headers={"User-Agent": self._settings.user_agent},
                follow_redirects=True,
                limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
            )
        return self._client

    async def __aenter__(self) -> HttpFetcher:
        self._ensure_client()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
        self._client = None

    def _retrying(self) -> AsyncRetrying:
        return AsyncRetrying(
            stop=stop_after_attempt(max(1, self._settings.max_retries)),
            wait=wait_exponential(multiplier=self._settings.backoff_base_seconds, max=30),
            retry=retry_if_exception(should_retry),
            reraise=True,
        )

    # ------------------------------------------------------------------ text
    async def get_text(
        self,
        url: str,
        params: QueryParamTypes | None = None,
        *,
        max_bytes: int | None = None,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        """Rate-limited GET returning a fully buffered response."""
        await self._api_limiter.acquire()
        logger.debug("http_get", extra={"url": url, "params": params})
        return await self._request("GET", url, params=params, max_bytes=max_bytes, headers=headers)

    async def _request(
        self,
        method: str,
        url: str,
        *,
        params: QueryParamTypes | None = None,
        max_bytes: int | None = None,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        client = self._ensure_client()
        async for attempt in self._retrying():
            with attempt:
                response = await client.request(
                    method, url, params=params, headers=headers,
                    timeout=self._settings.read_timeout_seconds,
                )
                self._raise_for_status(response, url)
                if max_bytes and len(response.content) > max_bytes:
                    raise ContentTooLarge(
                        f"{url} returned {len(response.content)} bytes > limit {max_bytes}",
                        url=url,
                    )
                return response
        raise HttpError(f"{method} {url} exhausted all retries", url=url)

    def _raise_for_status(self, response: httpx.Response, url: str) -> None:
        if response.status_code < 400:
            return
        if response.status_code == 429:
            raise HttpError(
                f"{url} rate limited (429)", status_code=429, url=url
            )
        raise HttpError(
            f"{url} failed with {response.status_code}",
            status_code=response.status_code,
            url=url,
        )

    # ----------------------------------------------------------------- exists
    async def exists(self, url: str, *, accept: tuple[str, ...] = ("text/html",)) -> bool:
        """HEAD probe used to decide whether an HTML rendering exists."""
        await self._api_limiter.acquire()
        try:
            response = await self._ensure_client().head(url, timeout=15.0)
        except httpx.HTTPError as exc:
            logger.debug("head_failed", extra={"url": url, "error": str(exc)})
            return False
        if response.status_code >= 400:
            return False
        if not accept:
            return True
        content_type = response.headers.get("content-type", "")
        return any(token in content_type for token in accept) or not content_type

    # -------------------------------------------------------------- download
    async def download(
        self,
        url: str,
        destination: Path,
        *,
        max_bytes: int | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[str, int]:
        """Stream a URL to disk, returning ``(sha256, size_in_bytes)``."""
        import hashlib

        limit = max_bytes or self._settings.max_pdf_bytes
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        staging = path.with_suffix(path.suffix + ".part")

        await self._download_limiter.acquire()
        client = self._ensure_client()

        async for attempt in self._retrying():
            with attempt:
                digest = hashlib.sha256()
                size = 0
                async with client.stream(
                    "GET", url, timeout=self._settings.download_timeout_seconds, headers=headers
                ) as response:
                    self._raise_for_status(response, url)
                    declared = response.headers.get("content-length")
                    if declared and int(declared) > limit:
                        raise ContentTooLarge(
                            f"{url} declares {declared} bytes > limit {limit}", url=url
                        )
                    with staging.open("wb") as handle:
                        async for block in response.aiter_bytes(chunk_size=1 << 16):
                            size += len(block)
                            if size > limit:
                                raise ContentTooLarge(f"{url} exceeded limit {limit}", url=url)
                            digest.update(block)
                            handle.write(block)
                staging.replace(path)
                logger.info("downloaded", extra={"url": url, "bytes": size, "path": str(path)})
                return digest.hexdigest(), size

        raise HttpError(f"GET {url} exhausted all retries", url=url)

    async def iter_bytes(
        self, url: str, *, max_bytes: int | None = None
    ) -> AsyncIterator[bytes]:  # pragma: no cover - convenience helper
        await self._download_limiter.acquire()
        async with self._ensure_client().stream("GET", url) as response:
            self._raise_for_status(response, url)
            yield response.content


@asynccontextmanager
async def build_fetcher(settings: ArxivSettings) -> AsyncIterator[HttpFetcher]:
    async with HttpFetcher(settings) as fetcher:
        yield fetcher