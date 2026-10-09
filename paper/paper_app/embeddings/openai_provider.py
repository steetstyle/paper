"""OpenAI embeddings provider (text-embedding-3-small/large, ada-002)."""

from __future__ import annotations

import asyncio
from typing import Any, ClassVar

from paper_app.embeddings.base import EmbeddingError, EmbeddingProvider
from paper_app.logging import get_logger

logger = get_logger(__name__)

# Known dimensions per model, used to sanity-check configuration.
MODEL_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class _Unset:
    """Sentinel distinguishing "not yet loaded" from a legitimately falsy value."""

    __slots__ = ()


_UNSET = _Unset()


class OpenAIEmbeddingProvider(EmbeddingProvider):
    name: ClassVar[str] = "openai"

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        dimensions: int | None = None,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        batch_size: int = 64,
        normalize: bool = True,
    ) -> None:
        super().__init__(
            model=model,
            dimensions=dimensions or MODEL_DIMENSIONS.get(model, 1536),
            batch_size=batch_size,
            normalize=normalize,
        )
        self._api_key = api_key
        self._base_url = base_url
        self._client: Any = None
        self._tokenizer: Any = _UNSET
        # Only the v3 family accepts a `dimensions` request parameter.
        self._supports_dimensions = model.startswith("text-embedding-3")

    # ------------------------------------------------------------------ setup
    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from openai import AsyncOpenAI  # noqa: PLC0415
            except ImportError as exc:  # pragma: no cover
                raise EmbeddingError(
                    "openai package not installed — pip install 'paper-app-backend[openai]'"
                ) from exc
            self._client = AsyncOpenAI(api_key=self._api_key, base_url=self._base_url)
        return self._client

    async def warmup(self) -> None:
        self._get_client()

    def count_tokens(self, text: str) -> int:
        if isinstance(self._tokenizer, _Unset):
            self._tokenizer = self._load_tokenizer()
        if self._tokenizer is None:
            from paper_app.infra.text import approx_tokens

            return approx_tokens(text)
        return len(self._tokenizer.encode(text, disallowed_special=()))

    def _load_tokenizer(self) -> Any:  # noqa: ANN401
        try:
            import tiktoken  # noqa: PLC0415
        except ImportError:
            return None
        try:
            return tiktoken.encoding_for_model(self._model)
        except KeyError:
            return tiktoken.get_encoding("cl100k_base")

    # ------------------------------------------------------------------ embed
    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        client = self._get_client()
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = [
                text if text.strip() else " "
                for text in texts[start : start + self.batch_size]
            ]
            try:
                response = await client.embeddings.create(
                    model=self._model,
                    input=batch,
                    dimensions=self.dimensions if self._supports_dimensions else None,
                )
            except Exception as exc:  # noqa: BLE001 - normalise SDK errors
                raise EmbeddingError(f"OpenAI embedding request failed: {exc}") from exc
            ordered = sorted(response.data, key=lambda item: item.index)
            vectors.extend(self._finalise([item.embedding for item in ordered]))
        logger.debug("openai_embedded", extra={"count": len(vectors)})
        return vectors


async def embed_many(
    provider: EmbeddingProvider, texts: list[str], *, concurrency: int = 4
) -> list[list[float]]:
    """Embed many texts with bounded concurrency (used by bulk ingestion)."""
    semaphore = asyncio.Semaphore(concurrency)

    async def worker(batch: list[str]) -> list[list[float]]:
        async with semaphore:
            return await provider.embed_documents(batch)

    size = provider.batch_size
    batches = [texts[i : i + size] for i in range(0, len(texts), size)]
    results = await asyncio.gather(*(worker(batch) for batch in batches))
    return [vector for batch_result in results for vector in batch_result]