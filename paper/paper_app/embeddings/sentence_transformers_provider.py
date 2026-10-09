"""Local ``sentence-transformers`` embeddings (no API key, no data egress)."""

from __future__ import annotations

import asyncio
from typing import Any, ClassVar

from paper_app.embeddings.base import EmbeddingError, EmbeddingProvider
from paper_app.logging import get_logger

logger = get_logger(__name__)


class SentenceTransformerProvider(EmbeddingProvider):
    name: ClassVar[str] = "sentence-transformers"

    def __init__(
        self,
        model: str = "sentence-transformers/all-MiniLM-L6-v2",
        dimensions: int | None = None,
        *,
        batch_size: int = 32,
        device: str | None = None,
        cache_dir: str | None = None,
        normalize: bool = True,
    ) -> None:
        # MiniLM defaults to 384; the real value is read from the model on load.
        super().__init__(
            model=model,
            dimensions=dimensions or 384,
            batch_size=batch_size,
            normalize=normalize,
        )
        # CPU unless asked. sentence-transformers moves the model to CUDA whenever
        # one is visible, which is right for a one-shot script and wrong for a
        # long-lived server: measured here, every `paper mcp` process pinned
        # ~1.4 GB of VRAM, so six idle tool processes held 8.8 GB of an 11.6 GB
        # card and left nothing for bulk work.
        #
        # CPU is also the faster choice for the *query* path, which is what these
        # processes mostly do: a single short embedding is dominated by host-to-
        # device transfer, not compute. `ST_DEVICE=cuda` is the opt-in for bulk
        # embedding, where it measured 8.6x faster (86 vs 10 chunks/s).
        self._device = device or "cpu"
        self._cache_dir = cache_dir
        self._encoder: Any = None

    def _load(self) -> Any:  # noqa: ANN401 - returns SentenceTransformer
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer  # noqa: PLC0415
            except ImportError as exc:  # pragma: no cover
                raise EmbeddingError(
                    "sentence-transformers not installed — "
                    "pip install 'paper-app-backend[local]'"
                ) from exc
            kwargs: dict[str, Any] = {"device": self._device} if self._device else {}
            if self._cache_dir:
                kwargs["cache_folder"] = str(self._cache_dir)
            logger.info("st_loading_model", extra={"model": self._model})
            self._encoder = SentenceTransformer(self._model, **kwargs)
            self._dimensions = int(self._encoder.get_sentence_embedding_dimension())
            logger.info("st_model_ready", extra={"model": self._model, "dims": self._dimensions})
        return self._encoder

    async def warmup(self) -> None:
        await asyncio.to_thread(self._load)

    def count_tokens(self, text: str) -> int:
        encoder = self._encoder
        if encoder is None:
            from paper_app.infra.text import approx_tokens

            return approx_tokens(text)
        try:
            return len(encoder.tokenizer.encode(text, add_special_tokens=False))
        except Exception:  # noqa: BLE001 - tokenizer shape varies by version
            from paper_app.infra.text import approx_tokens

            return approx_tokens(text)

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        encoder = await asyncio.to_thread(self._load)
        cleaned = [text if text.strip() else " " for text in texts]
        vectors = await asyncio.to_thread(
            encoder.encode,
            cleaned,
            batch_size=self.batch_size,
            normalize_embeddings=self._normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [list(map(float, row)) for row in vectors]