"""Composition root.

One place where concrete implementations are chosen. Everything else receives
its collaborators through constructor injection, which is what makes the whole
stack testable with fakes.

Embedding spaces are resolved here too: a space names a model plus the physical
table its vectors live in, and every space gets its own vector store, its own
pipeline steps and its own cache slot.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.clients.arxiv.client import ArxivClient
from app.clients.content.fetcher import ContentFetcher
from app.clients.content.mineru import MineruExtractor
from app.config import AppSettings, get_settings
from app.db.session import get_session_factory
from app.db.space_repository import EmbeddingSpaceRepository
from app.db.spaces import EmbeddingSpace
from app.db.vector_store import build_vector_store
from app.db.vector_store.base import VectorStore
from app.embeddings.base import EmbeddingProvider
from app.embeddings.registry import build_provider
from app.infra.http import HttpFetcher
from app.infra.storage import BlobStore, LocalBlobStore
from app.logging import get_logger
from app.pipeline.steps import (
    BuildSectionsStep,
    ChunkTextStep,
    EmbedChunksStep,
    ExtractAssetsStep,
    ExtractReferencesStep,
    ExtractTextStep,
    FetchContentStep,
    FetchMetadataStep,
    FinalizeStep,
    IndexVectorsStep,
    PersistMetadataStep,
    Step,
)
from app.services.chunker import ChunkingService

logger = get_logger(__name__)


@dataclass(slots=True)
class Container:
    """Lazily-built singleton graph for a process."""

    settings: AppSettings
    _http: HttpFetcher | None = None
    _arxiv: ArxivClient | None = None
    _fetcher: ContentFetcher | None = None
    _mineru: MineruExtractor | None = None
    _embeddings: dict[str, EmbeddingProvider] = field(default_factory=dict)
    _chunker: ChunkingService | None = None
    _blobs: BlobStore | None = None
    _stores: dict[str, VectorStore] = field(default_factory=dict)
    _active_space: EmbeddingSpace | None = None

    # ------------------------------------------------------------- transport
    @property
    def http(self) -> HttpFetcher:
        if self._http is None:
            self._http = HttpFetcher(self.settings.arxiv)
        return self._http

    # --------------------------------------------------------------- clients
    @property
    def arxiv(self) -> ArxivClient:
        if self._arxiv is None:
            self._arxiv = ArxivClient(self.settings.arxiv, self.http)
        return self._arxiv

    @property
    def blob_store(self) -> BlobStore:
        if self._blobs is None:
            self._blobs = LocalBlobStore(self.settings.storage)
        return self._blobs

    @property
    def fetcher(self) -> ContentFetcher:
        if self._fetcher is None:
            self._fetcher = ContentFetcher(self.settings.arxiv, self.http, self.blob_store)
        return self._fetcher

    @property
    def mineru(self) -> MineruExtractor:
        if self._mineru is None:
            self._mineru = MineruExtractor(self.settings.mineru)
        return self._mineru

    # ------------------------------------------------------------------ db
    @property
    def session_factory(self):  # noqa: ANN201 - async_sessionmaker[AsyncSession]
        return get_session_factory(self.settings.database)

    # ----------------------------------------------------------- embeddings
    @property
    def embeddings(self) -> EmbeddingProvider:
        """The provider described by ``EMBEDDING_*`` settings (the default space)."""
        return self.provider_for(self.default_space)

    def provider_for(
        self, space: EmbeddingSpace, device: str | None = None
    ) -> EmbeddingProvider:
        """Provider for a space, on a given device.

        A space naming the configured provider reuses the shared instance; any
        other space gets its own, built from the space's own model id.

        The cache is keyed by **space and device together**, because the model is
        loaded once and a loaded model belongs to the device it was loaded on —
        moving an existing encoder would leave the previous device holding a copy
        of the weights. Keying on the space alone would therefore make the second
        device silently reuse the first one's.

        The cost is real and worth stating: a second device means a second copy of
        the model, which is exactly the VRAM this process avoids holding by
        default. That is why ``device`` is an explicit per-operation argument
        rather than something that flips silently — asking for ``cuda`` is asking
        for GPU memory, and it happens only on the call that asked.
        """
        resolved = (device or self.settings.embedding.device or "cpu").strip().lower()
        key = f"{space.fingerprint}@{resolved}"
        if key in self._embeddings:
            return self._embeddings[key]
        provider = self._build_provider(space, resolved)
        self._embeddings[key] = provider
        logger.info(
            "embedding_provider_ready",
            extra={
                "space": space.name,
                "provider": provider.name,
                "model": provider.model,
                "dims": provider.dimensions,
                "device": resolved,
            },
        )
        return provider

    def _build_provider(self, space: EmbeddingSpace, device: str | None = None) -> EmbeddingProvider:
        if space.fingerprint == self.default_space.fingerprint:
            if device and device != (self.settings.embedding.device or "cpu"):
                return build_provider(
                    self.settings.embedding.model_copy(update={"device": device})
                )
            return build_provider(self.settings.embedding)
        # A managed space: same provider family, its own model/dimensions.
        from app.config import EmbeddingSettings  # noqa: PLC0415

        return build_provider(
            EmbeddingSettings(
                provider=space.provider,
                model=space.model,
                dimensions=space.dimensions,
                batch_size=self.settings.embedding.batch_size,
                normalize_embeddings=self.settings.embedding.normalize_embeddings,
                openai_api_key=self.settings.embedding.openai_api_key,
                openai_base_url=self.settings.embedding.openai_base_url,
                # The requested device, not the configured one: a managed space
                # asked for on cuda must not silently come back on cpu.
                device=device or self.settings.embedding.device,
                cache_dir=self.settings.embedding.cache_dir,
                space=space.name,
            )
        )

    @property
    def chunker(self) -> ChunkingService:
        if self._chunker is None:
            self._chunker = ChunkingService(
                self.settings.chunking, token_counter=self.embeddings.count_tokens
            )
        return self._chunker

    # ---------------------------------------------------------------- spaces
    @property
    def default_space(self) -> EmbeddingSpace:
        """The space described by settings, regardless of what is registered."""
        return EmbeddingSpace.default_for(self.settings)

    @property
    def active_space(self) -> EmbeddingSpace:
        if self._active_space is None:
            self._active_space = self.default_space
        return self._active_space

    def use_space(self, space: EmbeddingSpace) -> EmbeddingSpace:
        self._active_space = space
        return space

    # --------------------------------------------------------- vector stores
    @property
    def vector_store(self) -> VectorStore:
        return self.vector_store_for(self.active_space)

    def vector_store_for(self, space: EmbeddingSpace) -> VectorStore:
        key = f"{space.name}:{space.fingerprint}:{space.distance}"
        store = self._stores.get(key)
        if store is None:
            store = build_vector_store(self.settings, space)
            self._stores[key] = store
            logger.info(
                "vector_store_ready",
                extra={
                    "backend": store.name,
                    "space": space.name,
                    "table": space.resolved_table,
                },
            )
        return store

    # -------------------------------------------------------------- pipeline
    def build_steps(
        self,
        space: EmbeddingSpace | None = None,
        *,
        device: str | None = None,
    ) -> list[Step]:
        """The canonical ingestion pipeline for one embedding space.

        ``device`` applies to the embedding steps only. Extraction has its own
        device, per document, because a corpus holds both a scanned textbook that
        wants the GPU and a three-page note that does not.
        """
        space = space or self.active_space
        provider = self.provider_for(space, device)
        store = self.vector_store_for(space)
        return [
            FetchMetadataStep(self.arxiv),
            PersistMetadataStep(),
            FetchContentStep(self.fetcher),
            ExtractTextStep(self.mineru, self.blob_store),
            ExtractReferencesStep(self.blob_store),
            ExtractAssetsStep(self.blob_store),
            ChunkTextStep(self.chunker),
            BuildSectionsStep(),
            EmbedChunksStep(
                provider,
                space,
                batch_size=self.settings.embedding.batch_size,
                ensure_table=self.settings.vector.auto_create,
            ),
            IndexVectorsStep(store),
            FinalizeStep(),
        ]

    # -------------------------------------------------------------- registry
    async def space_repository(self) -> EmbeddingSpaceRepository:
        """Repository bound to a throwaway session for CLI-style calls.

        Prefer passing an explicit session in request handlers; this exists for
        scripts and the CLI.
        """
        return EmbeddingSpaceRepository(self.session_factory())

    async def resolve_space(self, name: str | None = None) -> EmbeddingSpace:
        """Look a space up in the registry, falling back to settings."""
        repo = await self.space_repository()
        try:
            return await repo.resolve(name, fallback=self.default_space)
        finally:
            await repo._session.close()  # noqa: SLF001 - short-lived helper

    async def startup(self) -> None:
        store = self.vector_store
        if self.settings.vector.auto_create:
            await store.ensure_ready()
        else:
            await store.health()
        await self.embeddings.warmup()

    async def aclose(self) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None
        for store in self._stores.values():
            await store.aclose()
        self._stores.clear()


_container: Container | None = None


def get_container(settings: AppSettings | None = None) -> Container:
    global _container
    if _container is None or (settings is not None and settings is not _container.settings):
        _container = Container(settings or get_settings())
    return _container


def set_container(container: Container | None) -> None:
    """Override the process-wide container (tests, workers)."""
    global _container
    _container = container