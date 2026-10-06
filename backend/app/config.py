"""Typed application settings (12-factor, environment driven).

Each sub-settings object owns an environment prefix so names never collide
across sections (``ARXIV_*``, ``EMBEDDING_*``, ``DATABASE_*`` …). Sub-sections
read ``.env`` themselves, which means an explicitly constructed
``ArxivSettings()`` behaves identically to the nested instance.

``get_settings()`` is cached so the CLI, API and worker share one object.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Literal

from pydantic import Field, computed_field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.domain.enums import VectorBackend

_REPO_ROOT = Path(__file__).resolve().parent.parent
_ENV_FILES = (".env", "../.env")


def _cfg(prefix: str) -> SettingsConfigDict:
    return SettingsConfigDict(
        env_prefix=prefix,
        env_file=_ENV_FILES,
        env_file_encoding="utf-8",
        extra="ignore",
    )


class ArxivSettings(BaseSettings):
    """ArXiv API + download politeness."""

    model_config = _cfg("ARXIV_")

    api_base_url: str = "https://export.arxiv.org/api/query"
    abs_base_url: str = "https://arxiv.org/abs"
    pdf_base_url: str = "https://arxiv.org/pdf"
    html_base_url: str = "https://arxiv.org/html"
    ar5iv_base_url: str = "https://ar5iv.labs.arxiv.org/html"
    user_agent: str = "paper-app-backend/0.1 (+https://github.com/paper-app)"

    max_results_per_request: int = Field(default=100, ge=1, le=2000)
    request_interval_seconds: float = Field(
        default=3.0,
        ge=0.0,
        description="ArXiv asks for at least one request every 3 seconds.",
    )
    connect_timeout_seconds: float = 10.0
    read_timeout_seconds: float = 60.0
    download_timeout_seconds: float = 300.0
    max_retries: int = 4
    backoff_base_seconds: float = 1.5

    max_pdf_bytes: int = 128 * 1024 * 1024
    max_html_bytes: int = 16 * 1024 * 1024
    allow_ar5iv_fallback: bool = True
    download_delay_seconds: float = 0.5


class MineruSettings(BaseSettings):
    """PDF text extraction via MinerU.

    MinerU has changed its public surface several times, so both the CLI shape
    and the Python API are detected at runtime:

    * ``magic-pdf`` 1.x - package ``magic-pdf``, CLI ``magic-pdf -p ... -o ...``
    * ``mineru`` 2.x - CLI ``mineru -p ... -o ...``, API ``do_parse(...)``
    * ``mineru`` 4.x - CLI ``mineru parse ...``, API ``doc_analyze(...)``

    ``backend``/``model_source``/``tier`` only apply where the installed version
    understands them; unsupported options are dropped automatically.
    """

    model_config = _cfg("MINERU_")

    backend_order: list[str] = Field(
        default_factory=lambda: ["python_api", "cli", "pypdf"],
        description="Tried in order until one produces usable text.",
    )
    cli_candidates: list[str] = Field(
        default_factory=lambda: ["mineru", "magic-pdf"], description="In preference order."
    )
    timeout_seconds: float = Field(default=900.0, ge=1.0)
    keep_artifacts: bool = False

    # ------------------------------------------------------------- 2.x options
    backend: str = "pipeline"  # pipeline | vlm-auto-engine | hybrid-auto-engine | ...
    language: str = "en"
    parse_method: str = "auto"  # auto | txt | ocr
    model_source: str = "huggingface"  # huggingface | modelscope | local
    device: str | None = None  # cpu | cuda | cuda:0 | mps | npu
    vram: int | None = None
    server_url: str | None = None
    formula_enable: bool = True
    table_enable: bool = True

    # ------------------------------------------------------------- 4.x options
    tier: str | None = None  # flash | basic | standard | advanced
    effort: str = "high"  # flash | medium | high | xhigh

    # ------------------------------------------------------- shared behaviour
    extract_images: bool = False
    """Extract figures and tables too. Off is text-only: cheaper, and it is
    what the embedding pipeline wants."""
    extra_args: list[str] = Field(default_factory=list)

    @property
    def effort_rank(self) -> int:
        return {"flash": 0, "medium": 1, "high": 2, "xhigh": 3}.get(self.effort, 2)


class EmbeddingSettings(BaseSettings):
    """Pluggable embedding provider."""

    model_config = _cfg("EMBEDDING_")

    provider: str = "hashing"
    model: str = "text-embedding-3-small"
    dimensions: int = Field(default=1536, ge=8)
    batch_size: int = Field(default=64, ge=1)
    normalize_embeddings: bool = True
    space: str = Field(
        default="default",
        description="Name of the embedding space these settings describe. "
        "One table per space, so two models can coexist.",
    )

    device: str | None = Field(
        default=None,
        validation_alias="ST_DEVICE",
        description=(
            "cpu | cuda | cuda:0 | mps. Defaults to cpu: a long-lived server "
            "process pins the model's worth of VRAM otherwise, and single-query "
            "latency is CPU-bound anyway. Set ST_DEVICE=cuda for bulk embedding."
        ),
    )
    cache_dir: Path = Field(default=Path("./.data/models"), validation_alias="ST_MODEL_CACHE_DIR")

    openai_api_key: str | None = Field(default=None, repr=False, validation_alias="OPENAI_API_KEY")
    openai_base_url: str | None = Field(default=None, validation_alias="OPENAI_BASE_URL")

    @field_validator("provider", mode="before")
    @classmethod
    def _normalise_provider(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip().lower().replace("_", "-")
        return value

    @property
    def fingerprint(self) -> str:
        return f"{self.provider}:{self.model}:{self.dimensions}"


class ChunkingSettings(BaseSettings):
    model_config = _cfg("CHUNK_")

    max_tokens: int = Field(default=1000, ge=16)
    overlap_tokens: int = Field(default=150, ge=0)
    min_tokens: int = Field(default=60, ge=1)
    respect_markdown: bool = True

    @model_validator(mode="after")
    def _validate_budget(self) -> ChunkingSettings:
        if self.overlap_tokens >= self.max_tokens:
            raise ValueError("CHUNK_OVERLAP_TOKENS must be smaller than CHUNK_MAX_TOKENS")
        if self.min_tokens > self.max_tokens:
            raise ValueError("CHUNK_MIN_TOKENS must be <= CHUNK_MAX_TOKENS")
        return self


class VectorSettings(BaseSettings):
    model_config = _cfg("VECTOR_")

    backend: VectorBackend = VectorBackend.PGVECTOR
    distance: Literal["cosine", "dot", "euclid"] = "cosine"
    hnsw_m: int = 16
    hnsw_ef_construction: int = 128
    auto_create: bool = Field(
        default=True,
        description="Create a space's table and index on first use. Set false where "
        "schema changes must go through reviewed migrations "
        "(see `paper embedding-space show --sql`).",
    )

    qdrant_url: str = Field(default="http://localhost:6333", validation_alias="QDRANT_URL")
    qdrant_api_key: str | None = Field(
        default=None, repr=False, validation_alias="QDRANT_API_KEY"
    )
    qdrant_collection: str = Field(
        default="paper_chunks", validation_alias="QDRANT_COLLECTION"
    )


class StorageSettings(BaseSettings):
    """Content-addressed blob store for PDFs/HTML/derived artefacts."""

    model_config = _cfg("STORAGE_")

    root: Path = Path("./.data/blobs")
    keep_downloads: bool = True

    @computed_field  # type: ignore[prop-decorator]
    @property
    def resolved_root(self) -> Path:
        path = Path(self.root)
        return path if path.is_absolute() else (_REPO_ROOT / path)


class DatabaseSettings(BaseSettings):
    model_config = _cfg("DATABASE_")

    url: str = "sqlite+aiosqlite:///./.data/paper.db"
    sync_url: str | None = None
    echo: bool = False
    pool_size: int = 10
    max_overflow: int = 20
    pool_pre_ping: bool = True
    create_all: bool = False

    @model_validator(mode="after")
    def _derive_sync_url(self) -> DatabaseSettings:
        if not self.sync_url:
            object.__setattr__(self, "sync_url", _to_sync_url(self.url))
        return self

    @property
    def is_postgres(self) -> bool:
        return "postgres" in self.url


class ApiSettings(BaseSettings):
    model_config = _cfg("API_")

    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    root_path: str = ""
    cors_origins: list[str] = Field(default_factory=list)


class IngestionSettings(BaseSettings):
    model_config = _cfg("INGEST_")

    concurrency: int = Field(default=2, ge=1, le=16)
    request_timeout_seconds: float = 900.0
    delete_existing_embeddings: bool = True
    embed_abstract_as_first_chunk: bool = True


class LoggingSettings(BaseSettings):
    model_config = _cfg("LOG_")

    level: str = "INFO"
    json_output: bool = Field(default=False, validation_alias="LOG_JSON")


class AppSettings(BaseSettings):
    """Root settings object; sub-settings are namespaced by env prefix."""

    model_config = SettingsConfigDict(
        env_file=_ENV_FILES,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    name: str = Field(default="paper-app-backend", validation_alias="APP_NAME")
    env: str = Field(default="development", validation_alias="APP_ENV")

    arxiv: ArxivSettings = Field(default_factory=ArxivSettings)
    mineru: MineruSettings = Field(default_factory=MineruSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    vector: VectorSettings = Field(default_factory=VectorSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    api: ApiSettings = Field(default_factory=ApiSettings)
    ingestion: IngestionSettings = Field(default_factory=IngestionSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)


def _to_sync_url(url: str) -> str:
    """Derive the blocking driver URL used by Alembic."""
    return (
        url.replace("+asyncpg", "+psycopg")
        .replace("+aiosqlite", "")
        .replace("postgresql+psycopg://", "postgresql://")
    )


@functools.lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    return AppSettings()


def reload_settings() -> AppSettings:
    """Clear the cache — used by tests and the ``paper config`` command."""
    get_settings.cache_clear()
    return get_settings()