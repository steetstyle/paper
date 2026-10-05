"""Shared pytest fixtures: fakes for every external dependency."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
os.environ.setdefault("VECTOR_BACKEND", "memory")
os.environ.setdefault("EMBEDDING_PROVIDER", "hashing")
os.environ.setdefault("EMBEDDING_DIMENSIONS", "64")
os.environ.setdefault("EMBEDDING_MODEL", "hashing-test")
os.environ.setdefault("CHUNK_MAX_TOKENS", "120")
os.environ.setdefault("CHUNK_OVERLAP_TOKENS", "20")
os.environ.setdefault("CHUNK_MIN_TOKENS", "5")
os.environ.setdefault("ARXIV_REQUEST_INTERVAL_SECONDS", "0")
os.environ.setdefault("ARXIV_DOWNLOAD_DELAY_SECONDS", "0")

from app.clients.arxiv.parser import render_entry_xml, render_feed_xml  # noqa: E402
from app.config import AppSettings, reload_settings  # noqa: E402
from app.db.session import create_all, dispose_engines, session_scope  # noqa: E402
from app.domain.models import PaperMetadata  # noqa: E402
from app.embeddings.hashing_provider import HashingEmbeddingProvider  # noqa: E402
from app.infra.storage import MemoryBlobStore  # noqa: E402

FIXTURE_ENTRY = {
    "id": "http://arxiv.org/abs/1706.03762v5",
    "title": "Attention Is All You Need",
    "summary": (
        "The dominant sequence transduction models are based on complex recurrent "
        "or convolutional neural networks. We propose a new simple network architecture, "
        "the Transformer, based solely on attention mechanisms."
    ),
    "authors": ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar"],
    "primary": "cs.CL",
    "categories": ["cs.CL", "cs.LG"],
    "published": "2017-06-12T00:00:00Z",
    "updated": "2023-08-02T00:00:00Z",
}


@pytest.fixture
def settings(tmp_path: Path) -> AppSettings:
    os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{tmp_path / 'test.db'}"
    os.environ["STORAGE_ROOT"] = str(tmp_path / "blobs")
    os.environ["VECTOR_BACKEND"] = "memory"
    os.environ["EMBEDDING_DIMENSIONS"] = "64"
    # Set before create_all seeds the registry, so both agree on the name.
    os.environ["EMBEDDING_MODEL"] = "hashing-test"
    os.environ["EMBEDDING_SPACE"] = "test"
    config = reload_settings()
    yield config


@pytest.fixture(autouse=True)
async def _database(settings: AppSettings):
    await create_all(settings.database)
    yield
    await dispose_engines()


@pytest.fixture
def feed_xml() -> str:
    return render_feed_xml([render_entry_xml(FIXTURE_ENTRY)], total=1)


@pytest.fixture
def feed_xml_multi() -> str:
    entries = []
    for index in range(3):
        spec = dict(FIXTURE_ENTRY)
        spec["id"] = f"http://arxiv.org/abs/1706.0376{index}v1"
        spec["title"] = f"Attention Is All You Need ({index})"
        entries.append(render_entry_xml(spec))
    return render_feed_xml(entries, total=3)


@pytest.fixture
def paper_metadata() -> PaperMetadata:
    from xml.etree import ElementTree as ET

    from app.clients.arxiv.parser import parse_entry

    entry = ET.fromstring(f"<entry xmlns:arxiv='http://arxiv.org/schemas/atom'>"
                          f"{render_entry_xml(FIXTURE_ENTRY)}</entry>")
    return parse_entry(entry).metadata


@pytest.fixture
def embedding_provider() -> HashingEmbeddingProvider:
    return HashingEmbeddingProvider(model="hashing-test", dimensions=64)


@pytest.fixture
def blob_store() -> MemoryBlobStore:
    return MemoryBlobStore()


@pytest.fixture
def now() -> datetime:
    return datetime.now(UTC)

@pytest.fixture
async def session(settings: AppSettings):
    """A transactional session for repository-level tests."""
    async with session_scope(settings.database) as session:
        yield session
