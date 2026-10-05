"""Extraction, embeddings and vector-store unit tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.clients.content.mineru import (
    ExtractionError,
    MineruExtractor,
    _build_document,
    _render_block,
)
from app.config import EmbeddingSettings, MineruSettings
from app.db.spaces import EmbeddingSpace
from app.db.vector_store.base import VectorFilter
from app.db.vector_store.memory_store import InMemoryVectorStore
from app.domain.enums import ContentSource, MineruBackend
from app.domain.models import VectorRecord
from app.embeddings.base import EmbeddingError
from app.embeddings.hashing_provider import HashingEmbeddingProvider
from app.embeddings.registry import build_provider, register_provider
from app.infra.text import split_sentences, strip_arxiv_noise


class TestMineruExtractor:
    def test_constructs_with_default_backend_order(self) -> None:
        """Regression: every backend takes settings, so this must not raise."""
        extractor = MineruExtractor()
        assert isinstance(extractor.available_backends, list)

    def test_unknown_backend_name_is_ignored(self) -> None:
        extractor = MineruExtractor(MineruSettings(backend_order=["nope", "pypdf"]))
        available = extractor.available_backends
        # "nope" is dropped; only real, importable backends survive.
        assert all(name != "nope" for name in available)
        assert all(name in {b.value for b in MineruBackend} for name in available)

    async def test_no_backend_raises_clear_error(self, tmp_path: Path) -> None:
        extractor = MineruExtractor(MineruSettings(backend_order=["cli"]))
        pdf = tmp_path / "x.pdf"
        pdf.write_bytes(b"%PDF-1.4")
        if extractor.available_backends:  # magic-pdf present in this environment
            pytest.skip("a cli backend is actually available")
        with pytest.raises(ExtractionError, match="no PDF extraction backend"):
            await extractor.extract_pdf(pdf)

    async def test_missing_file_raises(self) -> None:
        extractor = MineruExtractor()
        with pytest.raises(ExtractionError, match="not found"):
            await extractor.extract_pdf(Path("/nonexistent/file.pdf"))


class TestMineruBlockRendering:
    def test_title_becomes_h1(self) -> None:
        assert _render_block({"type": "title", "text": "Introduction"}) == "# Introduction"

    def test_section_header_respects_level(self) -> None:
        assert _render_block({"type": "section_header", "text": "Method", "level": 3}) == "### Method"

    def test_equation_is_wrapped(self) -> None:
        rendered = _render_block({"type": "equation", "latex": "x^2"})
        assert rendered.startswith("$$") and rendered.endswith("$$")

    def test_plain_text_passes_through(self) -> None:
        assert _render_block({"type": "text", "text": "Hello."}) == "Hello."

    def test_empty_block_is_dropped(self) -> None:
        assert _render_block({"type": "text", "text": "  "}) == ""

    def test_markdown_path_produces_text(self) -> None:
        document = _build_document(
            markdown="# Title\n\nSome **bold** prose with a [link](http://x).",
            blocks=[],
            source=ContentSource.PDF_MINERU,
            backend="test",
            meta={},
        )
        assert document.is_usable
        assert "**" not in document.text
        assert "http://x" not in document.text
        assert document.meta["tokens"] > 0


class TestSentenceSplitting:
    def test_splits_capitalised_sentences(self) -> None:
        assert len(split_sentences("One thing. Two things. Three things.")) == 3

    def test_keeps_abbreviations_intact(self) -> None:
        text = "See Vaswani et al. for details. We follow that setup exactly."
        sentences = split_sentences(text)
        assert len(sentences) == 2
        assert "et al. for details." in sentences[0]

    def test_keeps_lowercase_continuations(self) -> None:
        sentences = split_sentences("The result holds. as shown in the appendix.")
        assert len(sentences) == 1

    def test_handles_quoted_endings(self) -> None:
        sentences = split_sentences('He said "no." Then he left.')
        assert len(sentences) == 2

    def test_empty_input(self) -> None:
        assert split_sentences("   ") == []


class TestArxivNoise:
    def test_strips_boilerplate(self) -> None:
        noisy = "arXiv:2401.01234v1 [cs.LG] Download PDF\n\nReal content here."
        assert "arXiv:2401" not in strip_arxiv_noise(noisy)
        assert "Real content here." in strip_arxiv_noise(noisy)


class TestHashingProvider:
    async def test_deterministic_and_normalised(self) -> None:
        provider = HashingEmbeddingProvider(model="t", dimensions=32)
        first = await provider.embed_documents(["hello world", "attention"])
        second = await provider.embed_documents(["hello world", "attention"])
        assert first == second
        for vector in first:
            assert len(vector) == 32
            assert abs(sum(v * v for v in vector) ** 0.5 - 1.0) < 1e-5

    async def test_empty_string_is_safe(self) -> None:
        provider = HashingEmbeddingProvider(model="t", dimensions=16)
        vectors = await provider.embed_documents([""])
        assert len(vectors[0]) == 16

    async def test_empty_batch(self) -> None:
        provider = HashingEmbeddingProvider(model="t", dimensions=16)
        assert await provider.embed_documents([]) == []


class TestProviderRegistry:
    def test_builds_hashing(self) -> None:
        provider = build_provider(EmbeddingSettings(provider="hashing", model="t", dimensions=64))
        assert isinstance(provider, HashingEmbeddingProvider)
        assert provider.dimensions == 64

    def test_unknown_provider_raises(self) -> None:
        with pytest.raises(EmbeddingError, match="unknown embedding provider"):
            build_provider(EmbeddingSettings(provider="does-not-exist"))

    def test_custom_provider_can_be_registered(self) -> None:
        class Dummy(HashingEmbeddingProvider):
            pass

        register_provider("dummy", Dummy)
        provider = build_provider(EmbeddingSettings(provider="DUMMY", model="t", dimensions=32))
        assert isinstance(provider, Dummy)


def _space(dimensions: int = 4, name: str = "mem") -> EmbeddingSpace:
    return EmbeddingSpace(
        name=name, provider="hashing", model="t", dimensions=dimensions
    )


class TestInMemoryVectorStore:
    async def test_upsert_search_delete(self) -> None:
        store = InMemoryVectorStore(_space(4))
        records = [
            VectorRecord(chunk_id="a", paper_id="p1", vector=[1.0, 0, 0, 0], payload={"paper_id": "p1", "categories": ["cs.LG"]}),
            VectorRecord(chunk_id="b", paper_id="p2", vector=[0, 1.0, 0, 0], payload={"paper_id": "p2", "categories": ["cs.CL"]}),
        ]
        assert await store.upsert(records) == 2
        assert await store.count() == 2

        ranked = await store.search([1.0, 0, 0, 0], top_k=2)
        assert ranked[0][0] == "a"
        assert ranked[0][1] == pytest.approx(1.0)

        assert await store.delete_for_papers(["p1"]) == 1
        assert await store.count() == 1

    async def test_filters(self) -> None:
        store = InMemoryVectorStore(_space(2))
        await store.upsert(
            [
                VectorRecord(chunk_id="a", paper_id="p1", vector=[1.0, 0], payload={"paper_id": "p1", "categories": ["cs.LG"], "source": "arxiv_html"}),
                VectorRecord(chunk_id="b", paper_id="p2", vector=[1.0, 0], payload={"paper_id": "p2", "categories": ["cs.CL"], "source": "pdf_mineru"}),
            ]
        )
        assert [c for c, _ in await store.search([1, 0], filters=VectorFilter(categories=["cs.CL"]))] == ["b"]
        assert [c for c, _ in await store.search([1, 0], filters=VectorFilter(sources=["arxiv_html"]))] == ["a"]
        assert [c for c, _ in await store.search([1, 0], filters=VectorFilter(paper_ids=["p1"]))] == ["a"]
        assert await store.search([1, 0], filters=VectorFilter(paper_ids=["nope"])) == []

    async def test_dimension_mismatch_is_rejected(self) -> None:
        store = InMemoryVectorStore(_space(4))
        with pytest.raises(ValueError, match="space 'mem' expects 4-dim"):
            await store.upsert([VectorRecord(chunk_id="a", paper_id="p", vector=[1.0, 0])])

    async def test_upsert_replaces_same_chunk(self) -> None:
        store = InMemoryVectorStore(_space(2))
        await store.upsert([VectorRecord(chunk_id="a", paper_id="p", vector=[1.0, 0.0])])
        await store.upsert([VectorRecord(chunk_id="a", paper_id="p", vector=[0.0, 1.0])])
        assert await store.count() == 1