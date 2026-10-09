"""End-to-end pipeline test with every external dependency faked.

Runs the real eight-step pipeline against a real SQLite database and exercises
content fetching, HTML→markdown, chunking, embedding and vector indexing without
touching the network.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from sqlalchemy import func, select

from paper_app.clients.arxiv.client import ArxivClient
from paper_app.clients.arxiv.parser import parse_feed
from paper_app.config import ChunkingSettings, get_settings
from paper_app.db.models import Chunk, IngestionRun, Paper, RawDocument
from paper_app.db.repositories import RunRepository
from paper_app.db.session import get_session_factory
from paper_app.db.spaces import EmbeddingSpace
from paper_app.db.vector_store.base import VectorFilter, VectorStore
from paper_app.db.vector_store.schema import embedding_table
from paper_app.domain.enums import ContentKind, ContentSource, RunStatus, StepStatus
from paper_app.domain.models import ContentPayload, SearchQuery
from paper_app.embeddings.hashing_provider import HashingEmbeddingProvider
from paper_app.infra.storage import LocalBlobStore
from paper_app.pipeline.context import PipelineContext
from paper_app.pipeline.runner import PipelineRunner
from paper_app.pipeline.steps import (
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
)
from paper_app.services.chunker import ChunkingService
from paper_app.services.ingestion import IngestionService

PAPER_HTML = """<!DOCTYPE html>
<html><head><title>Attention Is All You Need</title>
<style>body{color:red}</style><script>alert('x')</script></head>
<body>
<article class="ltx_document">
<h1 class="ltx_title_document">Attention Is All You Need</h1>
<section><h2>1 Introduction</h2>
<p>Recurrent neural networks have long been the dominant approach for sequence
modelling. We propose the Transformer, a network architecture based solely on
attention mechanisms, dispensing with recurrence and convolutions entirely.</p>
</section>
<section><h2>2 Model Architecture</h2>
<p>The Transformer allows significantly more parallelization during training.
Scaled dot-product attention is computed as softmax of the query-key inner
product divided by the square root of the key dimension.</p>
<h3>2.1 Attention Complexity</h3>
<p>Self-attention is O(n^2) in sequence length, which motivates sparse variants
used in longer-context models.</p>
</section>
</article>
</body></html>
"""

FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">
  <opensearch:totalResults>1</opensearch:totalResults>
  <opensearch:startIndex>0</opensearch:startIndex>
  <opensearch:itemsPerPage>1</opensearch:itemsPerPage>
  <entry>
    <id>http://arxiv.org/abs/1706.03762v5</id>
    <published>2017-06-12T00:00:00Z</published>
    <updated>2023-08-02T00:00:00Z</updated>
    <title>Attention Is All You Need</title>
    <summary>We propose the Transformer, based solely on attention.</summary>
    <author><name>Ashish Vaswani</name></author>
    <author><name>Noam Shazeer</name></author>
    <arxiv:primary_category xmlns:arxiv="http://arxiv.org/schemas/atom" term="cs.CL"/>
    <category term="cs.CL"/>
    <category term="cs.LG"/>
    <link title="pdf" href="http://arxiv.org/pdf/1706.03762v5" rel="related" type="application/pdf"/>
  </entry>
</feed>"""


class FakeHttp:
    """Returns a canned feed for any ArXiv API call."""

    async def get_text(self, url: str, params=None, **kwargs):  # noqa: ANN001, ANN003
        class Response:
            status_code = 200
            content = FEED.encode()
            headers = {"content-type": "application/atom+xml"}

        return Response()


class FakeContentFetcher:
    """Serves a canned source artefact and counts downloads."""

    def __init__(self, *, blob_store: LocalBlobStore, html: str) -> None:
        self._blobs = blob_store
        self._html = html
        self.downloads = 0

    async def fetch_full_text(self, paper, *, prefer_html: bool = True):  # noqa: ANN001
        from paper_app.clients.content.fetcher import FetchOutcome

        self.downloads += 1
        if prefer_html:
            raw = self._html.encode("utf-8")
            kind, content_type, suffix = ContentKind.HTML, "text/html", "html"
            source = ContentSource.ARXIV_HTML
        else:
            raw = b"%PDF-1.4 fake pdf payload"
            kind, content_type, suffix = ContentKind.PDF, "application/pdf", "pdf"
            source = ContentSource.PDF_MINERU

        blob = self._blobs.put_bytes(raw, prefix="raw")
        return FetchOutcome(
            payload=ContentPayload(
                kind=kind,
                uri=blob.uri,
                content_type=content_type,
                size_bytes=len(raw),
                sha256=blob.sha256,
                source_url=f"https://arxiv.org/{suffix}/{paper.versioned_id}",
                local_path=None,
            ),
            blob=blob,
            temp_path=None,
            source=source,
        )


class FakeVectorStore(VectorStore):
    """Scores by lexical overlap so assertions stay deterministic."""

    name = "fake"

    def __init__(self, space: EmbeddingSpace) -> None:
        super().__init__(space)
        self._records: dict[str, tuple[dict, list[float]]] = {}
        self._texts: dict[str, str] = {}

    @property
    def dimensions(self) -> int:
        return self.space.dimensions

    async def ensure_ready(self) -> None:
        return None

    async def upsert(self, records) -> int:  # noqa: ANN001
        for record in records:
            self._records[record.chunk_id] = (dict(record.payload), list(record.vector))
        return len(records)

    async def search(self, vector, *, top_k=10, filters=None, min_score=None):  # noqa: ANN001, ANN003
        filters = filters or VectorFilter()
        scored: list[tuple[str, float]] = []
        for chunk_id, (payload, stored) in self._records.items():
            if filters.paper_ids and payload.get("paper_id") not in filters.paper_ids:
                continue
            scored.append((chunk_id, _cosine(vector, stored)))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]

    async def delete_for_papers(self, paper_ids) -> int:  # noqa: ANN001
        targets = set(paper_ids)
        doomed = [k for k, (p, _) in self._records.items() if p.get("paper_id") in targets]
        for chunk_id in doomed:
            del self._records[chunk_id]
        return len(doomed)

    async def count(self) -> int:
        return len(self._records)

    def register_text(self, chunk_id: str, text: str) -> None:
        self._texts[chunk_id] = text


def _cosine(a: list[float], b: list[float]) -> float:
    import math

    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


@pytest.fixture
def blob_store(tmp_path: Path) -> LocalBlobStore:
    return LocalBlobStore(root=tmp_path / "blobs")


@pytest.fixture
def provider() -> HashingEmbeddingProvider:
    return HashingEmbeddingProvider(model="hashing-test", dimensions=64)


@pytest.fixture
def chunker() -> ChunkingService:
    return ChunkingService(
        ChunkingSettings(max_tokens=80, overlap_tokens=15, min_tokens=5),
        token_counter=lambda text: max(1, len(text) // 4),
    )


@pytest.fixture
def space() -> EmbeddingSpace:
    return EmbeddingSpace(
        name="test", provider="hashing", model="hashing-test", dimensions=64
    )


@pytest.fixture
def vector_store(space: EmbeddingSpace) -> FakeVectorStore:
    return FakeVectorStore(space)


@pytest.fixture
def content_fetcher(blob_store: LocalBlobStore) -> FakeContentFetcher:
    return FakeContentFetcher(blob_store=blob_store, html=PAPER_HTML)


@pytest.fixture
def pipeline_steps(  # noqa: PLR0913
    settings, provider, chunker, vector_store, content_fetcher, blob_store, space
) -> list:
    arxiv_client = ArxivClient(get_settings().arxiv, FakeHttp())
    return [
        FetchMetadataStep(arxiv_client),
        PersistMetadataStep(),
        FetchContentStep(content_fetcher),
        ExtractTextStep(StubExtractor(), blob_store),
        ExtractReferencesStep(blob_store),
        ExtractAssetsStep(blob_store),
        ChunkTextStep(chunker),
        EmbedChunksStep(provider, space, batch_size=8),
        IndexVectorsStep(vector_store),
        FinalizeStep(),
    ]


class StubExtractor:
    """Pretends MinerU exists so the test does not need a PDF toolchain."""

    available_backends = ["stub"]

    async def extract_pdf(self, path: Path, request=None):  # noqa: ARG002
        from paper_app.clients.content.mineru import ExtractionError

        raise ExtractionError("no PDF in this test")


async def _run_pipeline(settings, steps, vector_store, arxiv_id="1706.03762", *, prefer_html=True):
    session_factory = get_session_factory(settings.database)
    async with session_factory() as session:
        run = await RunRepository(session).create(arxiv_id=arxiv_id, trigger="test")
        await session.commit()
        run_id = run.id

    async with session_factory() as session:
        ctx = PipelineContext(
            session=session, run_id=run_id, arxiv_id=arxiv_id, prefer_html=prefer_html
        )
        await PipelineRunner(steps).run(ctx)
        return run_id, ctx


class TestFullPipeline:
    async def test_end_to_end_html_path(
        self, settings, pipeline_steps, vector_store, space
    ) -> None:
        run_id, ctx = await _run_pipeline(settings, pipeline_steps, vector_store)

        assert ctx.content_kind == "html"
        assert ctx.document is not None
        assert ctx.document.source == "arxiv_html"
        assert "Transformer" in ctx.document.text
        assert len(ctx.chunks) >= 2
        assert ctx.embedded_count == len(ctx.chunks)

        session_factory = get_session_factory(settings.database)
        async with session_factory() as session:
            run = await RunRepository(session).get(run_id)
            assert run is not None
            assert run.status == RunStatus.SUCCEEDED
            assert run.error is None
            assert {s.name for s in run.steps} == {
                "fetch_metadata", "persist_metadata", "fetch_content", "extract_text",
                "extract_references", "extract_assets", "chunk_text",
                "embed_chunks", "index_vectors", "finalize",
            }
            assert all(s.status == StepStatus.SUCCEEDED for s in run.steps)
            assert run.chunk_count == len(ctx.chunks)
            assert run.embedding_count == len(ctx.chunks)
            assert all(t >= 0 for t in run.timings.values())

            paper = await session.get(Paper, ctx.paper_id)
            assert paper is not None
            assert paper.arxiv_id == "1706.03762"
            assert paper.version == 5
            assert paper.title == "Attention Is All You Need"
            # Authors are rows now, not a JSON blob: assert the join, not a column.
            assert paper.author_names[0] == "Ashish Vaswani"
            assert paper.categories == ["cs.CL", "cs.LG"]
            assert paper.primary_category == "cs.CL"
            assert paper.ingested_at is not None

            chunks = (await session.execute(
                Chunk.__table__.select().where(Chunk.paper_id == ctx.paper_id)
            )).mappings().all()
            assert len(chunks) == len(ctx.chunks)
            assert [c["ordinal"] for c in chunks] == list(range(len(chunks)))
            assert all(c["token_count"] > 0 for c in chunks)

            # Vectors live in this space's own table, not the legacy one.
            assert space.resolved_table == "embeddings__test"
            space_table = embedding_table(space)
            embeddings = (await session.execute(
                space_table.select().where(space_table.c.paper_id == ctx.paper_id)
            )).mappings().all()
            assert len(embeddings) == len(ctx.chunks)
            assert all(e["dimensions"] == 64 for e in embeddings)
            assert all(len(e["vector"]) == 64 for e in embeddings)
            assert {e["fingerprint"] for e in embeddings} == {space.fingerprint}

            contents = (await session.execute(
                RawDocument.__table__.select().where(RawDocument.paper_id == ctx.paper_id)
            )).mappings().all()
            kinds = {c["kind"] for c in contents}
            assert "html" in kinds and "text" in kinds

        assert await vector_store.count() == len(ctx.chunks)

    async def test_chunks_are_searchable(self, settings, pipeline_steps, vector_store) -> None:
        _, ctx = await _run_pipeline(settings, pipeline_steps, vector_store)
        assert ctx.embedded_count > 0
        assert await vector_store.count() == ctx.embedded_count
        ranked = await vector_store.search([0.1] * 64, top_k=3)
        assert len(ranked) == 3

    async def test_reingest_is_idempotent(self, settings, pipeline_steps, vector_store) -> None:
        first_id, first_ctx = await _run_pipeline(settings, pipeline_steps, vector_store)
        second_id, second_ctx = await _run_pipeline(settings, pipeline_steps, vector_store)

        assert first_id != second_id
        assert first_ctx.paper_id == second_ctx.paper_id
        assert [c.text for c in first_ctx.chunks] == [c.text for c in second_ctx.chunks]
        # Chunks are replaced, not appended.
        assert await vector_store.count() == len(second_ctx.chunks)

        session_factory = get_session_factory(settings.database)
        async with session_factory() as session:
            stored = await session.scalar(
                select(func.count())
                .select_from(Chunk)
                .where(Chunk.paper_id == first_ctx.paper_id)
            )
            assert stored == len(second_ctx.chunks)

    async def test_cached_download_is_skipped(self, settings, pipeline_steps, vector_store) -> None:
        await _run_pipeline(settings, pipeline_steps, vector_store)
        _, ctx = await _run_pipeline(settings, pipeline_steps, vector_store)

        session_factory = get_session_factory(settings.database)
        async with session_factory() as session:
            run = await RunRepository(session).get(
                (await session.execute(
                    IngestionRun.__table__.select().order_by(
                        IngestionRun.created_at.desc()
                    ).limit(1)
                )).mappings().first()["id"]
            )
            fetch_step = next(s for s in run.steps if s.name == "fetch_content")
            assert fetch_step.status == StepStatus.SKIPPED
            assert fetch_step.meta["reason"] == "already_downloaded"

        # Content was reused, not re-downloaded.
        assert len(ctx.chunks) > 0

    async def test_abstract_fallback_when_extraction_fails(
        self, settings, provider, chunker, vector_store, content_fetcher, blob_store, space
    ) -> None:
        from paper_app.clients.content.mineru import ExtractionError

        class FailingExtractor:
            """Stands in for an environment where MinerU cannot run."""

            available_backends: list[str] = []

            async def extract_pdf(self, path: Path, request=None):  # noqa: ARG002
                raise ExtractionError("no extraction backend available")

        steps = [
            FetchMetadataStep(ArxivClient(get_settings().arxiv, FakeHttp())),
            PersistMetadataStep(),
            FetchContentStep(content_fetcher),
            ExtractTextStep(FailingExtractor(), blob_store),
            ChunkTextStep(chunker),
            EmbedChunksStep(provider, space, batch_size=8),
            IndexVectorsStep(vector_store),
            FinalizeStep(),
        ]
        # `prefer_html=False` forces the PDF path, which is what calls MinerU.
        session_factory = get_session_factory(settings.database)
        async with session_factory() as session:
            run = await RunRepository(session).create(arxiv_id="1706.03762", trigger="test")
            await session.commit()
            run_id = run.id
        async with session_factory() as session:
            ctx = PipelineContext(
                session=session, run_id=run_id, arxiv_id="1706.03762", prefer_html=False
            )
            await PipelineRunner(steps).run(ctx)

        assert ctx.content_kind == "pdf"
        assert ctx.document is not None
        assert ctx.document.source == "abstract_only"
        assert "Transformer" in ctx.document.text  # from the abstract
        assert ctx.warnings
        assert ctx.chunks, "a paper must never be dropped even if extraction fails"
        assert ctx.embedded_count == len(ctx.chunks)

        async with session_factory() as session:
            run = await RunRepository(session).get(run_id)
            assert run is not None
            assert run.status == RunStatus.PARTIAL
            extract_step = next(s for s in run.steps if s.name == "extract_text")
            assert extract_step.status == StepStatus.SUCCEEDED
            assert extract_step.meta["fallback"] == "abstract_only"

    async def test_pdf_path_uses_mineru(self, settings, provider, chunker, vector_store, content_fetcher, blob_store, space) -> None:
        from paper_app.clients.content.mineru import ExtractedDocument

        class StubMineru:
            available_backends = ["stub"]

            async def extract_pdf(self, path: Path, request=None):
                return ExtractedDocument(
                    markdown="# 1 Introduction\n\nAttention is all you need.\n\n"
                    "## 2 Results\n\nWe reach 28.4 BLEU on WMT 2014 English-to-German.",
                    source=ContentSource.PDF_MINERU,
                    backend="mineru:stub",
                    text=(
                        "# 1 Introduction\n\nAttention is all you need.\n\n"
                        "## 2 Results\n\nWe reach 28.4 BLEU on WMT 2014 English-to-German."
                    ),
                    meta={"blocks": 2},
                )

        steps = [
            FetchMetadataStep(ArxivClient(get_settings().arxiv, FakeHttp())),
            PersistMetadataStep(),
            FetchContentStep(content_fetcher),
            ExtractTextStep(StubMineru(), blob_store),
            ChunkTextStep(chunker),
            EmbedChunksStep(provider, space, batch_size=8),
            IndexVectorsStep(vector_store),
            FinalizeStep(),
        ]
        run_id, ctx = await _run_pipeline(settings, steps, vector_store, prefer_html=False)
        assert ctx.content_kind == "pdf"
        assert ctx.document is not None
        assert ctx.document.source == "pdf_mineru"
        assert ctx.document.backend == "mineru:stub"
        assert not ctx.warnings

        session_factory = get_session_factory(settings.database)
        async with session_factory() as session:
            run = await RunRepository(session).get(run_id)
            assert run is not None
            assert run.status == RunStatus.SUCCEEDED
            assert run.content_kind == "pdf"
            extract_step = next(s for s in run.steps if s.name == "extract_text")
            assert extract_step.meta["backend"] == "mineru:stub"
            kinds = {
                row["kind"]
                for row in (await session.execute(
                    RawDocument.__table__.select().where(RawDocument.paper_id == ctx.paper_id)
                )).mappings().all()
            }
            assert "pdf" in kinds
            assert "mineru_markdown" in kinds

    async def test_ingestion_service_batch(self, settings, pipeline_steps, vector_store) -> None:
        service = IngestionService(
            arxiv_client=ArxivClient(get_settings().arxiv, FakeHttp()),
            steps=pipeline_steps,
            session_factory=get_session_factory(settings.database),
            vector_store=vector_store,
        )
        result = await service.ingest_paper("1706.03762", requested_by="test")
        assert result.status == RunStatus.SUCCEEDED
        assert len(result.run_ids) == 1
        assert len(result.paper_ids) == 1
        assert result.succeeded


class TestSemanticSearch:
    async def test_query_returns_the_paper(self, settings, provider, chunker, content_fetcher, blob_store, space) -> None:
        from paper_app.db.vector_store.memory_store import InMemoryVectorStore
        from paper_app.services.semantic_search import SemanticSearchService

        # The real in-memory store + hashing provider give deterministic lexical overlap.
        store = InMemoryVectorStore(space)
        steps = [
            FetchMetadataStep(ArxivClient(get_settings().arxiv, FakeHttp())),
            PersistMetadataStep(),
            FetchContentStep(content_fetcher),
            ExtractTextStep(StubExtractor(), blob_store),
            ChunkTextStep(chunker),
            EmbedChunksStep(provider, space, batch_size=8),
            IndexVectorsStep(store),
            FinalizeStep(),
        ]
        _, ctx = await _run_pipeline(settings, steps, store)
        assert ctx.paper_id and ctx.embedded_count > 0

        service = SemanticSearchService(
            vector_store=store,
            embeddings=provider,
            session_factory=get_session_factory(settings.database),
        )
        hits = await service.search("softmax query key dimension", top_k=5)
        assert hits
        top = hits[0]
        assert top.metadata["arxiv_id"] == "1706.03762"
        assert top.metadata["title"] == "Attention Is All You Need"
        assert top.metadata["authors"] == ["Ashish Vaswani", "Noam Shazeer"]
        assert top.metadata["abs_url"].startswith("https://arxiv.org/abs/1706.03762")
        assert top.metadata["categories"]
        assert 0.0 <= top.score <= 1.0
        assert top.text
        # Scores are monotonically non-increasing.
        assert all(hits[i].score >= hits[i + 1].score for i in range(len(hits) - 1))

    async def test_search_respects_paper_filter(
        self, settings, provider, chunker, content_fetcher, blob_store, space
    ) -> None:
        from paper_app.db.vector_store.memory_store import InMemoryVectorStore
        from paper_app.services.semantic_search import SemanticSearchService

        store = InMemoryVectorStore(space)
        steps = [
            FetchMetadataStep(ArxivClient(get_settings().arxiv, FakeHttp())),
            PersistMetadataStep(),
            FetchContentStep(content_fetcher),
            ExtractTextStep(StubExtractor(), blob_store),
            ChunkTextStep(chunker),
            EmbedChunksStep(provider, space, batch_size=8),
            IndexVectorsStep(store),
            FinalizeStep(),
        ]
        _, ctx = await _run_pipeline(settings, steps, store)

        service = SemanticSearchService(
            vector_store=store,
            embeddings=provider,
            session_factory=get_session_factory(settings.database),
        )
        # An unresolvable id used to mean "filter on it" and so matched nothing.
        # Resolution now happens in the service, so a paper id that was never
        # ingested is an error rather than a silently empty result — a search
        # returning nothing because an id was wrong is indistinguishable from one
        # that found no matching text.
        with pytest.raises(LookupError) as excinfo:
            await service.search("attention", top_k=5, paper_ids=["does-not-exist"])
        assert "does-not-exist" in str(excinfo.value)

        # Both spellings reach the same paper: the internal id, and the arXiv id
        # a caller actually types.
        by_internal = await service.search("attention", top_k=5, paper_ids=[ctx.paper_id])
        by_arxiv = await service.search("attention", top_k=5, paper_ids=["1706.03762"])
        assert by_internal and by_arxiv
        assert {hit.paper_id for hit in by_internal} == {ctx.paper_id}
        assert [h.chunk_id for h in by_internal] == [h.chunk_id for h in by_arxiv]


class TestQueryToIngestion:
    async def test_harvest_from_query(self, settings, pipeline_steps, vector_store) -> None:
        service = IngestionService(
            arxiv_client=ArxivClient(get_settings().arxiv, FakeHttp()),
            steps=pipeline_steps,
            session_factory=get_session_factory(settings.database),
            vector_store=vector_store,
        )
        result = await service.ingest_query(
            SearchQuery(raw="cat:cs.CL", max_results=1), limit=1, requested_by="test"
        )
        assert result.status == RunStatus.SUCCEEDED
        assert len(result.paper_ids) == 1


def test_blob_store_is_content_addressed(tmp_path: Path) -> None:
    store = LocalBlobStore(root=tmp_path / "blobs")
    first = store.put_bytes(PAPER_HTML.encode(), prefix="raw")
    second = store.put_bytes(PAPER_HTML.encode(), prefix="raw")
    assert first.sha256 == second.sha256 == hashlib.sha256(PAPER_HTML.encode()).hexdigest()
    assert first.path == second.path
    assert store.path_for(first.sha256).read_text() == PAPER_HTML


def test_html_extraction_drops_chrome() -> None:
    from paper_app.clients.content.html_extractor import html_to_markdown

    markdown = html_to_markdown(PAPER_HTML)
    assert "alert(" not in markdown
    assert "color:red" not in markdown
    assert "## 1 Introduction" in markdown or "# 1 Introduction" in markdown
    assert "softmax" in markdown


def test_feed_parse_sanity() -> None:
    page = parse_feed(FEED)
    assert page.total_results == 1
    assert page.hits[0].metadata.version == 5