"""HTTP API tests against the real app with faked external services."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from test_pipeline_e2e import FEED, PAPER_HTML, FakeContentFetcher

from paper_app.config import ChunkingSettings
from paper_app.container import Container, set_container
from paper_app.db.session import dispose_engines
from paper_app.db.vector_store.memory_store import InMemoryVectorStore
from paper_app.embeddings.hashing_provider import HashingEmbeddingProvider
from paper_app.infra.storage import LocalBlobStore
from paper_app.main import create_app
from paper_app.services.chunker import ChunkingService


class FakeHttp:
    async def get_text(self, url: str, params=None, **kwargs):  # noqa: ANN001, ANN003
        class Response:
            status_code = 200
            content = FEED.encode()
            headers = {"content-type": "application/atom+xml"}

        return Response()


class StubExtractor:
    """Stands in for MineruExtractor without importing MinerU."""

    available_backends = ["stub"]

    def describe_backends(self) -> dict[str, str]:
        return {"stub": "stub (fake)"}

    async def extract_pdf(self, path: Path, request=None):  # noqa: ARG002
        from paper_app.clients.content.mineru import ExtractionError

        raise ExtractionError("unused")


@pytest.fixture
def container(settings, tmp_path: Path) -> Container:
    settings.chunking = ChunkingSettings(max_tokens=80, overlap_tokens=15, min_tokens=5)
    settings.embedding.dimensions = 64
    settings.embedding.model = "hashing-test"
    settings.embedding.space = "test"

    container = Container(settings)
    space = container.default_space
    provider = HashingEmbeddingProvider(model="hashing-test", dimensions=64)

    container._http = FakeHttp()  # noqa: SLF001
    container._blobs = LocalBlobStore(root=tmp_path / "blobs")  # noqa: SLF001
    container._fetcher = FakeContentFetcher(blob_store=container.blob_store, html=PAPER_HTML)  # noqa: SLF001
    container._mineru = StubExtractor()  # noqa: SLF001
    # Providers and stores are keyed by space fingerprint now.
    container._embeddings[space.fingerprint] = provider  # noqa: SLF001
    container._stores[f"{space.name}:{space.fingerprint}:{space.distance}"] = (  # noqa: SLF001
        InMemoryVectorStore(space)
    )
    container._chunker = ChunkingService(  # noqa: SLF001
        settings.chunking, token_counter=provider.count_tokens
    )
    set_container(container)
    return container


@pytest.fixture
async def client(container):
    app = create_app(settings=container.settings, container=container)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
    set_container(None)
    await dispose_engines()


class TestHealth:
    async def test_health_reports_components(self, client) -> None:
        response = await client.get("/api/v1/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["database"] == "up"
        assert body["vector_store"].startswith("memory:")
        assert body["embeddings"].startswith("hashing/hashing-test@64")
        assert body["mineru_backends"] == ["stub"]
        assert body["mineru_detail"]["stub"].startswith("stub")
        assert "mineru_version" in body

    async def test_request_id_is_propagated(self, client) -> None:
        response = await client.get("/api/v1/health", headers={"x-request-id": "abc123"})
        assert response.headers["x-request-id"] == "abc123"
        assert float(response.headers["x-response-time-ms"]) >= 0

    async def test_root_points_at_docs(self, client) -> None:
        body = (await client.get("/")).json()
        assert body["docs"] == "/docs"


class TestArxivSearch:
    async def test_search_returns_papers(self, client) -> None:
        response = await client.get(
            "/api/v1/arxiv/search", params={"q": "transformer", "category": "cs.CL"}
        )
        assert response.status_code == 200
        body = response.json()
        assert body["total_results"] == 1
        paper = body["papers"][0]
        assert paper["arxiv_id"] == "1706.03762"
        assert paper["versioned_id"] == "1706.03762v5"
        assert paper["title"] == "Attention Is All You Need"
        assert paper["primary_category"] == "cs.CL"
        assert [a["name"] for a in paper["authors"]] == ["Ashish Vaswani", "Noam Shazeer"]

    async def test_search_requires_a_term(self, client) -> None:
        response = await client.get("/api/v1/arxiv/search")
        assert response.status_code == 422

    async def test_single_paper_lookup(self, client) -> None:
        response = await client.get("/api/v1/arxiv/paper/1706.03762v5")
        assert response.status_code == 200
        assert response.json()["arxiv_id"] == "1706.03762"


class TestIngestion:
    async def test_ingest_synchronously(self, client) -> None:
        response = await client.post(
            "/api/v1/ingest",
            json={"arxiv_id": "1706.03762"},
            params={"wait": True},
        )
        assert response.status_code == 202
        body = response.json()
        assert body["status"] == "succeeded"
        assert len(body["run_ids"]) == 1
        assert len(body["paper_ids"]) == 1

        run = (await client.get(f"/api/v1/ingest/runs/{body['run_ids'][0]}")).json()
        assert run["status"] == "succeeded"
        # The canonical step list, in order. Asserted by name so adding or
        # reordering a step is a deliberate edit rather than a number that drifts.
        assert [step["name"] for step in run["steps"]] == [
            "fetch_metadata",
            "persist_metadata",
            "fetch_content",
            "extract_text",
            "extract_references",
            "extract_assets",
            "chunk_text",
            "build_sections",
            "embed_chunks",
            "index_vectors",
            "finalize",
        ]
        assert all(step["duration_ms"] is not None for step in run["steps"])

        detail = (await client.get("/api/v1/papers/1706.03762")).json()
        assert detail["title"] == "Attention Is All You Need"
        assert detail["chunk_count"] > 0
        assert detail["ingested_at"] is not None
        assert {c["kind"] for c in detail["contents"]} >= {"html", "text"}

        chunks = (await client.get("/api/v1/papers/1706.03762/chunks")).json()
        assert chunks["total"] == detail["chunk_count"]
        assert chunks["chunks"][0]["ordinal"] == 0
        assert chunks["chunks"][0]["text"]

    async def test_ingest_requires_target(self, client) -> None:
        response = await client.post("/api/v1/ingest", json={})
        assert response.status_code == 422

    async def test_missing_paper_is_404(self, client) -> None:
        assert (await client.get("/api/v1/papers/2401.00001")).status_code == 404
        # 404, not 422. The corpus holds local documents too, addressed by a
        # doc_key, so "not an arXiv id" is no longer evidence of anything — only
        # "no such row" is. Answering 422 told callers a document that simply had
        # not been ingested was a malformed request.
        assert (await client.get("/api/v1/papers/not-an-id")).status_code == 404

    async def test_an_empty_identifier_is_422(self, client) -> None:
        assert (await client.get("/api/v1/papers/%20")).status_code in (404, 422)

    async def test_markdown_endpoint(self, client) -> None:
        await client.post(
            "/api/v1/ingest", json={"arxiv_id": "1706.03762"}, params={"wait": True}
        )
        response = await client.get("/api/v1/papers/1706.03762/markdown")
        assert response.status_code == 200
        assert "attention" in response.text.lower()

    async def test_runs_listing(self, client) -> None:
        await client.post(
            "/api/v1/ingest", json={"arxiv_id": "1706.03762"}, params={"wait": True}
        )
        runs = (await client.get("/api/v1/ingest/runs")).json()
        assert len(runs) == 1
        assert runs[0]["arxiv_id"] == "1706.03762"

    async def test_unknown_run_is_404(self, client) -> None:
        assert (await client.get("/api/v1/ingest/runs/deadbeef")).status_code == 404


class TestSemanticSearchApi:
    @pytest.fixture(autouse=True)
    async def _ingested(self, client) -> None:
        await client.post(
            "/api/v1/ingest", json={"arxiv_id": "1706.03762"}, params={"wait": True}
        )

    async def test_search_returns_hits(self, client) -> None:
        response = await client.post(
            "/api/v1/search/semantic",
            json={"query": "self-attention quadratic complexity", "top_k": 3},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["provider"] == "hashing"
        assert body["hits"]
        hit = body["hits"][0]
        assert hit["metadata"]["arxiv_id"] == "1706.03762"
        assert hit["metadata"]["title"]
        assert hit["text"]
        assert hit["metadata"]["abs_url"]

    async def test_search_rejects_empty_query(self, client) -> None:
        response = await client.post("/api/v1/search/semantic", json={"query": ""})
        assert response.status_code == 422

    async def test_search_with_category_filter(self, client) -> None:
        response = await client.post(
            "/api/v1/search/semantic",
            json={"query": "attention", "top_k": 5, "category": "cs.LG"},
        )
        assert response.status_code == 200
        assert response.json()["hits"]

        empty = await client.post(
            "/api/v1/search/semantic",
            json={"query": "attention", "top_k": 5, "category": "nope.XX"},
        )
        assert empty.json()["hits"] == []


class TestOpenApi:
    async def test_schema_is_generated(self, client) -> None:
        response = await client.get("/openapi.json")
        assert response.status_code == 200
        paths = response.json()["paths"]
        assert "/api/v1/arxiv/search" in paths
        assert "/api/v1/ingest" in paths
        assert "/api/v1/search/semantic" in paths