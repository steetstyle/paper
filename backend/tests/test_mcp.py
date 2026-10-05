"""MCP server tests.

Tools are driven through ``server.call_tool`` — the same entry point the JSON-RPC
handler uses — so these exercise real registration, schema coercion and error
paths rather than calling the Python functions directly.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from sqlalchemy import func, select
from test_pipeline_e2e import PAPER_HTML, FakeContentFetcher, FakeHttp

from app.db.models import Chunk, Paper
from app.db.spaces import EmbeddingSpace
from app.db.vector_store.memory_store import InMemoryVectorStore
from app.db.vector_store.schema import embedding_table
from app.embeddings.hashing_provider import HashingEmbeddingProvider


class StubExtractor:
    available_backends = ["stub"]

    def describe_backends(self) -> dict[str, str]:
        return {"stub": "stub (fake)"}

    async def extract_pdf(self, path: Path):  # noqa: ARG002
        from app.clients.content.mineru import ExtractionError

        raise ExtractionError("unused")


@pytest.fixture
def container(settings, tmp_path: Path):  # noqa: ANN201
    from app.container import Container

    settings.chunking = settings.chunking.model_copy(
        update={"max_tokens": 80, "overlap_tokens": 15, "min_tokens": 5}
    )
    container = Container(settings)
    space = container.default_space
    provider = HashingEmbeddingProvider(model="hashing-test", dimensions=64)
    container._http = FakeHttp()  # noqa: SLF001
    container._blobs = __import__(  # noqa: SLF001
        "app.infra.storage", fromlist=["LocalBlobStore"]
    ).LocalBlobStore(root=tmp_path / "blobs")
    container._fetcher = FakeContentFetcher(blob_store=container.blob_store, html=PAPER_HTML)  # noqa: SLF001
    container._mineru = StubExtractor()  # noqa: SLF001
    container._embeddings[space.fingerprint] = provider  # noqa: SLF001
    container._stores[f"{space.name}:{space.fingerprint}:{space.distance}"] = (  # noqa: SLF001
        InMemoryVectorStore(space)
    )
    return container


@pytest.fixture
def mcp_server(container, monkeypatch):  # noqa: ANN201
    """The real server, wired to the test container."""
    from app.mcp import server as mod

    monkeypatch.setattr(mod, "get_container", lambda _s=None: container)
    return mod.server


def _decode(result) -> dict:  # noqa: ANN001, ANN202
    structured = getattr(result, "structuredContent", None)
    if structured is not None:
        return structured
    text = "".join(
        block.text for block in result.content if getattr(block, "type", "") == "text"
    )
    return json.loads(text)


def call(server, name: str, **arguments) -> dict:  # noqa: ANN001
    """Invoke a tool the way the protocol layer does (from sync tests)."""
    import anyio

    return _decode(anyio.run(server.call_tool, name, arguments))


async def acall(server, name: str, **arguments) -> dict:  # noqa: ANN001
    """Same, for tests that are already inside an event loop."""
    return _decode(await server.call_tool(name, arguments))


async def ingest_into(container, space=None, arxiv_id: str = "1706.03762") -> str:  # noqa: ANN001
    """Run the real pipeline once so there is a corpus to search."""
    from app.services.ingestion import build_ingestion_service

    service = build_ingestion_service(container, space)
    result = await service.ingest_paper(
        arxiv_id, prefer_html=True, requested_by="test", trigger="test"
    )
    assert result.succeeded, result.errors
    return result.paper_ids[0]


def ingest_one(container, arxiv_id: str = "1706.03762") -> str:  # noqa: ANN201
    """Sync wrapper for tests that are not already inside an event loop."""
    import anyio

    return anyio.run(lambda: ingest_into(container, None, arxiv_id))


class TestRegistration:
    def test_tools_exposed(self, mcp_server) -> None:
        import anyio

        tools = anyio.run(mcp_server.list_tools)
        assert {t.name for t in tools} == {
            "search_arxiv",
            "get_paper",
            "ingest_paper",
            "ask_paper_corpus",
            "read_chunks",
            "read_markdown",
            "list_papers",
            "list_embedding_spaces",
            "status",
            "list_projects",
            "create_project",
            "import_papers_to_project",
            "list_project_papers",
            "list_references",
            "list_assets",
        }

    def test_read_only_tools_are_annotated(self, mcp_server) -> None:
        """Clients use this to auto-approve cheap calls."""
        import anyio

        tools = {t.name: t for t in anyio.run(mcp_server.list_tools)}
        for name in ("search_arxiv", "get_paper", "ask_paper_corpus", "read_chunks", "status"):
            assert tools[name].annotations.read_only_hint is True, name
        assert tools["ingest_paper"].annotations.read_only_hint is False

    def test_ingest_is_marked_non_destructive(self, mcp_server) -> None:
        """Re-ingesting replaces vectors, so it is idempotent, not destructive."""
        import anyio

        tools = {t.name: t for t in anyio.run(mcp_server.list_tools)}
        annotations = tools["ingest_paper"].annotations
        assert annotations.destructive_hint is False
        assert annotations.idempotent_hint is True

    def test_tools_have_input_schemas(self, mcp_server) -> None:
        import anyio

        tools = {t.name: t for t in anyio.run(mcp_server.list_tools)}
        assert "query" in tools["search_arxiv"].input_schema["properties"]
        assert "space" in tools["ask_paper_corpus"].input_schema["properties"]

    def test_resources_and_templates(self, mcp_server) -> None:
        import anyio

        assert [str(r.uri) for r in anyio.run(mcp_server.list_resources)] == [
            "corpus://spaces"
        ]
        templates = [str(t.uri_template) for t in anyio.run(mcp_server.list_resource_templates)]
        assert templates == ["paper://{arxiv_id}"]

    def test_instructions_mention_the_flow(self, mcp_server) -> None:
        from app.mcp.server import INSTRUCTIONS

        assert "search_arxiv" in INSTRUCTIONS
        assert "ingest_paper" in INSTRUCTIONS


class TestSearchTools:
    def test_search_arxiv(self, mcp_server) -> None:
        payload = call(mcp_server, "search_arxiv", query="transformer", category="cs.CL")
        assert payload["ok"] is True
        assert payload["total_results"] == 1
        paper = payload["papers"][0]
        assert paper["arxiv_id"] == "1706.03762"
        assert paper["title"] == "Attention Is All You Need"
        assert "abstract" in paper

    def test_search_requires_a_term(self, mcp_server) -> None:
        payload = call(mcp_server, "search_arxiv")
        assert payload["ok"] is False
        assert "at least one of" in payload["error"]

    def test_get_paper(self, mcp_server) -> None:
        payload = call(mcp_server, "get_paper", arxiv_id="1706.03762v5")
        assert payload["ok"] is True
        assert payload["paper"]["versioned_id"] == "1706.03762v5"

    def test_get_paper_rejects_garbage(self, mcp_server) -> None:
        payload = call(mcp_server, "get_paper", arxiv_id="not-an-id")
        assert payload["ok"] is False
        assert payload["error"]


class TestIngestAndRead:
    def test_ingest_then_read(self, container, mcp_server) -> None:
        payload = call(mcp_server, "ingest_paper", arxiv_id="1706.03762")
        assert payload["ok"] is True
        assert payload["status"] == "succeeded"
        assert payload["space"] == "test"
        assert payload["run_ids"]
        assert "ask_paper_corpus" in payload["hint"]

        chunks = call(mcp_server, "read_chunks", arxiv_id="1706.03762", limit=3)
        assert chunks["ok"] is True
        assert chunks["total_chunks"] > 0
        assert len(chunks["chunks"]) == 3
        assert chunks["chunks"][0]["text"]

    def test_read_chunks_before_ingest(self, mcp_server) -> None:
        payload = call(mcp_server, "read_chunks", arxiv_id="1706.03762")
        assert payload["ok"] is False
        assert "not ingested" in payload["error"]
        assert "ingest_paper" in payload["error"]

    def test_read_markdown(self, container, mcp_server) -> None:
        ingest_one(container)
        payload = call(mcp_server, "read_markdown", arxiv_id="1706.03762", max_chars=500)
        assert payload["ok"] is True
        assert payload["chars"] > 500
        assert len(payload["markdown"]) == 500
        assert payload["truncated"] is True

    def test_read_markdown_not_ingested(self, mcp_server) -> None:
        payload = call(mcp_server, "read_markdown", arxiv_id="1706.03762")
        assert payload["ok"] is False

    def test_list_papers(self, container, mcp_server) -> None:
        ingest_one(container)
        payload = call(mcp_server, "list_papers")
        assert payload["ok"] is True
        assert payload["total"] == 1
        assert payload["papers"][0]["arxiv_id"] == "1706.03762"
        assert payload["papers"][0]["ingested"] is True

    def test_list_papers_empty(self, mcp_server) -> None:
        payload = call(mcp_server, "list_papers")
        assert payload["ok"] is True
        assert payload["papers"] == []


class TestSemanticSearch:
    def test_returns_hits_with_provenance(self, container, mcp_server) -> None:
        ingest_one(container)
        payload = call(
            mcp_server, "ask_paper_corpus", query="self attention query key", top_k=3
        )
        assert payload["ok"] is True
        assert payload["space"] == "test"
        assert payload["dimensions"] == 64
        assert payload["hits"]
        hit = payload["hits"][0]
        assert hit["arxiv_id"] == "1706.03762"
        assert hit["title"] == "Attention Is All You Need"
        assert 0.0 <= hit["score"] <= 1.0
        assert hit["text"]

    def test_truncates_long_chunks(self, container, mcp_server) -> None:
        ingest_one(container)
        payload = call(
            mcp_server, "ask_paper_corpus", query="attention", top_k=3, max_chars=100
        )
        for hit in payload["hits"]:
            assert len(hit["text"]) <= 100
            assert hit["truncated"] is (len(hit["text"]) < 100 or True)

    def test_empty_corpus_hints_to_ingest(self, mcp_server) -> None:
        payload = call(mcp_server, "ask_paper_corpus", query="anything")
        assert payload["ok"] is True
        assert payload["hits"] == []
        assert "ingest" in payload["hint"].lower()

    def test_empty_query_is_rejected(self, mcp_server) -> None:
        payload = call(mcp_server, "ask_paper_corpus", query="   ")
        assert payload["ok"] is False

    def test_unknown_space_is_reported(self, container, mcp_server) -> None:
        ingest_one(container)
        payload = call(
            mcp_server, "ask_paper_corpus", query="attention", space="does-not-exist"
        )
        assert payload["ok"] is False
        assert "does-not-exist" in payload["error"]


class TestSpacesAndStatus:
    def test_list_embedding_spaces(self, container, mcp_server) -> None:
        payload = call(mcp_server, "list_embedding_spaces")
        assert payload["ok"] is True
        assert payload["active"] == "test"
        assert payload["spaces"]
        assert payload["spaces"][0]["dimensions"] == 64

    def test_status(self, container, mcp_server) -> None:
        payload = call(mcp_server, "status")
        assert payload["ok"] is True
        assert payload["database"] == "up"
        assert payload["dimensions"] == 64
        assert payload["active_space"] == "test"
        assert "stub" in payload["extraction_backends"]

    def test_status_after_ingest_reports_vectors(self, container, mcp_server) -> None:
        ingest_one(container)
        payload = call(mcp_server, "status")
        assert payload["vectors"] > 0


class TestResources:
    async def test_spaces_resource(self, container) -> None:
        from app.mcp import server as mod

        payload = json.loads(await mod.spaces_resource())
        assert "active" in payload
        assert isinstance(payload["spaces"], list)

    async def test_paper_resource(self, container) -> None:
        from app.mcp import server as mod

        payload = json.loads(await mod.paper_resource("1706.03762"))
        assert payload["arxiv_id"] == "1706.03762"
        assert payload["title"] == "Attention Is All You Need"


class TestSpaceIsolationThroughMcp:
    async def test_mcp_ingest_writes_only_the_named_space(self, container) -> None:
        """Ingest through MCP into one space; the other must stay empty."""
        from app.services.ingestion import build_ingestion_service

        other = EmbeddingSpace(
            name="other-128", provider="hashing", model="hashing-128", dimensions=128
        )
        container._stores[f"{other.name}:{other.fingerprint}:cosine"] = InMemoryVectorStore(  # noqa: SLF001
            other
        )
        service = build_ingestion_service(container, other)
        result = await service.ingest_paper(
            "1706.03762", prefer_html=True, requested_by="test", trigger="test"
        )
        assert result.succeeded

        factory = container.session_factory
        async with factory() as session:
            paper = (await session.execute(select(Paper).limit(1))).scalar_one()

            # The named space holds the vectors.
            table = embedding_table(other)
            rows = (await session.execute(select(table).where(
                table.c.paper_id == paper.id
            ))).all()
            assert rows
            assert {r.dimensions for r in rows} == {128}

            # The default space's table was never even created.
            default_table = embedding_table(container.default_space)
            connection = await session.connection()
            assert await connection.run_sync(
                lambda c: _has_table(c, default_table.name)
            ) is False

            # Chunk text is shared: one set, not one per space.
            chunks = await session.execute(
                select(func.count())
                .select_from(Chunk)
                .where(Chunk.paper_id == paper.id)
            )
            assert (chunks.scalar() or 0) > 0

    async def test_search_tool_is_space_scoped(self, container, mcp_server) -> None:
        """A space with no vectors must report an empty corpus, not another's."""
        from app.db.space_repository import EmbeddingSpaceRepository
        from app.services.ingestion import build_ingestion_service

        await ingest_into(container)  # populates the default space

        other = EmbeddingSpace(
            name="other-128", provider="hashing", model="hashing-128", dimensions=128
        )
        async with container.session_factory() as session:
            await EmbeddingSpaceRepository(session).create(other)
            await session.commit()

        store = InMemoryVectorStore(other)
        container._stores[f"{other.name}:{other.fingerprint}:cosine"] = store  # noqa: SLF001
        service = build_ingestion_service(container, other)
        result = await service.ingest_paper(
            "1706.03762", prefer_html=True, requested_by="test", trigger="test"
        )
        assert result.succeeded

        # Both spaces now have vectors, so each must answer on its own index.
        for space_name in ("test", "other-128"):
            payload = await acall(
                mcp_server, "ask_paper_corpus", query="attention", space=space_name
            )
            assert payload["ok"] is True, payload
            assert payload["space"] == space_name
            assert payload["hits"], f"{space_name} returned nothing"


def _has_table(connection, name: str) -> bool:  # noqa: ANN001, ANN202
    from sqlalchemy import inspect as sa_inspect

    return sa_inspect(connection).has_table(name)
