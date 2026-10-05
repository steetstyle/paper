"""Scoping search three ways, and moving a corpus into a second space.

The features under test:

* ``content_kind`` filters what a chunk *is* — equations, figures, tables,
  abstracts, code — and all three surfaces agree on it
* a project scope limits a query to that project's papers
* the two compose ("the equations in this project")
* re-embedding fills a second space from stored chunks, downloading nothing,
  and leaves the first space untouched
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
from test_filter_parity import StubExtractor
from test_mcp import acall
from test_pipeline_e2e import FakeContentFetcher, FakeHttp

from app.container import Container, set_container
from app.db.models import Chunk, ProjectPaper
from app.db.project_repository import ProjectRepository
from app.db.repositories import ChunkRepository, PaperRepository
from app.db.session import dispose_engines, get_session_factory
from app.db.spaces import EmbeddingSpace
from app.db.vector_store import schema
from app.db.vector_store.memory_store import InMemoryVectorStore
from app.domain.enums import ChunkKind
from app.domain.models import Author, PaperMetadata
from app.embeddings.hashing_provider import HashingEmbeddingProvider
from app.infra.storage import LocalBlobStore
from app.main import create_app
from app.services.reembed import ReembedService
from app.services.semantic_search import SemanticSearchService

# (text, heading, kind) per paper. Covers every kind the filter can ask for.
ALPHA: list[tuple[str, str, ChunkKind]] = [
    (
        "Paper Alpha > Abstract\n\nWe study spectral methods on graphs.",
        "Abstract",
        ChunkKind.ABSTRACT,
    ),
    (
        "Paper Alpha > Model\n\nAttention(Q,K,V)=softmax(QK^T/sqrt(d_k))V",
        "Model",
        ChunkKind.EQUATION,
    ),
    (
        "Paper Alpha > Results\n\n| Method | F1 |\n| --- | --- |\n| Ours | 91.2 |",
        "Results",
        ChunkKind.TABLE,
    ),
    (
        "Paper Alpha > Results\n\nFigure 2: The learned diffusion map.",
        "Results",
        ChunkKind.FIGURE,
    ),
    (
        "Paper Alpha > Model\n\nThe encoder maps inputs to vectors.",
        "Model",
        ChunkKind.BODY,
    ),
]
BETA: list[tuple[str, str, ChunkKind]] = [
    (
        "Paper Beta > Abstract\n\nA short note on topological learning.",
        "Abstract",
        ChunkKind.ABSTRACT,
    ),
    (
        "Paper Beta > Method\n\nLaplacian smoothing on cellular sheaves.",
        "Method",
        ChunkKind.BODY,
    ),
]


@pytest.fixture(autouse=True)
def _clear_schema_cache():
    schema.clear_cache()
    yield
    schema.clear_cache()


def _metadata(arxiv_id: str, title: str) -> PaperMetadata:
    return PaperMetadata(
        arxiv_id=arxiv_id,
        versioned_id=f"{arxiv_id}v1",
        version=1,
        title=title,
        abstract=f"Abstract of {title}.",
        authors=(Author(name="Ada Lovelace"),),
        categories=("cs.LG",),
        primary_category="cs.LG",
        published_at=datetime(2021, 1, 1, tzinfo=UTC),
        updated_at=datetime(2023, 1, 1, tzinfo=UTC),
        abs_url=f"https://arxiv.org/abs/{arxiv_id}",
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
        html_url=None,
    )


@pytest.fixture
def container(settings, tmp_path: Path):
    container = Container(settings)
    space = container.default_space
    container._http = FakeHttp()  # type: ignore[assignment]  # noqa: SLF001
    blob_store = LocalBlobStore(root=tmp_path / "blobs")
    container._blobs = blob_store  # type: ignore[assignment]  # noqa: SLF001
    container._fetcher = FakeContentFetcher(  # type: ignore[assignment]  # noqa: SLF001
        blob_store=blob_store, html="<html><body><p>hi</p></body></html>"
    )
    container._mineru = StubExtractor()  # type: ignore[assignment]  # noqa: SLF001
    container._embeddings[space.fingerprint] = HashingEmbeddingProvider(  # type: ignore[SLF001]
        model="hashing-test", dimensions=64
    )
    container._stores[f"{space.name}:{space.fingerprint}:{space.distance}"] = (  # noqa: SLF001
        InMemoryVectorStore(space)
    )
    set_container(container)
    yield container
    set_container(None)


@pytest.fixture
async def corpus(container):
    """Two papers with explicitly-kinded chunks, one of them in a project.

    Chunks are written straight to the repository instead of through the
    pipeline: these tests are about what the *filter* does, and chunking has its
    own tests. The vectors are added too, so ranking is exercised rather than
    only the filter plumbing.
    """
    factory = get_session_factory(container.settings.database)
    async with factory() as session:
        alpha_id = await _insert_paper(session, "2401.00001", "Paper Alpha", ALPHA)
        beta_id = await _insert_paper(session, "2401.00002", "Paper Beta", BETA)
        projects = ProjectRepository(session)
        project = await projects.create(name="sheaf-papers", description="only alpha")
        await projects.add_papers(project, [alpha_id])
        await session.commit()
        return {"project": project.slug, "alpha": alpha_id, "beta": beta_id}


async def _insert_paper(session, arxiv_id: str, title: str, rows: list[tuple[str, str, ChunkKind]]) -> str:  # noqa: ANN001
    paper = await PaperRepository(session).upsert(_metadata(arxiv_id, title))
    await session.flush()
    rows_to_write = [
        Chunk(
            paper_id=paper.id,
            ordinal=ordinal,
            text=text,
            token_count=len(text.split()),
            char_start=0,
            char_end=len(text),
            heading=heading,
            section_path=[heading],
            content_hash=f"{arxiv_id}-{ordinal}".ljust(64, "0"),
            source="arxiv_html",
            content_kind=kind.value,
        )
        for ordinal, (text, heading, kind) in enumerate(rows)
    ]
    session.add_all(rows_to_write)
    await session.flush()
    return paper.id


def service_for(container) -> SemanticSearchService:  # noqa: ANN001
    space = container.default_space
    return SemanticSearchService(
        vector_store=container.vector_store_for(space),
        embeddings=container.provider_for(space),
        session_factory=get_session_factory(container.settings.database),
        space=space,
    )


def reembed_for(container, space: EmbeddingSpace, provider, store) -> ReembedService:  # noqa: ANN001
    return ReembedService(
        provider=provider,
        space=space,
        session_factory=get_session_factory(container.settings.database),
        store=store,
        batch_size=4,
    )


@pytest.fixture
async def embedded(container, corpus):
    """The same chunks, embedded into the active space, via re-embedding.

    This is the real path a second embedding model takes, so the filters below
    are exercised against actual vectors rather than an empty index.
    """
    space = container.default_space
    service = reembed_for(
        container, space, container.provider_for(space), container.vector_store_for(space)
    )
    report = await service.reembed()
    assert report.chunks_embedded == len(ALPHA) + len(BETA)
    return report


@pytest.fixture
async def api_client(container):
    app = create_app(settings=container.settings, container=container)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
    await dispose_engines()


@pytest.fixture
def mcp_server(container, monkeypatch):
    from app.mcp import server as mod

    monkeypatch.setattr(mod, "get_container", lambda _s=None: container)
    return mod.server


class TestContentKindIsStored:
    async def test_every_kind_lands_in_the_column(self, container, corpus) -> None:  # noqa: ANN001
        from sqlalchemy import select

        async with get_session_factory(container.settings.database)() as session:
            kinds = set(
                (
                    await session.execute(
                        select(Chunk.content_kind).where(Chunk.paper_id == corpus["alpha"])
                    )
                ).scalars()
            )
        assert kinds == {kind.value for _t, _h, kind in ALPHA}


class TestProjectScope:
    async def test_scope_is_echoed(self, container, corpus) -> None:  # noqa: ANN001
        service = service_for(container)
        await service.search("attention", project=corpus["project"])
        assert service.last_scope["project"] == corpus["project"]
        assert service.last_scope["project_papers"] == 1

    async def test_project_limits_the_papers_returned(self, container, embedded, corpus) -> None:  # noqa: ANN001
        service = service_for(container)
        hits = await service.search("diffusion map", top_k=10, project=corpus["project"])
        assert hits
        assert {hit.metadata["arxiv_id"] for hit in hits} == {"2401.00001"}

    async def test_without_a_project_the_scope_is_corpus_wide(
        self, container, embedded
    ) -> None:  # noqa: ANN001
        service = service_for(container)
        await service.search("smoothing sheaves", top_k=10)
        assert service.last_scope["project"] is None
        assert service.last_scope["project_papers"] is None

    async def test_unknown_project_raises(self, container, corpus) -> None:  # noqa: ANN001
        service = service_for(container)
        with pytest.raises(LookupError):
            await service.search("x", project="does-not-exist")

    async def test_empty_project_does_not_widen_to_the_corpus(
        self, container, embedded, corpus
    ) -> None:  # noqa: ANN001
        factory = get_session_factory(container.settings.database)
        async with factory() as session:
            empty = await ProjectRepository(session).create(name="empty")
            await session.commit()
            slug = empty.slug
        service = service_for(container)
        assert await service.search("diffusion", top_k=5, project=slug) == []
        assert service.last_scope["project_papers"] == 0

    async def test_project_and_papers_intersect(self, container, embedded, corpus) -> None:  # noqa: ANN001
        """Beta is not in the project, so intersecting yields nothing."""
        service = service_for(container)
        hits = await service.search(
            "diffusion", top_k=5, project=corpus["project"], paper_ids=[corpus["beta"]]
        )
        assert hits == []

    async def test_project_with_a_matching_paper_id_works(
        self, container, embedded, corpus
    ) -> None:  # noqa: ANN001
        service = service_for(container)
        hits = await service.search(
            "diffusion map", top_k=5, project=corpus["project"], paper_ids=[corpus["alpha"]]
        )
        assert {hit.metadata["arxiv_id"] for hit in hits} == {"2401.00001"}


class TestContentKindFilter:
    async def test_only_equations(self, container, embedded) -> None:  # noqa: ANN001
        service = service_for(container)
        hits = await service.search("attention", top_k=10, content_kinds=["equation"])
        assert hits
        assert {hit.metadata["content_kind"] for hit in hits} == {"equation"}

    async def test_only_tables(self, container, embedded) -> None:  # noqa: ANN001
        service = service_for(container)
        hits = await service.search("method f1", top_k=10, content_kinds=["table"])
        assert {hit.metadata["content_kind"] for hit in hits} == {"table"}

    async def test_only_abstracts(self, container, embedded) -> None:  # noqa: ANN001
        service = service_for(container)
        hits = await service.search("spectral methods", top_k=10, content_kinds=["abstract"])
        assert hits
        assert {hit.metadata["content_kind"] for hit in hits} == {"abstract"}

    async def test_kinds_compose_with_a_project(self, container, embedded, corpus) -> None:  # noqa: ANN001
        service = service_for(container)
        hits = await service.search(
            "attention", top_k=10, project=corpus["project"], content_kinds=["equation"]
        )
        assert hits
        assert {hit.metadata["arxiv_id"] for hit in hits} == {"2401.00001"}
        assert {hit.metadata["content_kind"] for hit in hits} == {"equation"}

    async def test_kinds_that_do_not_exist_return_nothing(
        self, container, embedded
    ) -> None:  # noqa: ANN001
        service = service_for(container)
        hits = await service.search("diffusion", top_k=10, content_kinds=["code"])
        assert hits == []

    @pytest.mark.parametrize(
        "kind", [kind for kind in ChunkKind if kind in {k for _t, _h, k in ALPHA}]
    )
    async def test_every_present_kind_is_reachable(
        self, container, embedded, kind: ChunkKind
    ) -> None:  # noqa: ANN001
        service = service_for(container)
        hits = await service.search("paper", top_k=20, content_kinds=[kind.value])
        assert hits, kind
        assert {hit.metadata["content_kind"] for hit in hits} == {kind.value}

    @pytest.mark.parametrize(
        "kind", [kind for kind in ChunkKind if kind not in {k for _t, _h, k in ALPHA}]
    )
    async def test_a_kind_with_no_chunks_returns_nothing(
        self, container, embedded, kind: ChunkKind
    ) -> None:  # noqa: ANN001
        service = service_for(container)
        assert await service.search("paper", top_k=20, content_kinds=[kind.value]) == []


class TestReembed:
    async def test_fills_the_space_from_stored_chunks(self, container, corpus) -> None:  # noqa: ANN001
        space = container.default_space
        report = await reembed_for(
            container,
            space,
            container.provider_for(space),
            container.vector_store_for(space),
        ).reembed()
        assert report.papers_embedded == 2
        assert report.chunks_embedded == len(ALPHA) + len(BETA)
        assert not report.failures

    async def test_second_run_is_a_no_op(self, container, embedded) -> None:  # noqa: ANN001
        space = container.default_space
        again = await reembed_for(
            container,
            space,
            container.provider_for(space),
            container.vector_store_for(space),
        ).reembed()
        assert again.chunks_embedded == 0
        assert again.papers_skipped == 2

    async def test_force_re_embeds(self, container, embedded) -> None:  # noqa: ANN001
        space = container.default_space
        forced = await reembed_for(
            container,
            space,
            container.provider_for(space),
            container.vector_store_for(space),
        ).reembed(force=True)
        assert forced.chunks_embedded == len(ALPHA) + len(BETA)

    async def test_project_scope_limits_the_work(self, container, corpus) -> None:  # noqa: ANN001
        space = container.default_space
        report = await reembed_for(
            container,
            space,
            container.provider_for(space),
            container.vector_store_for(space),
        ).reembed(project=corpus["project"])
        assert report.papers_embedded == 1
        assert report.chunks_embedded == len(ALPHA)

    async def test_arxiv_id_scope_limits_the_work(self, container, corpus) -> None:  # noqa: ANN001
        space = container.default_space
        report = await reembed_for(
            container,
            space,
            container.provider_for(space),
            container.vector_store_for(space),
        ).reembed(arxiv_ids=["2401.00002"])
        assert report.chunks_embedded == len(BETA)

    async def test_unknown_project_raises(self, container, corpus) -> None:  # noqa: ANN001
        space = container.default_space
        with pytest.raises(LookupError):
            await reembed_for(
                container,
                space,
                container.provider_for(space),
                container.vector_store_for(space),
            ).reembed(project="nope")

    async def test_limit_is_deterministic(self, container, corpus) -> None:  # noqa: ANN001
        space = container.default_space
        service = reembed_for(
            container, space, container.provider_for(space), container.vector_store_for(space)
        )
        assert await service.preview(limit=1) == ["2401.00001"]
        assert await service.preview(limit=1) == ["2401.00001"]

    async def test_preview_embeds_nothing(self, container, corpus) -> None:  # noqa: ANN001
        """`preview` lists targets and must not create the space's table either.

        Asserted against the store's own count rather than the relational table:
        with no ingest yet, the table legitimately does not exist, and querying
        it would fail with "no such table" for a reason that has nothing to do
        with what is being tested here.
        """
        space = container.default_space
        store = container.vector_store_for(space)
        listed = await reembed_for(
            container, space, container.provider_for(space), store
        ).preview()
        assert listed == ["2401.00001", "2401.00002"]
        assert await store.count() == 0

    async def test_a_second_space_gets_its_own_vectors(self, container, corpus) -> None:  # noqa: ANN001
        from app.db.repositories import EmbeddingRepository
        from app.db.vector_store.schema import embedding_table, ensure_space_table

        factory = get_session_factory(container.settings.database)
        first = container.default_space
        second = EmbeddingSpace(
            name="other-32", provider="hashing", model="hashing-32", dimensions=32
        )

        async def fill(space: EmbeddingSpace, dimensions: int) -> int:
            service = reembed_for(
                container,
                space,
                HashingEmbeddingProvider(model=f"hashing-{dimensions}", dimensions=dimensions),
                InMemoryVectorStore(space),
            )
            await service.reembed()
            async with factory() as session:
                connection = await session.connection()
                await connection.run_sync(lambda conn: ensure_space_table(conn, space))
                await session.commit()
                return await EmbeddingRepository(session, embedding_table(space)).count()

        assert await fill(first, 64) == len(ALPHA) + len(BETA)
        assert await fill(second, 32) == len(ALPHA) + len(BETA)
        # Same chunks, two tables, and neither clobbered the other.
        async with factory() as session:
            rows = await ChunkRepository(session).list_for_paper(corpus["alpha"])
        assert len(rows) == len(ALPHA)

    async def test_kind_backfill_labels_unlabelled_chunks(
        self, container, corpus
    ) -> None:  # noqa: ANN001
        from sqlalchemy import update

        factory = get_session_factory(container.settings.database)
        async with factory() as session:
            await session.execute(update(Chunk).values(content_kind="body"))
            await session.commit()
        space = container.default_space
        counts = await reembed_for(
            container,
            space,
            container.provider_for(space),
            container.vector_store_for(space),
        ).backfill_kinds()
        # Every chunk was just reset to the `body` default, so `only_body=True`
        # sees all seven and the counts are a full labelling.
        assert counts == {
            "abstract": 2,
            "equation": 1,
            "table": 1,
            "figure": 1,
            "body": 2,
        }

        # `only_body` can never reach zero while body chunks exist: body is the
        # column's own default, so a genuine body chunk always looks unlabelled.
        # Harmless — classifying one is idempotent — but it means "still to do"
        # is only meaningful for the non-body kinds.
        second = await reembed_for(
            container,
            space,
            container.provider_for(space),
            container.vector_store_for(space),
        ).backfill_kinds()
        assert second == {"body": 2}

    async def test_kind_backfill_is_idempotent(self, container, corpus) -> None:  # noqa: ANN001
        from sqlalchemy import update

        factory = get_session_factory(container.settings.database)
        space = container.default_space
        service = reembed_for(
            container, space, container.provider_for(space), container.vector_store_for(space)
        )
        async with factory() as session:
            await session.execute(update(Chunk).values(content_kind="body"))
            await session.commit()
        await service.backfill_kinds()
        again = await service.backfill_kinds(only_body=False)
        assert again["equation"] == 1

    async def test_kind_backfill_reclassifies_appendix_proofs(
        self, container, corpus
    ) -> None:  # noqa: ANN001
        """A chunk under 'Appendix B Proofs' is body, not references."""
        from sqlalchemy import select, update

        factory = get_session_factory(container.settings.database)
        async with factory() as session:
            await session.execute(
                update(Chunk).values(content_kind="reference", heading="Appendix B Proofs")
            )
            await session.commit()
        space = container.default_space
        await reembed_for(
            container,
            space,
            container.provider_for(space),
            container.vector_store_for(space),
        ).backfill_kinds(only_body=False)
        async with factory() as session:
            kinds = set((await session.execute(select(Chunk.content_kind))).scalars())
        assert "reference" not in kinds

    async def test_reembed_leaves_project_membership_alone(
        self, container, embedded, corpus
    ) -> None:  # noqa: ANN001
        from sqlalchemy import func, select

        async with get_session_factory(container.settings.database)() as session:
            count = int(await session.scalar(select(func.count()).select_from(ProjectPaper)))
        assert count == 1


QUERY = "diffusion map"


class TestSurfaceParity:
    """The same scope must mean the same thing on all three surfaces."""

    async def test_api_applies_project_and_kind(self, api_client, embedded, corpus) -> None:  # noqa: ANN001
        response = await api_client.post(
            "/api/v1/search/semantic",
            json={
                "query": QUERY,
                "top_k": 5,
                "project": corpus["project"],
                "content_kinds": ["equation"],
            },
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["scope"]["project"] == corpus["project"]
        assert body["scope"]["content_kinds"] == ["equation"]
        assert {hit["metadata"]["arxiv_id"] for hit in body["hits"]} == {"2401.00001"}
        for hit in body["hits"]:
            assert hit["metadata"]["content_kind"] == "equation"

    async def test_api_accepts_a_plural_alias(self, api_client, embedded) -> None:  # noqa: ANN001
        response = await api_client.post(
            "/api/v1/search/semantic", json={"query": QUERY, "content_kinds": ["equations"]}
        )
        assert response.status_code == 200, response.text
        assert response.json()["scope"]["content_kinds"] == ["equation"]

    async def test_api_rejects_an_unknown_kind(self, api_client) -> None:  # noqa: ANN001
        response = await api_client.post(
            "/api/v1/search/semantic", json={"query": QUERY, "content_kinds": ["eqaution"]}
        )
        assert response.status_code == 422
        assert "eqaution" in response.json()["detail"]

    async def test_api_404s_on_an_unknown_project(self, api_client) -> None:  # noqa: ANN001
        response = await api_client.post(
            "/api/v1/search/semantic", json={"query": QUERY, "project": "nope"}
        )
        assert response.status_code == 404

    async def test_api_hits_carry_their_kind(self, api_client, embedded) -> None:  # noqa: ANN001
        response = await api_client.post("/api/v1/search/semantic", json={"query": QUERY})
        assert response.status_code == 200, response.text
        kinds = {hit["metadata"]["content_kind"] for hit in response.json()["hits"]}
        assert kinds <= {kind.value for kind in ChunkKind}
        assert kinds  # not empty

    async def test_mcp_applies_project_and_kind(self, mcp_server, embedded, corpus) -> None:  # noqa: ANN001
        result = await acall(
            mcp_server,
            "ask_paper_corpus",
            query=QUERY,
            project=corpus["project"],
            content=["eqs"],
        )
        body = result
        assert body["ok"] is True, result
        assert body["scope"]["content_kinds"] == ["equation"]
        assert {hit["arxiv_id"] for hit in body["hits"]} == {"2401.00001"}
        for hit in body["hits"]:
            assert hit["kind"] == "equation"

    async def test_mcp_rejects_an_unknown_kind(self, mcp_server) -> None:  # noqa: ANN001
        result = await acall(mcp_server, "ask_paper_corpus", query=QUERY, content=["figurez"])
        body = result
        assert body["ok"] is False, result
        assert "figurez" in body["error"]

    async def test_mcp_reports_the_kind_on_every_hit(
        self, mcp_server, embedded
    ) -> None:  # noqa: ANN001
        result = await acall(mcp_server, "ask_paper_corpus", query=QUERY)
        body = result
        assert body["ok"] is True, result
        assert body["hits"]
        for hit in body["hits"]:
            assert hit["kind"] in {kind.value for kind in ChunkKind}

    def test_cli_agrees_with_the_api(
        self, api_client, container, embedded, corpus, monkeypatch
    ) -> None:  # noqa: ANN001
        import anyio
        from typer.testing import CliRunner

        from app.cli import app as cli_app

        monkeypatch.setattr("app.container.get_container", lambda *a, **k: container)
        result = CliRunner().invoke(
            cli_app,
            [
                "ask",
                QUERY,
                "--project",
                corpus["project"],
                "--content",
                "equations",
                "--text",
            ],
        )
        assert result.exit_code == 0, result.output

        response = anyio.run(
            lambda: api_client.post(
                "/api/v1/search/semantic",
                json={
                    "query": QUERY,
                    "top_k": 8,
                    "project": corpus["project"],
                    "content_kinds": ["equation"],
                },
            )
        )
        assert response.status_code == 200, response.text
        api_ids = {hit["metadata"]["arxiv_id"] for hit in response.json()["hits"]}
        cli_ids = set(re.findall(r"\b\d{4}\.\d{4,5}\b", result.output))
        assert cli_ids == api_ids

    def test_cli_rejects_an_unknown_kind(self, container, embedded, monkeypatch) -> None:  # noqa: ANN001
        from typer.testing import CliRunner

        from app.cli import app as cli_app

        monkeypatch.setattr("app.container.get_container", lambda *a, **k: container)
        result = CliRunner().invoke(cli_app, ["ask", QUERY, "--content", "eqaution"])
        assert result.exit_code == 2
        assert "eqaution" in result.output
