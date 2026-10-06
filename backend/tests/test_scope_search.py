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
        """A project narrows the paper set; it does not widen it.

        ``paper_ids`` is intersected with the project's papers, so naming a paper
        that the project does not hold yields nothing rather than escaping the
        project scope.
        """
        service = service_for(container)
        hits = await service.search(
            "diffusion",
            top_k=5,
            project=corpus["project"],
            paper_ids=["2401.00002"],
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


class TestScopeConventions:
    """The same scope must be spelled and validated the same way everywhere.

    These are the audit findings that were silent-wrong-behaviour, not cosmetics.
    """

    async def test_arxiv_ids_work_where_internal_ids_used_to_be_required(
        self, container, embedded, corpus
    ) -> None:  # noqa: ANN001
        """Regression: `--paper 2401.00001` filtered on `uuid` ids.

        The arXiv id was compared against `Paper.id`, so it matched nothing and
        reported zero hits with no error and no hint. Resolution now happens in
        the service, so every surface benefits.
        """
        service = service_for(container)
        by_arxiv = await service.search("diffusion", top_k=5, paper_ids=["2401.00001"])
        by_internal = await service.search("diffusion", top_k=5, paper_ids=[corpus["alpha"]])
        assert {h.metadata["arxiv_id"] for h in by_arxiv} == {"2401.00001"}
        assert [h.chunk_id for h in by_arxiv] == [h.chunk_id for h in by_internal]

    async def test_a_mistyped_paper_id_is_an_error_not_silence(
        self, container, embedded
    ) -> None:  # noqa: ANN001
        """A search returning nothing because an id was wrong is indistinguishable
        from one that found no matching text, so it must fail loudly."""
        service = service_for(container)
        with pytest.raises(LookupError) as excinfo:
            await service.search("diffusion", paper_ids=["2401.99999"])
        assert "2401.99999" in str(excinfo.value)

    async def test_versioned_and_url_spellings_resolve(self, container, embedded) -> None:  # noqa: ANN001
        service = service_for(container)
        for spelling in ("2401.00001", "2401.00001v1", "arXiv:2401.00001"):
            hits = await service.search("diffusion", top_k=5, paper_ids=[spelling])
            assert hits, spelling

    async def test_project_is_found_by_name(self, container, embedded) -> None:  # noqa: ANN001
        """`paper projects add "My Reading List"` promises name lookup."""
        service = service_for(container)
        by_slug = await service.search("diffusion", top_k=5, project="sheaf-papers")
        by_name = await service.search("diffusion", top_k=5, project="Sheaf-papers")
        assert [h.chunk_id for h in by_slug] == [h.chunk_id for h in by_name]

    async def test_an_unknown_source_is_rejected(self, container, embedded) -> None:  # noqa: ANN001
        service = service_for(container)
        with pytest.raises(ValueError) as excinfo:
            await service.search("diffusion", sources=["arxiv_htmll"])
        assert "arxiv_htmll" in str(excinfo.value)

    async def test_an_unknown_kind_is_rejected_in_the_service_too(
        self, container, embedded
    ) -> None:  # noqa: ANN001
        service = service_for(container)
        with pytest.raises(ValueError):
            await service.search("diffusion", content_kinds=["figurez"])

    async def test_scope_reports_the_resolved_filter(self, container, embedded, corpus) -> None:  # noqa: ANN001
        """The echoed scope must describe what reached the store."""
        service = service_for(container)
        await service.search(
            "diffusion", top_k=5, project=corpus["project"], content_kinds=["equation"]
        )
        scope = service.last_scope
        assert scope["project_papers"] == 1
        assert scope["paper_ids"] == 1
        assert scope["content_kinds"] == ["equation"]


class TestVectorFilterSemantics:
    """`None` means "no filter"; an empty list means "match nothing"."""

    def test_empty_paper_ids_is_not_empty(self) -> None:
        from app.db.vector_store.base import VectorFilter

        assert VectorFilter(paper_ids=[]).matches_nothing
        assert not VectorFilter(paper_ids=[]).is_empty()
        assert VectorFilter().is_empty()
        assert not VectorFilter().matches_nothing

    def test_empty_kinds_is_not_empty(self) -> None:
        from app.db.vector_store.base import VectorFilter

        assert VectorFilter(content_kinds=[]).matches_nothing

    async def test_an_explicit_empty_kind_list_widens_nothing(self, container, embedded) -> None:  # noqa: ANN001
        """`content_kinds=[]` must not read as "no filter"."""
        from app.db.vector_store.base import VectorFilter

        store = container.vector_store_for(container.default_space)
        service = service_for(container)
        await service.search("diffusion", top_k=20)
        vector = await container.provider_for(container.default_space).embed_query("diffusion")
        assert await store.search(vector, top_k=20, filters=VectorFilter(content_kinds=[])) == []

    async def test_unfiltered_search_still_returns_rows(self, container, embedded) -> None:  # noqa: ANN001
        """The guard above must not have broken the ordinary path."""
        service = service_for(container)
        assert await service.search("diffusion", top_k=20)


class TestChunkKindOnReadPaths:
    async def test_chunk_repository_filters_by_kind(self, container, corpus) -> None:  # noqa: ANN001
        factory = get_session_factory(container.settings.database)
        async with factory() as session:
            repo = ChunkRepository(session)
            equations = await repo.list_for_paper(
                corpus["alpha"], content_kinds=["equation"]
            )
            assert [c.content_kind for c in equations] == ["equation"]
            assert await repo.count(corpus["alpha"], content_kinds=["equation"]) == 1
            assert await repo.count(corpus["alpha"]) == len(ALPHA)

    async def test_empty_kind_list_matches_nothing(self, container, corpus) -> None:  # noqa: ANN001
        factory = get_session_factory(container.settings.database)
        async with factory() as session:
            repo = ChunkRepository(session)
            assert await repo.list_for_paper(corpus["alpha"], content_kinds=[]) == []
            assert await repo.count(corpus["alpha"], content_kinds=[]) == 0

    async def test_limit_zero_returns_nothing(self, container, corpus) -> None:  # noqa: ANN001
        """`limit=0` used to read as "no limit" and hand back the whole paper."""
        factory = get_session_factory(container.settings.database)
        async with factory() as session:
            assert await ChunkRepository(session).list_for_paper(corpus["alpha"], limit=0) == []

    async def test_api_chunk_listing_filters_and_reports_totals(
        self, api_client, corpus
    ) -> None:  # noqa: ANN001
        response = await api_client.get(
            "/api/v1/papers/2401.00001/chunks", params={"content": ["equation"]}
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["total"] == 1
        assert body["paper_total"] == len(ALPHA)
        assert body["content_kinds"] == ["equation"]
        assert [c["content_kind"] for c in body["chunks"]] == ["equation"]

    async def test_api_chunk_listing_rejects_an_unknown_kind(
        self, api_client, corpus
    ) -> None:  # noqa: ANN001
        response = await api_client.get(
            "/api/v1/papers/2401.00001/chunks", params={"content": ["equationz"]}
        )
        assert response.status_code == 422
        assert "equationz" in response.json()["detail"]

    async def test_api_chunk_payload_carries_the_kind(self, api_client, corpus) -> None:  # noqa: ANN001
        response = await api_client.get("/api/v1/papers/2401.00001/chunks")
        assert response.status_code == 200
        kinds = {c["content_kind"] for c in response.json()["chunks"]}
        assert kinds <= {k.value for k in ChunkKind}
        assert "equation" in kinds

    async def test_mcp_read_chunks_filters_by_kind(self, mcp_server, corpus) -> None:  # noqa: ANN001
        result = await acall(
            mcp_server, "read_chunks", arxiv_id="2401.00001", content=["eqs"]
        )
        body = result
        assert body["ok"] is True, result
        assert body["content_kinds"] == ["equation"]
        assert body["total_chunks"] == 1
        assert body["paper_total_chunks"] == len(ALPHA)
        assert [c["kind"] for c in body["chunks"]] == ["equation"]

    async def test_mcp_read_chunks_reports_the_kind_unfiltered(
        self, mcp_server, corpus
    ) -> None:  # noqa: ANN001
        body = await acall(mcp_server, "read_chunks", arxiv_id="2401.00001", limit=50)
        assert body["ok"] is True
        assert all("kind" in c for c in body["chunks"])

    async def test_mcp_read_chunks_rejects_an_unknown_kind(self, mcp_server) -> None:  # noqa: ANN001
        body = await acall(mcp_server, "read_chunks", arxiv_id="2401.00001", content=["figz"])
        assert body["ok"] is False
        assert "figz" in body["error"]

    def test_cli_show_filters_by_kind(self, container, corpus, monkeypatch) -> None:  # noqa: ANN001
        from typer.testing import CliRunner

        from app.cli import app as cli_app

        monkeypatch.setattr("app.container.get_container", lambda *a, **k: container)
        result = CliRunner().invoke(
            cli_app, ["show", "2401.00001", "-C", "equations", "-c", "20"]
        )
        assert result.exit_code == 0, result.output
        assert "equation" in result.output
        # The body chunk must be filtered out.
        assert "The encoder maps inputs to vectors." not in result.output

    def test_cli_show_rejects_an_unknown_kind(self, container, corpus, monkeypatch) -> None:  # noqa: ANN001
        from typer.testing import CliRunner

        from app.cli import app as cli_app

        monkeypatch.setattr("app.container.get_container", lambda *a, **k: container)
        result = CliRunner().invoke(cli_app, ["show", "2401.00001", "-C", "eqation"])
        assert result.exit_code == 2
        assert "eqation" in result.output

    def test_cli_ask_rejects_an_unknown_source(self, container, embedded, monkeypatch) -> None:  # noqa: ANN001
        from typer.testing import CliRunner

        from app.cli import app as cli_app

        monkeypatch.setattr("app.container.get_container", lambda *a, **k: container)
        result = CliRunner().invoke(cli_app, ["ask", "diffusion", "--source", "html"])
        assert result.exit_code == 2
        assert "html" in result.output

    def test_cli_ask_reports_an_unknown_project_cleanly(
        self, container, embedded, monkeypatch
    ) -> None:  # noqa: ANN001
        """Was a raw traceback out of asyncio.run; HTTP 404s and MCP errors."""
        from typer.testing import CliRunner

        from app.cli import app as cli_app

        monkeypatch.setattr("app.container.get_container", lambda *a, **k: container)
        result = CliRunner().invoke(cli_app, ["ask", "diffusion", "--project", "nope"])
        assert result.exit_code == 2, result.output
        assert "nope" in result.output
        assert "Traceback" not in result.output


class TestEquationSearchScope:
    @pytest.fixture
    def db(self, container):
        """The test database, reached through the same factory the app uses."""
        return get_session_factory(container.settings.database)

    @pytest.fixture(autouse=True)
    async def _equations(self, db, corpus) -> None:  # noqa: ANN001
        """One display equation per paper, so scoping is observable."""
        from app.db.asset_models import PaperEquation

        async with db() as session:
            session.add(
                PaperEquation(
                    id="eq-alpha",
                    paper_id=corpus["alpha"],
                    ordinal=1,
                    latex="\\mathrm{Attention}(Q,K,V)",
                    is_display=True,
                    source="html",
                )
            )
            session.add(
                PaperEquation(
                    id="eq-beta",
                    paper_id=corpus["beta"],
                    ordinal=1,
                    latex="\\mathrm{Attention}(W,V)",
                    is_display=True,
                    source="html",
                )
            )
            await session.commit()

    async def test_equation_search_accepts_a_project(self, api_client, corpus) -> None:  # noqa: ANN001
        scoped = await api_client.get(
            "/api/v1/equations",
            params={"q": "Attention", "project": "sheaf-papers"},
        )
        assert scoped.status_code == 200, scoped.text
        body = scoped.json()
        assert body["scope_papers"] == 1
        assert [e["latex"] for e in body["equations"]] == ["\\mathrm{Attention}(Q,K,V)"]

    async def test_equation_search_unscoped_finds_both(self, api_client) -> None:  # noqa: ANN001
        body = (await api_client.get("/api/v1/equations", params={"q": "Attention"})).json()
        assert body["scope_papers"] is None
        assert len(body["equations"]) == 2

    async def test_equation_search_by_arxiv_id(self, api_client) -> None:  # noqa: ANN001
        response = await api_client.get(
            "/api/v1/equations", params={"q": "Attention", "arxiv_id": "2401.00002"}
        )
        assert response.status_code == 200
        assert response.json()["scope_papers"] == 1

    async def test_equation_search_404s_on_an_unknown_project(self, api_client) -> None:  # noqa: ANN001
        response = await api_client.get(
            "/api/v1/equations", params={"q": "x", "project": "nope"}
        )
        assert response.status_code == 404

    async def test_equation_search_404s_on_an_unknown_paper(self, api_client) -> None:  # noqa: ANN001
        response = await api_client.get(
            "/api/v1/equations", params={"q": "x", "arxiv_id": "2401.99999"}
        )
        assert response.status_code == 404

    def test_qdrant_delete_with_no_papers_is_a_no_op(self) -> None:
        """An empty `MatchAny` is rejected by Qdrant, so it cannot be pushed down."""
        import inspect

        from app.db.vector_store.qdrant_store import QdrantVectorStore

        source = inspect.getsource(QdrantVectorStore.delete_for_papers)
        assert source.index("if not paper_ids") < source.index("_get_client")


class TestProjectWideAssets:
    """`paper assets --project` answers the corpus-level asset question.

    Per-paper paging through 36 papers to answer "which papers have figures"
    is the wrong tool; the counts are one query.
    """

    @pytest.fixture(autouse=True)
    async def _assets(self, container, corpus) -> None:  # noqa: ANN001
        from app.db.asset_models import PaperFigure, PaperTable

        factory = get_session_factory(container.settings.database)
        async with factory() as session:
            session.add(
                PaperFigure(
                    id="fig-a1",
                    paper_id=corpus["alpha"],
                    ordinal=1,
                    label="Figure 1",
                    caption="The pipeline.",
                    source="html",
                )
            )
            session.add(
                PaperTable(
                    id="tbl-b1",
                    paper_id=corpus["beta"],
                    ordinal=1,
                    label="Table 1",
                    caption="Results.",
                    row_count=3,
                    column_count=2,
                    source="html",
                )
            )
            await session.commit()

    def _invoke(self, container, monkeypatch, *args: str):  # noqa: ANN001, ANN202
        from typer.testing import CliRunner

        from app.cli import app as cli_app

        monkeypatch.setattr("app.container.get_container", lambda *a, **k: container)
        return CliRunner().invoke(cli_app, ["assets", *args])

    def test_lists_asset_counts_per_paper(self, container, corpus, monkeypatch) -> None:  # noqa: ANN001
        result = self._invoke(container, monkeypatch, "x", "--project", corpus["project"])
        assert result.exit_code == 0, result.output
        # Only alpha is in the project, and only alpha has a figure.
        assert "2401.00001" in result.output
        assert "2401.00002" not in result.output

    def test_counts_come_from_the_asset_tables(
        self, container, corpus, monkeypatch
    ) -> None:  # noqa: ANN001
        """The numbers are read from `paper_figures` / `paper_tables`, not guessed."""
        result = self._invoke(container, monkeypatch, "x", "--project", corpus["project"])
        assert result.exit_code == 0
        row = next(
            line for line in result.output.splitlines() if line.startswith("2401.00001")
        )
        assert row.split() == ["2401.00001", "1", "0", "0", "0"]

    def test_unknown_project_is_exit_2(self, container, monkeypatch) -> None:  # noqa: ANN001
        result = self._invoke(container, monkeypatch, "x", "--project", "yok")
        assert result.exit_code == 2
        assert "yok" in result.output

    def test_single_paper_listing_still_works(self, container, corpus, monkeypatch) -> None:  # noqa: ANN001
        result = self._invoke(container, monkeypatch, "2401.00001", "--kind", "figures")
        assert result.exit_code == 0, result.output
        assert "Figure 1" in result.output
        assert "The pipeline." in result.output



class TestProjectSession:
    """`paper session` keeps one project's scope in force across commands.

    Exists because `paper ask --project X` throws the scope away after one
    question: reading a list is mostly sequencing, and re-typing the flag each
    time is what made it unusable.
    """

    @staticmethod
    def _session(container, *lines: str):  # noqa: ANN001, ANN205
        from typer.testing import CliRunner

        from app.cli import app as cli_app

        # No monkeypatching needed: the container fixture calls `set_container`,
        # and every command resolves `get_container` at call time.
        return CliRunner().invoke(
            cli_app, ["session", "sheaf-papers"], input="\n".join([*lines, "exit"])
        )

    def test_papers_lists_the_project(self, container, corpus) -> None:  # noqa: ANN001
        result = self._session(container, "papers")
        assert result.exit_code == 0, result.output
        assert "2401.00001" in result.output
        assert "2401.00002" not in result.output  # not in this project

    def test_ask_stays_inside_the_project(self, container, embedded, corpus) -> None:  # noqa: ANN001
        result = self._session(container, "set top_k=3", "ask diffusion map")
        assert result.exit_code == 0, result.output
        assert "2401.00001" in result.output
        assert "2401.00002" not in result.output

    def test_content_filter_persists_across_questions(
        self, container, embedded, corpus
    ) -> None:  # noqa: ANN001
        result = self._session(container, "set -c equation", "ask diffusion map")
        assert result.exit_code == 0, result.output
        assert "equation" in result.output

    def test_content_filter_can_be_cleared(self, container, embedded, corpus) -> None:  # noqa: ANN001
        result = self._session(
            container, "set -c equation", "set -c none", "ask diffusion map"
        )
        assert result.exit_code == 0, result.output
        assert "content filter cleared" in result.output

    @pytest.mark.parametrize(
        ("command", "expected"),
        [
            ("set content=figures", "content = figure"),
            ("set -c equations", "content = equation"),
            ("set c=equations,tables", "content = equation+table"),
            ("set top_k=3", "top_k = 3"),
        ],
    )
    def test_set_spellings(
        self, container, corpus, command: str, expected: str
    ) -> None:  # noqa: ANN001
        """`set -c x` and `set content=x` both work; the separator is optional."""
        result = self._session(container, command)
        assert result.exit_code == 0, result.output
        assert expected in result.output

    def test_bad_setting_does_not_kill_the_session(
        self, container, corpus
    ) -> None:  # noqa: ANN001
        result = self._session(container, "set -c eqaution", "set -c equations")
        assert result.exit_code == 0, result.output
        assert "eqaution" in result.output
        # The session kept going and accepted the corrected value.
        assert "content = equation" in result.output

    def test_unknown_command_does_not_kill_the_session(
        self, container, corpus
    ) -> None:  # noqa: ANN001
        result = self._session(container, "bogus", "papers")
        assert result.exit_code == 0, result.output
        assert "unknown command 'bogus'" in result.output
        assert "2401.00001" in result.output

    def test_unknown_paper_is_reported_not_raised(
        self, container, corpus
    ) -> None:  # noqa: ANN001
        result = self._session(container, "show 9999.99999")
        assert result.exit_code == 0, result.output
        assert "not ingested" in result.output

    def test_unknown_project_exits_2(self, container) -> None:  # noqa: ANN001
        from typer.testing import CliRunner

        from app.cli import app as cli_app

        result = CliRunner().invoke(cli_app, ["session", "yok-boyle"])
        assert result.exit_code == 2
        assert "yok-boyle" in result.output

    def test_no_argument_lists_projects(self, container, corpus) -> None:  # noqa: ANN001
        from typer.testing import CliRunner

        from app.cli import app as cli_app

        result = CliRunner().invoke(cli_app, ["session"])
        assert result.exit_code == 1
        assert "sheaf-papers" in result.output

    def test_read_toggles_one_paper(self, container, corpus) -> None:  # noqa: ANN001
        result = self._session(container, "read 2401.00001")
        assert result.exit_code == 0, result.output
        assert "1 paper(s) toggled" in result.output

    def test_help_lists_the_commands(self, container, corpus) -> None:  # noqa: ANN001
        result = self._session(container, "help")
        assert result.exit_code == 0, result.output
        for command in ("papers", "ask", "show", "assets", "read", "set"):
            assert command in result.output

    def test_malformed_markup_is_not_raised(self, container, corpus, monkeypatch) -> None:  # noqa: ANN001
        """Rich raises MarkupError on mismatched tags; the session must survive."""
        result = self._session(container, "bogus", "help")
        assert result.exit_code == 0
        assert "MarkupError" not in result.output


class TestReembedOfflineAndScope:
    """Re-embedding is a local operation over stored chunks.

    The whole point is that moving a corpus to another model does not go back to
    ArXiv: chunks are already in the `chunks` table, and each space has its own
    vector table. Proven live by re-embedding 25 chunks in 0.1s with every
    outbound socket blocked.
    """

    async def test_a_versioned_paper_id_resolves(self, container, corpus) -> None:  # noqa: ANN001
        """Regression: `--paper 1706.03762v7` matched nothing, silently.

        `Paper.arxiv_id` stores the versionless id, so comparing the raw
        argument against it found no rows and reported "0 papers" with no error —
        the same trap `SemanticSearchService` fell into.
        """
        space = container.default_space
        report = await reembed_for(
            container, space, container.provider_for(space), container.vector_store_for(space)
        ).reembed(arxiv_ids=["2401.00001v1"])
        assert report.papers_embedded == 1
        assert report.chunks_embedded == len(ALPHA)

    async def test_an_arxiv_url_resolves(self, container, corpus) -> None:  # noqa: ANN001
        space = container.default_space
        report = await reembed_for(
            container, space, container.provider_for(space), container.vector_store_for(space)
        ).reembed(arxiv_ids=["https://arxiv.org/abs/2401.00002v1"])
        assert report.chunks_embedded == len(BETA)

    async def test_an_unknown_paper_is_an_error_not_silence(
        self, container, corpus
    ) -> None:  # noqa: ANN001
        """Nothing to re-embed is a different answer from a mistyped id."""
        space = container.default_space
        with pytest.raises(LookupError) as excinfo:
            await reembed_for(
                container,
                space,
                container.provider_for(space),
                container.vector_store_for(space),
            ).reembed(arxiv_ids=["2401.99999"])
        assert "2401.99999" in str(excinfo.value)

    async def test_an_empty_project_does_not_widen_to_the_corpus(
        self, container, corpus
    ) -> None:  # noqa: ANN001
        """Falling through would re-embed everything: the opposite of the ask."""
        factory = get_session_factory(container.settings.database)
        async with factory() as session:
            empty = await ProjectRepository(session).create(name="empty-two")
            await session.commit()
            slug = empty.slug
        space = container.default_space
        service = reembed_for(
            container, space, container.provider_for(space), container.vector_store_for(space)
        )
        assert await service.preview(project=slug) == []
        report = await service.reembed(project=slug)
        assert report.papers_seen == 0
        assert report.chunks_embedded == 0

    async def test_project_and_papers_intersect(self, container, corpus) -> None:  # noqa: ANN001
        """A paper the project does not hold drops out rather than adding."""
        space = container.default_space
        report = await reembed_for(
            container, space, container.provider_for(space), container.vector_store_for(space)
        ).reembed(project=corpus["project"], arxiv_ids=["2401.00002"])
        assert report.papers_seen == 0

    def test_cli_reports_an_unknown_paper_as_bad_input(
        self, container, corpus, monkeypatch
    ) -> None:  # noqa: ANN001
        from typer.testing import CliRunner

        from app.cli import app as cli_app

        monkeypatch.setattr("app.container.get_container", lambda *a, **k: container)
        result = CliRunner().invoke(
            cli_app, ["reembed", "--space", "test", "--paper", "2401.99999"]
        )
        assert result.exit_code == 2
        assert "2401.99999" in result.output

    def test_cli_dry_run_writes_nothing(
        self, container, corpus, monkeypatch
    ) -> None:  # noqa: ANN001
        """`--dry-run` must not even create the space's table."""
        import anyio
        from sqlalchemy import inspect as sa_inspect
        from typer.testing import CliRunner

        from app.cli import app as cli_app

        monkeypatch.setattr("app.container.get_container", lambda *a, **k: container)
        result = CliRunner().invoke(
            cli_app, ["reembed", "--space", "test", "--dry-run", "--project", corpus["project"]]
        )
        assert result.exit_code == 0, result.output
        assert "2401.00001" in result.output

        async def table_exists() -> bool:
            factory = get_session_factory(container.settings.database)
            async with factory() as session:
                connection = await session.connection()
                return await connection.run_sync(
                    lambda conn: sa_inspect(conn).has_table("embeddings__test")
                )

        # The table itself must not exist yet: `preview()` resolves targets and
        # stops, before the DDL that a real run would issue.
        assert anyio.run(table_exists) is False


class TestNewMcpTools:
    """The corpus-maintenance and discovery tools, driven through MCP.

    Each one existed as a CLI command or HTTP route before; these check the
    model-facing surface answers the same question and fails the same way.
    """

    async def test_search_equations_is_scoped(self, mcp_server, container, corpus) -> None:  # noqa: ANN001
        from app.db.asset_models import PaperEquation

        factory = get_session_factory(container.settings.database)
        async with factory() as session:
            session.add(
                PaperEquation(
                    id="eq-a",
                    paper_id=corpus["alpha"],
                    ordinal=1,
                    latex="\\mathrm{Attention}(Q,K,V)",
                    is_display=True,
                    source="html",
                )
            )
            session.add(
                PaperEquation(
                    id="eq-b",
                    paper_id=corpus["beta"],
                    ordinal=1,
                    latex="\\mathrm{Attention}(W,V)",
                    is_display=True,
                    source="html",
                )
            )
            await session.commit()

        scoped = await acall(
            mcp_server, "search_equations", q="Attention", project="sheaf-papers"
        )
        assert scoped["ok"] is True, scoped
        assert scoped["scope_papers"] == 1
        # Named, so a formula is traceable to its source.
        assert [e["arxiv_id"] for e in scoped["equations"]] == ["2401.00001"]

    async def test_search_equations_rejects_an_unknown_project(self, mcp_server) -> None:  # noqa: ANN001
        body = await acall(mcp_server, "search_equations", q="x", project="yok")
        assert body["ok"] is False
        assert "yok" in body["error"]

    async def test_list_categories_reports_counts(self, mcp_server, corpus) -> None:  # noqa: ANN001
        body = await acall(mcp_server, "list_categories")
        assert body["ok"] is True, body
        codes = {row["code"]: row["paper_count"] for row in body["categories"]}
        assert codes.get("cs.LG") == 2

    async def test_list_authors_returns_the_corpus(self, mcp_server, corpus) -> None:  # noqa: ANN001
        body = await acall(mcp_server, "list_authors")
        assert body["ok"] is True, body
        assert any("Lovelace" in row["name"] for row in body["authors"])

    async def test_list_authors_resolves_one(self, mcp_server, corpus) -> None:  # noqa: ANN001
        body = await acall(mcp_server, "list_authors", name="Ada Lovelace")
        assert body["ok"] is True, body
        assert {p["arxiv_id"] for p in body["papers"]} == {"2401.00001", "2401.00002"}

    async def test_list_authors_unknown_name(self, mcp_server) -> None:  # noqa: ANN001
        body = await acall(mcp_server, "list_authors", name="Nobody At All")
        assert body["ok"] is False

    async def test_most_cited_references(self, mcp_server, corpus) -> None:  # noqa: ANN001
        body = await acall(mcp_server, "most_cited_references")
        assert body["ok"] is True, body
        assert isinstance(body["papers"], list)

    async def test_list_ingest_runs(self, mcp_server, corpus) -> None:  # noqa: ANN001
        body = await acall(mcp_server, "list_ingest_runs")
        assert body["ok"] is True, body
        assert "runs" in body

    async def test_chunk_kinds_reports_without_writing(self, mcp_server, corpus) -> None:  # noqa: ANN001
        body = await acall(mcp_server, "chunk_kinds")
        assert body["ok"] is True, body
        assert body["total"] == len(ALPHA) + len(BETA)
        assert any(row["kind"] == "equation" for row in body["kinds"])

    async def test_chunk_kinds_relabels_one_paper(self, mcp_server, corpus) -> None:  # noqa: ANN001
        body = await acall(mcp_server, "chunk_kinds", relabel=True, scope="2401.00001")
        assert body["ok"] is True, body
        assert body["scanned"] == len(ALPHA)
        assert body["counts"]["equation"] == 1

    async def test_chunk_kinds_rejects_a_bad_scope(self, mcp_server) -> None:  # noqa: ANN001
        body = await acall(mcp_server, "chunk_kinds", relabel=True, scope="yok-boyle")
        assert body["ok"] is False
        assert "yok-boyle" in body["error"]

    async def test_reembed_space_dry_run(self, mcp_server, corpus) -> None:  # noqa: ANN001
        body = await acall(
            mcp_server, "reembed_space", space="test", project="sheaf-papers", dry_run=True
        )
        assert body["ok"] is True, body
        assert body["dry_run"] is True
        assert body["papers"] == ["2401.00001"]

    async def test_reembed_space_reports_an_unknown_paper(self, mcp_server) -> None:  # noqa: ANN001
        body = await acall(mcp_server, "reembed_space", space="test", arxiv_id=["2401.99999"])
        assert body["ok"] is False
        assert "2401.99999" in body["error"]

    async def test_set_paper_read_refuses_a_paper_outside_the_project(
        self, mcp_server, corpus
    ) -> None:  # noqa: ANN001
        """Regression: it reported success while updating zero rows."""
        body = await acall(
            mcp_server, "set_paper_read", project="sheaf-papers", arxiv_id="2401.00002"
        )
        assert body["ok"] is False
        assert "not in project" in body["error"]

    async def test_set_paper_read_sets_a_member(self, mcp_server, corpus) -> None:  # noqa: ANN001
        body = await acall(
            mcp_server,
            "set_paper_read",
            project="sheaf-papers",
            arxiv_id="2401.00001",
            is_read=True,
        )
        assert body["ok"] is True, body
        assert body["changed"] == 1

    async def test_delete_project_keeps_the_papers(self, mcp_server, corpus) -> None:  # noqa: ANN001
        from sqlalchemy import func, select

        from app.db.models import Paper

        body = await acall(mcp_server, "delete_project", project="sheaf-papers")
        assert body["ok"] is True, body
        assert body["papers_retained"] == 1
        factory = get_session_factory()
        async with factory() as session:
            count = int(await session.scalar(select(func.count()).select_from(Paper)))
        assert count == 2  # both papers survive the project going away


class TestMcpScopeParameters:
    """Parameters the surfaces had gained, reachable over MCP too."""

    async def test_ask_filters_by_source(self, mcp_server, embedded) -> None:  # noqa: ANN001
        body = await acall(
            mcp_server, "ask_paper_corpus", query="diffusion", source=["arxiv_html"]
        )
        assert body["ok"] is True, body
        assert body["scope"]["sources"] == ["arxiv_html"]

    async def test_ask_rejects_an_unknown_source(self, mcp_server) -> None:  # noqa: ANN001
        body = await acall(mcp_server, "ask_paper_corpus", query="x", source=["html"])
        assert body["ok"] is False
        assert "html" in body["error"]

    async def test_ask_accepts_min_score(self, mcp_server, embedded) -> None:  # noqa: ANN001
        loose = await acall(mcp_server, "ask_paper_corpus", query="diffusion", top_k=10)
        strict = await acall(
            mcp_server, "ask_paper_corpus", query="diffusion", top_k=10, min_score=0.99
        )
        assert strict["ok"] is True, strict
        assert len(strict["hits"]) < len(loose["hits"]) + 1
        assert strict["scope"]["min_score"] == 0.99

    async def test_list_assets_accepts_a_project(self, mcp_server, corpus) -> None:  # noqa: ANN001
        from app.db.asset_models import PaperFigure

        factory = get_session_factory()
        async with factory() as session:
            session.add(
                PaperFigure(
                    id="f1",
                    paper_id=corpus["alpha"],
                    ordinal=1,
                    label="Figure 1",
                    caption="A figure.",
                    source="html",
                )
            )
            await session.commit()
        body = await acall(mcp_server, "list_assets", project="sheaf-papers")
        assert body["ok"] is True, body
        assert body["project"] == "sheaf-papers"
        assert [p["arxiv_id"] for p in body["papers"]] == ["2401.00001"]

    async def test_list_assets_rejects_both_scopes(self, mcp_server) -> None:  # noqa: ANN001
        body = await acall(
            mcp_server, "list_assets", arxiv_id="2401.00001", project="sheaf-papers"
        )
        assert body["ok"] is False
        assert "not both" in body["error"]

    async def test_list_assets_needs_one_scope(self, mcp_server) -> None:  # noqa: ANN001
        body = await acall(mcp_server, "list_assets")
        assert body["ok"] is False
        assert "either" in body["error"]

