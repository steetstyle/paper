"""The same filter must mean the same thing in the CLI, the HTTP API and MCP.

Parity is the whole point of routing all three surfaces through
``search_query_from``, so these tests exercise each surface for real and compare
the ``search_query`` that reaches ArXiv.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from test_mcp import call
from test_pipeline_e2e import PAPER_HTML, FakeContentFetcher, FakeHttp

from paper_app.clients.arxiv.filters import search_query_from
from paper_app.container import set_container
from paper_app.db.session import dispose_engines
from paper_app.db.vector_store.memory_store import InMemoryVectorStore
from paper_app.embeddings.hashing_provider import HashingEmbeddingProvider
from paper_app.infra.storage import LocalBlobStore
from paper_app.main import create_app


class StubExtractor:
    available_backends = ["stub"]

    def describe_backends(self) -> dict[str, str]:
        return {"stub": "stub (fake)"}

    async def extract_pdf(self, path: Path):  # noqa: ARG002
        from paper_app.clients.content.mineru import ExtractionError

        raise ExtractionError("unused")


@pytest.fixture
def container(settings, tmp_path: Path):  # noqa: ANN201
    from paper_app.container import Container

    settings.chunking = settings.chunking.model_copy(
        update={"max_tokens": 80, "overlap_tokens": 15, "min_tokens": 5}
    )
    container = Container(settings)
    space = container.default_space
    # The real collaborators are swappable at construction, so the fake
    # assignments below are the intended way to inject them.
    container._http = FakeHttp()  # type: ignore[assignment]  # noqa: SLF001
    blob_store = LocalBlobStore(root=tmp_path / "blobs")
    container._blobs = blob_store  # noqa: SLF001
    container._fetcher = FakeContentFetcher(  # type: ignore[assignment]  # noqa: SLF001
        blob_store=blob_store, html=PAPER_HTML
    )
    container._mineru = StubExtractor()  # type: ignore[assignment]  # noqa: SLF001
    container._embeddings[space.fingerprint] = HashingEmbeddingProvider(  # noqa: SLF001
        model="hashing-test", dimensions=64
    )
    container._stores[f"{space.name}:{space.fingerprint}:{space.distance}"] = (  # noqa: SLF001
        InMemoryVectorStore(space)
    )
    set_container(container)
    return container


@pytest.fixture
async def api_client(container):  # noqa: ANN201
    app = create_app(settings=container.settings, container=container)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
    set_container(None)
    await dispose_engines()


@pytest.fixture
def mcp_server(container, monkeypatch):  # noqa: ANN201
    from paper_app.mcp import server as mod

    monkeypatch.setattr(mod, "get_container", lambda _s=None: container)
    return mod.server


def cli_query(*args: str) -> str:
    """Run `paper search --json --show-query` and return the compiled query.

    `--show-query` short-circuits before any network access, so this measures
    the CLI's compilation alone.
    """
    from typer.testing import CliRunner

    from paper_app.cli import app

    result = CliRunner().invoke(app, ["search", *args, "--show-query"])
    assert result.exit_code == 0, result.output
    return result.output.strip()


def cli_json(container, monkeypatch, *args: str) -> dict[str, Any]:  # noqa: ANN001
    """Run a real `paper search --json` against the test container.

    The command imports `get_container` lazily inside its body, so the patch has
    to land on `paper_app.container` rather than on `paper_app.cli`.
    """
    from typer.testing import CliRunner

    from paper_app.cli import app
    from paper_app.container import get_container

    monkeypatch.setattr("paper_app.container.get_container", lambda *a, **k: container)
    assert get_container() is container
    result = CliRunner().invoke(app, ["search", *args, "--json"])
    assert result.exit_code == 0, result.output
    return json.loads(result.output)


class TestStructuredParity:
    """The same intent, expressed three ways, compiles to one string."""

    STRUCTURED: dict[str, Any] = {
        "fields": {
            "title": ["transformer"],
            "author": ["vaswani"],
            "abstract": ["attention"],
            "comment": ["accepted at neurips"],
            "journal": ["neurips"],
            "category": ["cs.CL"],
            "report_number": ["LA-UR"],
        },
        "submitted_from": "2023-01-01",
        "submitted_to": "2024-01-01",
        "max_results": 5,
        "sort_by": "submittedDate",
        "sort_order": "ascending",
    }

    EXPECTED_FIELDS = (
        'ti:transformer AND au:vaswani AND abs:attention AND co:"accepted at neurips" '
        'AND jr:neurips AND cat:cs.CL AND rn:LA-UR'
    )

    def test_expected_shape_is_stable(self) -> None:
        """Anchor the exact query the three surfaces are compared against."""
        request = search_query_from(**self.STRUCTURED)
        compiled = request.filter.compile()
        assert self.EXPECTED_FIELDS in compiled
        assert "submittedDate:[202301010000 TO 202401010000]" in compiled

    async def test_api_compiles_the_structured_fields(self, api_client) -> None:  # noqa: ANN001
        response = await api_client.get(
            "/api/v1/arxiv/search",
            params=[
                ("title", "transformer"),
                ("author", "vaswani"),
                ("abstract", "attention"),
                ("comment", "accepted at neurips"),
                ("journal", "neurips"),
                ("category", "cs.CL"),
                ("report_number", "LA-UR"),
                ("submitted_from", "2023-01-01"),
                ("submitted_to", "2024-01-01"),
                ("max_results", 5),
                ("sort_by", "submittedDate"),
                ("sort_order", "ascending"),
            ],
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert self.EXPECTED_FIELDS in body["search_query"]
        assert "submittedDate:[202301010000 TO 202401010000]" in body["search_query"]

    def test_cli_compiles_the_same_structured_fields(self) -> None:
        compiled = cli_query(
            "--title", "transformer",
            "--author", "vaswani",
            "--abstract", "attention",
            "--comment", "accepted at neurips",
            "--journal", "neurips",
            "--category", "cs.CL",
            "--report-number", "LA-UR",
            "--since", "2023-01-01",
            "--until", "2024-01-01",
        )
        assert self.EXPECTED_FIELDS in compiled
        assert "submittedDate:[202301010000 TO 202401010000]" in compiled

    def test_mcp_compiles_the_same_structured_fields(self, mcp_server) -> None:  # noqa: ANN001
        payload = call(
            mcp_server,
            "search_arxiv",
            title="transformer",
            author="vaswani",
            abstract="attention",
            comment="accepted at neurips",
            journal="neurips",
            category="cs.CL",
            report_number="LA-UR",
            submitted_from="2023-01-01",
            submitted_to="2024-01-01",
        )
        assert payload["ok"] is True, payload
        assert self.EXPECTED_FIELDS in payload["search_query"]
        assert "submittedDate:[202301010000 TO 202401010000]" in payload["search_query"]


class TestOperatorParity:
    def test_or_within_a_field_agrees_everywhere(self) -> None:
        expected = "ti:a OR ti:b"
        assert search_query_from(
            fields={"title": ["a", "b"]}, operator="OR"
        ).filter.compile() == expected
        assert cli_query("--title", "a", "--title", "b", "--op", "OR") == expected

    async def test_api_or_matches(self, api_client) -> None:  # noqa: ANN001
        response = await api_client.get(
            "/api/v1/arxiv/search",
            params=[("title", "a"), ("title", "b"), ("operator", "OR")],
        )
        assert response.status_code == 200
        assert response.json()["search_query"] == "ti:a OR ti:b"

    def test_operator_does_not_widen_across_fields(self) -> None:
        compiled = cli_query("--title", "a", "--title", "b", "--author", "ho", "--op", "OR")
        assert compiled == "ti:a OR ti:b AND au:ho"


class TestRawExpressionParity:
    """A hand-written expression must survive byte-for-byte on every surface."""

    EXPRESSIONS = [
        "cat:cs.CL",
        "ti:transformer AND cat:cs.LG",
        "(ti:a OR ti:b) ANDNOT abs:survey",
        'ti:"attention is all you need"',
        "au:del_maestro AND submittedDate:[202301010600+TO+202401010600]",
        "all:electron ANDNOT all:proton",
        "jr:NeurIPS OR co:accepted",
        "rn:LA-UR-*",
    ]

    @pytest.mark.parametrize("expression", EXPRESSIONS)
    def test_builder_passes_it_through_untouched(self, expression: str) -> None:
        assert search_query_from(raw=expression).filter.compile() == expression

    @pytest.mark.parametrize("expression", EXPRESSIONS)
    def test_cli_passes_it_through_untouched(self, expression: str) -> None:
        assert cli_query(expression) == expression

    @pytest.mark.parametrize("expression", EXPRESSIONS)
    async def test_api_passes_it_through_untouched(self, api_client, expression: str) -> None:  # noqa: ANN001
        response = await api_client.get(
            "/api/v1/arxiv/search", params={"filter": expression}
        )
        assert response.status_code == 200, response.text
        assert response.json()["search_query"] == expression

    @pytest.mark.parametrize("expression", EXPRESSIONS)
    def test_mcp_passes_it_through_untouched(self, mcp_server, expression: str) -> None:  # noqa: ANN001
        payload = call(mcp_server, "search_arxiv", query=expression)
        assert payload["ok"] is True, payload
        assert payload["search_query"] == expression

    def test_quotes_survive_shell_quoting_in_the_cli(self) -> None:
        assert cli_query('ti:"attention is all you need"') == 'ti:"attention is all you need"'

    def test_a_raw_expression_anded_with_a_field(self) -> None:
        assert (
            cli_query("cat:cs.CL", "--title", "transformer")
            == "cat:cs.CL AND ti:transformer"
        )


class TestIdListParity:
    def test_ids_alone_send_no_search_query(self) -> None:
        request = search_query_from(id_list=["1706.03762v5"])
        assert request.to_search_query().raw is None

    async def test_api_returns_the_requested_ids(self, api_client) -> None:  # noqa: ANN001
        response = await api_client.get(
            "/api/v1/arxiv/search", params={"arxiv_id": "1706.03762"}
        )
        assert response.status_code == 200, response.text
        body = response.json()
        # Nothing to search on, so only the id lookup is used.
        assert body["search_query"] is None
        assert body["items_per_page"] == 1
        assert body["papers"][0]["arxiv_id"] == "1706.03762"

    def test_mcp_returns_the_requested_ids(self, mcp_server) -> None:  # noqa: ANN001
        payload = call(mcp_server, "search_arxiv", arxiv_id="1706.03762")
        assert payload["ok"] is True, payload
        assert payload["search_query"] is None
        assert payload["returned"] == 1


class TestValidationParity:
    """All three surfaces refuse a broken filter, and say the same thing."""

    async def test_api_rejects_an_unknown_field_prefix(self, api_client) -> None:  # noqa: ANN001
        response = await api_client.get(
            "/api/v1/arxiv/search", params={"filter": "keyword:x"}
        )
        assert response.status_code == 422
        assert "keyword" in response.json()["detail"]

    def test_cli_rejects_an_unknown_field_prefix(self) -> None:
        from typer.testing import CliRunner

        from paper_app.cli import app

        result = CliRunner().invoke(app, ["search", "keyword:x", "--show-query"])
        assert result.exit_code == 2
        assert "keyword" in result.output

    def test_mcp_rejects_an_unknown_field_prefix(self, mcp_server) -> None:  # noqa: ANN001
        payload = call(mcp_server, "search_arxiv", query="keyword:x")
        assert payload["ok"] is False
        assert "keyword" in payload["error"]

    async def test_api_rejects_unbalanced_parentheses(self, api_client) -> None:  # noqa: ANN001
        response = await api_client.get(
            "/api/v1/arxiv/search", params={"filter": "(ti:a AND ti:b"}
        )
        assert response.status_code == 422

    def test_mcp_rejects_an_unknown_operator(self, mcp_server) -> None:  # noqa: ANN001
        payload = call(mcp_server, "search_arxiv", title="x", operator="XOR")
        assert payload["ok"] is False
        assert "AND" in payload["error"]

    def test_cli_rejects_an_unknown_operator(self) -> None:
        from typer.testing import CliRunner

        from paper_app.cli import app

        result = CliRunner().invoke(app, ["search", "--title", "x", "--op", "XOR"])
        assert result.exit_code == 2
        assert "AND" in result.output

    async def test_api_rejects_an_empty_search(self, api_client) -> None:  # noqa: ANN001
        response = await api_client.get("/api/v1/arxiv/search")
        assert response.status_code == 422

    def test_mcp_rejects_an_empty_search(self, mcp_server) -> None:  # noqa: ANN001
        payload = call(mcp_server, "search_arxiv")
        assert payload["ok"] is False
        assert "at least one of" in payload["error"]


class TestLowercaseOperatorWarningParity:
    """Lowercase `and` is the most common mistake, so all three point it out."""

    QUERY = "ti:transformer and cat:cs.CL"

    async def test_api_warns(self, api_client) -> None:  # noqa: ANN001
        response = await api_client.get("/api/v1/arxiv/search", params={"filter": self.QUERY})
        assert response.status_code == 200
        warnings = response.json()["warnings"]
        assert any("UPPERCASE" in w for w in warnings)
        # Reported, not silently rewritten.
        assert response.json()["search_query"] == self.QUERY

    def test_cli_warns(self) -> None:
        from typer.testing import CliRunner

        from paper_app.cli import app

        result = CliRunner().invoke(app, ["search", self.QUERY, "--show-query"])
        assert "UPPERCASE" in result.output

    def test_mcp_warns(self, mcp_server) -> None:  # noqa: ANN001
        payload = call(mcp_server, "search_arxiv", query=self.QUERY)
        assert any("UPPERCASE" in w for w in payload["warnings"])
        assert payload["search_query"] == self.QUERY

    async def test_a_correct_query_produces_no_warnings(self, api_client) -> None:  # noqa: ANN001
        response = await api_client.get(
            "/api/v1/arxiv/search", params={"filter": "ti:a AND ti:b OR ti:c ANDNOT ti:d"}
        )
        assert response.json()["warnings"] == []


class TestPostFilterParity:
    """Client-side filters are reported identically everywhere."""

    async def test_api_reports_the_post_filter_it_used(self, api_client) -> None:  # noqa: ANN001
        response = await api_client.get(
            "/api/v1/arxiv/search",
            params={"category": "cs.CL", "has_pdf": "true"},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["post_filter"]["has_pdf"] is True

    def test_mcp_reports_the_post_filter_it_used(self, mcp_server) -> None:  # noqa: ANN001
        payload = call(mcp_server, "search_arxiv", category="cs.CL", has_pdf=True)
        assert payload["ok"] is True, payload

    def test_has_pdf_false_keeps_only_entries_without_a_pdf(self, mcp_server) -> None:  # noqa: ANN001
        payload = call(mcp_server, "search_arxiv", category="cs.CL", has_pdf=False)
        assert payload["ok"] is True, payload
        # The fixture entry does carry a PDF, so it is filtered out.
        assert payload["returned"] == 0
        assert payload["filtered_out"] == 1

    async def test_api_has_pdf_false_matches(self, api_client) -> None:  # noqa: ANN001
        response = await api_client.get(
            "/api/v1/arxiv/search",
            params={"category": "cs.CL", "has_pdf": "false"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["items_per_page"] == 0
        assert body["filtered_out"] == 1

    def test_pagination_offset_is_carried_through(self, mcp_server) -> None:  # noqa: ANN001
        payload = call(mcp_server, "search_arxiv", category="cs.CL", start=0, max_results=1)
        assert payload["ok"] is True, payload
        assert payload["returned"] == 1


class TestHarvestUsesTheSameFilter:
    """`paper harvest` must accept exactly what `paper search` accepts."""

    def test_harvest_exposes_the_same_filter_flags(self) -> None:
        from typer.testing import CliRunner

        from paper_app.cli import app

        def flags(command: str) -> set[str]:
            result = CliRunner().invoke(app, [command, "--help"])
            assert result.exit_code == 0
            return {
                token.strip().split()[0]
                for token in result.output.split()
                if token.strip().startswith("--")
            }

        search_flags = flags("search")
        harvest_flags = flags("harvest")
        # Everything search understands for filtering, harvest understands too.
        for shared in [
            "--title", "--author", "--abstract", "--comment", "--journal",
            "--category", "--report-number", "--all", "--op", "--id",
            "--since", "--until", "--sort-by", "--sort-order", "--has-pdf",
            "--has-html",
        ]:
            assert shared in search_flags, f"{shared} missing from search"
            assert shared in harvest_flags, f"{shared} missing from harvest"

    def test_harvest_rejects_a_broken_filter_before_ingesting(self) -> None:
        from typer.testing import CliRunner

        from paper_app.cli import app

        result = CliRunner().invoke(app, ["harvest", "keyword:x"])
        assert result.exit_code == 2
        assert "keyword" in result.output