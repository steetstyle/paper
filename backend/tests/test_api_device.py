"""Per-operation ``device`` on the embedding routes.

The point of the option is that a device belongs to a call rather than to the
process, so these tests assert on which device the provider and the ingestion
service were asked for — not merely that the request succeeded.
"""

from __future__ import annotations

import pytest

from app.api.devices import parse_device

# The app-level `container` and `client` fixtures live in test_api.py; loading
# that module as a plugin reuses them instead of rebuilding the app and its
# fakes here. Named per test_api.py's docstring: "the real app with faked
# external services".
pytest_plugins = ("test_api",)

ARXIV_ID = "1706.03762"


async def _import_with_ingest(client, device: str | None = None) -> None:  # noqa: ANN001
    """Drive the project import far enough to build an ingestion service."""
    params = [("ingest", "true")]
    if device is not None:
        params.append(("device", device))
    await client.post(
        "/api/v1/projects/reading/papers",
        json=[ARXIV_ID],
        params=params,
    )


@pytest.fixture
def devices_requested(monkeypatch):  # noqa: ANN201
    """Every ``device`` the routes hand to the ingestion service builder.

    Raw rather than resolved: ``None`` must reach the service intact, because
    deciding that "no device" means "the configured one" is that function's job
    and not the route's.
    """
    import app.services.ingestion as ingestion

    real = ingestion.build_ingestion_service
    seen: list[str | None] = []

    def spy(container, space=None, *, device=None):  # noqa: ANN001, ANN202
        seen.append(device)
        return real(container, space, device=device)

    monkeypatch.setattr(ingestion, "build_ingestion_service", spy)
    return seen


@pytest.fixture
def devices_embedded_on(monkeypatch):  # noqa: ANN201
    """Every ``device`` the pipeline is actually built with."""
    from app.container import Container

    real = Container.build_steps
    seen: list[str | None] = []

    def spy(self, space=None, *, device=None):  # noqa: ANN001, ANN202
        seen.append(device)
        return real(self, space, device=device)

    monkeypatch.setattr(Container, "build_steps", spy)
    return seen


class TestParseDevice:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("cpu", "cpu"),
            ("CUDA", "cuda"),
            ("  cuda:1  ", "cuda:1"),
            ("mps", "mps"),
            ("npu", "npu"),
            ("npu:3", "npu:3"),
            ("1", "1"),
            (1, "1"),
            (None, None),
        ],
    )
    def test_accepted_forms_normalise(self, raw, expected) -> None:
        assert parse_device(raw) == expected

    @pytest.mark.parametrize(
        "raw", ["cud", "gpu", "cuda:", "cuda:x", "-1", "cpu:0", "0:1", "", "  "]
    )
    def test_typos_are_refused(self, raw) -> None:
        with pytest.raises(ValueError, match="choose"):
            parse_device(raw)

    def test_the_error_names_the_accepted_set(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            parse_device("cud")
        message = str(excinfo.value)
        assert "cpu, cuda, mps, npu" in message
        assert "cuda:1" in message

    @pytest.mark.parametrize("raw", [True, False, 1.5, ["cpu"], {"a": 1}])
    def test_wrongly_shaped_values_are_refused(self, raw) -> None:
        with pytest.raises(ValueError):
            parse_device(raw)


class TestSemanticSearchDevice:
    @pytest.fixture(autouse=True)
    async def _ingested(self, client) -> None:
        await client.post("/api/v1/ingest", json={"arxiv_id": ARXIV_ID}, params={"wait": True})

    async def _search(self, client, **body):  # noqa: ANN003, ANN202
        return await client.post(
            "/api/v1/search/semantic", json={"query": "self-attention", "top_k": 3, **body}
        )

    @pytest.mark.parametrize("device", ["cpu", "cuda", "cuda:1", "mps", "npu", "1"])
    async def test_valid_device_is_accepted(self, client, device) -> None:
        response = await self._search(client, device=device)
        assert response.status_code == 200
        assert response.json()["hits"]

    async def test_device_reaches_the_provider_cache(self, client, container) -> None:
        """The device is what distinguishes two providers for one space.

        Asserted on the cache key, because that key is the mechanism: one space
        on a second device must not hand back the first device's instance, which
        is the mistake the per-operation option exists to make impossible.
        """
        await self._search(client, device="cuda:1")
        keys = [key for key in container._embeddings if "cuda:1" in key]  # noqa: SLF001
        assert keys, "no provider was cached for the requested device"

    async def test_without_a_device_the_configured_provider_is_reused(
        self, client, container
    ) -> None:
        """Omitting the field must not build anything new."""
        before = dict(container._embeddings)  # noqa: SLF001
        response = await self._search(client)
        assert response.status_code == 200
        assert set(container._embeddings) - set(before) == set()  # noqa: SLF001

    @pytest.mark.parametrize("device", ["cud", "gpu", "cuda:x", "cuda:", "0:1"])
    async def test_invalid_device_is_422(self, client, device) -> None:
        response = await self._search(client, device=device)
        assert response.status_code == 422
        # Names the accepted set, so a typo is answerable without the docs.
        assert "cpu, cuda, mps, npu" in response.text

    @pytest.mark.parametrize("device", [1.5, True, ["cpu"]])
    async def test_wrongly_shaped_device_is_422(self, client, device) -> None:
        assert (await self._search(client, device=device)).status_code == 422

    async def test_an_empty_device_is_422_not_a_silent_default(self, client) -> None:
        """Blank is not "no opinion": omitting the field is how a caller asks for
        the configured device, so a blank value must not quietly mean the same.
        """
        for device in ("", "  "):
            assert (await self._search(client, device=device)).status_code == 422


class TestIngestDevice:
    async def test_valid_device_reaches_the_service(
        self, client, devices_requested
    ) -> None:
        response = await client.post(
            "/api/v1/ingest",
            json={"arxiv_id": ARXIV_ID, "device": "cuda"},
            params={"wait": True},
        )
        assert response.status_code == 202
        assert devices_requested == ["cuda"]

    async def test_omitted_device_keeps_the_configured_one(
        self, client, container, devices_requested, devices_embedded_on
    ) -> None:
        response = await client.post(
            "/api/v1/ingest", json={"arxiv_id": ARXIV_ID}, params={"wait": True}
        )
        assert response.status_code == 202
        assert devices_requested == [None]
        # And the service resolves that against settings rather than the route
        # having already pinned it.
        assert devices_embedded_on == [container.settings.embedding.device or "cpu"]

    async def test_the_device_is_the_one_the_pipeline_uses(
        self, client, devices_embedded_on
    ) -> None:
        response = await client.post(
            "/api/v1/ingest",
            json={"arxiv_id": ARXIV_ID, "device": "cuda:1"},
            params={"wait": True},
        )
        assert response.status_code == 202
        assert devices_embedded_on == ["cuda:1"]

    @pytest.mark.parametrize("device", ["cud", "gpu", "cuda:gpu", "0:1"])
    async def test_invalid_device_is_422(self, client, device) -> None:
        response = await client.post(
            "/api/v1/ingest", json={"arxiv_id": ARXIV_ID, "device": device}
        )
        assert response.status_code == 422
        assert "cpu, cuda, mps, npu" in response.text

    async def test_invalid_device_is_refused_before_any_work(
        self, client, monkeypatch
    ) -> None:
        """A bad device must not reach torch, and must not start a run either."""
        called = False

        def fail(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
            nonlocal called
            called = True
            raise AssertionError("ingestion started despite an invalid device")

        monkeypatch.setattr("app.services.ingestion.build_ingestion_service", fail)
        response = await client.post(
            "/api/v1/ingest", json={"arxiv_id": ARXIV_ID, "device": "nope"}
        )
        assert response.status_code == 422
        assert called is False


class TestProjectImportDevice:
    """``?device=`` on the project import, which embeds only when it ingests.

    Same service as ``/ingest``, but the argument arrives in the query string, so
    this exercises the dependency form of the validation rather than pydantic's.
    """

    @pytest.fixture(autouse=True)
    async def _project(self, client) -> None:
        await client.post("/api/v1/projects", json={"name": "Reading"})

    async def test_device_query_reaches_the_service(
        self, client, devices_requested
    ) -> None:
        await _import_with_ingest(client, device="cuda")
        assert devices_requested == ["cuda"]

    async def test_omitted_device_is_passed_through(
        self, client, devices_requested
    ) -> None:
        await _import_with_ingest(client)
        assert devices_requested == [None]

    async def test_invalid_device_query_is_422(self, client) -> None:
        response = await client.post(
            "/api/v1/projects/reading/papers",
            json=[ARXIV_ID],
            params=[("ingest", "true"), ("device", "cud")],
        )
        assert response.status_code == 422
        assert "cpu, cuda, mps, npu" in response.text


class TestDeviceSchemaDocs:
    async def test_openapi_documents_both_body_fields(self, client) -> None:
        components = (await client.get("/openapi.json")).json()["components"]["schemas"]
        for model in ("SemanticSearchRequest", "IngestRequest"):
            device = components[model]["properties"]["device"]
            assert "cpu, cuda" in device["description"]
            assert device.get("default") is None

    async def test_openapi_documents_the_project_import_query_param(
        self, client
    ) -> None:
        operation = (await client.get("/openapi.json")).json()["paths"][
            "/api/v1/projects/{slug}/papers"
        ]["post"]
        device = next(p for p in operation["parameters"] if p["name"] == "device")
        assert device["in"] == "query"
        assert device["required"] is False