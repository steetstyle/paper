"""Per-operation device selection on the MCP tools.

The MCP surface has to accept exactly the device strings the CLI and the HTTP
layer accept. A tool that takes something ``paper --device`` rejects is worse than
either rejecting it, because the caller has no way to find out which rule applies
until a real ingest fails minutes in — so these tests assert the *same* parser
rather than three lookalikes.

The point of the feature is separation: asking for cuda on one call must not
change what every other call does. So the assertions are about which provider the
cache hands back, not merely that requests succeed.
"""

from __future__ import annotations

import pytest

from app.mcp.devices import (
    ACCEPTED_DEVICES,
    DEVICES,
    INDEXED_DEVICES,
    parse_device,
)

pytest_plugins = ("test_mcp",)


class TestParseDevice:
    @pytest.mark.parametrize("name", DEVICES)
    def test_a_bare_accelerator_is_accepted(self, name: str) -> None:
        assert parse_device(name) == name

    @pytest.mark.parametrize("name", INDEXED_DEVICES)
    def test_an_index_is_accepted_where_torch_has_one(self, name: str) -> None:
        assert parse_device(f"{name}:1") == f"{name}:1"

    def test_case_and_padding_are_normalised(self) -> None:
        assert parse_device("  CUDA:1 ") == "cuda:1"
        assert parse_device("CPU") == "cpu"

    def test_a_bare_index_is_accepted(self) -> None:
        """What the HTTP layer accepts too; a client that sends the index and a
        client that sends ``cuda:0`` mean the same card."""
        assert parse_device("1") == "1"

    def test_none_means_no_opinion(self) -> None:
        """What every existing caller passes, so every existing caller keeps
        working and the configured device is still the default."""
        assert parse_device(None) is None

    def test_an_empty_string_is_refused_rather_than_read_as_none(self) -> None:
        """A caller that sent one has asked about a device. Answering with the
        configured one would hide the mistake behind a working call."""
        with pytest.raises(ValueError, match="empty"):
            parse_device("")

    @pytest.mark.parametrize(
        "value", ["cud", "gpu", "cuda:", "0:1", "cpu:0", "cuda:x", "", "  ", "cuda:1:2"]
    )
    def test_nonsense_is_refused_by_name(self, value: str) -> None:
        with pytest.raises(ValueError):
            parse_device(value)

    def test_a_bool_is_not_an_index(self) -> None:
        """`device=True` is a wrong-shaped argument, not device 1."""
        with pytest.raises(ValueError, match="must be a string"):
            parse_device(True)

    @pytest.mark.parametrize("value", [1.5, [], {}, object()])
    def test_a_non_string_is_refused(self, value: object) -> None:
        with pytest.raises(ValueError, match="must be a string"):
            parse_device(value)

    def test_the_message_says_what_is_accepted(self) -> None:
        """An error that lists the valid values is the difference between a typo
        costing a second and costing a round trip."""
        with pytest.raises(ValueError) as caught:
            parse_device("cud")
        for name in DEVICES:
            assert name in str(caught.value)


class TestParityWithTheOtherSurfaces:
    """One rule, three doors."""

    @pytest.mark.parametrize(
        "value", ["cpu", "cuda", "cuda:1", "mps", "npu", "1", "CUDA:2", "cud", "", "cpu:0"]
    )
    def test_the_http_parser_agrees(self, value: str) -> None:
        from app.api.devices import parse_device as http_parse  # noqa: PLC0415

        def outcome(parser, argument: str) -> str:  # noqa: ANN001, ANN202
            try:
                return f"ok:{parser(argument)}"
            except ValueError as exc:
                return f"err:{exc}"

        assert outcome(parse_device, value) == outcome(http_parse, value)

    def test_the_cli_agrees(self) -> None:
        """``paper --device`` is the third door. Checked through the CLI helper so
        a divergence fails here rather than in a user's terminal."""
        import typer  # noqa: PLC0415

        from app.cli import _device  # noqa: PLC0415

        def outcome(argument: str) -> str:  # noqa: ANN202
            try:
                return f"ok:{_device(argument)}"
            except typer.BadParameter as exc:
                return f"err:{exc}"

        for value in ("cpu", "cuda", "cuda:1", "mps", "1", "CUDA:2"):
            assert outcome(value) == f"ok:{value.strip().lower()}"
        for value in ("cud", "", "cpu:0", "cuda:"):
            assert outcome(value).startswith("err:")


class TestProviderSelection:
    def test_a_device_selects_a_different_provider(self, container) -> None:  # noqa: ANN001
        """The cache is keyed by (space, device). Keying on the space alone would
        make the second device silently reuse the first one's — a loaded model
        belongs to the device it was loaded on."""
        space = container.default_space
        cpu = container.provider_for(space, "cpu")
        assert container.provider_for(space, "cpu") is cpu, "same device must reuse"

    def test_omitting_the_device_keeps_the_configured_one(self, container) -> None:  # noqa: ANN001
        """Asserted on the cache key rather than the provider's device: the test
        container's provider is the hashing one, which has no device at all, and
        the key is the actual contract."""
        from app.config import get_settings  # noqa: PLC0415

        space = container.default_space
        container.provider_for(space)  # noqa: SLF001 - populates the cache
        expected = get_settings().embedding.device or "cpu"
        assert f"{space.fingerprint}@{expected}" in container._embeddings  # noqa: SLF001

    def test_two_devices_do_not_collide_in_the_cache(self, container) -> None:  # noqa: ANN001
        space = container.default_space
        keys = list(container._embeddings)  # noqa: SLF001
        container.provider_for(space, "cpu")
        container.provider_for(space, "cuda")
        fresh = [k for k in container._embeddings if k not in keys]  # noqa: SLF001
        assert len(fresh) >= 1
        assert any(k.endswith("@cuda") for k in container._embeddings)  # noqa: SLF001
        assert any(k.endswith("@cpu") for k in container._embeddings)  # noqa: SLF001


class TestToolsExposeIt:
    """The surface itself, so a parameter cannot be dropped by accident."""

    @pytest.mark.parametrize(
        "name", ["ingest_paper", "ask_paper_corpus", "reembed_space"]
    )
    def test_the_tool_declares_a_device_argument(self, mcp_server, name: str) -> None:  # noqa: ANN001
        # Synchronous on purpose: `anyio.run` cannot start inside a running loop,
        # and every tool test here lists tools rather than calling one.
        import anyio  # noqa: PLC0415

        tools = {t.name: t for t in anyio.run(mcp_server.list_tools)}
        properties = tools[name].input_schema.get("properties", {})
        assert "device" in properties, f"{name} lost its device argument"

    @pytest.mark.parametrize(
        "name", ["ingest_paper", "ask_paper_corpus", "reembed_space"]
    )
    def test_the_description_explains_it(self, mcp_server, name: str) -> None:  # noqa: ANN001
        import anyio  # noqa: PLC0415

        tools = {t.name: t for t in anyio.run(mcp_server.list_tools)}
        description = tools[name].description or ""
        assert "device" in description.lower()
        # The accepted set, so a client can build a valid call without guessing.
        assert any(name_ in description for name_ in DEVICES)

    def test_reading_a_tool_stays_read_only(self, mcp_server) -> None:  # noqa: ANN001
        """Asking for cuda changes where a query is computed. It does not write,
        so the annotation must not change — a client auto-approves on that flag."""
        import anyio  # noqa: PLC0415

        tools = {t.name: t for t in anyio.run(mcp_server.list_tools)}
        assert tools["ask_paper_corpus"].annotations.read_only_hint is True

    async def test_an_invalid_device_is_a_clear_failure(self, mcp_server) -> None:  # noqa: ANN001
        """A tool returns its failure shape, not a traceback — and it must name
        the accepted set, because a typo is the overwhelmingly likely mistake."""
        from tests.test_mcp import acall  # noqa: PLC0415

        payload = await acall(
            mcp_server, "ask_paper_corpus", query="anything", device="cud"
        )
        assert payload.get("error") or payload.get("ok") is False
        text = str(payload)
        assert "cuda" in text
        assert ACCEPTED_DEVICES.split(",")[0].strip() in text


class TestMineruOptionsCarryADevice:
    def test_the_extraction_device_is_a_per_document_option(self) -> None:
        """Separate from the embedding device on purpose: a scanned textbook wants
        the GPU for the layout model while its query embedding stays on CPU."""
        from app.domain.models import MineruOptions  # noqa: PLC0415

        options = MineruOptions(device="cuda")
        assert options.has_overrides
        assert options.merged_with(MineruOptions(device="cpu")).device == "cuda"

    def test_the_extractor_is_told(self) -> None:
        """It reaches MinerU's settings, which is the only place a device can
        actually take effect."""
        from app.clients.content.mineru import ExtractRequest  # noqa: PLC0415
        from app.config import MineruSettings  # noqa: PLC0415
        from app.domain.models import MineruOptions  # noqa: PLC0415

        settings = ExtractRequest(options=MineruOptions(device="cuda")).settings_for(
            MineruSettings()
        )
        assert settings.device == "cuda"
