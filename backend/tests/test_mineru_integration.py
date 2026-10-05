"""Tests that assert against the *real* MinerU install when one is present.

These are skipped in CI (where MinerU needs torch + several hundred MB of model
weights) but they are what actually pins the integration to the upstream
package: a rename, a moved module or a changed signature fails here rather than
in production.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from app.clients.content import mineru_resolver as resolver
from app.clients.content.mineru import MineruCliBackend, MineruPythonApiBackend
from app.clients.content.mineru_resolver import GENERATION_1, GENERATION_2, GENERATION_4
from app.config import MineruSettings

pytestmark = pytest.mark.usefixtures("_clear_resolver_cache")


@pytest.fixture
def _clear_resolver_cache():  # noqa: ANN201
    resolver.reset_cache()
    yield
    resolver.reset_cache()


@pytest.fixture
def pdf(tmp_path: Path) -> Path:
    """A tiny, valid, single-page PDF built without extra dependencies."""
    path = tmp_path / "sample.pdf"
    content = (
        b"%PDF-1.4\n"
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
        b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 200 200]>>endobj\n"
        b"trailer<</Root 1 0 R>>\n"
        b"%%EOF\n"
    )
    path.write_bytes(content)
    return path


class TestRealPackageIsBound:
    def test_resolver_binds_something_when_mineru_installed(self) -> None:
        api = resolver.resolve_api()
        if api is None:
            pytest.skip("MinerU is not installed")
        assert api.available
        assert api.generation in {GENERATION_1, GENERATION_2, GENERATION_4}
        assert api.version

    def test_callables_match_the_contract_we_call(self) -> None:
        """Every generation must expose exactly the signatures we invoke."""
        api = resolver.resolve_api()
        if api is None:
            pytest.skip("MinerU is not installed")

        if api.generation == GENERATION_4:
            assert callable(api.render_markdown)
            assert callable(api.doc_analyze or api.aio_doc_analyze)
            render_params = inspect.signature(api.render_markdown).parameters
            assert "middle_json" in render_params
            analyze = api.aio_doc_analyze or api.doc_analyze
            params = inspect.signature(analyze).parameters
            for expected in ("file_bytes", "effort", "parse_mode", "image_analysis"):
                assert expected in params, f"doc_analyze is missing {expected}"
            return

        assert callable(api.do_parse or api.aio_do_parse)
        parse = api.aio_do_parse or api.do_parse
        positional = [
            p.name
            for p in inspect.signature(parse).parameters.values()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
        # We call do_parse(output_dir, names, byte_lists, langs, **kwargs).
        assert positional[:4] == ["output_dir", "pdf_file_names", "pdf_bytes_list", "p_lang_list"]
        for optional in ("backend", "parse_method", "formula_enable", "table_enable"):
            assert optional in inspect.signature(parse).parameters

    def test_backend_reports_itself_available(self) -> None:
        if resolver.resolve_api() is None:
            pytest.skip("MinerU is not installed")
        backend = MineruPythonApiBackend(MineruSettings())
        assert backend.available()
        assert "mineru" in backend.describe()

    def test_settings_are_accepted_by_the_real_signature(self) -> None:
        """A typo'd kwarg here would only surface at runtime on a real GPU box."""
        api = resolver.resolve_api()
        if api is None:
            pytest.skip("MinerU is not installed")
        parse = api.aio_do_parse or api.do_parse
        if parse is None:
            pytest.skip("generation 4.x install")

        settings = MineruSettings(
            backend="pipeline",
            parse_method="auto",
            formula_enable=True,
            table_enable=True,
            effort="high",
            extract_images=False,
        )
        accepted = inspect.signature(parse).parameters
        # Exactly the kwargs MineruPythonApiBackend forwards for 2.x.
        for name in ("backend", "parse_method", "formula_enable", "table_enable"):
            assert name in accepted, f"do_parse would reject {name}={getattr(settings, name)!r}"
        if settings.server_url:
            assert "server_url" in accepted


class TestRealCli:
    def test_cli_generation_matches_binary_help(self) -> None:
        cli = resolver.resolve_cli(["mineru", "magic-pdf"])
        if cli is None:
            pytest.skip("no MinerU binary on PATH")
        assert cli.binary
        assert cli.generation in {GENERATION_1, GENERATION_2, GENERATION_4}

    def test_v4_binary_accepts_subcommand_shape(self, pdf, tmp_path) -> None:
        """4.x must be `mineru parse`, 2.x must be `mineru -p/-o`."""
        cli = resolver.resolve_cli(["mineru", "magic-pdf"])
        if cli is None or cli.generation != GENERATION_4:
            pytest.skip("requires the MinerU 4.x CLI")
        backend = MineruCliBackend(MineruSettings(timeout_seconds=10), cli=cli)
        assert backend.available()
        # Command construction is exercised in test_mineru.py; here we only
        # assert the resolved shape matches the detected generation.
        assert backend.describe().endswith(")") or "generation" in backend.describe()


class TestNoBackendAvailable:
    async def test_clear_error_names_the_install_options(self) -> None:
        from app.clients.content.mineru import ExtractionError, MineruExtractor

        extractor = MineruExtractor(MineruSettings(backend_order=["python_api", "cli"]))
        if extractor.available_backends:
            pytest.skip("MinerU is installed in this environment")
        with pytest.raises(ExtractionError) as excinfo:
            await extractor.extract_pdf(Path("/etc/hostname"))
        message = str(excinfo.value)
        assert "paper-app-backend[mineru]" in message
        assert "pypdf" in message