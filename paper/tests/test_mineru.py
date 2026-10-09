"""MinerU version detection and version-adaptive extraction.

MinerU is not installable in CI (torch + model weights), so every generation is
exercised through an injected fake API surface. What these tests pin down is the
contract with MinerU: which callable we reach for, and with which arguments.
"""

from __future__ import annotations

import types
from pathlib import Path

import pytest

from paper_app.clients.content import mineru_resolver as resolver
from paper_app.clients.content.mineru import (
    ExtractionError,
    MineruCliBackend,
    MineruExtractor,
    MineruPythonApiBackend,
    PyPdfBackend,
    _build_document,
    _count_pages,
)
from paper_app.clients.content.mineru_resolver import (
    GENERATION_1,
    GENERATION_2,
    GENERATION_4,
    MineruApi,
    MineruCli,
)
from paper_app.config import MineruSettings
from paper_app.domain.enums import ContentSource, MineruBackend

SAMPLE_MARKDOWN = """# Attention Is All You Need

## 1 Introduction

Recurrent networks have long been dominant.

## 2 Method

We use multi-head attention. $$\\text{softmax}(QK^T/\\sqrt{d})$$
"""

SAMPLE_BLOCKS = [
    {"type": "title", "text": "Attention Is All You Need"},
    {"type": "section_header", "text": "1 Introduction", "level": 2},
    {"type": "text", "text": "Recurrent networks have long been dominant."},
]


@pytest.fixture
def pdf(tmp_path: Path) -> Path:
    path = tmp_path / "paper.pdf"
    path.write_bytes(b"%PDF-1.7 fake")
    return path


@pytest.fixture(autouse=True)
def _clear_resolver_cache() -> None:
    """Both probes are lru_cached; monkeypatching them would otherwise be a no-op."""
    resolver.reset_cache()
    yield
    resolver.reset_cache()


# --------------------------------------------------------------------- resolvers
class TestGenerationDetection:
    def test_major_parsing(self) -> None:
        assert resolver._major("4.0.10") == 4
        assert resolver._major("2.7.6") == 2
        assert resolver._major("1.3.9") == 1
        assert resolver._major(None) == 0
        assert resolver._major("weird") == 0

    def test_prefers_generation_4_when_render_layer_present(self, monkeypatch) -> None:
        analyze = types.SimpleNamespace(doc_analyze=object(), aio_doc_analyze=object())
        markdown = types.SimpleNamespace(render_markdown=object())
        content = types.SimpleNamespace(render_content_list=object())
        monkeypatch.setattr(resolver, "installed_version", lambda: "4.0.10")
        monkeypatch.setattr(resolver, "_try_import", _fake_importer(
            {"mineru.backend.analyze": analyze, "mineru.render.markdown": markdown,
             "mineru.render.content_list": content}
        ))

        api = resolver.resolve_api()
        assert api is not None
        assert api.generation == GENERATION_4
        assert api.available
        assert api.render_markdown is markdown.render_markdown
        assert "4.0.10" in api.label

    def test_falls_back_to_generation_2(self, monkeypatch) -> None:
        common = types.SimpleNamespace(read_fn=object(), do_parse=object(), aio_do_parse=object())
        monkeypatch.setattr(resolver, "installed_version", lambda: "2.7.6")
        monkeypatch.setattr(resolver, "_try_import", _fake_importer({"mineru.cli.common": common}))

        api = resolver.resolve_api()
        assert api is not None
        assert api.generation == GENERATION_2
        assert api.read_fn is common.read_fn
        assert api.prefers_async is True

    def test_falls_back_to_generation_1_magic_pdf(self, monkeypatch) -> None:
        legacy = types.SimpleNamespace(do_parse=object(), read_fn=object())
        monkeypatch.setattr(resolver, "installed_version", lambda: None)
        monkeypatch.setattr(resolver, "_magic_pdf_version", lambda: "1.3.9")
        monkeypatch.setattr(resolver, "_try_import", _fake_importer(
            {"magic_pdf": legacy, "magic_pdf.cli.common": legacy}
        ))

        api = resolver.resolve_api()
        assert api is not None
        assert api.generation == GENERATION_1
        assert api.prefers_async is False

    def test_returns_none_when_absent(self, monkeypatch) -> None:
        _no_mineru_installed(monkeypatch)
        monkeypatch.setattr(resolver, "_try_import", _fake_importer({}))
        assert resolver.resolve_api() is None

    def test_render_layer_missing_is_not_usable(self, monkeypatch) -> None:
        """4.x analyze without render_markdown must not be selected."""
        analyze = types.SimpleNamespace(doc_analyze=object())
        monkeypatch.setattr(resolver, "installed_version", lambda: "4.0.10")
        monkeypatch.setattr(resolver, "_try_import", _fake_importer({"mineru.backend.analyze": analyze}))
        assert resolver.resolve_api() is None

    def test_falls_through_when_probed_module_raises(self, monkeypatch) -> None:
        """A partially-installed MinerU must not crash the probe."""
        monkeypatch.setattr(resolver, "installed_version", lambda: "4.0.10")

        def _import(module_path: str, *_a: object, **_k: object):  # noqa: ANN202
            if module_path == "mineru.backend.analyze":
                raise RuntimeError("shared object mismatch")
            return None

        monkeypatch.setattr(resolver, "_try_import", _import)
        assert resolver.resolve_api() is None

    def test_broken_install_does_not_raise(self, monkeypatch) -> None:
        def boom(_path: str, *_a: object, **_k: object) -> None:
            raise RuntimeError("shared object mismatch")

        monkeypatch.setattr(resolver, "installed_version", lambda: "4.0.10")
        monkeypatch.setattr(resolver, "_try_import", boom)
        assert resolver.resolve_api() is None

    def test_cli_binary_lookup_prefers_mineru(self, monkeypatch) -> None:
        monkeypatch.setattr(resolver.shutil, "which", lambda name: f"/usr/bin/{name}" if name == "mineru" else None)
        cli = resolver.resolve_cli(["mineru", "magic-pdf"])
        assert cli is not None and cli.binary.endswith("mineru")

    def test_cli_lookup_returns_none_when_absent(self, monkeypatch) -> None:
        _no_mineru_installed(monkeypatch)
        assert resolver.resolve_cli(["mineru"]) is None

    def test_cli_found_next_to_interpreter(self, tmp_path, monkeypatch) -> None:
        """A venv whose bin/ is not on PATH still resolves its own binary."""
        bin_dir = tmp_path / "venv" / "bin"
        bin_dir.mkdir(parents=True)
        binary = bin_dir / "mineru"
        binary.write_text("#!/bin/sh\n")

        _no_mineru_installed(monkeypatch)
        monkeypatch.setattr(resolver.sys, "executable", str(bin_dir / "python"))

        assert resolver.find_cli_binary(["mineru", "magic-pdf"]) == str(binary)

    def test_which_takes_priority_over_interpreter_dir(self, tmp_path, monkeypatch) -> None:
        bin_dir = tmp_path / "venv" / "bin"
        bin_dir.mkdir(parents=True)
        (bin_dir / "mineru").write_text("#!/bin/sh\n")
        monkeypatch.setattr(resolver.sys, "executable", str(bin_dir / "python"))
        monkeypatch.setattr(resolver.shutil, "which", lambda _n: "/usr/local/bin/mineru")

        assert resolver.find_cli_binary(["mineru"]) == "/usr/local/bin/mineru"

    def test_cli_version_extraction(self, monkeypatch) -> None:
        class Result:
            stdout = b"mineru, version 4.0.10\n"
            stderr = b""

        monkeypatch.setattr(resolver.subprocess, "run", lambda *a, **k: Result())
        assert resolver._cli_version("/usr/bin/mineru") == "4.0.10"


def _fake_importer(modules: dict[str, object]):  # noqa: ANN202
    """A ``_try_import`` replacement backed by an explicit module table."""

    def _import(module_path: str, *_args: object, **_kwargs: object):  # noqa: ANN202
        return modules.get(module_path)

    return _import


def _no_mineru_installed(monkeypatch) -> None:
    """Neutralise every lookup path so tests do not depend on the host having
    (or not having) MinerU installed."""
    monkeypatch.setattr(resolver, "installed_version", lambda: None)
    monkeypatch.setattr(resolver.shutil, "which", lambda _name: None)
    monkeypatch.setattr(resolver.sys, "executable", "/nonexistent/python")
    monkeypatch.setattr(resolver.sys, "prefix", "/nonexistent")


# ------------------------------------------------------------------- 4.x python
class TestPythonApiV4:
    @pytest.fixture
    def calls(self) -> list[tuple]:
        return []

    def _api(self, calls: list[tuple], *, with_async: bool = True) -> MineruApi:
        # 4.x keys pdf_info by page index.
        middle = {"pdf_info": {"0": {"page_idx": 0}, "1": {"page_idx": 1}, "2": {"page_idx": 2}}}
        model = {"pdf_info": middle["pdf_info"]}

        def doc_analyze(file_bytes, **kwargs):  # noqa: ANN001, ANN003
            calls.append(("doc_analyze", file_bytes, kwargs))
            return middle, model

        async def aio_doc_analyze(file_bytes, **kwargs):  # noqa: ANN001, ANN003
            calls.append(("aio_doc_analyze", file_bytes, kwargs))
            return middle, model

        return MineruApi(
            generation=GENERATION_4,
            version="4.0.10",
            doc_analyze=doc_analyze,
            aio_doc_analyze=aio_doc_analyze if with_async else None,
            render_markdown=lambda mj: SAMPLE_MARKDOWN,
            render_content_list=lambda mj: SAMPLE_BLOCKS,
        )

    async def test_uses_aio_doc_analyze_when_available(self, pdf, calls) -> None:
        backend = MineruPythonApiBackend(
            MineruSettings(extract_images=False), api=self._api(calls)
        )
        doc = await backend.extract(pdf, Path("/tmp/opencode/work"))

        assert calls[0][0] == "aio_doc_analyze"
        assert calls[0][1] == b"%PDF-1.7 fake"
        assert calls[0][2] == {
            "effort": "high",
            "parse_mode": "auto",
            "image_analysis": False,
            "file_suffix": "pdf",
        }
        assert doc.backend == "mineru:python_api:v4"
        assert doc.is_usable
        assert "multi-head attention" in doc.text
        assert doc.blocks == SAMPLE_BLOCKS
        assert doc.meta["pdf_pages"] == 3
        assert doc.meta["generation"] == 4

    async def test_falls_back_to_sync_doc_analyze(self, pdf, calls) -> None:
        backend = MineruPythonApiBackend(
            MineruSettings(), api=self._api(calls, with_async=False)
        )
        doc = await backend.extract(pdf, Path("/tmp/opencode/work"))
        assert calls[0][0] == "doc_analyze"
        assert doc.is_usable

    async def test_settings_are_forwarded(self, pdf, calls) -> None:
        settings = MineruSettings(effort="xhigh", parse_method="ocr", extract_images=True)
        backend = MineruPythonApiBackend(settings, api=self._api(calls))
        await backend.extract(pdf, Path("/tmp/opencode/work"))
        assert calls[0][2]["effort"] == "xhigh"
        assert calls[0][2]["parse_mode"] == "ocr"
        assert calls[0][2]["image_analysis"] is True

    async def test_content_list_failure_is_not_fatal(self, pdf, calls) -> None:
        api = self._api(calls)
        api_with_broken_blocks = MineruApi(
            generation=GENERATION_4,
            version=api.version,
            doc_analyze=api.doc_analyze,
            aio_doc_analyze=api.aio_doc_analyze,
            render_markdown=api.render_markdown,
            render_content_list=lambda _mj: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        backend = MineruPythonApiBackend(MineruSettings(), api=api_with_broken_blocks)
        doc = await backend.extract(pdf, Path("/tmp/opencode/work"))
        assert doc.is_usable
        assert doc.blocks == []

    def test_unavailable_when_resolver_finds_nothing(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "paper_app.clients.content.mineru.resolve_api", lambda: None
        )
        assert MineruPythonApiBackend(MineruSettings()).available() is False


# ------------------------------------------------------------------- 2.x python
class TestPythonApiV2:
    def _api(self, calls: list, markdown: str = SAMPLE_MARKDOWN) -> MineruApi:
        def read_fn(path):  # noqa: ANN001
            calls.append(("read_fn", str(path)))
            return b"%PDF-1.7 fake"

        async def aio_do_parse(out, names, byte_lists, langs, **kwargs):  # noqa: ANN001
            calls.append(("aio_do_parse", out, names, byte_lists, langs, kwargs))
            target = Path(out)
            (target / f"{names[0]}.md").write_text(markdown)
            (target / f"{names[0]}_content_list.json").write_text(
                __import__("json").dumps(SAMPLE_BLOCKS)
            )

        return MineruApi(
            generation=GENERATION_2,
            version="2.7.6",
            read_fn=read_fn,
            aio_do_parse=aio_do_parse,
        )

    async def test_positional_do_parse_signature(self, pdf, tmp_path) -> None:
        calls: list = []
        backend = MineruPythonApiBackend(MineruSettings(), api=self._api(calls))
        doc = await backend.extract(pdf, tmp_path)

        assert [c[0] for c in calls] == ["read_fn", "aio_do_parse"]
        parse_call = calls[1]
        # do_parse(output_dir, pdf_file_names, pdf_bytes_list, p_lang_list, **kwargs)
        assert parse_call[2] == ["paper"]           # names list
        assert parse_call[3] == [b"%PDF-1.7 fake"]   # bytes list
        assert parse_call[4] == ["en"]              # lang list
        assert parse_call[5]["backend"] == "pipeline"
        assert parse_call[5]["parse_method"] == "auto"
        assert parse_call[5]["formula_enable"] is True
        assert parse_call[5]["table_enable"] is True

        assert doc.backend == "mineru:python_api:v2"
        assert "multi-head attention" in doc.text
        assert doc.blocks == SAMPLE_BLOCKS
        assert doc.meta["generation"] == 2

    async def test_server_url_forwarded(self, pdf, tmp_path) -> None:
        calls: list = []
        settings = MineruSettings(server_url="http://gpu:30000", backend="vlm-auto-engine")
        backend = MineruPythonApiBackend(settings, api=self._api(calls))
        await backend.extract(pdf, tmp_path)
        assert calls[1][5]["server_url"] == "http://gpu:30000"
        assert calls[1][5]["backend"] == "vlm-auto-engine"

    async def test_missing_markdown_raises(self, pdf, tmp_path) -> None:
        api = MineruApi(
            generation=GENERATION_2,
            version="2.7.6",
            read_fn=lambda _p: b"x",
            aio_do_parse=_noop,
        )
        backend = MineruPythonApiBackend(MineruSettings(), api=api)
        with pytest.raises(ExtractionError, match="produced no markdown"):
            await backend.extract(pdf, tmp_path)

    async def test_read_fn_failure_falls_back_to_raw_bytes(self, pdf, tmp_path) -> None:
        def bad_read_fn(_path):  # noqa: ANN001
            raise TypeError("read_fn wants bytes here")

        async def aio_do_parse(out, names, byte_lists, langs, **kwargs):  # noqa: ANN001
            assert byte_lists == [b"%PDF-1.7 fake"]
            (Path(out) / f"{names[0]}.md").write_text(SAMPLE_MARKDOWN)

        api = MineruApi(
            generation=GENERATION_2, version="2.7.6",
            read_fn=bad_read_fn, aio_do_parse=aio_do_parse,
        )
        backend = MineruPythonApiBackend(MineruSettings(), api=api)
        doc = await backend.extract(pdf, tmp_path)
        assert doc.is_usable


async def _noop(*_args, **_kwargs) -> None:
    return None


# --------------------------------------------------------------------------- CLI
class TestCliBackend:
    def _run_spy(self, recorded: list, *, files: dict[str, str] | None = None):  # noqa: ANN202
        async def _run(self, command):  # noqa: ANN001, ANN001
            recorded.append(command)
            output = Path(command[command.index("-o") + 1])
            for name, content in (files or {}).items():
                # 4.x writes a single markdown file; 2.x writes into a directory.
                target = output if output.suffix == ".md" else output / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content)

        return _run

    def test_v4_command_shape(self, pdf, tmp_path, monkeypatch) -> None:
        recorded: list = []
        backend = MineruCliBackend(
            MineruSettings(tier="standard", timeout_seconds=120),
            cli=MineruCli(binary="/usr/bin/mineru", generation=GENERATION_4, version="4.0.10"),
        )
        monkeypatch.setattr(MineruCliBackend, "_run", self._run_spy(recorded, files={"paper.md": SAMPLE_MARKDOWN}))

        doc = asyncio_run(backend.extract(pdf, tmp_path))
        command = recorded[0]
        assert command[0].endswith("mineru")
        assert command[1] == "parse"
        assert "--tier" in command and "standard" in command
        assert "--format" in command and "markdown" in command
        assert "--force" in command
        assert doc.backend == "mineru:cli:v4"
        assert "multi-head attention" in doc.text
        assert doc.meta["tier"] == "standard"

    def test_v2_command_shape(self, pdf, tmp_path, monkeypatch) -> None:
        recorded: list = []
        settings = MineruSettings(
            backend="pipeline", language="en", parse_method="auto",
            model_source="modelscope", device="cpu", vram=8,
        )
        backend = MineruCliBackend(
            settings, cli=MineruCli(binary="/usr/bin/mineru", generation=GENERATION_2, version="2.7.6")
        )
        monkeypatch.setattr(MineruCliBackend, "_run", self._run_spy(recorded, files={"paper.md": SAMPLE_MARKDOWN}))

        doc = asyncio_run(backend.extract(pdf, tmp_path))
        command = recorded[0]
        assert "-p" in command and str(pdf) in command
        assert "-o" in command
        # Long flags only: `-s` is `--start` in MinerU 2.x, not `--source`.
        assert "-s" not in command
        assert command[command.index("--backend") + 1] == "pipeline"
        assert command[command.index("--lang") + 1] == "en"
        assert command[command.index("--method") + 1] == "auto"
        assert command[command.index("--source") + 1] == "modelscope"
        assert command[command.index("--device") + 1] == "cpu"
        assert command[command.index("--vram") + 1] == "8"
        assert doc.backend == "mineru:cli:v2"

    def test_v2_degrades_to_simpler_command(self, pdf, tmp_path, monkeypatch) -> None:
        recorded: list = []

        async def _run(self, command):  # noqa: ANN001, ANN001
            recorded.append(command)
            if "--backend" in command:
                # A 1.x binary rejects the modern flags; the backend must retry.
                raise ExtractionError("no such option: -b")
            target = Path(command[command.index("-o") + 1])
            (target / "paper.md").write_text(SAMPLE_MARKDOWN)

        monkeypatch.setattr(MineruCliBackend, "_run", _run)
        backend = MineruCliBackend(
            MineruSettings(),
            cli=MineruCli(binary="/usr/bin/magic-pdf", generation=GENERATION_1, version="1.3.9"),
        )
        doc = asyncio_run(backend.extract(pdf, tmp_path))
        # Three shapes are tried: modern flags, then reduced, then bare.
        assert len(recorded) == 3
        assert recorded[-1] == ["/usr/bin/magic-pdf", "-p", str(pdf), "-o", recorded[-1][-1]]
        assert doc.is_usable

    def test_env_is_exported(self) -> None:
        backend = MineruCliBackend(
            MineruSettings(model_source="modelscope", device="cuda:0", vram=12)
        )
        env = backend._env()
        assert env["MINERU_MODEL_SOURCE"] == "modelscope"
        assert env["MINERU_DEVICE"] == "cuda:0"
        assert env["MINERU_VRAM"] == "12"

    def test_unavailable_without_binary(self, monkeypatch) -> None:
        _no_mineru_installed(monkeypatch)
        backend = MineruCliBackend(MineruSettings(cli_candidates=["definitely-not-here"]))
        assert backend.available() is False


def asyncio_run(coro):  # noqa: ANN201, ANN202
    import asyncio

    return asyncio.run(coro)


# ------------------------------------------------------------------ orchestrator
class TestExtractorOrchestration:
    def test_respects_backend_order(self, monkeypatch, pdf) -> None:
        monkeypatch.setattr(PyPdfBackend, "available", lambda self: True)
        order: list[str] = []

        class Failing(MineruPythonApiBackend):
            def available(self) -> bool:
                return True

            async def extract(self, pdf_path, workdir, request=None):  # noqa: ANN001, ARG002
                order.append("python_api")
                raise ExtractionError("gpu oom")

        class PypdfOk(PyPdfBackend):
            async def extract(self, pdf_path, workdir, request=None):  # noqa: ANN001, ARG002
                order.append("pypdf")
                return _build_document(
                    markdown=SAMPLE_MARKDOWN, blocks=[],
                    source=ContentSource.PDF_PYPDF, backend="pypdf", meta={},
                )

        settings = MineruSettings(backend_order=["python_api", "pypdf"], keep_artifacts=False)
        extractor = MineruExtractor(settings)
        extractor._backends = {  # noqa: SLF001 - inject fakes
            MineruBackend.PYTHON_API: Failing(settings),
            MineruBackend.PYPDF: PypdfOk(settings),
        }

        doc = asyncio_run(extractor.extract_pdf(pdf))
        assert order == ["python_api", "pypdf"]
        assert doc.backend == "pypdf"

    async def test_all_backends_failing_raises(self, pdf) -> None:
        class AlwaysFails(PyPdfBackend):
            def available(self) -> bool:
                return True

            async def extract(self, pdf_path, workdir, request=None):  # noqa: ANN001, ARG002
                raise ExtractionError("nope")

        settings = MineruSettings(backend_order=["pypdf"])
        extractor = MineruExtractor(settings)
        extractor._backends = {MineruBackend.PYPDF: AlwaysFails(settings)}  # noqa: SLF001
        with pytest.raises(ExtractionError, match="all PDF backends failed"):
            await extractor.extract_pdf(pdf)

    async def test_empty_output_triggers_fallback(self, pdf) -> None:
        class Empty(MineruPythonApiBackend):
            def available(self) -> bool:
                return True

            async def extract(self, pdf_path, workdir, request=None):  # noqa: ANN001, ARG002
                return _build_document(
                    markdown="   ", blocks=[], source=ContentSource.PDF_MINERU,
                    backend="fake", meta={},
                )

        class PypdfOk(PyPdfBackend):
            async def extract(self, pdf_path, workdir, request=None):  # noqa: ANN001, ARG002
                return _build_document(
                    markdown=SAMPLE_MARKDOWN, blocks=[],
                    source=ContentSource.PDF_PYPDF, backend="pypdf", meta={},
                )

        settings = MineruSettings(backend_order=["python_api", "pypdf"])
        extractor = MineruExtractor(settings)
        extractor._backends = {  # noqa: SLF001
            MineruBackend.PYTHON_API: Empty(settings),
            MineruBackend.PYPDF: PypdfOk(settings),
        }
        doc = await extractor.extract_pdf(pdf)
        assert doc.backend == "pypdf"

    async def test_missing_pdf_checked_before_backends(self, tmp_path) -> None:
        extractor = MineruExtractor(MineruSettings(backend_order=[]))
        with pytest.raises(ExtractionError, match="not found"):
            await extractor.extract_pdf(tmp_path / "nope.pdf")

    def test_describe_reports_versions(self) -> None:
        extractor = MineruExtractor(MineruSettings(backend_order=["pypdf"]))
        described = extractor.describe_backends()
        assert "pypdf" in described
        assert described["pypdf"].startswith("pypdf")


class TestCountPages:
    """model_json page counting must survive every shape MinerU has shipped."""

    def test_dict_keyed_pdf_info(self) -> None:
        assert _count_pages({"pdf_info": {str(i): {} for i in range(7)}}) == 7

    def test_list_pdf_info(self) -> None:
        assert _count_pages({"pdf_info": [1, 2, 3]}) == 3

    def test_page_info_key(self) -> None:
        assert _count_pages({"page_info": [{}, {}]}) == 2

    def test_nested_page_markers(self) -> None:
        assert _count_pages({"stuff": [{"page_idx": 0}, {"page_idx": 1}]}) == 2

    def test_unknown_shape(self) -> None:
        assert _count_pages({"unrelated": 1}) is None
        assert _count_pages(None) is None
        assert _count_pages([1, 2]) is None
