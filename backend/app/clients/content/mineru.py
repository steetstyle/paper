"""MinerU (``magic-pdf``) integration for PDF -> Markdown/Text.

Backends are tried in the configured order and all return the same
:class:`~app.domain.models.ExtractedDocument`, so swapping one changes nothing
upstream:

``python_api``
    Calls MinerU in-process. Handles both the 4.x ``doc_analyze`` API and the
    2.x ``do_parse`` API (see :mod:`app.clients.content.mineru_resolver`).
``cli``
    Shells out to ``mineru`` / ``magic-pdf``, auto-detecting the command shape.
``pypdf``
    Last-resort text extraction so ingestion degrades instead of failing.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from app.clients.content.mineru_resolver import (
    GENERATION_1,
    GENERATION_2,
    GENERATION_4,
    MineruApi,
    resolve_api,
    resolve_cli,
)
from app.config import MineruSettings
from app.domain.enums import ContentSource, MineruBackend
from app.domain.models import ExtractedDocument
from app.infra.text import (
    approx_tokens,
    collapse_whitespace,
    iter_paragraphs,
    strip_markdown,
)
from app.logging import get_logger

logger = get_logger(__name__)


class ExtractionError(RuntimeError):
    """All configured backends failed to produce text."""


class MineruBackendUnavailable(RuntimeError):
    """A backend cannot run in the current environment."""


class ExtractionBackend(ABC):
    """A single PDF -> Markdown strategy.

    Every backend takes the same settings object so
    :class:`MineruExtractor` can build them uniformly and report which ones are
    actually usable in the current environment.
    """

    name: MineruBackend

    def __init__(self, settings: MineruSettings) -> None:
        self.settings = settings

    @abstractmethod
    def available(self) -> bool:
        """Whether this backend can run here."""

    @abstractmethod
    async def extract(self, pdf_path: Path, workdir: Path) -> ExtractedDocument:
        """Convert one PDF into markdown + text."""

    def describe(self) -> str:
        return self.name.value

    def cleanup(self, workdir: Path) -> None:
        if not workdir.exists():
            return
        shutil.rmtree(workdir, ignore_errors=True)


# --------------------------------------------------------------------------- API
class MineruPythonApiBackend(ExtractionBackend):
    """In-process MinerU. Version-adaptive via :class:`MineruApi`.

    4.x  ``doc_analyze(file_bytes)`` -> ``render_markdown(middle_json)``
    2.x  ``read_fn(path)`` + ``do_parse(out, names, byte_lists, langs)``
    1.x  ``do_parse(output_dir=..., pdf_bytes=..., parse_method=...)``
    """

    name = MineruBackend.PYTHON_API

    def __init__(self, settings: MineruSettings, api: MineruApi | None = None) -> None:
        super().__init__(settings)
        self._api = api

    @property
    def api(self) -> MineruApi | None:
        # Resolved lazily so importing MinerU never happens at app startup.
        return self._api if self._api is not None else resolve_api()

    def available(self) -> bool:
        api = self.api
        return bool(api and api.available)

    def describe(self) -> str:
        api = self.api
        return f"python_api ({api.label})" if api else "python_api (not installed)"

    async def extract(self, pdf_path: Path, workdir: Path) -> ExtractedDocument:
        api = self.api
        if api is None or not api.available:
            raise MineruBackendUnavailable("MinerU is not installed")
        if api.generation == GENERATION_4:
            return await self._extract_v4(api, pdf_path)
        return await self._extract_legacy(api, pdf_path, workdir)

    # -- 4.x ------------------------------------------------------------------
    async def _extract_v4(self, api: MineruApi, pdf_path: Path) -> ExtractedDocument:
        file_bytes = await asyncio.to_thread(pdf_path.read_bytes)
        settings = self.settings

        kwargs: dict[str, Any] = {
            "effort": settings.effort,
            "parse_mode": settings.parse_method,
            "image_analysis": settings.extract_images,
            "file_suffix": "pdf",
        }
        if api.aio_doc_analyze is not None:
            middle_json, model_json = await api.aio_doc_analyze(file_bytes, **kwargs)
        elif api.doc_analyze is not None:
            middle_json, model_json = await _with_timeout(
                asyncio.to_thread(api.doc_analyze, file_bytes, **kwargs),
                settings.timeout_seconds,
            )
        else:  # pragma: no cover - guarded by available()
            raise MineruBackendUnavailable("mineru 4.x exposes neither doc_analyze variant")

        if api.render_markdown is None:
            raise MineruBackendUnavailable("mineru.render.markdown.render_markdown missing")
        markdown = api.render_markdown(middle_json)
        blocks = []
        if api.render_content_list is not None:
            try:
                blocks = list(api.render_content_list(middle_json) or [])
            except Exception as exc:  # noqa: BLE001 - blocks are an optional extra
                logger.warning("content_list_render_failed", extra={"error": str(exc)})

        document = _build_document(
            markdown=markdown,
            blocks=blocks,
            source=ContentSource.PDF_MINERU,
            backend="mineru:python_api:v4",
            meta={
                "generation": GENERATION_4,
                "effort": settings.effort,
                "parse_mode": settings.parse_method,
                "pdf_pages": _count_pages(model_json),
            },
        )
        logger.info(
            "mineru_api_done",
            extra={"generation": 4, "blocks": len(blocks), "chars": document.char_count},
        )
        return document

    # -- 2.x / 1.x ------------------------------------------------------------
    async def _extract_legacy(
        self, api: MineruApi, pdf_path: Path, workdir: Path
    ) -> ExtractedDocument:
        output_dir = workdir / "mineru"
        output_dir.mkdir(parents=True, exist_ok=True)
        settings = self.settings

        if api.generation == GENERATION_1:
            return await self._extract_v1(api, pdf_path, output_dir)

        if api.read_fn is None or (api.do_parse is None and api.aio_do_parse is None):
            raise MineruBackendUnavailable("mineru 2.x requires read_fn + do_parse")

        read_fn = api.read_fn
        try:
            file_bytes = await asyncio.to_thread(read_fn, pdf_path)
        except Exception:  # noqa: BLE001 - read_fn takes bytes on some builds
            file_bytes = await asyncio.to_thread(pdf_path.read_bytes)

        names = [pdf_path.stem]
        byte_lists = [file_bytes]
        langs = [settings.language]

        kwargs: dict[str, Any] = {
            "backend": settings.backend,
            "parse_method": settings.parse_method,
            "formula_enable": settings.formula_enable,
            "table_enable": settings.table_enable,
        }
        if settings.server_url:
            kwargs["server_url"] = settings.server_url

        if api.aio_do_parse is not None:
            await _with_timeout(
                api.aio_do_parse(str(output_dir), names, byte_lists, langs, **kwargs),
                settings.timeout_seconds,
            )
        elif api.do_parse is not None:
            await _with_timeout(
                asyncio.to_thread(
                    api.do_parse, str(output_dir), names, byte_lists, langs, **kwargs
                ),
                settings.timeout_seconds,
            )
        else:  # pragma: no cover - guarded by available()
            raise MineruBackendUnavailable("mineru 2.x exposes neither do_parse variant")

        markdown_path = _find_markdown(output_dir)
        if markdown_path is None:
            raise ExtractionError(f"MinerU 2.x produced no markdown in {output_dir}")
        blocks = _find_blocks(output_dir)
        return _build_document(
            markdown=markdown_path.read_text(errors="replace"),
            blocks=blocks,
            source=ContentSource.PDF_MINERU,
            backend="mineru:python_api:v2",
            meta={
                "generation": GENERATION_2,
                "backend": settings.backend,
                # Surfaced so the asset-extraction step can read
                # content_list.json and the cropped images in this run.
                "mineru_output_dir": str(output_dir),
            },
        )

    async def _extract_v1(
        self, api: MineruApi, pdf_path: Path, output_dir: Path
    ) -> ExtractedDocument:
        """magic-pdf 1.x: keyword-only ``do_parse`` with ``parse_method``."""
        assert api.do_parse is not None  # guarded by available()
        kwargs: dict[str, Any] = {
            "output_dir": str(output_dir),
            "pdf_file_name": pdf_path.stem,
            "pdf_bytes": await asyncio.to_thread(pdf_path.read_bytes),
            "parse_method": self.settings.parse_method,
            "return_images": self.settings.extract_images,
        }
        try:
            result = await _with_timeout(
                asyncio.to_thread(api.do_parse, **kwargs), self.settings.timeout_seconds
            )
        except TypeError:
            # Some 1.x builds take positional args.
            result = await _with_timeout(
                asyncio.to_thread(
                    api.do_parse,
                    str(output_dir),
                    pdf_path.stem,
                    await asyncio.to_thread(pdf_path.read_bytes),
                    [],
                    self.settings.parse_method,
                ),
                self.settings.timeout_seconds,
            )
        blocks = _normalise_result(result)
        markdown = "\n\n".join(p for p in (_render_block(b) for b in blocks) if p)
        markdown_path = _find_markdown(output_dir)
        if markdown_path is not None:
            markdown = markdown_path.read_text(errors="replace")
        return _build_document(
            markdown=markdown,
            blocks=blocks,
            source=ContentSource.PDF_MINERU,
            backend="mineru:python_api:v1",
            meta={"generation": GENERATION_1, "mineru_output_dir": str(output_dir)},
        )


async def _with_timeout(awaitable: Any, seconds: float) -> Any:
    return await asyncio.wait_for(awaitable, timeout=seconds)


# --------------------------------------------------------------------------- CLI
class MineruCliBackend(ExtractionBackend):
    """Subprocess backend — the reliable choice inside Docker images.

    Supports both command shapes::

        mineru parse <pdf> -o <out.md> --tier standard   # 4.x
        mineru -p <pdf> -o <outdir> -b pipeline -l en   # 2.x / 1.x
    """

    name = MineruBackend.CLI

    def __init__(self, settings: MineruSettings, cli: Any | None = None) -> None:
        super().__init__(settings)
        self._cli = cli

    @property
    def cli(self) -> Any:
        return self._cli if self._cli is not None else resolve_cli(self.settings.cli_candidates)

    def available(self) -> bool:
        return self.cli is not None

    def describe(self) -> str:
        cli = self.cli
        return f"cli ({cli.label})" if cli else "cli (not on PATH)"

    async def extract(self, pdf_path: Path, workdir: Path) -> ExtractedDocument:
        cli = self.cli
        if cli is None:
            raise MineruBackendUnavailable("no `mineru` or `magic-pdf` binary on PATH")

        output_dir = workdir / "mineru_cli"
        output_dir.mkdir(parents=True, exist_ok=True)

        if cli.generation == GENERATION_4:
            return await self._run_v4(cli, pdf_path, output_dir)
        return await self._run_legacy(cli, pdf_path, output_dir)

    async def _run_v4(self, cli: Any, pdf_path: Path, output_dir: Path) -> ExtractedDocument:
        settings = self.settings
        target = output_dir / f"{pdf_path.stem}.md"

        command = [cli.binary, "parse", str(pdf_path), "-o", str(target), "--format", "markdown"]
        if settings.tier:
            command += ["--tier", settings.tier]
        if settings.parse_method != "auto":
            command += ["--pages", "all"]
        command += ["--wait", str(int(settings.timeout_seconds)), "--force"]
        command += settings.extra_args

        await self._run(command)
        if not target.exists():
            raise ExtractionError(f"MinerU 4.x wrote no markdown to {target}")
        return _build_document(
            markdown=target.read_text(errors="replace"),
            blocks=[],
            source=ContentSource.PDF_MINERU,
            backend="mineru:cli:v4",
            meta={
                "generation": GENERATION_4,
                "tier": settings.tier or "server-default",
                "mineru_output_dir": str(output_dir),
            },
        )

    async def _run_legacy(self, cli: Any, pdf_path: Path, output_dir: Path) -> ExtractedDocument:
        settings = self.settings
        # Long flags only beyond -p/-o: in MinerU 2.x `-s` is `--start` (a page
        # index), NOT `--source`, so short flags are actively misleading here.
        # `-p`/`-o` are the only short forms stable across 1.x and 2.x.
        full = [
            cli.binary,
            "-p", str(pdf_path),
            "-o", str(output_dir),
            "--backend", settings.backend,
            "--lang", settings.language,
            "--method", settings.parse_method,
            "--source", settings.model_source,
        ]
        if settings.device:
            full += ["--device", settings.device]
        if settings.vram:
            full += ["--vram", str(settings.vram)]
        if settings.server_url:
            full += ["--url", settings.server_url]
        if settings.extra_args:
            full += settings.extra_args

        # Progressively drop flags for builds that reject them.
        commands: list[list[str]] = [
            full,
            [cli.binary, "-p", str(pdf_path), "-o", str(output_dir), "--backend", settings.backend],
            [cli.binary, "-p", str(pdf_path), "-o", str(output_dir)],
        ]

        errors: list[str] = []
        for command in commands:
            try:
                await self._run(command)
            except ExtractionError as exc:
                errors.append(str(exc))
                continue
            markdown = _find_markdown(output_dir)
            if markdown is not None:
                return _build_document(
                    markdown=markdown.read_text(errors="replace"),
                    blocks=_find_blocks(output_dir),
                    source=ContentSource.PDF_MINERU,
                    backend=f"mineru:cli:v{cli.generation}",
                    meta={
                        "generation": cli.generation,
                        "backend": settings.backend,
                        "mineru_output_dir": str(output_dir),
                    },
                )
            errors.append(f"{' '.join(command)} produced no markdown")

        raise ExtractionError("; ".join(errors))

    async def _run(self, command: list[str]) -> None:
        env = self._env()
        logger.debug("mineru_cli_start", extra={"command": " ".join(command)})
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=self.settings.timeout_seconds
            )
        except TimeoutError:
            process.kill()
            await process.wait()
            raise ExtractionError(
                f"MinerU CLI timed out after {self.settings.timeout_seconds}s: "
                f"{' '.join(command[:3])}"
            ) from None

        if process.returncode != 0:
            tail = stderr.decode("utf-8", "replace")[-2000:] or stdout.decode(
                "utf-8", "replace"
            )[-2000:]
            raise ExtractionError(f"exit {process.returncode}: {tail}")

        logger.info("mineru_cli_ok", extra={"command": " ".join(command[:3])})

    def _env(self) -> dict[str, str]:
        """Honour the model source / device MinerU reads from the environment."""
        env = dict(os.environ)
        if self.settings.model_source:
            env["MINERU_MODEL_SOURCE"] = self.settings.model_source
        if self.settings.device:
            env["MINERU_DEVICE"] = self.settings.device
        if self.settings.vram:
            env.setdefault("MINERU_VRAM", str(self.settings.vram))
        return env


# ------------------------------------------------------------------------ pypdf
class PyPdfBackend(ExtractionBackend):
    """Fallback so ingestion never hard-fails on a missing MinerU install."""

    name = MineruBackend.PYPDF

    def __init__(self, settings: MineruSettings) -> None:
        super().__init__(settings)

    def available(self) -> bool:
        try:
            import pypdf  # noqa: F401, PLC0415

            return True
        except ImportError:
            return False

    def describe(self) -> str:
        return "pypdf" if self.available() else "pypdf (not installed)"

    async def extract(self, pdf_path: Path, workdir: Path) -> ExtractedDocument:
        return await asyncio.to_thread(self._extract_sync, pdf_path)

    def _extract_sync(self, pdf_path: Path) -> ExtractedDocument:
        try:
            from pypdf import PdfReader  # noqa: PLC0415
        except ImportError:  # pragma: no cover
            from PyPDF2 import PdfReader  # type: ignore[no-redef] # noqa: PLC0415

        reader = PdfReader(str(pdf_path))
        pages: list[str] = []
        for index, page in enumerate(reader.pages):
            try:
                text = page.extract_text() or ""
            except Exception as exc:  # noqa: BLE001 - one bad page must not kill the doc
                logger.warning("pypdf_page_failed", extra={"page": index + 1, "error": str(exc)})
                text = ""
            pages.append(f"## Page {index + 1}\n\n{text}")
        return _build_document(
            markdown="\n\n".join(pages),
            blocks=[],
            source=ContentSource.PDF_PYPDF,
            backend="pypdf",
            meta={"pages": len(reader.pages), "page_count": len(reader.pages)},
        )


# ------------------------------------------------------------------- helpers
def _count_pages(model_json: Any) -> int | None:
    """Page count from MinerU's model_json, tolerating both known shapes.

    4.x keys ``pdf_info`` by page index; some builds use a positional list, and
    others nest pages one level deeper under ``page_idx``/``page_no`` markers.
    """
    if not isinstance(model_json, dict):
        return None

    pdf_info = model_json.get("pdf_info")
    if isinstance(pdf_info, (dict, list)):
        return len(pdf_info)

    for key in ("page_info", "pages"):
        value = model_json.get(key)
        if isinstance(value, list):
            return len(value)

    for value in model_json.values():
        if (
            isinstance(value, list)
            and value
            and isinstance(value[0], dict)
            and ("page_idx" in value[0] or "page_no" in value[0])
        ):
            return len(value)
    return None


def _normalise_result(result: Any) -> list[dict[str, Any]]:
    """1.x returns ``(blocks, ...)`` or a content list depending on version."""
    if result is None:
        return []
    if isinstance(result, dict):
        if "content_list" in result:
            return list(result["content_list"] or [])
        if "blocks" in result:
            return list(result["blocks"] or [])
        return [result]
    if isinstance(result, (list, tuple)):
        for item in result:
            if isinstance(item, list):
                return [b for b in item if isinstance(b, dict)]
            if isinstance(item, dict):
                return [item]
    return []


def _render_block(block: dict[str, Any]) -> str:
    block_type = str(block.get("type", "")).lower()

    # Equations carry LaTeX in a dedicated field and often no `text` at all,
    # so they must be handled before the empty-text guard.
    if block_type == "equation" or (block.get("latex") and not block.get("text")):
        latex = str(block.get("latex") or block.get("text") or "").strip()
        return f"$$\n{latex}\n$$" if latex else ""

    text = block.get("text") or block.get("content") or ""
    if isinstance(text, list):
        text = " ".join(str(item) for item in text)
    text = str(text).strip()

    if block_type in {"image", "image_caption", "figure", "figure_caption"}:
        caption = block.get("caption") or []
        if isinstance(caption, list):
            caption = " ".join(str(item) for item in caption)
        return str(caption).strip()
    if not text:
        return ""
    if block_type == "title":
        return f"# {text}"
    if block_type == "section_header":
        level = int(block.get("level", 2) or 2)
        return f"{'#' * min(6, max(1, level))} {text}"
    if block_type == "table":
        return _render_table(block)
    if block_type in {"list", "list_item"}:
        return f"- {text}"
    return text


def _render_table(block: dict[str, Any]) -> str:
    body = block.get("body") or ""
    if isinstance(body, list):
        body = "\n".join(str(line) for line in body)
    return str(body).strip() or str(block.get("text", "")).strip()


def _find_markdown(output_dir: Path) -> Path | None:
    """Newest ``*.md`` under the output tree, ignoring README-style noise."""
    best: Path | None = None
    for pattern in ("*.md", "*/*.md", "*/*/*.md", "*/*/*/*.md"):
        for candidate in output_dir.glob(pattern):
            name = candidate.name.lower()
            if name.startswith("readme") or candidate.stat().st_size == 0:
                continue
            if best is None or candidate.stat().st_mtime > best.stat().st_mtime:
                best = candidate
    return best


def _find_blocks(output_dir: Path) -> list[dict[str, Any]]:
    for pattern in ("*_content_list.json", "*.json", "*/*.json", "*/*/*.json"):
        for path in sorted(output_dir.glob(pattern)):
            if "content_list" not in path.name:
                continue
            try:
                data = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            if isinstance(data, list):
                return [b for b in data if isinstance(b, dict)]
            if isinstance(data, dict) and isinstance(data.get("content_list"), list):
                return data["content_list"]
    return []


def _build_document(
    *,
    markdown: str | Path,
    blocks: list[dict[str, Any]],
    source: ContentSource,
    backend: str,
    meta: dict[str, Any],
) -> ExtractedDocument:
    raw = markdown if isinstance(markdown, str) else markdown.read_text(errors="replace")
    collapsed = collapse_whitespace(raw)
    text = strip_markdown(collapsed)
    warnings: list[str] = []
    if not text.strip():
        warnings.append("extraction produced no text")
    return ExtractedDocument(
        markdown=collapsed,
        source=source,
        backend=backend,
        text=text,
        blocks=blocks,
        meta={
            **meta,
            "tokens": approx_tokens(text),
            "paragraphs": len(iter_paragraphs(text)),
            "block_count": len(blocks),
        },
        warnings=tuple(warnings),
    )


# ---------------------------------------------------------------- orchestrator
class MineruExtractor:
    """Facade over the configured backend order."""

    def __init__(self, settings: MineruSettings | None = None) -> None:
        self._settings = settings or MineruSettings()
        self._backends = self._build_backends(self._settings)

    def _build_backends(self, settings: MineruSettings) -> dict[MineruBackend, ExtractionBackend]:
        candidates: dict[MineruBackend, ExtractionBackend] = {
            MineruBackend.PYTHON_API: MineruPythonApiBackend(settings),
            MineruBackend.CLI: MineruCliBackend(settings),
            MineruBackend.PYPDF: PyPdfBackend(settings),
        }
        selected: dict[MineruBackend, ExtractionBackend] = {}
        for name in settings.backend_order:
            try:
                key = MineruBackend(name)
            except ValueError:
                logger.warning("unknown_mineru_backend", extra={"backend": name})
                continue
            backend = candidates[key]
            if backend.available():
                selected[key] = backend
            else:
                logger.info("mineru_backend_unavailable", extra={"backend": key.value})
        return selected

    @property
    def available_backends(self) -> list[str]:
        return [backend.name.value for backend in self._backends.values()]

    def describe_backends(self) -> dict[str, str]:
        """Human-readable backend status for `paper doctor`."""
        return {backend.name.value: backend.describe() for backend in self._backends.values()}

    async def extract_pdf(self, pdf_path: Path) -> ExtractedDocument:
        """Extract text from a PDF using the first backend that succeeds."""
        pdf_path = Path(pdf_path)
        # Check the input before the environment: a missing file is a caller bug
        # and must never be masked by "no backend installed".
        if not pdf_path.exists():
            raise ExtractionError(f"pdf not found: {pdf_path}")
        if not self._backends:
            raise ExtractionError(
                "no PDF extraction backend available — install one of: "
                "`mineru` (pip install 'paper-app-backend[mineru]'), the "
                "`mineru` CLI, or `pypdf` ('paper-app-backend[fallback]')"
            )

        errors: list[str] = []
        for backend in self._backends.values():
            workdir = Path(tempfile.mkdtemp(prefix=f"mineru-{backend.name.value}-"))
            try:
                document = await backend.extract(pdf_path, workdir)
                if document.is_usable:
                    logger.info(
                        "pdf_extracted",
                        extra={
                            "backend": backend.name.value,
                            "chars": document.char_count,
                            "blocks": len(document.blocks),
                            "warnings": list(document.warnings),
                        },
                    )
                    if not self._settings.keep_artifacts:
                        backend.cleanup(workdir)
                    return document
                errors.append(f"{backend.name.value}: {', '.join(document.warnings) or 'empty'}")
            except Exception as exc:  # noqa: BLE001 - fall through to the next backend
                logger.warning(
                    "mineru_backend_failed",
                    extra={"backend": backend.name.value, "error": str(exc)},
                )
                errors.append(f"{backend.name.value}: {exc}")
            finally:
                if not self._settings.keep_artifacts:
                    backend.cleanup(workdir)

        raise ExtractionError("all PDF backends failed: " + "; ".join(errors))