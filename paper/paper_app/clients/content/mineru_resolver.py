"""MinerU resolution: detect which public surface is installed.

MinerU renamed and restructured itself several times:

===========  ==========================  ================================
Generation   Python package / CLI        Python entry points
===========  ==========================  ================================
``1.x``      ``magic-pdf`` / ``magic-pdf``  ``mineru.cli.common.do_parse``
``2.x``      ``mineru`` / ``mineru``        ``read_fn`` + ``do_parse``
``4.x``      ``mineru`` / ``mineru parse``  ``doc_analyze`` + ``render_markdown``
===========  ==========================  ================================

Rather than pinning one, we probe once and adapt. The result is cached because
importing MinerU is expensive.
"""

from __future__ import annotations

import importlib
import importlib.metadata as importlib_metadata
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from paper_app.logging import get_logger

logger = get_logger(__name__)

GENERATION_1 = 1  # magic-pdf
GENERATION_2 = 2  # mineru 2.x
GENERATION_4 = 4  # mineru 4.x


@dataclass(frozen=True, slots=True)
class MineruApi:
    """Resolved callables for whichever generation is installed."""

    generation: int
    version: str | None = None

    # 4.x
    doc_analyze: Callable[..., Any] | None = None
    aio_doc_analyze: Callable[..., Any] | None = None
    render_markdown: Callable[..., str] | None = None
    render_content_list: Callable[..., list[dict]] | None = None

    # 2.x / 1.x
    read_fn: Callable[..., bytes] | None = None
    do_parse: Callable[..., Any] | None = None
    aio_do_parse: Callable[..., Any] | None = None

    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def available(self) -> bool:
        if self.generation == GENERATION_4:
            return bool(self.doc_analyze or self.aio_doc_analyze)
        return bool(self.do_parse or self.aio_do_parse)

    @property
    def label(self) -> str:
        return f"mineru {self.version or '?'} (generation {self.generation})"

    @property
    def prefers_async(self) -> bool:
        """4.x and 2.x ship a native coroutine; 1.x does not."""
        if self.generation == GENERATION_4:
            return self.aio_doc_analyze is not None
        return self.aio_do_parse is not None


def installed_version() -> str | None:
    for package in ("mineru", "magic-pdf"):
        try:
            return importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            continue
    return None


def _major(version: str | None) -> int:
    if not version:
        return 0
    head = version.split(".", 1)[0]
    digits = "".join(ch for ch in head if ch.isdigit())
    return int(digits) if digits else 0


def _try_import(module_path: str, *, candidate: bool = False):  # noqa: ANN202
    """Import ``module_path``; ``candidate`` marks a path we expect to exist.

    Failure reasons are logged at debug level so an operator can tell
    "MinerU is not installed" apart from "MinerU is installed but a transitive
    dependency is missing".
    """
    try:
        return importlib.import_module(module_path)
    except ImportError as exc:
        if candidate:
            logger.debug(
                "mineru_import_unavailable",
                extra={"module_path": module_path, "reason": str(exc)},
            )
        return None
    except Exception as exc:  # noqa: BLE001 - a broken install must not crash us
        logger.warning("mineru_import_failed", extra={"module_path": module_path, "error": str(exc)})
        return None


@lru_cache(maxsize=1)
def resolve_api() -> MineruApi | None:
    """Probe the installed MinerU and return its callable surface, or ``None``."""
    try:
        return _probe_api()
    except Exception as exc:  # noqa: BLE001 - a broken install must not crash startup
        logger.error("mineru_probe_failed", extra={"error": str(exc)})
        return None


def _probe_api() -> MineruApi | None:
    version = installed_version()
    major = _major(version)

    # 4.x: the render layer and the analyze entry point live in fixed modules.
    analyze = _try_import("mineru.backend.analyze", candidate=True)
    if analyze is not None and hasattr(analyze, "doc_analyze"):
        markdown_mod = _try_import("mineru.render.markdown", candidate=True)
        content_mod = _try_import("mineru.render.content_list", candidate=True)
        if markdown_mod is None:
            return None
        return MineruApi(
            generation=GENERATION_4,
            version=version,
            doc_analyze=getattr(analyze, "doc_analyze", None),
            aio_doc_analyze=getattr(analyze, "aio_doc_analyze", None),
            render_markdown=getattr(markdown_mod, "render_markdown", None),
            render_content_list=(
                getattr(content_mod, "render_content_list", None) if content_mod else None
            ),
        )

    # 2.x: read_fn + do_parse in mineru.cli.common
    common = _try_import("mineru.cli.common", candidate=True)
    if common is not None and hasattr(common, "do_parse"):
        return MineruApi(
            generation=GENERATION_2,
            version=version,
            read_fn=getattr(common, "read_fn", None),
            do_parse=getattr(common, "do_parse", None),
            aio_do_parse=getattr(common, "aio_do_parse", None),
        )

    # 1.x: the package was called magic-pdf
    legacy = _try_import("magic_pdf", candidate=True)
    if legacy is not None:
        cli_common = _try_import("magic_pdf.cli.common", candidate=True) or _try_import(
            "magic_pdf.common", candidate=True
        )
        if cli_common is not None and hasattr(cli_common, "do_parse"):
            return MineruApi(
                generation=GENERATION_1,
                version=version or _magic_pdf_version(),
                read_fn=getattr(cli_common, "read_fn", None),
                do_parse=getattr(cli_common, "do_parse", None),
            )

    if major >= 4:
        logger.warning("mineru_unrecognised_generation", extra={"version": version})
    return None


def _magic_pdf_version() -> str | None:
    try:
        return importlib_metadata.version("magic-pdf")
    except importlib_metadata.PackageNotFoundError:
        return None


def reset_cache() -> None:
    """Clear the probe caches (tests, and after installing MinerU at runtime)."""
    resolve_api.cache_clear()
    _cli_version.cache_clear()
    _cli_generation.cache_clear()


# --------------------------------------------------------------------------- CLI
@dataclass(frozen=True, slots=True)
class MineruCli:
    """A resolved MinerU CLI binary plus the command shape it expects."""

    binary: str
    generation: int
    version: str | None = None

    @property
    def label(self) -> str:
        return f"{self.binary} {self.version or '?'} (generation {self.generation})"


def find_cli_binary(candidates: list[str]) -> str | None:
    """Locate a MinerU binary.

    ``shutil.which`` is tried first, then this interpreter's ``bin`` directory:
    when the app runs from a virtualenv whose ``bin`` is not on ``PATH`` (for
    example ``.venv/bin/python -m app``), the matching ``mineru`` lives right
    next to the interpreter — and is exactly the one whose Python API we bound.
    """
    for candidate in candidates:
        resolved = shutil.which(candidate)
        if resolved:
            return resolved

    for directory in (Path(sys.executable).parent, Path(sys.prefix) / "bin"):
        if not directory.is_dir():
            continue
        for candidate in candidates:
            for path in (directory / candidate, directory / f"{candidate}.exe"):
                if path.is_file():
                    return str(path)
    return None


@lru_cache(maxsize=8)
def _cli_version(binary: str) -> str | None:
    try:
        result = subprocess.run(  # noqa: S603
            [binary, "--version"],
            capture_output=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    output = f"{result.stdout.decode('utf-8', 'replace')} {result.stderr.decode('utf-8', 'replace')}"
    for token in output.replace(",", " ").split():
        digits = ""
        for char in token:
            if char.isdigit() or (char == "." and digits):
                digits += char
            else:
                break
        if digits.count(".") >= 1:
            return digits.strip(".")
    return None


@lru_cache(maxsize=8)
def _cli_generation(binary: str) -> int:
    """4.x renamed the CLI to subcommands, so `parse` is a reliable probe."""
    try:
        probe = subprocess.run(  # noqa: S603
            [binary, "parse", "--help"],
            capture_output=True,
            timeout=20,
            check=False,
        )
        if probe.returncode == 0 and b"tier" in (probe.stdout + probe.stderr):
            return GENERATION_4
    except (OSError, subprocess.SubprocessError):
        pass
    major = _major(_cli_version(binary))
    if major >= 4:
        return GENERATION_4
    if major == 2:
        return GENERATION_2
    if major == 1:
        return GENERATION_1
    # Unknown version: the `-p/-o` shape has been stable since 1.x.
    return GENERATION_2


def resolve_cli(candidates: list[str]) -> MineruCli | None:
    binary = find_cli_binary(candidates)
    if binary is None:
        return None
    return MineruCli(
        binary=binary, generation=_cli_generation(binary), version=_cli_version(binary)
    )