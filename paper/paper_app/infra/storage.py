"""Content-addressed blob store.

PDFs, raw HTML and MinerU artefacts are written once and referenced by
``sha256`` so re-ingestion never duplicates bytes. Works on any local path;
swap for S3 by implementing :class:`BlobStore`.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from paper_app.config import StorageSettings
from paper_app.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class StoredBlob:
    sha256: str
    uri: str
    size_bytes: int
    path: Path | None


class BlobStore(ABC):
    @abstractmethod
    def put_bytes(self, data: bytes, *, prefix: str = "raw") -> StoredBlob: ...

    @abstractmethod
    def put_file(self, source: Path, *, prefix: str = "raw") -> StoredBlob: ...

    @abstractmethod
    def open(self, uri: str) -> BinaryIO: ...

    @abstractmethod
    def exists(self, sha256: str) -> bool: ...

    @abstractmethod
    def path_for(self, sha256: str) -> Path | None: ...


class LocalBlobStore(BlobStore):
    """``<root>/<prefix>/<aa>/<sha256>`` with a sidecar ``<sha256>.json``."""

    def __init__(self, settings: StorageSettings | None = None, root: Path | None = None) -> None:
        if root is not None:
            self._root = Path(root)
        elif settings is not None:
            self._root = Path(settings.resolved_root)
        else:  # pragma: no cover - only via DI
            from paper_app.config import get_settings

            self._root = Path(get_settings().storage.resolved_root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._settings = settings

    @property
    def root(self) -> Path:
        return self._root

    def _target(self, sha256: str, prefix: str) -> Path:
        return self._root / prefix / sha256[:2] / sha256

    def put_bytes(self, data: bytes, *, prefix: str = "raw") -> StoredBlob:
        digest = hashlib.sha256(data).hexdigest()
        target = self._target(digest, prefix)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            tmp = target.with_suffix(".tmp")
            tmp.write_bytes(data)
            tmp.replace(target)
        self._write_meta(target, {"size_bytes": len(data), "prefix": prefix})
        return StoredBlob(digest, target.as_uri(), len(data), target)

    def put_file(self, source: Path, *, prefix: str = "raw") -> StoredBlob:
        source = Path(source)
        digest = hashlib.sha256()
        size = 0
        # Stream through a temp file so a 40 MB PDF never lands in memory.
        with (
            source.open("rb") as handle,
            tempfile.NamedTemporaryFile(dir=self._root, delete=False, suffix=".upload") as tmp,
        ):
            staging = Path(tmp.name)
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
                size += len(block)
                tmp.write(block)

        sha256 = digest.hexdigest()
        target = self._target(sha256, prefix)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            staging.unlink(missing_ok=True)
        else:
            staging.replace(target)
        self._write_meta(target, {"size_bytes": size, "prefix": prefix})
        logger.debug("blob_stored", extra={"sha256": sha256, "size": size, "prefix": prefix})
        return StoredBlob(sha256, target.as_uri(), size, target)

    def open(self, uri: str) -> BinaryIO:
        return Path(self._local_path(uri)).open("rb")

    #: Every prefix `put_*` can write to. A blob written under one of them is
    #: invisible to `exists`/`path_for` if they do not enumerate all of them.
    PREFIXES = ("raw", "derived", "assets")

    def exists(self, sha256: str) -> bool:
        return any(self._target(sha256, prefix).exists() for prefix in self.PREFIXES)

    def path_for(self, sha256: str) -> Path | None:
        for prefix in self.PREFIXES:
            candidate = self._target(sha256, prefix)
            if candidate.exists():
                return candidate
        return None

    def read_meta(self, sha256: str) -> dict:
        for prefix in ("raw", "derived"):
            meta_path = self._target(sha256, prefix).with_name(f"{sha256}.json")
            if meta_path.exists():
                return json.loads(meta_path.read_text())
        return {}

    def _local_path(self, uri: str) -> str:
        if uri.startswith("file://"):
            return uri[len("file://") :]
        return uri

    def _write_meta(self, target: Path, payload: dict) -> None:
        meta_path = target.with_name(f"{target.name}.json")
        if not meta_path.exists():
            meta_path.write_text(json.dumps(payload, indent=2))

    def clear(self) -> None:  # pragma: no cover - test/dev helper
        shutil.rmtree(self._root, ignore_errors=True)
        self._root.mkdir(parents=True, exist_ok=True)


class MemoryBlobStore(BlobStore):  # pragma: no cover - test helper
    def __init__(self) -> None:
        self._blobs: dict[str, bytes] = {}

    def put_bytes(self, data: bytes, *, prefix: str = "raw") -> StoredBlob:
        digest = hashlib.sha256(data).hexdigest()
        self._blobs[digest] = data
        return StoredBlob(digest, f"memory://{digest}", len(data), None)

    def put_file(self, source: Path, *, prefix: str = "raw") -> StoredBlob:
        return self.put_bytes(Path(source).read_bytes(), prefix=prefix)

    def open(self, uri: str) -> BinaryIO:
        import io

        return io.BytesIO(self._blobs[uri.split("://", 1)[-1]])

    def exists(self, sha256: str) -> bool:
        return sha256 in self._blobs

    def path_for(self, sha256: str) -> Path | None:
        return None


def build_blob_store(settings: StorageSettings | None = None) -> BlobStore:
    return LocalBlobStore(settings)