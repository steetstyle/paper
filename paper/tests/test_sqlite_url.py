"""SQLite URLs follow the checkout, not whichever directory launched the process."""

from __future__ import annotations

from paper_app.config import DatabaseSettings, _REPO_ROOT


def test_relative_sqlite_urls_are_anchored_at_the_repo(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    settings = DatabaseSettings(
        url="sqlite+aiosqlite:///./.data/paper.db",
        sync_url="sqlite:///./.data/paper.db",
    )
    expected = (_REPO_ROOT / ".data" / "paper.db").resolve().as_posix()
    assert settings.url == f"sqlite+aiosqlite:///{expected}"
    assert settings.sync_url == f"sqlite:///{expected}"


def test_missing_sync_url_is_derived_from_the_anchored_url(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    settings = DatabaseSettings(url="sqlite+aiosqlite:///./.data/paper.db", sync_url=None)
    expected = (_REPO_ROOT / ".data" / "paper.db").resolve().as_posix()
    assert settings.sync_url == f"sqlite:///{expected}"


def test_memory_absolute_and_postgres_urls_stay_put() -> None:
    memory = DatabaseSettings(url="sqlite+aiosqlite:///:memory:", sync_url="sqlite:///:memory:")
    assert memory.url == "sqlite+aiosqlite:///:memory:"
    assert memory.sync_url == "sqlite:///:memory:"

    absolute = "/tmp/paper-test.db"
    pinned = DatabaseSettings(
        url=f"sqlite+aiosqlite:///{absolute}",
        sync_url=f"sqlite:///{absolute}",
    )
    assert pinned.url == "sqlite+aiosqlite:////tmp/paper-test.db"
    assert pinned.sync_url == "sqlite:////tmp/paper-test.db"

    postgres = "postgresql+asyncpg://paper:paper@localhost:5432/paper"
    remote = DatabaseSettings(
        url=postgres,
        sync_url="postgresql://paper:paper@localhost:5432/paper",
    )
    assert remote.url == postgres
