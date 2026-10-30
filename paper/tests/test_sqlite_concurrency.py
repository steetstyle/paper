"""File SQLite waits for a busy writer instead of failing the second ingest."""

from __future__ import annotations

from sqlalchemy import text

from paper_app.config import DatabaseSettings
from paper_app.db.session import _SQLITE_BUSY_TIMEOUT_MS, _engines, get_engine


async def test_file_sqlite_uses_wal_and_a_busy_timeout(tmp_path) -> None:
    url = f"sqlite+aiosqlite:///{tmp_path / 'lock.db'}"
    settings = DatabaseSettings(url=url, sync_url=f"sqlite:///{tmp_path / 'lock.db'}")
    engine = get_engine(settings)
    try:
        async with engine.connect() as connection:
            mode = (await connection.execute(text("PRAGMA journal_mode"))).scalar()
            timeout = (await connection.execute(text("PRAGMA busy_timeout"))).scalar()
        assert mode == "wal"
        assert timeout == _SQLITE_BUSY_TIMEOUT_MS
    finally:
        await engine.dispose()
        _engines.pop(settings.url, None)
