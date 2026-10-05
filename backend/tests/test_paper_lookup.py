"""Looking a paper up by whatever spelling of its arXiv id you have.

The database stores the *versionless* id; people type every other form. This
was a live bug: ``paper ingest 2408.05245v1`` stored ``2408.05245``, and
``paper show 2408.05245v1`` then reported "not ingested" for the paper ingested
seconds earlier, because the raw argument was compared against the versionless
column.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.db.repositories import PaperRepository
from app.db.session import get_session_factory
from app.domain.models import Author, PaperMetadata

VERSIONS = [
    "1706.03762",
    "1706.03762v1",
    "1706.03762v7",
    "1706.03762V7",
    "arXiv:1706.03762v5",
    "arxiv:1706.03762",
    "https://arxiv.org/abs/1706.03762v5",
    "https://arxiv.org/abs/1706.03762",
    "https://arxiv.org/pdf/1706.03762v5.pdf",
    "  1706.03762v2  ",
]


@pytest.fixture
async def stored(container):  # noqa: ANN201
    factory = get_session_factory(container.settings.database)
    async with factory() as session:
        paper = await PaperRepository(session).upsert(
            PaperMetadata(
                arxiv_id="1706.03762",
                versioned_id="1706.03762v7",
                version=7,
                title="Attention Is All You Need",
                abstract="A summary.",
                authors=(Author(name="Ashish Vaswani"),),
                categories=("cs.CL",),
                primary_category="cs.CL",
                published_at=datetime(2017, 6, 12, tzinfo=UTC),
                updated_at=datetime(2023, 8, 2, tzinfo=UTC),
                abs_url="https://arxiv.org/abs/1706.03762v7",
                pdf_url="https://arxiv.org/pdf/1706.03762v7",
            )
        )
        await session.commit()
        return paper.id


@pytest.fixture
def container(settings, tmp_path):  # noqa: ANN201
    from app.container import Container, set_container

    container = Container(settings)
    container._blobs = None  # type: ignore[assignment]  # noqa: SLF001
    set_container(container)
    yield container
    set_container(None)


class TestVersionedLookup:
    @pytest.mark.parametrize("probe", VERSIONS)
    async def test_every_spelling_resolves(self, container, stored, probe: str) -> None:  # noqa: ANN001
        factory = get_session_factory(container.settings.database)
        async with factory() as session:
            found = await PaperRepository(session).get_by_arxiv_id(probe)
        assert found is not None, probe
        assert found.id == stored

    @pytest.mark.parametrize(
        "probe", ["", "   ", "not an id", "solv-int/9712001", "9999.99999"]
    )
    async def test_unresolvable_input_is_a_miss_not_an_exception(
        self, container, stored, probe: str
    ) -> None:  # noqa: ANN001
        """A lookup miss is not an error; raising here would break every caller."""
        factory = get_session_factory(container.settings.database)
        async with factory() as session:
            assert await PaperRepository(session).get_by_arxiv_id(probe) is None

    async def test_old_style_ids_keep_their_archive_case(
        self, container, stored
    ) -> None:  # noqa: ANN001
        """``math.CO/0309136`` is case-sensitive; ``split('v')[0]`` mangled it.

        Replaced by a proper parse, so ``solv-int/9712001`` stays intact instead
        of becoming ``sol``.
        """
        from app.domain.ids import normalize_arxiv_id

        assert normalize_arxiv_id("solv-int/9712001") == "solv-int/9712001"
        assert normalize_arxiv_id("math.CO/0309136v2") == "math.CO/0309136"

    def test_no_version_stripping_shortcut_remains(self) -> None:
        """Guard against the ``split("v")[0]`` pattern coming back."""
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent / "app"
        offenders = [
            str(path.relative_to(root.parent))
            for path in root.rglob("*.py")
            if 'split("v")' in path.read_text(encoding="utf-8")
        ]
        assert offenders == []
