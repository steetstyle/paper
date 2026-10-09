"""Authors and categories as relational tables.

These tests exercise the tables through the repository, so they cover the
junction rows, the deduplication of authors across papers, the category
taxonomy, and the queries that replace the old jsonb containment filter.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

import pytest
from sqlalchemy import func, select

from paper_app.db.models import Author, Category, Paper, PaperAuthor, PaperCategory
from paper_app.db.repositories import PaperRepository, normalize_author_name
from paper_app.db.session import get_session_factory
from paper_app.domain.models import Author as AuthorSpec
from paper_app.domain.models import PaperMetadata


def metadata(
    arxiv_id: str = "1706.03762",
    *,
    authors: tuple[str, ...] = ("Ashish Vaswani", "Noam Shazeer"),
    affiliations: tuple[str | None, ...] | None = None,
    categories: tuple[str, ...] = ("cs.CL", "cs.LG"),
    primary: str | None = "cs.CL",
    version: int = 5,
) -> PaperMetadata:
    affs = affiliations if affiliations is not None else (None,) * len(authors)
    return PaperMetadata(
        arxiv_id=arxiv_id,
        versioned_id=f"{arxiv_id}v{version}",
        version=version,
        title="Attention Is All You Need",
        abstract="We propose the Transformer.",
        authors=tuple(
            AuthorSpec(name=name, affiliation=aff)
            for name, aff in zip(authors, affs, strict=False)
        ),
        categories=categories,
        primary_category=primary,
        published_at=datetime(2017, 6, 12, tzinfo=UTC),
        updated_at=datetime(2023, 8, 2, tzinfo=UTC),
        abs_url=f"https://arxiv.org/abs/{arxiv_id}",
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
        html_url=None,
    )


@asynccontextmanager
async def repo_session(settings) -> AsyncIterator[PaperRepository]:  # noqa: ANN001
    """A repository on a real session, committed so reads see the writes.

    Wrapped in a context manager because the tests need to commit between the
    seeding step and the assertions, and a bare object attribute would lose the
    type of ``__aexit__``.
    """
    factory = get_session_factory(settings.database)
    async with factory() as session:
        yield PaperRepository(session)
        await session.commit()


@pytest.fixture
def repo(settings) -> Callable[[], Any]:  # noqa: ANN201
    """A zero-argument factory returning a repository context manager.

    Bound to `settings` here so tests read `async with repo() as r`.
    """
    return lambda: repo_session(settings)
    @pytest.mark.parametrize(
        ("left", "right"),
        [
            ("Ashish Vaswani", "ashish vaswani"),
            ("Ashish  Vaswani", "Ashish Vaswani"),
            ("  Ashish Vaswani  ", "Ashish Vaswani"),
            ("Noam Shazeer", "noam   shazeer"),
        ],
    )
    def test_spellings_that_should_collapse(self, left: str, right: str) -> None:
        assert normalize_author_name(left) == normalize_author_name(right)

    def test_different_people_do_not_collapse(self) -> None:
        assert normalize_author_name("Ashish Vaswani") != normalize_author_name("Ashish Vashni")


class TestAuthorRows:
    async def test_authors_become_rows_in_order(self, repo) -> None:  # noqa: ANN001
        async with repo() as r:
            paper = await r.upsert(metadata())
            assert paper.author_names == ["Ashish Vaswani", "Noam Shazeer"]

    async def test_junction_rows_carry_the_ordinal(self, repo) -> None:  # noqa: ANN001
        async with repo() as r:
            paper = await r.upsert(metadata())
            rows = (
                await r.session.execute(
                    select(PaperAuthor)
                    .where(PaperAuthor.paper_id == paper.id)
                    .order_by(PaperAuthor.ordinal)
                )
            ).scalars().all()
            assert [row.ordinal for row in rows] == [0, 1]
            assert all(row.author_id for row in rows)

    async def test_affiliation_lives_on_the_link(self, repo) -> None:  # noqa: ANN001
        async with repo() as r:
            await r.upsert(metadata(affiliations=("Google Brain", "University of Toronto")))
            stmt = (
                select(PaperAuthor.affiliation, PaperAuthor.ordinal)
                .order_by(PaperAuthor.ordinal)
            )
            rows = (await r.session.execute(stmt)).all()
            assert rows == [("Google Brain", 0), ("University of Toronto", 1)]

    async def test_one_row_per_person_across_papers(self, repo) -> None:  # noqa: ANN001
        """The reason this table exists: an author's papers are reachable by id."""
        async with repo() as r:
            await r.upsert(metadata("1706.03762"))
            await r.upsert(
                metadata("2104.00001", authors=("Ashish Vaswani", "Jane Doe"))
            )
            count = await r.session.scalar(
                select(func.count()).select_from(Author).where(Author.name == "Ashish Vaswani")
            )
            assert count == 1

    async def test_case_and_spacing_variants_share_one_author(self, repo) -> None:  # noqa: ANN001
        async with repo() as r:
            await r.upsert(metadata("1706.03762", authors=("Ashish Vaswani",)))
            await r.upsert(metadata("2104.00001", authors=("ashish  vaswani",)))
            count = await r.session.scalar(
                select(func.count())
                .select_from(Author)
                .where(Author.normalized_name == "ashish vaswani")
            )
            assert count == 1

    async def test_reingest_replaces_rather_than_duplicates(self, repo) -> None:  # noqa: ANN001
        async with repo() as r:
            await r.upsert(metadata())
            await r.upsert(metadata(version=6))
            links = await r.session.scalar(
                select(func.count()).select_from(PaperAuthor)
            )
            assert links == 2

    async def test_changing_the_author_list_replaces_the_links(self, repo) -> None:  # noqa: ANN001
        async with repo() as r:
            await r.upsert(metadata(authors=("A One", "B Two", "C Three")))
            await r.upsert(metadata(version=6, authors=("A One", "D Four")))
            paper = await r.get_by_arxiv_id("1706.03762")
            assert paper is not None
            assert paper.author_names == ["A One", "D Four"]

    async def test_lookup_by_author_name(self, repo) -> None:  # noqa: ANN001
        async with repo() as r:
            await r.upsert(metadata("1706.03762"))
            await r.upsert(metadata("2104.00001"))
            papers = await r.papers_by_author("  ashish   vaswani ")
            assert len(papers) == 2

    async def test_prolific_author_listing(self, repo) -> None:  # noqa: ANN001
        async with repo() as r:
            await r.upsert(metadata("1706.03762", authors=("Ashish Vaswani",)))
            await r.upsert(metadata("2104.00001", authors=("Ashish Vaswani", "Jane Doe")))
            authors = await r.list_authors()
            top = authors[0]
            assert top["name"] == "Ashish Vaswani"
            assert top["paper_count"] == 2

    async def test_authors_helper_returns_the_old_dict_shape(self, repo) -> None:  # noqa: ANN001
        async with repo() as r:
            paper = await r.upsert(metadata(affiliations=("Google Brain", None)))
            assert paper.authors() == [
                {"name": "Ashish Vaswani", "affiliation": "Google Brain"},
                {"name": "Noam Shazeer", "affiliation": None},
            ]


class TestCategoryRows:
    async def test_categories_become_rows_in_order(self, repo) -> None:  # noqa: ANN001
        async with repo() as r:
            paper = await r.upsert(metadata())
            assert paper.categories == ["cs.CL", "cs.LG"]

    async def test_exactly_one_row_is_primary(self, repo) -> None:  # noqa: ANN001
        async with repo() as r:
            paper = await r.upsert(metadata())
            rows = (
                await r.session.execute(
                    select(PaperCategory).where(PaperCategory.paper_id == paper.id)
                )
            ).scalars().all()
            primaries = [row for row in rows if row.is_primary]
            assert len(primaries) == 1
            assert primaries[0].category == "cs.CL"

    async def test_codes_are_registered_once(self, repo) -> None:  # noqa: ANN001
        async with repo() as r:
            await r.upsert(metadata("1706.03762"))
            await r.upsert(metadata("2104.00001"))
            count = await r.session.scalar(
                select(func.count()).select_from(Category).where(Category.code == "cs.CL")
            )
            assert count == 1

    async def test_subcodes_register_their_ancestors(self, repo) -> None:  # noqa: ANN001
        async with repo() as r:
            await r.upsert(metadata(categories=("cs.CL.MS",), primary="cs.CL.MS"))
            rows = (
                await r.session.execute(
                    select(Category.code, Category.parent, Category.depth).order_by(Category.code)
                )
            ).all()
            # cs.CL.MS -> cs.CL -> cs
            assert {row[0] for row in rows} == {"cs", "cs.CL", "cs.CL.MS"}
            assert dict((row[0], row[1]) for row in rows)["cs.CL.MS"] == "cs.CL"
            assert dict((row[0], row[2]) for row in rows)["cs"] == 0

    async def test_category_filter_matches_exactly(self, repo) -> None:  # noqa: ANN001
        """`cs.LG` must not match `cs.LG.MS` — the substring bug jsonb had."""
        async with repo() as r:
            await r.upsert(metadata(categories=("cs.CL", "cs.LG"), primary="cs.CL"))
            await r.upsert(
                metadata("2104.00001", categories=("cs.LG.MS",), primary="cs.LG.MS")
            )
            found, total = await r.list_papers(category="cs.LG")
            assert total == 1
            assert [p.arxiv_id for p in found] == ["1706.03762"]

            deep, deep_total = await r.list_papers(category="cs.LG.MS")
            assert deep_total == 1
            assert [p.arxiv_id for p in deep] == ["2104.00001"]

    async def test_filter_count_is_not_inflated_by_the_join(self, repo) -> None:  # noqa: ANN001
        """A paper with several categories must still be counted once."""
        async with repo() as r:
            await r.upsert(metadata(categories=("cs.CL", "cs.LG", "stat.ML"), primary="cs.CL"))
            _, total = await r.list_papers(category="cs.CL")
            assert total == 1

    async def test_subtree_lookup(self, repo) -> None:  # noqa: ANN001
        async with repo() as r:
            await r.upsert(metadata("1706.03762", categories=("cs.CL",), primary="cs.CL"))
            await r.upsert(
                metadata("2104.00001", categories=("cs.LG.MS",), primary="cs.LG.MS")
            )
            shallow = await r.papers_in_category("cs")
            deep = await r.papers_in_category("cs", include_subcategories=True)
            assert len(shallow) == 0
            assert len(deep) == 2

    async def test_category_listing_carries_counts(self, repo) -> None:  # noqa: ANN001
        async with repo() as r:
            await r.upsert(metadata("1706.03762"))
            await r.upsert(metadata("2104.00001"))
            listing = {row["code"]: row["paper_count"] for row in await r.list_categories()}
            assert listing["cs.CL"] == 2
            assert listing["cs.LG"] == 2

    async def test_counts_track_deletions(self, repo) -> None:  # noqa: ANN001
        async with repo() as r:
            await r.upsert(metadata())
            paper = await r.get_by_arxiv_id("1706.03762")
            await r.session.delete(paper)
            await r.session.commit()
            listing = {row["code"]: row["paper_count"] for row in await r.list_categories()}
            assert listing["cs.CL"] == 0


class TestReingestKeepsTablesConsistent:
    async def test_reingest_does_not_duplicate_either_junction(self, repo) -> None:  # noqa: ANN001
        async with repo() as r:
            for version in (5, 6, 7):
                await r.upsert(metadata(version=version))
            links = await r.session.scalar(select(func.count()).select_from(PaperAuthor))
            cats = await r.session.scalar(select(func.count()).select_from(PaperCategory))
            assert links == 2
            assert cats == 2

    async def test_categories_can_change_between_versions(self, repo) -> None:  # noqa: ANN001
        async with repo() as r:
            await r.upsert(metadata(categories=("cs.CL", "cs.LG"), primary="cs.CL"))
            await r.upsert(
                metadata(version=6, categories=("cs.CL", "cs.AI"), primary="cs.AI")
            )
            paper = await r.get_by_arxiv_id("1706.03762")
            assert paper.categories == ["cs.CL", "cs.AI"]
            links = await r.session.scalar(
                select(func.count()).select_from(PaperCategory).where(PaperCategory.paper_id == paper.id)
            )
            assert links == 2

    async def test_papers_without_authors_are_allowed(self, repo) -> None:  # noqa: ANN001
        async with repo() as r:
            paper = await r.upsert(metadata(authors=()))
            assert paper.author_names == []


class TestPaperRow:
    async def test_primary_category_column_matches_the_flag(self, repo) -> None:  # noqa: ANN001
        async with repo() as r:
            paper = await r.upsert(metadata(primary="cs.LG"))
            assert paper.primary_category == "cs.LG"
            flagged = (
                await r.session.execute(
                    select(PaperCategory.category).where(
                        PaperCategory.paper_id == paper.id,
                        PaperCategory.is_primary.is_(True),
                    )
                )
            ).scalar_one()
            assert flagged == "cs.LG"

    async def test_papers_table_no_longer_stores_json_lists(self, repo) -> None:  # noqa: ANN001
        """Guards against the columns creeping back in."""
        async with repo() as r:
            await r.upsert(metadata())
        column_names = set(Paper.__table__.columns.keys())
        assert "authors_json" not in column_names
        assert "categories" not in column_names

class TestAuthorResolution:
    """`find_author` turns a loose string into one stored row, or nothing."""

    async def _seed(self, repo) -> None:  # noqa: ANN001
        """Seed in a session of its own, committed so the next one sees it."""
        async with repo() as r:
            await r.upsert(
                metadata(authors=("Ashish Vaswani", "Noam Shazeer", "Jane Doe"))
            )
            await r.session.commit()

    @pytest.mark.parametrize(
        "query",
        ["Ashish Vaswani", "ashish vaswani", "  ASHISH   VASWANI ", "Vaswani, Ashish"],
    )
    async def test_equivalent_spellings_resolve(self, repo, query: str) -> None:  # noqa: ANN001
        await self._seed(repo)
        async with repo() as r:
            match = await r.find_author(query)
            assert match is not None
            assert match["name"] == "Ashish Vaswani"

    async def test_surname_resolves_when_unique(self, repo) -> None:  # noqa: ANN001
        await self._seed(repo)
        async with repo() as r:
            match = await r.find_author("vaswani")
            assert match is not None
            assert match["name"] == "Ashish Vaswani"

    async def test_ambiguous_surname_resolves_to_nobody(self, repo) -> None:  # noqa: ANN001
        """Guessing between two people would silently answer wrongly."""
        async with repo() as r:
            await r.upsert(metadata(authors=("John Smith", "Jane Smith")))
            assert await r.find_author("smith") is None

    async def test_unknown_name_resolves_to_none(self, repo) -> None:  # noqa: ANN001
        await self._seed(repo)
        async with repo() as r:
            assert await r.find_author("Nobody Atall") is None

    async def test_blank_resolves_to_none(self, repo) -> None:  # noqa: ANN001
        await self._seed(repo)
        async with repo() as r:
            assert await r.find_author("   ") is None

    async def test_papers_by_author_uses_the_same_resolution(self, repo) -> None:  # noqa: ANN001
        await self._seed(repo)
        async with repo() as r:
            assert len(await r.papers_by_author("vaswani")) == 1
            assert await r.papers_by_author("nobody") == []

    async def test_resolved_row_reports_its_stored_name(self, repo) -> None:  # noqa: ANN001
        await self._seed(repo)
        async with repo() as r:
            match = await r.find_author("VASWANI")
            assert match["name"] == "Ashish Vaswani"
            assert match["paper_count"] == 1
            assert match["id"]
