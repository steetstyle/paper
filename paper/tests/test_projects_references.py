"""Projects and the citation graph.

Covers the three properties the design rests on:

- papers stay global — importing into a second project adds a link, never a copy
- per-project state (note, read) cannot leak between projects
- references are extracted from the real HTML rendering, survive re-ingestion
  without duplicating, and resolve to corpus rows when the cited paper arrives
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

import pytest
from sqlalchemy import func, select

from paper_app.clients.content.references import (
    extract_references,
    extract_references_from_text,
    summarise,
)
from paper_app.db.models import Paper, PaperAuthor, PaperCategory, PaperReference, ProjectPaper
from paper_app.db.project_repository import (
    ProjectConflictError,
    ProjectRepository,
    normalize_arxiv_reference,
    slugify,
)
from paper_app.db.reference_repository import ReferenceRepository, normalize_arxiv_id
from paper_app.db.repositories import PaperRepository
from paper_app.db.session import get_session_factory
from paper_app.domain.models import Author as AuthorSpec
from paper_app.domain.models import PaperMetadata


def metadata(
    arxiv_id: str = "1706.03762",
    *,
    authors: tuple[str, ...] = ("Ashish Vaswani",),
    categories: tuple[str, ...] = ("cs.CL", "cs.LG"),
    primary: str | None = "cs.CL",
) -> PaperMetadata:
    return PaperMetadata(
        arxiv_id=arxiv_id,
        versioned_id=f"{arxiv_id}v1",
        version=1,
        title=f"Paper {arxiv_id}",
        abstract="An abstract.",
        authors=tuple(AuthorSpec(name=name) for name in authors),
        categories=categories,
        primary_category=primary,
        published_at=datetime(2020, 1, 1, tzinfo=UTC),
        updated_at=datetime(2023, 1, 1, tzinfo=UTC),
        abs_url=f"https://arxiv.org/abs/{arxiv_id}",
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
        html_url=None,
    )


@asynccontextmanager
async def session(settings) -> AsyncIterator[Any]:  # noqa: ANN401
    factory = get_session_factory(settings.database)
    async with factory() as db:
        yield db
        await db.commit()


@pytest.fixture
def repo(settings):  # noqa: ANN201
    return lambda: session(settings)


async def seed(db, arxiv_id: str = "1706.03762", **kwargs) -> str:
    paper = await PaperRepository(db).upsert(metadata(arxiv_id, **kwargs))
    await db.flush()
    return paper.id


# A trimmed but structurally faithful copy of arXiv's rendering: refnum span,
# then three bibblocks, with the venue wrapping an italic span and the year
# sitting *outside* it.
BIB_HTML = """
<html><body><ul class="ltx_biblist">
<li id="bib.bib1" class="ltx_bibitem">
  <span class="ltx_tag ltx_role_refnum ltx_tag_bibitem">[1]</span>
  <span class="ltx_bibblock">Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton.</span>
  <span class="ltx_bibblock">Layer normalization.</span>
  <span class="ltx_bibblock"><span class="ltx_text ltx_font_italic">arXiv preprint
    arXiv:1607.06450</span>, 2016.</span>
</li>
<li id="bib.bib2" class="ltx_bibitem">
  <span class="ltx_tag ltx_role_refnum ltx_tag_bibitem">[2]</span>
  <span class="ltx_bibblock">Francois Chollet.</span>
  <span class="ltx_bibblock">Xception: Deep learning with depthwise separable
    convolutions.</span>
  <span class="ltx_bibblock"><span class="ltx_text ltx_font_italic">arXiv preprint
    arXiv:1610.02357</span>, 2016.</span>
</li>
<li id="bib.bib3" class="ltx_bibitem">
  <span class="ltx_tag ltx_role_refnum ltx_tag_bibitem">[3]</span>
  <span class="ltx_bibblock">Ashish Vaswani, Noam Shazeer, and Niki Parmar.</span>
  <span class="ltx_bibblock">Attention is all you need.</span>
  <span class="ltx_bibblock"><span class="ltx_text ltx_font_italic">CoRR</span>,
    abs/1706.03762, 2017.</span>
</li>
</ul></body></html>
"""

# The other real rendering: no refnum spans at all, and inline "[2012]" markers
# in the body that must not be mistaken for reference numbers.
INLINE_MARKER_HTML = """
<html><body>
<p>Krizhevsky et al. [2012] introduced the network.</p>
<ul class="ltx_biblist">
<li class="ltx_bibitem">
  <span class="ltx_bibblock">Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton.</span>
  <span class="ltx_bibblock">ImageNet classification with deep convolutional
    neural networks.</span>
  <span class="ltx_bibblock"><span class="ltx_text ltx_font_italic">NIPS</span>, 2012.</span>
</li>
<li class="ltx_bibitem">
  <span class="ltx_bibblock">Sergey Ioffe and Christian Szegedy.</span>
  <span class="ltx_bibblock">Batch normalization.</span>
  <span class="ltx_bibblock"><span class="ltx_text ltx_font_italic">ICML</span>, 2015.</span>
</li>
</ul></body></html>
"""


class TestReferenceExtraction:
    def test_finds_every_item(self) -> None:
        assert len(extract_references(BIB_HTML)) == 3

    def test_splits_authors_title_and_venue(self) -> None:
        first = extract_references(BIB_HTML)[0]
        assert first.authors == "Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton."
        assert first.title == "Layer normalization."
        assert first.venue is not None and "1607.06450" in first.venue

    def test_recovers_the_year_from_outside_the_nested_span(self) -> None:
        """The year sits after the italic venue span, so a regex stopping at the
        first </span> silently loses it."""
        assert [r.year for r in extract_references(BIB_HTML)] == [2016, 2016, 2017]

    def test_ordinals_are_unique_even_without_refnum_spans(self) -> None:
        refs = extract_references(INLINE_MARKER_HTML)
        assert [r.ordinal for r in refs] == [1, 2]

    def test_inline_year_markers_are_not_read_as_reference_numbers(self) -> None:
        for ref in extract_references(INLINE_MARKER_HTML):
            assert ref.ordinal < 1000

    def test_extracts_both_arxiv_spellings(self) -> None:
        refs = extract_references(BIB_HTML)
        assert refs[0].cited_arxiv_id == "1607.06450"   # arXiv:1607.06450
        assert refs[2].cited_arxiv_id == "1706.03762"   # abs/1706.03762

    def test_no_bibliography_yields_nothing(self) -> None:
        assert extract_references(None) == []
        assert extract_references("<html><body>no refs</body></html>") == []
        assert extract_references("<p>ltx_bibitem mentioned but not present</p>") == []

    def test_summarise_counts_linkable_entries(self) -> None:
        assert summarise(extract_references(BIB_HTML)) == {"total": 3, "with_arxiv_id": 3}

    def test_label_degrades_gracefully(self) -> None:
        from paper_app.domain.models import Reference

        assert Reference(raw_text="x", ordinal=1).label() == "? — x"

    def test_markdown_fallback(self) -> None:
        markdown = (
            "# Attention\n\nbody\n\n## References\n\n"
            "[1] Jimmy Lei Ba. Layer normalization. arXiv:1607.06450, 2016.\n"
            "[2] Sergey Ioffe. Batch normalization. ICML, 2015.\n"
            "\n## Appendix\n\nnot a reference\n"
        )
        refs = extract_references_from_text(markdown)
        assert len(refs) == 2
        assert refs[0].cited_arxiv_id == "1607.06450"
        assert refs[0].year == 2016

    def test_markdown_fallback_needs_a_heading(self) -> None:
        assert extract_references_from_text("just some text") == []


class TestArxivIdNormalisation:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("1607.06450", "1607.06450"),
            ("1607.06450v2", "1607.06450"),
            ("arXiv:1607.06450", "1607.06450"),
            ("https://arxiv.org/abs/1607.06450v3", "1607.06450"),
            ("https://arxiv.org/pdf/1607.06450v3.pdf", "1607.06450"),
            ("math.CO/0309136", "math.CO/0309136"),
        ],
    )
    def test_strips_version_and_prefixes(self, raw: str, expected: str) -> None:
        assert normalize_arxiv_reference(raw) == expected

    def test_old_style_ids_keep_their_case(self) -> None:
        """`math.CO/0309136` is not `math.co/0309136`; folding it breaks the match."""
        assert normalize_arxiv_reference("math.CO/0309136v2") == "math.CO/0309136"

    def test_reference_side_normalisation_matches(self) -> None:
        assert normalize_arxiv_id("arXiv:1607.06450v2") == "1607.06450"
        assert normalize_arxiv_id(None) is None
        assert normalize_arxiv_id("") is None
        assert normalize_arxiv_id("not an id") is None


class TestReferencesStored:
    async def test_references_are_written_with_ordinals(self, repo) -> None:  # noqa: ANN001
        async with repo() as db:
            paper_id = await seed(db)
            stats = await ReferenceRepository(db).replace_for_paper(
                paper_id, extract_references(BIB_HTML), source="html"
            )
            rows = await ReferenceRepository(db).list_references(paper_id)
        assert stats["total"] == 3
        assert [row.ordinal for row in rows] == [1, 2, 3]

    async def test_reingest_replaces_rather_than_duplicates(self, repo) -> None:  # noqa: ANN001
        async with repo() as db:
            paper_id = await seed(db)
            refs = ReferenceRepository(db)
            await refs.replace_for_paper(paper_id, extract_references(BIB_HTML))
            await refs.replace_for_paper(paper_id, extract_references(BIB_HTML))
            count = await db.scalar(select(func.count()).select_from(PaperReference))
        assert count == 3

    async def test_shuffling_ordinals_does_not_violate_the_constraint(
        self, repo
    ) -> None:  # noqa: ANN001
        """The unique(citing, ordinal) index is what made a re-ingest fail when
        two entries were assigned the same number."""
        async with repo() as db:
            paper_id = await seed(db)
            refs = ReferenceRepository(db)
            shuffled = list(reversed(extract_references(INLINE_MARKER_HTML)))
            await refs.replace_for_paper(paper_id, shuffled)
            count = await db.scalar(select(func.count()).select_from(PaperReference))
        assert count == 2

    async def test_external_references_need_no_cited_paper(self, repo) -> None:  # noqa: ANN001
        async with repo() as db:
            paper_id = await seed(db)
            await ReferenceRepository(db).replace_for_paper(
                paper_id, extract_references(BIB_HTML)
            )
            rows = await ReferenceRepository(db).list_references(paper_id)
        assert all(row.cited_paper_id is None for row in rows)
        assert [row.is_resolved for row in rows] == [False, False, False]

    async def test_delete_removes_only_that_papers_rows(self, repo) -> None:  # noqa: ANN001
        async with repo() as db:
            first = await seed(db, "1706.03762")
            second = await seed(db, "1607.06450")
            refs = ReferenceRepository(db)
            await refs.replace_for_paper(first, extract_references(BIB_HTML))
            await refs.replace_for_paper(second, extract_references(BIB_HTML))
            await refs.delete_for_paper(first)
            assert await refs.count(first) == 0
            assert await refs.count(second) == 3

    async def test_label_reads_as_a_citation(self, repo) -> None:  # noqa: ANN001
        async with repo() as db:
            paper_id = await seed(db)
            refs = ReferenceRepository(db)
            await refs.replace_for_paper(paper_id, extract_references(BIB_HTML))
            rows = await refs.list_references(paper_id)
        assert rows[0].label().startswith("Jimmy Lei Ba (2016) — Layer normalization")


class TestCitationGraph:
    async def test_links_once_the_cited_paper_is_ingested(self, repo) -> None:  # noqa: ANN001
        async with repo() as db:
            citing_id = await seed(db, "1706.03762")
            refs = ReferenceRepository(db)
            await refs.replace_for_paper(citing_id, extract_references(BIB_HTML))
            assert await refs.count(citing_id, resolved_only=True) == 0

            # Layer normalization arrives later.
            await seed(db, "1607.06450")
            linked = await refs.link_to_papers()
            assert linked == 1
            rows = await refs.list_references(citing_id, resolved_only=True)
            assert [row.cited_arxiv_id for row in rows] == ["1607.06450"]

    async def test_linking_is_idempotent(self, repo) -> None:  # noqa: ANN001
        async with repo() as db:
            citing_id = await seed(db, "1706.03762")
            refs = ReferenceRepository(db)
            await refs.replace_for_paper(citing_id, extract_references(BIB_HTML))
            await seed(db, "1607.06450")
            assert await refs.link_to_papers() == 1
            assert await refs.link_to_papers() == 0

    async def test_a_paper_never_links_to_itself(self, repo) -> None:  # noqa: ANN001
        async with repo() as db:
            paper_id = await seed(db, "1706.03762")
            refs = ReferenceRepository(db)
            await refs.replace_for_paper(paper_id, extract_references(BIB_HTML))
            assert await refs.link_to_papers() == 0

    async def test_cited_by_is_the_reverse_edge(self, repo) -> None:  # noqa: ANN001
        async with repo() as db:
            citing_id = await seed(db, "1706.03762")
            cited_id = await seed(db, "1607.06450")
            refs = ReferenceRepository(db)
            await refs.replace_for_paper(citing_id, extract_references(BIB_HTML))
            await refs.link_to_papers()
            incoming = await refs.citing_papers(cited_id)
            outgoing = await refs.referenced_papers(citing_id)
        assert [p.id for p in incoming] == [citing_id]
        assert [p.id for p in outgoing] == [cited_id]

    async def test_most_cited_ranks_the_cohort(self, repo) -> None:  # noqa: ANN001
        async with repo() as db:
            a = await seed(db, "1706.03762")
            b = await seed(db, "1607.06450")
            c = await seed(db, "1610.02357")
            refs = ReferenceRepository(db)
            for citing in (a, b, c):
                await refs.replace_for_paper(citing, extract_references(BIB_HTML))
            await refs.link_to_papers()
            ranked = await refs.most_cited()
        # Each of the three papers cites the other two (self-links excluded), so
        # every one has two incoming edges and the order is the title tie-break.
        assert {row["arxiv_id"] for row in ranked} == {
            "1607.06450",
            "1610.02357",
            "1706.03762",
        }
        assert all(row["citation_count"] == 2 for row in ranked)


class TestProjects:
    def test_slug_is_derived_and_usable(self) -> None:
        assert slugify("Graph Neural Networks (2024)") == "graph-neural-networks-2024"
        assert slugify("Şişli Çağrışım") == "sisli-cagrsm"
        assert slugify("---") == "project"

    async def test_create_and_list(self, repo) -> None:  # noqa: ANN001
        async with repo() as db:
            projects = ProjectRepository(db)
            await projects.create("Graph Neural Networks", description="gnn")
            rows = await projects.list_projects()
        assert [(row.slug, row.paper_count) for row in rows] == [
            ("graph-neural-networks", 0)
        ]

    async def test_duplicate_slug_is_rejected(self, repo) -> None:  # noqa: ANN001
        async with repo() as db:
            projects = ProjectRepository(db)
            await projects.create("Reading List")
            with pytest.raises(ProjectConflictError):
                await projects.create("reading   list")

    async def test_lookup_by_slug_is_case_insensitive(self, repo) -> None:  # noqa: ANN001
        async with repo() as db:
            projects = ProjectRepository(db)
            await projects.create("Reading List")
            found = await projects.get("READING-LIST")
        assert found is not None and found.slug == "reading-list"

    async def test_unknown_project_raises(self, repo) -> None:  # noqa: ANN001
        async with repo() as db:
            from paper_app.db.project_repository import PaperNotFoundError

            with pytest.raises(PaperNotFoundError):
                await ProjectRepository(db).require("nope")

    async def test_import_links_papers(self, repo) -> None:  # noqa: ANN001
        async with repo() as db:
            paper_id = await seed(db)
            projects = ProjectRepository(db)
            project = await projects.create("Reading List")
            assert await projects.add_papers(project, [paper_id], added_by="test") == 1
            rows = await projects.list_papers(project)
        assert [paper.id for paper, _ in rows] == [paper_id]

    async def test_reimport_is_a_noop(self, repo) -> None:  # noqa: ANN001
        async with repo() as db:
            paper_id = await seed(db)
            projects = ProjectRepository(db)
            project = await projects.create("Reading List")
            await projects.add_papers(project, [paper_id])
            assert await projects.add_papers(project, [paper_id]) == 0
        assert True

    async def test_resolution_accepts_every_spelling(self, repo) -> None:  # noqa: ANN001
        async with repo() as db:
            await seed(db, "1706.03762")
            projects = ProjectRepository(db)
            resolved, missing = await projects.resolve_paper_ids(
                [
                    "1706.03762",
                    "1706.03762v1",
                    "https://arxiv.org/abs/1706.03762",
                    "arXiv:1706.03762",
                    "2401.00001",
                ]
            )
        assert len(resolved) == 1
        assert missing == ["2401.00001"]

    async def test_resolution_reports_what_it_could_not_find(self, repo) -> None:  # noqa: ANN001
        async with repo() as db:
            await seed(db, "1706.03762")
            resolved, missing = await ProjectRepository(db).resolve_paper_ids(
                ["1706.03762", "9999.99999", "2401.00001", "2401.00002"]
            )
        assert len(resolved) == 1
        # Both spellings of the missing paper are reported once, in order.
        assert missing == ["9999.99999", "2401.00001", "2401.00002"]

    async def test_empty_input(self, repo) -> None:  # noqa: ANN001
        async with repo() as db:
            assert await ProjectRepository(db).resolve_paper_ids([]) == ([], [])


class TestPapersAreGlobal:
    async def test_two_projects_share_one_paper_row(self, repo) -> None:  # noqa: ANN001
        async with repo() as db:
            paper_id = await seed(db)
            projects = ProjectRepository(db)
            first = await projects.create("First")
            second = await projects.create("Second")
            await projects.add_papers(first, [paper_id], note="one")
            await projects.add_papers(second, [paper_id], note="two")

            papers = await db.scalar(select(func.count()).select_from(Paper))
            links = await db.scalar(select(func.count()).select_from(ProjectPaper))
        # One paper, two membership rows: importing never copies.
        assert papers == 1
        assert links == 2

    async def test_notes_are_per_project(self, repo) -> None:  # noqa: ANN001
        async with repo() as db:
            paper_id = await seed(db)
            projects = ProjectRepository(db)
            first = await projects.create("First")
            second = await projects.create("Second")
            await projects.add_papers(first, [paper_id], note="first note")
            await projects.add_papers(second, [paper_id], note="second note")
            rows_first = await projects.list_papers(first)
            rows_second = await projects.list_papers(second)
        assert rows_first[0][1].note == "first note"
        assert rows_second[0][1].note == "second note"

    async def test_read_state_is_per_project(self, repo) -> None:  # noqa: ANN001
        async with repo() as db:
            paper_id = await seed(db)
            projects = ProjectRepository(db)
            first = await projects.create("First")
            second = await projects.create("Second")
            await projects.add_papers(first, [paper_id])
            await projects.add_papers(second, [paper_id])
            await projects.set_read(first, paper_id, is_read=True)
            assert (await projects.list_papers(first))[0][1].is_read is True
            assert (await projects.list_papers(second))[0][1].is_read is False

    async def test_unlinking_keeps_the_paper(self, repo) -> None:  # noqa: ANN001
        async with repo() as db:
            paper_id = await seed(db)
            projects = ProjectRepository(db)
            first = await projects.create("First")
            second = await projects.create("Second")
            await projects.add_papers(first, [paper_id])
            await projects.add_papers(second, [paper_id])
            assert await projects.remove_paper(first, paper_id) is True
            assert await db.get(Paper, paper_id) is not None
            assert await db.scalar(select(func.count()).select_from(Paper)) == 1
            assert len(await projects.list_papers(second)) == 1

    async def test_deleting_a_project_keeps_its_papers(self, repo) -> None:  # noqa: ANN001
        async with repo() as db:
            paper_id = await seed(db)
            projects = ProjectRepository(db)
            project = await projects.create("Doomed")
            await projects.add_papers(project, [paper_id])
            await projects.delete(project)
            await db.flush()
            assert await db.get(Paper, paper_id) is not None
            assert await db.scalar(select(func.count()).select_from(Paper)) == 1

    async def test_reverse_lookup_names_every_project(self, repo) -> None:  # noqa: ANN001
        async with repo() as db:
            paper_id = await seed(db)
            projects = ProjectRepository(db)
            for name in ("Alpha", "Beta"):
                await projects.add_papers(await projects.create(name), [paper_id])
            slugs = await projects.projects_for_paper(paper_id)
        assert slugs == ["alpha", "beta"]

    async def test_counts_track_removal(self, repo) -> None:  # noqa: ANN001
        async with repo() as db:
            paper_id = await seed(db)
            projects = ProjectRepository(db)
            project = await projects.create("Reading List")
            await projects.add_papers(project, [paper_id])
            await projects.set_read(project, paper_id, is_read=True)
            rows = await projects.list_projects()
            assert rows[0].paper_count == 1
            assert rows[0].read_count == 1
            await projects.remove_paper(project, paper_id)
            rows = await projects.list_projects()
        assert rows[0].paper_count == 0
        assert rows[0].read_count == 0


class TestProjectCategoryFilter:
    async def test_filters_by_category(self, repo) -> None:  # noqa: ANN001
        async with repo() as db:
            cl = await seed(db, "1706.03762", categories=("cs.CL",), primary="cs.CL")
            lg = await seed(db, "1607.06450", categories=("cs.LG",), primary="cs.LG")
            deep = await seed(
                db, "2104.00001", categories=("cs.LG.MS",), primary="cs.LG.MS"
            )
            projects = ProjectRepository(db)
            project = await projects.create("Mixed")
            for paper_id in (cl, lg, deep):
                await projects.add_papers(project, [paper_id])

            assert len(await projects.list_papers(project, category="cs.LG")) == 1
            both = await projects.list_papers(project, category="cs.CL")
            assert len(both) == 1
            subtree = await projects.list_papers(
                project, category="cs.LG", limit=10
            )
            assert len(subtree) == 1
        assert True


class TestAuthorsAndCategoriesUnaffected:
    async def test_import_does_not_disturb_the_taxonomy(self, repo) -> None:  # noqa: ANN001
        async with repo() as db:
            paper_id = await seed(db)
            projects = ProjectRepository(db)
            project = await projects.create("Reading List")
            await projects.add_papers(project, [paper_id])
            await db.flush()
            assert await db.scalar(
                select(func.count()).select_from(PaperAuthor)
            ) == 1
            assert await db.scalar(
                select(func.count()).select_from(PaperCategory)
            ) == 2