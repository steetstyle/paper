"""Sections must be navigable, or they are not sections.

Measured on this corpus before the rule existed: 177 arXiv papers held **4168
"sections"**, of which **65 had a page**. The rest were the arXiv page's own
furniture — ``Submission history``, ``Access Paper:``, ``BibTeX formatted
citation``, ``Demos``, ``arXivLabs: experimental projects``, ``Current browse
context`` — recovered as headings because the pipeline extracted them from an HTML
rendering and read them as document structure.

``paper outline 1211.4482v1`` listed fourteen of them under a paper's title, and
``paper show`` announced "sections: 14". The failure is not ugliness: a reader who
believes a table of contents is structure will look for a chapter that does not
exist.

The rule that fixes it asks a question about each row rather than about arXiv's
template — *can this be pointed at on a page?* — which is why it survives the
website changing rather than needing a blocklist.
"""

from __future__ import annotations

from app.db.models import DocumentSection, Paper
from app.db.session import get_session_factory
from app.services.section_prune import (
    documents_with_unplaceable_sections,
    prune_unplaceable_sections,
)
from app.services.sections import build_sections

#: arXiv's abs-page furniture, verbatim from the bug report.
ARXIV_CHROME = [
    "Title: Phononics in Low-Dimensions: Engineering Phonons in Nanostructures and Graphene",
    "Submission history",
    "Access Paper:",
    "Current browse context:",
    "References & Citations",
    "BibTeX formatted citation",
    "Bookmark",
    "Bibliographic and Citation Tools",
    "Code, Data and Media Associated with this Article",
    "Demos",
    "Recommenders and Search Tools",
    "arXivLabs: experimental projects with community collaborators",
    "Condensed Matter > Mesoscale and Nanoscale Physics",
]


class TestOnlyNavigableSectionsSurvive:
    def test_page_furniture_is_never_a_section(self) -> None:
        markdown = "\n\n".join(f"# {heading}\n\nbody text." for heading in ARXIV_CHROME)
        # No page map: nothing here can be placed, which is exactly the situation.
        assert build_sections(markdown=markdown, page_map=None) == []

    def test_a_real_section_still_survives(self) -> None:
        blocks = [
            {"type": "text", "page_idx": 0, "text": "2.1 Introduction and body."},
        ]
        markdown = "# 2.1 Introduction\n\nand body."
        sections = build_sections(
            markdown=markdown,
            page_map=_map(blocks),
            document_pages=305,
        )
        assert [s.title for s in sections] == ["2.1 Introduction"]
        assert sections[0].page_start == 1

    def test_the_whole_list_goes_when_none_can_be_placed(self) -> None:
        """Not a partial result: half a table of contents is worse than none,
        because the reader cannot tell which half is real."""
        markdown = "\n\n".join(f"# {heading}\n\nbody." for heading in ARXIV_CHROME[:3])
        assert build_sections(markdown=markdown, page_map=None) == []

    def test_a_section_with_no_page_drops_out_of_an_outline_too(self) -> None:
        """Bookmarks always carry pages, so this exercises the markdown half —
        but the filter is applied to the merged list, not per source, which is
        what stops a chrome heading attaching itself to a real chapter."""
        from app.clients.content.pdf_info import OutlineNode  # noqa: PLC0415

        outline = (OutlineNode(title="1 Overview", depth=0, page=10),)
        markdown = "# Submission history\n\nbody."
        sections = build_sections(outline=outline, markdown=markdown, document_pages=20)
        assert [s.title for s in sections] == ["1 Overview"]


def _map(blocks):  # noqa: ANN001, ANN202
    from app.services.page_map import build_page_map  # noqa: PLC0415

    return build_page_map(blocks)


class TestPruningWhatIsAlreadyStored:
    """The corpus is full of rows the fixed pipeline will never write again."""

    async def _seed_chrome(self, doc_key: str = "1211.4482") -> str:
        async with get_session_factory()() as session:
            paper = Paper(
                arxiv_id=doc_key,
                versioned_id=f"{doc_key}v1",
                doc_key=doc_key,
                kind="paper",
                title="Phononics in Low-Dimensions",
                abstract="",
            )
            session.add(paper)
            await session.flush()
            session.add_all(
                [
                    DocumentSection(
                        paper_id=paper.id,
                        ordinal=index,
                        title=heading,
                        level=1,
                        page_start=None,
                        page_end=None,
                        source="markdown",
                    )
                    for index, heading in enumerate(ARXIV_CHROME, start=1)
                ]
                + [
                    DocumentSection(
                        paper_id=paper.id,
                        ordinal=99,
                        title="2 Methods",
                        level=2,
                        page_start=4,
                        page_end=9,
                        source="outline",
                    )
                ]
            )
            await session.commit()
            return paper.id

    async def test_it_reports_before_it_changes(self) -> None:
        paper_id = await self._seed_chrome()
        async with get_session_factory()() as session:
            report = await prune_unplaceable_sections(session, apply=False)
        assert report.sections == len(ARXIV_CHROME)
        assert report.applied is False
        async with get_session_factory()() as session:
            from sqlalchemy import func, select  # noqa: PLC0415

            total = await session.scalar(
                select(func.count()).select_from(DocumentSection).where(
                    DocumentSection.paper_id == paper_id
                )
            )
        assert total == len(ARXIV_CHROME) + 1, "a dry run must change nothing"

    async def test_it_removes_only_the_unplaceable_ones(self) -> None:
        paper_id = await self._seed_chrome()
        async with get_session_factory()() as session:
            report = await prune_unplaceable_sections(session, apply=True)
        assert report.sections == len(ARXIV_CHROME)
        assert report.kept == 1
        async with get_session_factory()() as session:
            from sqlalchemy import select  # noqa: PLC0415

            rows = (
                await session.execute(
                    select(DocumentSection).where(DocumentSection.paper_id == paper_id)
                )
            ).scalars().all()
        assert [row.title for row in rows] == ["2 Methods"]

    async def test_it_names_the_documents_affected(self) -> None:
        """\"4103 rows across 88 papers\" is a number; naming the paper is a
        decision."""
        await self._seed_chrome()
        async with get_session_factory()() as session:
            affected = await documents_with_unplaceable_sections(session)
        mine = [row for row in affected if row["doc_key"] == "1211.4482"]
        assert mine and mine[0]["sections"] == len(ARXIV_CHROME)

    async def test_pruning_nothing_is_not_an_error(self) -> None:
        async with get_session_factory()() as session:
            report = await prune_unplaceable_sections(session, apply=True)
        assert report.sections == 0
        assert report.applied is True


class TestTheStepRefusesHtmlEntirely:
    async def test_an_html_source_does_not_even_try(self) -> None:
        """The step says *why* rather than reporting "no structure found", which
        reads like the document has none when the truth is that its headings are
        the renderer's."""
        from app.domain.enums import ContentKind  # noqa: PLC0415
        from app.domain.models import ContentPayload  # noqa: PLC0415
        from app.pipeline.context import PipelineContext  # noqa: PLC0415
        from app.pipeline.steps import BuildSectionsStep  # noqa: PLC0415

        async with get_session_factory()() as session:
            paper = Paper(doc_key="html-doc", kind="paper", title="T", abstract="")
            session.add(paper)
            await session.commit()
            ctx = PipelineContext(session=session, run_id="r")
            ctx.paper_id = paper.id
            ctx.content = ContentPayload(
                kind=ContentKind.HTML,
                uri="x",
                content_type="text/html",
                size_bytes=1,
                sha256="a" * 64,
                source_url="https://arxiv.org/abs/1211.4482",
            )
            result = await BuildSectionsStep().run(ctx)

        assert result.ok
        assert result.data["sections"] == 0
        assert "no pages" in result.data["reason"]


class TestOutlineSaysWhichReason:
    """The two reasons need saying separately.

    "No bookmarks" sent a reader hunting for a PDF that was never downloaded, when
    the real answer was that the document came from an HTML rendering and an HTML
    page has no pages to point at.
    """

    @staticmethod
    def _outline(doc_key: str, contents_kinds: list[str]) -> str:
        from click.testing import CliRunner  # noqa: PLC0415
        from typer.main import get_command  # noqa: PLC0415

        from app.cli import app  # noqa: PLC0415
        from app.db.models import RawDocument  # noqa: PLC0415
        from app.db.session import get_session_factory  # noqa: PLC0415

        async def seed() -> None:
            async with get_session_factory()() as session:
                paper = Paper(
                    arxiv_id=doc_key,
                    versioned_id=f"{doc_key}v1",
                    doc_key=doc_key,
                    kind="paper",
                    title="A Paper",
                    abstract="",
                )
                session.add(paper)
                await session.flush()
                session.add_all(
                    [
                        RawDocument(
                            paper_id=paper.id,
                            kind=kind,
                            uri="x",
                            content_type="text/plain",
                            size_bytes=1,
                            sha256=f"{index:064d}",
                        )
                        for index, kind in enumerate(contents_kinds)
                    ]
                )
                await session.commit()

        import anyio  # noqa: PLC0415

        anyio.run(seed)
        result = CliRunner().invoke(get_command(app), ["outline", doc_key])
        return result.output

    def test_an_html_document_is_told_why(self) -> None:
        text = self._outline("html-only", ["html"]).lower()
        assert "no structure recorded" in text
        assert "html" in text
        assert "no pages" in text

    def test_a_pdf_document_is_told_the_other_why(self) -> None:
        text = self._outline("pdf-only", ["pdf"]).lower()
        assert "no structure recorded" in text
        assert "bookmarks" in text
