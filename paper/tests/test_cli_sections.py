"""The section surface of the CLI, driven the way a reader drives it.

``paper outline`` / ``paper section`` / ``paper ask --section`` / ``paper
ask --by-section`` are one feature with two halves: what a book *contains*, and
where a passage *is*. :mod:`tests.test_sections_query` settles how a name is
resolved to a section; what is under test here is what the terminal shows once
it has been — exit codes, page numbers, section titles, and the difference
between a filter that matched nothing and a corpus that does not cover the topic.

Everything goes through ``CliRunner`` against the real Typer app rather than
calling the renderers directly, because what is worth pinning down is
presentation and exit codes, and neither survives being tested at the function
boundary.

The corpus is written straight to the ORM and the vectors straight to the
in-memory store, so nothing here depends on MinerU, a PDF or a real embedding
model: the CLI reads real rows, and only the *ranking* is synthetic.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import anyio
import pytest
from click.testing import CliRunner, Result
from typer.main import get_command

from paper_app.cli import app
from paper_app.container import Container, set_container

DOC = "solid-state-basics"
FLAT_DOC = "pethick-superconductivity"

#: A spine shaped like the books it was measured on rather than like a tidy
#: fixture. Chapter 2's own title carries the word ``Debye``, so asking for
#: "Debye" has something to outrank. And the two structural sources disagree
#: about depth, which is the case ``--with-subsections`` is written around: the
#: PDF's bookmarks put ``2.2`` at level 3, and the markdown read of the same book
#: read ``2.2.1`` as level 3 too, so the child is not deeper and a level
#: comparison finds nothing for a section that visibly has subsections.
BOOK_SECTIONS = [
    (0, "1 Introduction", 1, 3, 9, "outline"),
    (1, "2 Specific Heat of Solids: Boltzmann, Einstein, and Debye", 1, 34, 120, "outline"),
    (2, "2.1 Drude's Model", 2, 34, 60, "outline"),
    (3, "2.2 Debye's Calculation", 3, 61, 90, "outline"),
    (4, "2.2.1 Periodic Debye Heat Capacity", 3, 61, 70, "markdown"),
    (5, "2.3 Phonons in Metals", 2, 91, 120, "markdown"),
    (6, "3 Phonons", 1, 121, 200, "outline"),
]

#: (section ordinal or ``None``, page_start, page_end, sentence). Chapter 2 has
#: no chunk at all — a heading the corpus holds no text for — and the last row
#: belongs to no heading, which is what the front matter and the index are.
BOOK_CHUNKS = [
    (0, 4, 5, "The solid state is what a crystal leaves behind when it freezes."),
    (2, 34, 35, "Drude treated the conduction electrons as a classical gas."),
    (2, 36, 37, "The mean free path is what sets the conductivity of the metal."),
    (3, 61, 62, "Debye replaced the classical cutoff with the speed of sound."),
    (3, 63, 64, "Integrating the phonon density of states to three modes per atom."),
    (4, 67, 68, "The boundary condition quantises the lattice in one dimension."),
    (5, 91, 92, "In a metal the same vibrations are what carry the current."),
    (6, 121, 122, "Phonons are the quantised vibrations of the lattice."),
    (None, 250, 252, "Numerical constants used throughout the book are listed here."),
]

FLAT_CHUNKS = [
    (0, 40, 41, "The superconducting state is described by a macroscopic wave function."),
]


@pytest.fixture
def container(settings, tmp_path: Path) -> Any:  # noqa: ANN401
    """The real container, with the in-memory vector store the suite uses."""
    from paper_app.db.vector_store.memory_store import InMemoryVectorStore  # noqa: PLC0415

    container = Container(settings)
    space = container.default_space
    container._stores[f"{space.name}:{space.fingerprint}:{space.distance}"] = (  # noqa: SLF001
        InMemoryVectorStore(space)
    )
    set_container(container)
    try:
        yield container
    finally:
        set_container(None)


async def _seed(container) -> None:  # noqa: ANN001
    """Two documents, one of them with no structure recorded at all."""
    from paper_app.db.repositories import (  # noqa: PLC0415
        ChunkRepository,
        PaperRepository,
        RawDocumentRepository,
        SectionRepository,
    )
    from paper_app.domain.enums import ChunkKind  # noqa: PLC0415
    from paper_app.domain.models import ContentSource, PaperMetadata, TextChunk  # noqa: PLC0415
    from paper_app.services.sections import Section  # noqa: PLC0415

    async with container.session_factory() as session:
        papers = PaperRepository(session)
        book = await papers.upsert(
            PaperMetadata.from_file(
                path=Path(f"{DOC}.pdf"),
                title="Solid State Basics",
                doc_key=DOC,
                kind="book",
                page_count=305,
                file_sha256=DOC.ljust(64, "0"),
            )
        )
        await SectionRepository(session).replace_for_paper(
            book.id,
            [
                Section(ordinal, title, level, page_start, page_end, source)
                for ordinal, title, level, page_start, page_end, source in BOOK_SECTIONS
            ],
        )
        await ChunkRepository(session).replace_for_paper(
            book.id,
            [
                TextChunk(
                    ordinal=index,
                    text=text,
                    token_count=len(text.split()),
                    char_start=0,
                    char_end=len(text),
                    heading=BOOK_SECTIONS[ordinal][1] if ordinal is not None else "Index",
                    source=ContentSource.PDF_MINERU,
                    kind=ChunkKind.BODY,
                    page_start=page_start,
                    page_end=page_end,
                    section_ordinal=ordinal,
                ).with_hash()
                for index, (ordinal, page_start, page_end, text) in enumerate(BOOK_CHUNKS)
            ],
        )

        # A PDF whose bookmarks and headings both yielded nothing: two of the
        # four real books here have no bookmarks, and the honest answer to
        # `paper outline` is that there is no structure — not an empty table the
        # reader has to interpret. The raw-document row is what makes that
        # different from an arXiv HTML rendering, which has no pages to point at
        # either, and the CLI says which of the two it is looking at.
        flat = await papers.upsert(
            PaperMetadata.from_file(
                path=Path(f"{FLAT_DOC}.pdf"),
                title="Superconductivity of Metals and Alloys",
                doc_key=FLAT_DOC,
                kind="book",
                page_count=578,
                file_sha256=FLAT_DOC.ljust(64, "0"),
            )
        )
        await RawDocumentRepository(session).record(
            paper_id=flat.id,
            kind="pdf",
            uri=str(Path(f"{FLAT_DOC}.pdf")),
            content_type="application/pdf",
            size_bytes=24_000_000,
            sha256=FLAT_DOC.ljust(64, "0"),
        )
        await ChunkRepository(session).replace_for_paper(
            flat.id,
            [
                TextChunk(
                    ordinal=index,
                    text=text,
                    token_count=len(text.split()),
                    char_start=0,
                    char_end=len(text),
                    source=ContentSource.PDF_PYPDF,
                    kind=ChunkKind.BODY,
                    page_start=page_start,
                    page_end=page_end,
                ).with_hash()
                for index, (_ordinal, page_start, page_end, text) in enumerate(FLAT_CHUNKS)
            ],
        )
        await session.commit()


async def _index(container) -> None:  # noqa: ANN001
    """Vectors for the stored chunks, the way an ingest would have written them.

    The payload carries the fields a *filter* reads, ``section_ordinal`` above
    all: ``--section`` is resolved against the section table first and then
    answered by the vector store, which is only possible because a chunk carries
    the ordinal of the section it belongs to.
    """
    from sqlalchemy import select  # noqa: PLC0415

    from paper_app.db.models import Chunk  # noqa: PLC0415
    from paper_app.domain.models import VectorRecord  # noqa: PLC0415

    space = container.default_space
    provider = container.provider_for(space)
    store = container.vector_store_for(space)
    async with container.session_factory() as session:
        rows = list((await session.execute(select(Chunk).order_by(Chunk.id))).scalars())
        vectors = await provider.embed_documents([row.text for row in rows])
        await store.upsert(
            [
                VectorRecord(
                    chunk_id=row.id,
                    paper_id=row.paper_id,
                    vector=vector,
                    payload={
                        "provider": space.provider,
                        "model": space.model,
                        "fingerprint": space.fingerprint,
                        "paper_id": row.paper_id,
                        "ordinal": row.ordinal,
                        "source": row.source,
                        "content_kind": row.content_kind,
                        "section_ordinal": row.section_ordinal,
                        "page_start": row.page_start,
                        "heading": row.heading,
                        "categories": [],
                    },
                )
                for row, vector in zip(rows, vectors, strict=True)
            ]
        )
    assert await store.count() == len(BOOK_CHUNKS) + len(FLAT_CHUNKS)


@pytest.fixture
def corpus(container) -> str:  # noqa: ANN201
    """Seed and index. Sync, like every test in this file: the CLI runs
    ``asyncio.run`` itself, so a coroutine test could not invoke it."""
    anyio.run(_seed, container)
    anyio.run(_index, container)
    return DOC


@pytest.fixture
def run(corpus):  # noqa: ANN001, ANN201
    """Invoke the real CLI, the way a shell does."""
    command = get_command(app)

    def invoke(*args: str, read_only: bool = False) -> Result:
        prefix = ["--read-only"] if read_only else []
        return CliRunner().invoke(command, [*prefix, *args])

    return invoke


@pytest.fixture(autouse=True)
def _wide_console(monkeypatch) -> None:  # noqa: ANN001
    """Unwrap the rendered tables.

    Rich takes its width from the terminal and ``CliRunner`` is not one, so
    without this a 305-page book's headings are ellipsised mid-title and every
    assertion in this file would be about column arithmetic rather than content.
    """
    from paper_app.cli import console  # noqa: PLC0415

    monkeypatch.setattr(console, "width", 200)


@pytest.fixture(autouse=True)
def _clear_read_only() -> Any:  # noqa: ANN401
    """``--read-only`` sets a process-wide flag that nothing clears.

    Left set, every later test in the process would run as though a mode it never
    asked for were in force, so it is restored here rather than only in the tests
    that set it.
    """
    from paper_app.cli import read_mode  # noqa: PLC0415

    yield
    read_mode(False)


def _rows(output: str, doc_key: str) -> list[str]:
    """Rendered table rows belonging to one document.

    ``box=None`` tables have no borders to key on, so the document handle is the
    discriminator — and the header row does not contain it.
    """
    return [line for line in output.splitlines() if doc_key in line]


def _outline_rows(output: str) -> list[str]:
    """Outline rows, past the title line and the column header."""
    return output.splitlines()[2:]


def _hits_cell(row: str, doc_key: str) -> str:
    """The hit count on a ``--by-section`` row: everything after the handle."""
    return row.rsplit(doc_key, 1)[-1].split()[0]


def _first_line(output: str) -> str:
    """The heading line of a ``paper section`` render."""
    return output.splitlines()[0]


async def _levels(container, doc_key: str) -> dict[str, int]:  # noqa: ANN001
    """Stored heading levels of one document, by title."""
    from paper_app.db.repositories import PaperRepository, SectionRepository  # noqa: PLC0415

    async with container.session_factory() as session:
        paper = await PaperRepository(session).resolve(doc_key)
        rows = await SectionRepository(session).list_for_paper(paper.id)
        return {row.title: row.level for row in rows}


class TestPaperSection:
    """``paper section <doc_key> 2.2`` — the section itself, in order, with pages."""

    def test_the_heading_its_page_range_and_the_text_under_it(self, run) -> None:
        result = run("section", DOC, "2.2")
        assert result.exit_code == 0, result.output
        out = result.output
        # Heading, the range it spans, and which source claimed it exists: two
        # of the four real books have no bookmarks, so the reader deserves to
        # know which structure a heading came from.
        assert "2.2 Debye's Calculation" in out
        assert "p61-90" in out
        assert "from outline" in out
        # Both chunks of the section, in document order, each with its own page.
        assert "Debye replaced the classical cutoff" in out
        assert "Integrating the phonon density of states" in out
        assert out.index("Debye replaced") < out.index("Integrating the phonon")
        # The subsection's body is not smuggled in with the parent's.
        assert "The boundary condition quantises the lattice" not in out

    def test_the_page_range_is_not_the_list_of_pages_shown(self, run) -> None:
        """``p61-90`` is the heading's claim; ``p61`` and ``p63`` are the facts.

        They are different things, and printing the range as though it were the
        text would promise thirty pages of material the corpus does not hold — a
        section can span a page range and have two chunks inside it.
        """
        out = run("section", DOC, "2.2").output
        shown = re.findall(r"p(\d+) body$", out, flags=re.MULTILINE)
        assert shown == ["61", "63"]
        assert "61-90" in out

    def test_a_section_the_corpus_holds_no_text_for_says_so(self, run) -> None:
        """Chapter 2 has a heading and no chunks: a zero, not a hole."""
        result = run("section", DOC, "2")
        assert result.exit_code == 0, result.output
        assert "2 Specific Heat of Solids" in result.output
        assert "no text stored for this section" in result.output
        # It says the pages were never indexed rather than printing an empty body.
        assert "34-120" in result.output

    def test_a_word_from_the_title_finds_the_same_section(self, run) -> None:
        """Two spellings, because both are how a passage is referred to."""
        by_word = run("section", DOC, "Debye")
        by_number = run("section", DOC, "2.2")
        assert by_word.exit_code == 0, by_word.output
        assert "2.2 Debye's Calculation" in by_word.output
        assert "Debye replaced the classical cutoff" in by_word.output
        assert by_number.exit_code == 0
        # The section named by words and the one named by number print the same
        # way: the render reports which section was chosen, not how.
        assert _first_line(by_word.output) == _first_line(by_number.output)

    def test_level_narrows_which_headings_a_word_may_match(self, run) -> None:
        """``--level 1`` keeps the deep headings out of the answer.

        Chapter 2's own title contains "Debye", so this is the difference between
        the chapter a reader did not ask for and the section they did name.
        """
        result = run("section", DOC, "Debye", "--level", "1")
        assert result.exit_code == 0, result.output
        assert "Debye's Calculation" not in result.output
        assert "2 Specific Heat of Solids" in result.output

    def test_an_unknown_section_points_at_the_outline(self, run) -> None:
        result = run("section", DOC, "Fermi liquid")
        assert result.exit_code != 0
        out = result.output
        assert "no section matching" in out
        assert "Fermi liquid" in out
        assert DOC in out
        # The overwhelmingly likely mistake is a title this book does not use,
        # and the command that lists the titles it does use is one flag away.
        assert f"paper outline {DOC}" in out

    def test_a_document_that_is_not_ingested_is_a_clean_failure(self, run) -> None:
        result = run("section", "not-a-book", "2.2")
        assert result.exit_code != 0
        assert "not ingested" in result.output
        # Nothing was searched for, so nothing is claimed about sections.
        assert "no section matching" not in result.output


class TestSubsections:
    """``--with-subsections``, and the case it exists for."""

    def test_nested_sections_are_listed_below_the_text(self, run) -> None:
        result = run("section", DOC, "2.2", "--with-subsections")
        assert result.exit_code == 0, result.output
        out = result.output
        assert "2.2.1 Periodic Debye Heat Capacity" in out
        assert "p61" in out
        # Listed as structure, not printed as text: the subsection's body is one
        # more command away, and mixing the two would blur which words came
        # from where.
        assert out.index("Integrating the phonon") < out.index("2.2.1 Periodic Debye")

    def test_the_child_is_found_by_page_because_the_sources_disagree_on_level(
        self, run, container
    ) -> None:
        """Containment here is decided by page range, and only by page range.

        In this book the PDF's bookmarks put ``2.2 Debye's Calculation`` at level
        3, and the markdown read of the same book read ``2.2.1 …`` as level 3 as
        well — so the child is *not* deeper, and a level comparison returns
        nothing at all for a section that visibly has subsections. What both
        sources do agree on is pages: ``2.2.1`` starts on 61, inside 61-90.

        The levels are read back from the database first, so this cannot pass by
        accident on a fixture whose levels became consistent — which would be the
        one change that would silently retire the rule.
        """
        levels = anyio.run(_levels, container, DOC)
        assert levels["2.2 Debye's Calculation"] == 3
        assert levels["2.2.1 Periodic Debye Heat Capacity"] == 3
        assert levels["2.2.1 Periodic Debye Heat Capacity"] <= levels["2.2 Debye's Calculation"]

        out = run("section", DOC, "2.2", "--with-subsections").output
        assert "2.2.1 Periodic Debye Heat Capacity" in out

    def test_a_sibling_is_not_reported_as_a_child(self, run) -> None:
        """``2.3`` starts on page 91, past the end of 2.2's 61-90."""
        out = run("section", DOC, "2.2", "--with-subsections").output
        assert "2.3 Phonons in Metals" not in out
        assert "2.1 Drude's Model" not in out

    def test_subsections_are_left_out_unless_asked_for(self, run) -> None:
        out = run("section", DOC, "2.2").output
        assert "2.2.1 Periodic Debye Heat Capacity" not in out


class TestAskSectionFilter:
    """``paper ask --section`` — and the failure that must not look like success."""

    def test_a_matched_section_narrows_the_search_to_it(self, run) -> None:
        result = run("ask", "phonons", "--section", "2.2", "--in", DOC, "-k", "10")
        assert result.exit_code == 0, result.output
        out = result.output
        # Echoed with its page range, so the reader can see what was searched.
        assert "2.2 Debye's Calculation" in out
        assert "(p61-90)" in out
        # Only chunks pointing at that section: the store filters on the section
        # ordinal, so a neighbouring section's passage cannot slip in.
        assert "§ 2.2 Debye's Calculation" in out
        assert "classical gas" not in out
        assert "quantises the lattice" not in out

    def test_top_k_counts_matches_inside_the_section(self, run) -> None:
        """``-k 1`` returns the section's best match, not one of a corpus-wide top-k.

        The filter is applied before ranking, which is what makes "the best
        passage in this section" a question the command can answer.
        """
        result = run("ask", "phonons", "--section", "2.2", "--in", DOC, "-k", "1")
        assert result.exit_code == 0, result.output
        assert result.output.count("score=") == 1
        assert "§ 2.2 Debye's Calculation" in result.output

    def test_a_number_finds_that_section_and_not_its_neighbours(self, run) -> None:
        """``2.2`` is one section: not ``2.2.1`` and not ``2.3``."""
        out = run("ask", "phonons", "--section", "2.2", "--in", DOC, "-k", "10").output
        assert "2.2.1" not in out
        assert "2.3 Phonons" not in out
        assert "§ 2.2 Debye's Calculation" in out

    def test_an_unknown_section_is_not_a_zero_hit_search(self, run) -> None:
        """The important one.

        A section filter that resolved to nothing and was then applied anyway
        would print ``no matches in space …`` — byte-identical to a question the
        corpus genuinely does not cover, and the reader would go hunting for a
        gap in a book that is really a typo in a flag. So an unresolved name
        fails before the search runs, says which name failed, and names the
        command that lists the names that do exist.
        """
        result = run("ask", "phonons", "--section", "Fermi liquid", "--in", DOC)
        assert result.exit_code != 0
        out = result.output
        assert "no section matching" in out
        assert "Fermi liquid" in out
        assert DOC in out
        assert f"paper outline {DOC}" in out
        # None of the zero-hit vocabulary, and no hits either.
        assert "no matches in space" not in out
        assert "score=" not in out

    def test_an_unknown_section_fails_the_same_way_without_in(self, run) -> None:
        """``--in`` is optional; the failure must not depend on it."""
        result = run("ask", "phonons", "--section", "Fermi liquid")
        assert result.exit_code != 0
        out = result.output
        assert "no section matching" in out
        assert "in the corpus" in out
        assert "no matches in space" not in out

    def test_ranking_puts_the_section_before_the_chapter_that_mentions_it(self, run) -> None:
        """``Debye`` names 2.2, not the chapter that merely contains the word.

        Both are matches and both are searched — the flag is not a filter on the
        heading's own name — but the order is what tells a reader which one is
        the answer, and without an order the corpus answers in whatever order the
        database returns.
        """
        out = run("ask", "Debye", "--section", "Debye", "--in", DOC, "-k", "10").output
        echoed = re.search(r"^section: (.+)$", out, flags=re.MULTILINE)
        assert echoed is not None, out
        # Split on the page range rather than on commas: the titles themselves
        # contain commas, which is most of why this is asserted on the render.
        names = [name.strip(" ,") for name in re.findall(r"([^(]+)\(p[\d-]+\)", echoed.group(1))]
        assert names[0] == "2.2 Debye's Calculation"
        assert "2 Specific Heat of Solids: Boltzmann, Einstein, and Debye" in names


class TestAskBySection:
    """``paper ask --by-section`` — which pages cover this, not which chunks did."""

    def test_one_row_per_section_showing_the_sections_own_pages(self, run) -> None:
        result = run("ask", "phonons", "--by-section", "-k", "20")
        assert result.exit_code == 0, result.output
        rows = _rows(result.output, DOC)
        # Nine chunks of this document fall into six headings plus one stretch
        # that belongs to none of them; the point of the view is that the two
        # chunks of 2.2 are one row rather than two.
        assert len(rows) == 7, rows

        debye = [line for line in rows if "Debye's Calculation" in line]
        assert len(debye) == 1, rows
        # 61-90 is the section's span, not the two pages that scored: a reader
        # opens the book at the section, not at the winning sentence.
        assert "61-90" in debye[0]
        assert _hits_cell(debye[0], DOC) == "2"

        assert any("34-60" in line and "Drude" in line for line in rows), rows
        assert any("121-200" in line and "3 Phonons" in line for line in rows), rows

    def test_grouping_is_by_section_not_by_page_or_chunk(self, run) -> None:
        """Stated as the count it changes: nine chunks in one document, seven rows.

        Two of those chunks share one heading and the ninth belongs to none, so a
        view grouped by the page a chunk starts on would print a row per chunk —
        the opposite of the answer this view exists to give.
        """
        rows = _rows(run("ask", "phonons", "--by-section", "-k", "20").output, DOC)
        assert sum(int(_hits_cell(row, DOC)) for row in rows) == len(BOOK_CHUNKS)
        assert len(rows) < len(BOOK_CHUNKS)

    def test_a_subsection_reports_its_own_range_not_its_chapter_s(self, run) -> None:
        """Chapter 2 spans 34-120 and 2.1 spans 34-60.

        The grouped view is only useful if each row reports the section it is
        about rather than the chapter that contains it.
        """
        rows = _rows(run("ask", "phonons", "--by-section", "-k", "20").output, DOC)
        subsection = [line for line in rows if "2.2.1" in line]
        assert len(subsection) == 1, rows
        assert "61-70" in subsection[0]
        assert "34-120" not in subsection[0]

    def test_a_chunk_with_no_section_keeps_its_pages(self, run) -> None:
        """Front matter belongs to no heading, and its pages are still worth naming.

        Three consecutive pages merge into one range: without merging, one stretch
        of index would print as three rows of noise.
        """
        rows = _rows(run("ask", "phonons", "--by-section", "-k", "20").output, DOC)
        unsectioned = [line for line in rows if "250-252" in line]
        assert len(unsectioned) == 1, rows
        assert _hits_cell(unsectioned[0], DOC) == "1"

    def test_the_view_says_what_the_page_range_means(self, run) -> None:
        """Because ``61-90`` printed beside ``score=`` invites the wrong reading."""
        out = run("ask", "phonons", "--by-section", "-k", "20").output
        assert "the section's own range" in out


class TestOutline:
    """``paper outline`` — the map, and an honest absence."""

    def test_the_whole_spine_with_pages_and_its_source(self, run) -> None:
        result = run("outline", DOC)
        assert result.exit_code == 0, result.output
        out = result.output
        assert "Solid State Basics" in out
        assert DOC in out
        rows = _outline_rows(out)
        assert len(rows) == len(BOOK_SECTIONS), rows
        assert any("61-90" in line and "2.2 Debye's Calculation" in line for line in rows), rows
        assert any("61-70" in line and "2.2.1 Periodic Debye Heat Capacity" in line for line in rows)

    def test_each_row_says_which_structure_it_came_from(self, run) -> None:
        """``pdf`` for the document's own bookmarks, ``text`` for headings the
        extraction recovered.

        Two of the four real books have no bookmarks at all, so the column is the
        difference between a complete spine and half of one.
        """
        rows = _outline_rows(run("outline", DOC).output)

        def row_for(title: str) -> list[str]:
            found = [line for line in rows if title in line]
            assert len(found) == 1, rows
            return found[0].split()

        assert "pdf" in row_for("1 Introduction")
        assert "pdf" in row_for("2.2 Debye's Calculation")
        assert "text" in row_for("2.2.1 Periodic Debye Heat Capacity")
        assert "text" in row_for("2.3 Phonons in Metals")

    def test_the_chunk_column_distinguishes_a_heading_with_no_text(self, run) -> None:
        """Chapter 2 holds no chunks and says 0, rather than looking absent.

        A reader cannot tell "this heading has no text" from "this heading is
        missing" if the row simply is not there.
        """
        rows = _outline_rows(run("outline", DOC).output)
        chapter = [line for line in rows if line.strip().startswith("2 Specific Heat")]
        assert len(chapter) == 1, rows
        assert "34-120" in chapter[0]
        assert chapter[0].split()[-1] == "0"
        # …while the section beside it has two, so the zero is about this
        # heading and not about the column.
        debye = [line for line in rows if "2.2 Debye's Calculation" in line]
        assert debye[0].split()[-1] == "2"

    def test_a_document_with_no_structure_says_so_and_exits_non_zero(self, run) -> None:
        result = run("outline", FLAT_DOC)
        assert result.exit_code != 0
        out = result.output
        assert "no structure recorded" in out
        assert "bookmarks" in out
        # No table at all, rather than a header with no rows under it.
        assert "pages" not in out
        assert "src" not in out

    def test_a_document_that_is_not_ingested_is_a_clean_failure(self, run) -> None:
        result = run("outline", "not-a-book")
        assert result.exit_code != 0
        assert "not ingested" in result.output
        assert "no structure recorded" not in result.output


class TestReadOnlyMode:
    """Every command above, under ``paper --read-only``.

    These read; a flag that refused them would make the corpus un-navigable for
    the agent the flag exists to serve. The assertion is that the output is
    *identical* with and without it, since a reader of the answer should not see
    the mode that produced it.
    """

    @pytest.mark.parametrize(
        ("args", "code"),
        [
            (["outline", DOC], 0),
            (["section", DOC, "2.2"], 0),
            (["section", DOC, "2.2", "--with-subsections"], 0),
            (["ask", "phonons", "--by-section", "-k", "20"], 0),
            (["ask", "phonons", "--section", "2.2", "--in", DOC, "-k", "10"], 0),
            (["outline", FLAT_DOC], 1),
            (["section", DOC, "Fermi liquid"], 1),
            (["ask", "phonons", "--section", "Fermi liquid", "--in", DOC], 1),
        ],
    )
    def test_the_same_output_with_and_without_the_flag(self, run, args, code) -> None:
        plain = run(*args)
        guarded = run(*args, read_only=True)
        assert plain.exit_code == code, plain.output
        assert guarded.exit_code == code, guarded.output
        assert guarded.output == plain.output
        assert "read-only" not in guarded.output

    def test_refusing_writes_is_not_what_read_only_means(self, run) -> None:
        """The negative, so the test above is not passing for the wrong reason:
        a writer still refuses under the flag, and with the code a caller can
        branch on."""
        guarded = run("projects", "new", "nope", read_only=True)
        assert guarded.exit_code == 3
        assert "read-only" in guarded.output


def test_the_surface_exists_where_it_is_documented() -> None:
    """``--section``, ``--in`` and ``--by-section`` are real flags on ``ask``.

    Asserted on the command tree rather than by invoking it, so a rename fails
    here with a clear message instead of as four unrelated "no such option"
    failures below.
    """
    commands = get_command(app).commands
    assert {"section", "outline", "ask"} <= set(commands)
    ask_options = {opt for param in commands["ask"].params for opt in param.opts}
    assert {"--section", "-S", "--in", "--by-section", "--device"} <= ask_options
    section_options = {opt for param in commands["section"].params for opt in param.opts}
    assert {"--level", "--with-subsections"} <= section_options