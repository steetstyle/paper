"""Sections found by the name a reader uses for them.

:mod:`app.services.sections_query` answers a question no vector store can: which
rows of ``document_sections`` did "Debye" mean? The table is written here through
the ORM rather than by the pipeline, because building it from a PDF has its own
tests and what is at stake below is *matching* — the table is the input, not the
subject.

The documents are shaped like real ones rather than like a tidy fixture:
numbering that grows a digit (``2.2`` next to ``2.20``), a heading the same word
appears in three times at three depths, accented titles under an ASCII
``doc_key``, a book with no page numbers at all, and — the case
:func:`sections_within` is written around — two sources that disagree about how
deep a section is.
"""

from __future__ import annotations

import pytest

from app.db.models import DocumentSection, Paper
from app.db.repositories import SectionRepository
from app.services.sections_query import find_sections, is_section_number, sections_within

# (ordinal, title, level, page_start, page_end, source)
BOOK_SECTIONS = [
    (0, "1 Introduction", 1, 3, 9, "outline"),
    (
        1,
        "2 Specific Heat of Solids: Boltzmann, Einstein, and Debye",
        1,
        10,
        23,
        "outline",
    ),
    (2, "2.1 Einstein's Theory of the Specific Heat", 2, 12, 15, "outline"),
    # Level 3, not 2: the PDF's own bookmarks put it three deep. The markdown read
    # of the same book read its own subsections as level 3 too, so a level
    # comparison finds no children under a section that visibly has them. Both
    # sources agree on the pages, which is why pages decide containment.
    (3, "2.2 Debye's Calculation", 3, 16, 23, "outline"),
    (4, "2.2.1 Periodic Debye Heat Capacity", 3, 16, 19, "markdown"),
    (5, "2.2.2 Anharmonic Correction", 4, 20, 23, "markdown"),
    # The neighbours of "2.2": one digit longer, one component deeper. Both are
    # sections a reader would swear they did not ask for.
    (6, "2.20 High-Temperature Limit", 2, 300, 310, "outline"),
    (7, "3 Free Energy", 1, 31, 40, "outline"),
    (8, "7 Exercises", 1, 550, 560, "outline"),
    (9, "7.1 Exercises", 2, 560, 565, "markdown"),
    (10, "Exercises", 1, 566, 570, "markdown"),
]

NOTES_SECTIONS = [
    (0, "1 Giriş", 1, 1, 5, "outline"),
    (1, "2 Birim Hız", 2, 6, 9, "outline"),
    (2, "2.1 Vorteks Çekirdeği", 2, 7, 8, "markdown"),
    (3, "3 Abrikosov Vortices", 1, 12, 18, "outline"),
    (4, "4.4 Şekerleme Yöntemi", 2, 40, 44, "markdown"),
]

# A document with no page map at all, which is what one that ships without
# bookmarks *and* whose extracted text carries no page numbers produces: every row
# is unlocated, so level is the only thing left to say what is inside what.
UNLOCATED_SECTIONS = [
    (0, "A Methods", 1, None, None, "markdown"),
    (1, "A.1 Setup and notation", 2, None, None, "markdown"),
    (2, "B Results", 1, None, None, "markdown"),
]

BOOKS = {
    "solid-state-basics": ("Solid State Basics", "book", BOOK_SECTIONS),
    "tsoyuz-superfluidanlar": ("Süperakışkanlar Üzerine Notlar", "notes", NOTES_SECTIONS),
}

#: The third document, unlocated throughout.
UNLOCATED_BOOK = "superconductivity-notes"


async def _seed(session, doc_key: str, title: str, kind: str, rows) -> Paper:  # noqa: ANN001
    """Write one document and its sections straight to the ORM.

    Ids are assigned by the column default, so the paper is flushed before the
    sections that point at it are added.
    """
    paper = Paper(doc_key=doc_key, kind=kind, title=title, abstract="", page_count=600)
    session.add(paper)
    await session.flush()
    session.add_all(
        [
            DocumentSection(
                paper_id=paper.id,
                ordinal=ordinal,
                title=heading,
                level=level,
                page_start=page_start,
                page_end=page_end,
                source=source,
            )
            for ordinal, heading, level, page_start, page_end, source in rows
        ]
    )
    await session.flush()
    return paper


@pytest.fixture
async def corpus(session):
    """The three documents with their tables of contents, keyed by doc_key."""
    papers: dict[str, Paper] = {}
    for doc_key, (title, kind, rows) in BOOKS.items():
        papers[doc_key] = await _seed(session, doc_key, title, kind, rows)
    papers[UNLOCATED_BOOK] = await _seed(
        session, UNLOCATED_BOOK, "Notes on Superconductivity", "notes", UNLOCATED_SECTIONS
    )
    return papers


def _titles(matches) -> list[str]:  # noqa: ANN001
    return [match.title for match in matches]


class TestWordBoundaries:
    """A term matches on a word edge, never inside one.

    A substring match hands a reader a chapter they did not ask for, with total
    confidence and no way to tell it was a guess.
    """

    async def test_a_heading_is_matched_wherever_the_word_sits(self, session, corpus) -> None:
        """Not just the first one: books say "Exercises" in several places."""
        found = await find_sections(session, "Exercises")
        assert _titles(found) == ["Exercises", "7 Exercises", "7.1 Exercises"]

    async def test_a_word_matches_its_own_heading_exactly(self, session, corpus) -> None:
        """Only a heading that *is* the word can match exactly.

        The numbering is stripped before comparing, so "7 Exercises" is a prefix
        match and the bare heading is the exact one — which is what puts it first.
        """
        found = await find_sections(session, "exercises")
        assert [match.match_kind for match in found] == ["exact", "prefix", "prefix"]

    async def test_matching_ignores_case(self, session, corpus) -> None:
        assert _titles(await find_sections(session, "eXeRcIsEs")) == _titles(
            await find_sections(session, "Exercises")
        )

    @pytest.mark.parametrize("term", ["ex", "Deb", "xercises", "ercise"])
    async def test_a_fragment_of_a_word_matches_nothing(self, session, corpus, term) -> None:
        assert await find_sections(session, term) == []

    async def test_an_apostrophe_ends_the_word(self, session, corpus) -> None:
        """``Debye`` matches ``2.2 Debye's Calculation``…

        …because the apostrophe ends the word, not because matching is a substring
        scan: ``Debye's`` as one needle matches too, ``Deb`` does not.
        """
        assert "2.2 Debye's Calculation" in _titles(await find_sections(session, "Debye"))


class TestBookNumbering:
    """A book may be addressed by its own numbering, which is not a word search."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("2", True),
            ("2.2", True),
            ("2.2.1", True),
            ("IV", True),
            ("A", True),
            ("Debye", False),
            ("2.", False),
        ],
    )
    def test_what_counts_as_numbering(self, value, expected) -> None:
        assert is_section_number(value) is expected

    async def test_a_number_finds_that_section_and_not_its_neighbours(
        self, session, corpus
    ) -> None:
        """``2.2`` is one section, not ``2.20`` and not ``2.2.1``."""
        found = await find_sections(session, "2.2")
        assert _titles(found) == ["2.2 Debye's Calculation"]
        assert found[0].match_kind == "exact"

    async def test_a_deeper_number_is_its_own_section(self, session, corpus) -> None:
        found = await find_sections(session, "2.2.1")
        assert _titles(found) == ["2.2.1 Periodic Debye Heat Capacity"]

    async def test_a_number_absent_from_the_book_matches_nothing(self, session, corpus) -> None:
        assert await find_sections(session, "9.9") == []


class TestRanking:
    """Without an order the corpus answers in whatever order the database returns."""

    async def test_the_section_beats_the_chapter_that_merely_contains_the_word(
        self, session, corpus
    ) -> None:
        """The case in the module docstring: "Debye" names the section.

        The whole order, because the two rules behind it are both visible here: a
        word at the front of a heading outranks one buried in the middle, and
        among equals the shallower, earlier section wins.
        """
        found = await find_sections(session, "Debye")
        assert _titles(found) == [
            "2.2 Debye's Calculation",
            "2 Specific Heat of Solids: Boltzmann, Einstein, and Debye",
            "2.2.1 Periodic Debye Heat Capacity",
        ]
        assert [match.match_kind for match in found] == ["prefix", "contains", "contains"]

    async def test_shallower_and_earlier_wins_a_tie(self, session, corpus) -> None:
        """``1 Introduction`` at page 4 before ``9.2 Introduction`` at page 400.

        The reader almost always means the earlier, shallower one, so level and
        page break ties rather than the database.
        """
        await _seed(
            session,
            "later-introduction",
            "A Book With Another Introduction",
            "book",
            [(0, "9.2 Introduction", 2, 400, 405, "outline")],
        )
        found = await find_sections(session, "Introduction")
        assert _titles(found) == ["1 Introduction", "9.2 Introduction"]
        assert all(match.match_kind == "prefix" for match in found)

    async def test_max_level_keeps_the_deeper_headings_out(self, session, corpus) -> None:
        found = await find_sections(
            session, "Exercises", doc_keys=["solid-state-basics"], max_level=1
        )
        assert _titles(found) == ["Exercises", "7 Exercises"]


class TestDocKeys:
    """``doc_keys`` narrows to particular documents."""

    async def test_the_same_number_in_two_books_is_two_answers(self, session, corpus) -> None:
        """Two books, one ``2.1``. Which is why the corpus-wide answer is a question
        about the whole library, and why ``doc_keys`` exists.

        Compared as a set: with a number the tie-break is title length, which says
        something about the heading and nothing about which book was meant.
        """
        assert sorted(_titles(await find_sections(session, "2.1"))) == [
            "2.1 Einstein's Theory of the Specific Heat",
            "2.1 Vorteks Çekirdeği",
        ]

    async def test_a_doc_key_leaves_one_of_them(self, session, corpus) -> None:
        found = await find_sections(session, "2.1", doc_keys=["tsoyuz-superfluidanlar"])
        assert _titles(found) == ["2.1 Vorteks Çekirdeği"]
        assert {match.display_id for match in found} == {"tsoyuz-superfluidanlar"}

    async def test_narrowing_to_the_wrong_book_finds_nothing(self, session, corpus) -> None:
        """The honest empty answer, so a caller can widen the search themselves."""
        assert await find_sections(session, "Debye", doc_keys=["tsoyuz-superfluidanlar"]) == []

    async def test_an_ingested_key_that_names_no_document_finds_nothing(
        self, session, corpus
    ) -> None:
        assert await find_sections(session, "Debye", doc_keys=["not-ingested"]) == []

    async def test_several_keys_are_a_union(self, session, corpus) -> None:
        found = await find_sections(
            session, "Abrikosov", doc_keys=["solid-state-basics", "tsoyuz-superfluidanlar"]
        )
        assert _titles(found) == ["3 Abrikosov Vortices"]

    async def test_a_blank_query_asks_about_nothing(self, session, corpus) -> None:
        assert await find_sections(session, "   ") == []


class TestAccentedTitles:
    """Titles keep their accents; only the ``doc_key`` is folded to ASCII.

    So matching has to work on the accented spelling a reader copied out of the
    book, not on the slug the corpus stores the document under.
    """

    async def test_an_ASCII_name_finds_an_accented_heading(self, session, corpus) -> None:
        found = await find_sections(session, "Abrikosov")
        assert _titles(found) == ["3 Abrikosov Vortices"]
        assert found[0].match_kind == "prefix"

    @pytest.mark.parametrize("term", ["Vorteks", "vorteks", "Giriş", "giriş", "Hız"])
    async def test_Turkish_characters_survive_the_round_trip(self, session, corpus, term) -> None:
        assert await find_sections(session, term) != []

    @pytest.mark.parametrize("term", ["şeker", "Vortek", "Hı", "şekerlem"])
    async def test_a_fragment_of_an_accented_word_matches_nothing(
        self, session, corpus, term
    ) -> None:
        assert await find_sections(session, term) == []

    async def test_the_doc_key_is_the_folded_slug_not_the_title(self, session, corpus) -> None:
        """``doc_slug`` drops the diacritics; the heading that matches has not."""
        found = await find_sections(session, "Şekerleme", doc_keys=["tsoyuz-superfluidanlar"])
        assert _titles(found) == ["4.4 Şekerleme Yöntemi"]
        assert found[0].display_id == "tsoyuz-superfluidanlar"


class TestSectionMatch:
    """What a match carries, since it is what the caller is shown."""

    async def test_the_key_is_what_the_vector_filter_filters_on(self, session, corpus) -> None:
        book = corpus["solid-state-basics"]
        found = await find_sections(session, "2.2")
        assert found[0].key == (book.id, 3)

    async def test_pages_are_said_the_way_a_reader_would(self, session, corpus) -> None:
        found = await find_sections(session, "2.2", doc_keys=["solid-state-basics"])
        assert [match.pages for match in found] == ["16-23"]

    async def test_the_unlocated_document_reports_no_pages(self, session, corpus) -> None:
        found = await find_sections(session, "Methods", doc_keys=[UNLOCATED_BOOK])
        assert [match.pages for match in found] == ["-"]
        assert found[0].page_start is None and found[0].page_end is None

    async def test_as_dict_names_the_document_and_the_heading(self, session, corpus) -> None:
        """``title`` is the book's; ``section`` is the heading inside it."""
        entry = (await find_sections(session, "2.2"))[0].as_dict()
        assert entry == {
            "doc_key": "solid-state-basics",
            "title": "Solid State Basics",
            "ordinal": 3,
            "section": "2.2 Debye's Calculation",
            "level": 3,
            "page_start": 16,
            "page_end": 23,
            "pages": "16-23",
            "source": "outline",
            "matched": "exact",
        }


class TestSectionsWithin:
    """Containment, which is decided by pages and only falls back to levels."""

    async def test_children_are_found_though_the_sources_disagree_about_level(
        self, session, corpus
    ) -> None:
        """The case that actually occurs, on *Solid State Basics*.

        The PDF's bookmarks put ``2.2 Debye's Calculation`` at level 3 and the
        markdown read of the same book read ``2.2.1 …`` as level 3 too, so the
        child is *not* deeper and a level comparison returns nothing at all.
        """
        book = corpus["solid-state-basics"]
        match = (await find_sections(session, "2.2"))[0]
        rows = await SectionRepository(session).list_for_paper(book.id)
        children = sections_within(match, rows)
        assert [row.title for row in children] == [
            "2.2.1 Periodic Debye Heat Capacity",
            "2.2.2 Anharmonic Correction",
        ]
        # The disagreement itself: the child's level equals the parent's, so a
        # level comparison would have skipped it and reported no children.
        assert match.level == 3
        assert children[0].level == match.level

    async def test_a_section_outside_the_page_range_is_not_a_child(
        self, session, corpus
    ) -> None:
        """``2.20 …`` at page 300 and ``3 Free Energy`` at page 31 are siblings of
        a section spanning 16-23, not parts of it — and the scan stops at the
        first one to leave, since everything after it belongs to something else.
        """
        book = corpus["solid-state-basics"]
        match = (await find_sections(session, "2.2"))[0]
        rows = await SectionRepository(session).list_for_paper(book.id)
        titles = [row.title for row in sections_within(match, rows)]
        assert "2.20 High-Temperature Limit" not in titles
        assert "3 Free Energy" not in titles

    async def test_the_section_itself_is_never_its_own_child(self, session, corpus) -> None:
        book = corpus["solid-state-basics"]
        match = (await find_sections(session, "2"))[0]
        rows = await SectionRepository(session).list_for_paper(book.id)
        assert match.ordinal not in {row.ordinal for row in sections_within(match, rows)}

    async def test_level_is_the_fallback_when_there_are_no_pages(
        self, session, corpus
    ) -> None:
        """A document with neither bookmarks nor page numbers has levels only."""
        match = (await find_sections(session, "Methods", doc_keys=[UNLOCATED_BOOK]))[0]
        rows = await SectionRepository(session).list_for_paper(
            corpus[UNLOCATED_BOOK].id
        )
        children = sections_within(match, rows)
        assert [row.title for row in children] == ["A.1 Setup and notation"]
        assert match.page_start is None
