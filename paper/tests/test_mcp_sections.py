"""The document map and the text under it: ``read_sections`` and ``read_section``.

A book is navigated in two steps, and the two tools are one step each:

* ``read_sections`` says **what exists** — every chapter and section, the pages it
  spans, how much of it is stored. A 305-page textbook has no other way to be
  opened, because a section is addressed by name and nothing else.
* ``read_section`` says **what it says**, by the two names a reader actually uses:
  the book's own numbering (``"2.2"``) and words from its title (``"Debye"``).

``read_sections`` was registered twice under one name for a while, and the SDK's
answer to a duplicate registration is a warning and the *first* registration — so
a defect in the copy it shadowed would have been invisible from the outside. The
surface itself is asserted in ``test_mcp.py``; what these tests cover is the
behaviour of the copy that survived.

The chunk count on the map is asserted by counting SQL, not by reading the
answer: a table of contents with a count column is the case where the naive
implementation (one query per row) looks identical in the response and costs a
few hundred queries on a real book.
"""

from __future__ import annotations

import contextlib
import re
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import event

from paper_app.domain.enums import ChunkKind
from paper_app.domain.models import ContentSource, PaperMetadata, TextChunk
from paper_app.services.sections import Section
from tests.test_mcp import acall, call

pytest_plugins = ("test_mcp",)

DOC_KEY = "solid-state-basics"

# The spine of the book these tests pretend to hold: three chapters, a nested
# subsection, and a chapter with nothing indexed inside it — all three shapes the
# map has to report honestly.
SECTIONS = [
    Section(0, "1 Overview", 1, page_start=1, page_end=33, source="outline"),
    Section(1, "2 Electrons in Metals", 1, page_start=34, page_end=120, source="outline"),
    Section(2, "2.1 Drude", 2, page_start=34, page_end=60, source="merged"),
    Section(3, "2.2 Debye's Calculation", 2, page_start=61, page_end=90, source="merged"),
    Section(
        4,
        "2.2.1 Periodic (Born-von Karman) Boundary",
        3,
        page_start=61,
        page_end=70,
        source="text",
    ),
    Section(5, "2.3 Phonons in Metals", 2, page_start=91, page_end=120, source="text"),
    Section(6, "3 Phonons", 1, page_start=121, page_end=200, source="text"),
]

# (section ordinal, pages, sentence). Ordinal 1 is deliberately absent: a chapter
# nobody has extracted any text from must read as zero chunks, not as missing.
CHUNKS = [
    (0, (4, 5), "The solid state is what a crystal leaves behind when it freezes."),
    (2, (34, 35), "Drude treated the conduction electrons as a classical gas."),
    (2, (36, 37), "The mean free path sets the conductivity of the metal."),
    (3, (61, 62), "Debye replaced the classical cutoff with the sound speed."),
    (3, (63, 64), "Integrating the phonon density of states to three modes per atom."),
    (3, (65, 66), "The result is the Debye temperature of the solid."),
    (4, (67, 68), "The boundary condition quantises the lattice in one dimension."),
    (5, (91, 92), "In a metal the same vibrations carry the current."),
    (6, (121, 122), "Phonons are the quantised vibrations of the lattice."),
]


async def seed_book(container, doc_key: str = DOC_KEY) -> str:  # noqa: ANN001
    """A textbook in the corpus: sections, page ranges, and stored chunks."""
    from paper_app.db.repositories import ChunkRepository, PaperRepository, SectionRepository

    async with container.session_factory() as session:
        paper = await PaperRepository(session).upsert(
            PaperMetadata.from_file(
                path=Path(f"{doc_key}.pdf"),
                doc_key=doc_key,
                kind="book",
                page_count=305,
                file_sha256=doc_key.ljust(64, "0"),
            )
        )
        await SectionRepository(session).replace_for_paper(paper.id, SECTIONS)
        await ChunkRepository(session).replace_for_paper(
            paper.id,
            [
                TextChunk(
                    ordinal=index,
                    text=text,
                    token_count=len(text.split()),
                    char_start=0,
                    char_end=len(text),
                    heading=SECTIONS[ordinal].title,
                    source=ContentSource.PDF_MINERU,
                    kind=ChunkKind.BODY,
                    page_start=page_start,
                    page_end=page_end,
                    section_ordinal=ordinal,
                )
                for index, (ordinal, (page_start, page_end), text) in enumerate(CHUNKS)
            ],
        )
        await session.commit()
    return doc_key


def seed_book_sync(container, doc_key: str = DOC_KEY) -> str:  # noqa: ANN201
    """Sync wrapper: these tests are sync, like most of ``test_mcp.py``."""
    import anyio

    return anyio.run(seed_book, container, doc_key)


@pytest.fixture
def book(container) -> str:  # noqa: ANN201
    return seed_book_sync(container)


@contextlib.contextmanager
def count_chunk_queries(container) -> Any:  # noqa: ANN401
    """Every statement the engine runs, so a test can count them.

    Counts statements that read the ``chunks`` table rather than calls to a
    repository method: the point of the assertion is what the database is asked,
    not which object asked it.
    """
    from paper_app.db.session import get_engine

    seen: list[str] = []
    engine = get_engine(container.settings.database).sync_engine

    def record(_conn, _cursor, statement, *_rest) -> None:  # noqa: ANN001, ANN202
        if re.search(r"\bfrom chunks\b", statement.lower()):
            seen.append(statement)

    event.listen(engine, "before_cursor_execute", record)
    try:
        yield seen
    finally:
        event.remove(engine, "before_cursor_execute", record)


class TestTheMap:
    def test_every_section_is_listed_with_its_pages(self, mcp_server, book: str) -> None:
        payload = call(mcp_server, "read_sections", doc_key=book)
        assert payload["ok"] is True
        assert payload["doc_key"] == DOC_KEY
        assert payload["title"] == "Solid State Basics"
        assert payload["kind"] == "book"
        assert payload["page_count"] == 305
        assert payload["shown"] == payload["total"] == len(SECTIONS)
        assert [row["ordinal"] for row in payload["sections"]] == [0, 1, 2, 3, 4, 5, 6]
        assert payload["sections"][3] == {
            "ordinal": 3,
            "title": "2.2 Debye's Calculation",
            "level": 2,
            "page_start": 61,
            "page_end": 90,
            "source": "merged",
            "chunks": 3,
        }

    def test_a_section_nobody_extracted_reads_as_zero_chunks(
        self, mcp_server, book: str
    ) -> None:
        """Chapter 2 has its own row, so its absence of text is a zero and not a
        hole in the map — a reader cannot tell those apart from the row alone."""
        rows = {row["ordinal"]: row for row in call(mcp_server, "read_sections", doc_key=book)["sections"]}
        assert rows[1]["chunks"] == 0

    def test_the_source_of_each_row_is_reported(self, mcp_server, book: str) -> None:
        """Two structural sources disagree often enough that a reader deserves to
        know which one claimed a section exists."""
        payload = call(mcp_server, "read_sections", doc_key=book)
        assert {row["ordinal"]: row["source"] for row in payload["sections"]} == {
            0: "outline",
            1: "outline",
            2: "merged",
            3: "merged",
            4: "text",
            5: "text",
            6: "text",
        }

    def test_max_level_drops_the_subsections(self, mcp_server, book: str) -> None:
        payload = call(mcp_server, "read_sections", doc_key=book, max_level=2)
        assert [row["level"] for row in payload["sections"]] == [1, 1, 2, 2, 2, 1]
        # `total` is what the level filter left, not the whole book: it is the
        # number a caller pages against.
        assert payload["total"] == 6

    def test_after_page_narrows_to_what_was_indexed(self, mcp_server, book: str) -> None:
        payload = call(mcp_server, "read_sections", doc_key=book, after_page=100)
        assert [row["title"] for row in payload["sections"]] == ["3 Phonons"]
        assert payload["shown"] == 1
        assert payload["total"] == len(SECTIONS)

    def test_a_document_that_is_not_in_the_corpus_is_a_clean_failure(
        self, mcp_server, book: str
    ) -> None:
        payload = call(mcp_server, "read_sections", doc_key="not-a-book")
        assert payload["ok"] is False
        assert "not-a-book" in payload["error"]
        assert "list_documents" in payload["error"]

    def test_the_chunk_column_costs_one_query_not_one_per_row(
        self, mcp_server, container, book: str
    ) -> None:
        """A book's table of contents is hundreds of rows and the count is on every
        one, so a per-row query is a few hundred queries for one payload."""
        with count_chunk_queries(container) as statements:
            payload = call(mcp_server, "read_sections", doc_key=book)
        assert payload["sections"][3]["chunks"] == 3
        assert len(statements) == 1, statements


class TestTheText:
    def test_a_section_is_found_by_its_number(self, mcp_server, book: str) -> None:
        payload = call(mcp_server, "read_section", doc_key=book, name="2.2")
        assert payload["ok"] is True
        assert payload["count"] == 1
        found = payload["sections"][0]
        assert found["section"] == "2.2 Debye's Calculation"
        assert found["ordinal"] == 3
        assert found["matched"] == "exact"
        assert found["pages"] == "61-90"
        # The text is the chunks pointing at the section, in document order, and
        # the pages are reported beside them rather than inferred from the range.
        assert [part["page_start"] for part in found["chunks"]] == [61, 63, 65]
        assert found["text"].startswith("Debye replaced the classical cutoff")
        assert found["truncated"] is False

    def test_a_section_is_found_by_a_word_from_its_title(
        self, mcp_server, book: str
    ) -> None:
        payload = call(mcp_server, "read_section", doc_key=book, name="Debye")
        assert payload["ok"] is True
        found = payload["sections"][0]
        assert found["section"] == "2.2 Debye's Calculation"
        # A word is a prefix match here rather than an exact one: the reader named
        # part of the title, and the tool says which kind it was so a widened
        # scope is visible.
        assert found["matched"] == "prefix"
        assert "Debye temperature" in found["text"]

    def test_matching_is_by_word_not_by_substring(self, mcp_server, book: str) -> None:
        """"Deb" must not come back for "Debye": a reader would be handed a chapter
        they did not ask for, with total confidence."""
        payload = call(mcp_server, "read_section", doc_key=book, name="Deb")
        assert payload["ok"] is False
        assert "read_sections" in payload["error"]

    def test_a_word_naming_several_sections_returns_them_ranked(
        self, mcp_server, book: str
    ) -> None:
        """Two sections here carry the word "Phonons". Neither is titled exactly that,
        so both are prefix matches and the order is decided by depth: the chapter
        the word names beats the section that qualifies it."""
        payload = call(mcp_server, "read_section", doc_key=book, name="Phonons")
        assert payload["count"] == 2
        assert [row["section"] for row in payload["sections"]] == [
            "3 Phonons",
            "2.3 Phonons in Metals",
        ]
        assert [row["level"] for row in payload["sections"]] == [1, 2]
        assert [row["matched"] for row in payload["sections"]] == ["prefix", "prefix"]
        assert payload["omitted"] == 0

    def test_a_section_with_no_stored_text_returns_empty_text(
        self, mcp_server, book: str
    ) -> None:
        """Chapter 2 is a heading the corpus has no text for. The match is still a
        match: the caller learns the section exists and that nothing was indexed
        inside it."""
        payload = call(mcp_server, "read_section", doc_key=book, name="Electrons")
        assert payload["count"] == 1
        assert payload["sections"][0]["section"] == "2 Electrons in Metals"
        assert payload["sections"][0]["chunks"] == []
        assert payload["sections"][0]["text"] == ""

    def test_subsections_are_listed_without_their_text(
        self, mcp_server, book: str
    ) -> None:
        payload = call(
            mcp_server, "read_section", doc_key=book, name="2.2", include_subsections=True
        )
        found = payload["sections"][0]
        # Decided by page range, not by level: 2.2.1 starts on 61, inside 61-90.
        assert [row["title"] for row in found["subsections"]] == [
            "2.2.1 Periodic (Born-von Karman) Boundary"
        ]
        assert "Boundary condition" not in found["text"]

    def test_subsections_are_left_out_unless_asked_for(
        self, mcp_server, book: str
    ) -> None:
        assert call(mcp_server, "read_section", doc_key=book, name="2.2")["sections"][0][
            "subsections"
        ] is None

    def test_an_unmatched_name_names_the_next_step(self, mcp_server, book: str) -> None:
        """The overwhelmingly likely mistake is a title that is not in the corpus,
        and the tool that fixes it is one call away."""
        payload = call(mcp_server, "read_section", doc_key=book, name="Fermi liquid")
        assert payload["ok"] is False
        assert "Fermi liquid" in payload["error"]
        assert DOC_KEY in payload["error"]
        assert f"read_sections('{DOC_KEY}')" in payload["error"]

    def test_a_document_that_is_not_in_the_corpus_is_a_clean_failure(
        self, mcp_server, book: str
    ) -> None:
        payload = call(mcp_server, "read_section", doc_key="not-a-book", name="2.2")
        assert payload["ok"] is False
        assert "not-a-book" in payload["error"]

    def test_the_name_is_resolved_inside_the_named_document(
        self, mcp_server, container, book: str
    ) -> None:
        """Two books can hold the same section number; the tool was given one book
        and must not answer with the other's section."""
        seed_book_sync(container, "another-book")
        payload = call(mcp_server, "read_section", doc_key="another-book", name="2.2")
        assert payload["ok"] is True
        assert payload["doc_key"] == "another-book"


class TestTheCharBudget:
    def test_the_budget_is_the_call_s_not_each_section_s(
        self, mcp_server, container, book: str
    ) -> None:
        """Truncating per match would return N stubs for a name matching N
        sections; the allowance is spent in ranked order instead."""
        seed = call(mcp_server, "read_section", doc_key=book, name="Drude")
        whole = len(seed["sections"][0]["text"])
        assert whole > 40

        payload = call(mcp_server, "read_section", doc_key=book, name="Drude", max_chars=40)
        assert len(payload["sections"][0]["text"]) <= 40
        assert payload["sections"][0]["truncated"] is True

    def test_the_budget_stops_before_the_next_section(
        self, mcp_server, container, book: str
    ) -> None:
        """A name matching two sections and a budget for one returns the one the
        caller most likely meant, whole, and says how many it left out — rather
        than two half-sections."""
        ranked = call(mcp_server, "read_section", doc_key=book, name="Phonons")
        first = len(ranked["sections"][0]["text"])

        payload = call(
            mcp_server, "read_section", doc_key=book, name="Phonons", max_chars=first
        )
        assert payload["count"] == 1
        assert payload["omitted"] == 1
        assert payload["sections"][0]["truncated"] is False
        assert payload["sections"][0]["text"] == ranked["sections"][0]["text"]

    def test_the_parts_and_the_text_are_the_same_words(self, mcp_server, book: str) -> None:
        """A per-part cut keeps the reported pages in step with the text beside
        them; a cut on the joined string would leave a part claiming a page that
        holds none of the words shown."""
        payload = call(mcp_server, "read_section", doc_key=book, name="Debye", max_chars=40)
        found = payload["sections"][0]
        assert found["chunks"]
        assert found["text"] == "\n\n".join(part["text"] for part in found["chunks"])
        assert all(len(part["text"]) <= 40 for part in found["chunks"])


class TestTheSurfaceIsUnique:
    """One name, one implementation.

    The SDK's answer to a duplicate registration is a warning and the first copy
    wins, so a second copy is invisible from the outside until the two disagree.
    """

    @pytest.mark.parametrize("name", ["read_sections", "read_section"])
    def test_a_section_tool_is_registered_once(self, mcp_server, name: str) -> None:
        import anyio

        names = [tool.name for tool in anyio.run(mcp_server.list_tools)]
        assert names.count(name) == 1

    def test_the_map_tool_names_the_text_tool(self, mcp_server) -> None:
        """The two are one flow and the client is never told; the description is
        how it learns which to call next."""
        import anyio

        tools = {tool.name: tool for tool in anyio.run(mcp_server.list_tools)}
        assert "read_section" in (tools["read_sections"].description or "")
        assert "read_sections" in (tools["read_section"].description or "")

    def test_both_are_annotated_read_only(self, mcp_server) -> None:
        """They read. `read_section` reads a whole section, so a client that
        prompts for anything large would be right to prompt for something else."""
        import anyio

        tools = {tool.name: tool for tool in anyio.run(mcp_server.list_tools)}
        for name in ("read_sections", "read_section"):
            assert tools[name].annotations.read_only_hint is True, name
            assert tools[name].annotations.destructive_hint is False, name

    def test_both_take_a_doc_key(self, mcp_server) -> None:
        """Not `paper`: the argument resolves an arXiv id too, but these tools are
        about documents, and `doc_key` is the honest name for that."""
        import anyio

        tools = {tool.name: tool for tool in anyio.run(mcp_server.list_tools)}
        for name in ("read_sections", "read_section"):
            properties = tools[name].input_schema["properties"]
            assert "doc_key" in properties, name
            assert "paper" not in properties, name


class TestThroughTheProtocolLayer:
    async def test_the_map_and_the_text_agree(self, mcp_server, book: str) -> None:
        """Same entry point the JSON-RPC handler uses, so registration and schema
        coercion are exercised rather than the bare Python functions."""
        toc = await acall(mcp_server, "read_sections", doc_key=book, max_level=1)
        assert [row["title"] for row in toc["sections"]] == [
            "1 Overview",
            "2 Electrons in Metals",
            "3 Phonons",
        ]
        body = await acall(mcp_server, "read_section", doc_key=book, name="3")
        assert body["sections"][0]["section"] == "3 Phonons"
        assert "Phonons are the quantised vibrations" in body["sections"][0]["text"]