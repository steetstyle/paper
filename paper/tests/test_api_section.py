"""`section` and `doc_key` on the semantic search route.

Two things are being tested, and the second is the one that matters. First, that
a name a reader uses ("Debye", "2.2") reaches the vector store as the
``(paper_id, ordinal)`` pairs the section table is keyed by. Second, that a name
which matches nothing is an **error** rather than an empty result set: a section
that does not discuss the question and a name that names nothing produce the same
zero hits, so a route that let the second through would be answering a question
nobody asked — confidently, and with nothing to show for it.

The section tables are written straight to the ORM. A book whose pages were
ingested would also have to be extracted, chunked and embedded, and the
pipeline's construction of the table from a document is tested there; what is
under test here is the route resolving a name against it, so the table is the
input.
"""

from __future__ import annotations

import pytest

# The app-level `container` and `client` fixtures live in test_api.py; loading
# that module as a plugin reuses them instead of rebuilding the app and its
# fakes here, as test_api_device.py does for the device tests.
pytest_plugins = ("test_api",)

ARXIV_ID = "1706.03762"
BOOK_KEY = "solid-state-basics"
BOOK_TITLE = "Solid State Basics"

#: ``(ordinal, heading, level, page_start, page_end)`` — a book with the awkward
#: numbering a search has to survive: a section whose number is a prefix of its
#: neighbour's, and a subsection one component deeper.
BOOK_SECTIONS = [
    (0, "1 Introduction", 1, 3, 9),
    (1, "2 Specific Heat of Solids", 1, 10, 23),
    (2, "2.2 Debye's Calculation", 2, 16, 23),
    (3, "2.20 High-Temperature Limit", 2, 300, 310),
    (4, "2.2.1 Periodic Debye Heat Capacity", 3, 16, 19),
]

#: A second book, to give ``doc_key`` something to choose between.
OTHER_SECTIONS = [(0, "2.2 Drude Metals", 2, 10, 20)]

#: The headings of the fake paper — an HTML rendering, so it has no pages at all.
ARXIV_SECTIONS = [
    (0, "1 Introduction", 1, None, None),
    (1, "2 Model Architecture", 1, None, None),
    (2, "2.1 Attention Complexity", 2, None, None),
]


async def _seed_document(container, doc_key: str, title: str, rows, *, kind="book") -> str:  # noqa: ANN001
    """Write a document and its table of contents; return its internal id.

    Replaces the section rows rather than adding to them, so the table these tests
    assert against is the one written here rather than whatever an earlier run of
    the pipeline left behind.
    """
    from sqlalchemy import delete

    from paper_app.db.models import DocumentSection, Paper
    from paper_app.db.repositories import PaperRepository

    async with container.session_factory() as session:
        paper = await PaperRepository(session).resolve(doc_key)
        if paper is None:
            paper = Paper(
                doc_key=doc_key, kind=kind, title=title, abstract="", page_count=310
            )
            session.add(paper)
            await session.flush()
        await session.execute(
            delete(DocumentSection).where(DocumentSection.paper_id == paper.id)
        )
        session.add_all(
            [
                DocumentSection(
                    paper_id=paper.id,
                    ordinal=ordinal,
                    title=heading,
                    level=level,
                    page_start=page_start,
                    page_end=page_end,
                    source="outline",
                )
                for ordinal, heading, level, page_start, page_end in rows
            ]
        )
        await session.commit()
        return paper.id


@pytest.fixture
def sections_requested(monkeypatch):
    """Every ``sections`` the route hands to the search service.

    Raw rather than resolved: ``None`` must reach the service intact, because
    "no section filter" is the default, and turning it into an empty list would be
    a policy change rather than plumbing — an empty list is a filter that matches
    *nothing*.
    """
    from paper_app.services.semantic_search import SemanticSearchService

    real = SemanticSearchService.search
    seen: list[list[tuple[str, int]] | None] = []

    async def spy(self, query, **kwargs):  # noqa: ANN001, ANN003, ANN202
        seen.append(kwargs.get("sections"))
        return await real(self, query, **kwargs)

    monkeypatch.setattr(SemanticSearchService, "search", spy)
    return seen


@pytest.fixture(autouse=True)
async def corpus(client, container) -> dict[str, str]:
    """The fake paper, ingested through the API, with a book beside it.

    The paper is a real corpus row by the time it is searched — created by the
    pipeline, with chunks and vectors — and its sections are the headings its
    markdown carries.
    """
    await client.post("/api/v1/ingest", json={"arxiv_id": ARXIV_ID}, params={"wait": True})
    paper_id = await _seed_document(
        container, ARXIV_ID, "Attention Is All You Need", ARXIV_SECTIONS, kind="paper"
    )
    book_id = await _seed_document(container, BOOK_KEY, BOOK_TITLE, BOOK_SECTIONS)
    return {"paper": paper_id, "book": book_id}


async def _search(client, **body):  # noqa: ANN003, ANN202
    return await client.post(
        "/api/v1/search/semantic",
        json={"query": "specific heat of a solid", "top_k": 3, **body},
    )


class TestResolvingASectionName:
    async def test_a_word_resolves_to_the_sections_that_name_it(
        self, client, corpus, sections_requested
    ) -> None:
        response = await _search(client, section="Debye", doc_key=BOOK_KEY)
        assert response.status_code == 200
        # Forwarded as the pairs the section table is keyed by — an ordinal alone
        # would mean a different section in every other document.
        assert sections_requested == [[(corpus["book"], 2), (corpus["book"], 4)]]

    async def test_a_number_finds_that_section_and_not_its_neighbours(
        self, client, corpus, sections_requested
    ) -> None:
        """The book's own numbering finds "2.2" — not "2.20", and not the "2.2.1"
        nested inside it."""
        response = await _search(client, section="2.2", doc_key=BOOK_KEY)
        assert response.status_code == 200
        assert sections_requested == [[(corpus["book"], 2)]]

    async def test_the_response_shows_what_the_name_widened_to(
        self, client, sections_requested
    ) -> None:
        """A caller has to be able to see which sections answered.

        Two rows for one word: the section the word names first, then the
        subsection that merely contains it. A set would not say which one the
        search was really about.
        """
        response = await _search(client, section="Debye", doc_key=BOOK_KEY)
        body = response.json()
        assert [row["section"] for row in body["sections"]] == [
            "2.2 Debye's Calculation",
            "2.2.1 Periodic Debye Heat Capacity",
        ]
        assert body["sections"][0] == {
            "doc_key": BOOK_KEY,
            "title": BOOK_TITLE,
            "ordinal": 2,
            "section": "2.2 Debye's Calculation",
            "level": 2,
            "page_start": 16,
            "page_end": 23,
            "pages": "16-23",
            "source": "outline",
            "matched": "prefix",
        }
        # The count is in `scope` too, so a caller can see how narrow the search
        # really was without parsing the rows.
        assert body["scope"]["sections"] == 2

    async def test_a_paper_without_pages_is_addressable_by_its_headings(
        self, client, corpus, sections_requested
    ) -> None:
        """An HTML rendering has no page numbers, and ``pages`` says so rather than
        inventing one."""
        response = await _search(client, section="2.1", doc_key=ARXIV_ID)
        assert response.status_code == 200
        assert sections_requested == [[(corpus["paper"], 2)]]
        found = response.json()["sections"][0]
        assert found["doc_key"] == ARXIV_ID
        assert found["section"] == "2.1 Attention Complexity"
        assert found["pages"] == "-"

    async def test_without_a_doc_key_the_whole_corpus_is_read(
        self, client, container, corpus, sections_requested
    ) -> None:
        # A number two books hold is exactly what `doc_key` is for: this answer is
        # correct and rarely what was meant.
        other = await _seed_document(container, "another-book", "Another Book", OTHER_SECTIONS)
        response = await _search(client, section="2.2")
        assert response.status_code == 200
        assert sections_requested == [[(other, 0), (corpus["book"], 2)]]
        assert len(response.json()["sections"]) == 2


class TestDocKeyNarrows:
    async def test_it_decides_which_document_a_name_is_read_in(
        self, client, container, corpus, sections_requested
    ) -> None:
        """Two books hold a "2.2"; naming one must not answer with the other's."""
        other = await _seed_document(container, "another-book", "Another Book", OTHER_SECTIONS)
        assert (await _search(client, section="2.2", doc_key=BOOK_KEY)).status_code == 200
        assert sections_requested == [[(corpus["book"], 2)]]

        assert (await _search(client, section="2.2", doc_key="another-book")).status_code == 200
        assert sections_requested[-1] == [(other, 0)]

    async def test_an_arXiv_id_is_a_handle_too(self, client, corpus, sections_requested) -> None:
        response = await _search(client, section="2.1", doc_key=ARXIV_ID)
        assert response.status_code == 200
        assert sections_requested == [[(corpus["paper"], 2)]]

    async def test_an_internal_id_is_a_handle_too(
        self, client, corpus, sections_requested
    ) -> None:
        """One lookup for every spelling a paper can arrive under, so a caller
        holding an id from a previous response is not stuck."""
        response = await _search(client, section="2.1", doc_key=corpus["paper"])
        assert response.status_code == 200
        assert sections_requested == [[(corpus["paper"], 2)]]

    async def test_a_document_that_is_not_ingested_is_404(
        self, client, sections_requested
    ) -> None:
        """404, not 422: a corpus that also holds books cannot tell a malformed id
        from a document that simply is not here."""
        response = await _search(client, section="2.2", doc_key="not-ingested")
        assert response.status_code == 404
        assert "not-ingested" in response.text
        assert sections_requested == []

    async def test_it_alone_is_422_rather_than_a_field_silently_dropped(
        self, client, sections_requested
    ) -> None:
        """``doc_key`` says which document a *section name* belongs to. On its own
        it does nothing, and a field accepted and then dropped is worse than an
        absent one: to search one document outright, say ``paper_ids``."""
        response = await _search(client, doc_key=BOOK_KEY)
        assert response.status_code == 422
        assert "paper_ids" in response.text
        assert sections_requested == []

    @pytest.mark.parametrize("doc_key", ["", "  "])
    async def test_a_blank_doc_key_is_422_not_the_whole_corpus(
        self, client, sections_requested, doc_key
    ) -> None:
        response = await _search(client, section="Debye", doc_key=doc_key)
        assert response.status_code == 422
        assert sections_requested == []


class TestNoSection:
    async def test_the_field_is_optional(self, client, sections_requested) -> None:
        response = await _search(client)
        assert response.status_code == 200
        assert response.json()["hits"]
        # `None` reaches the service, not `[]`: an empty list is a filter that
        # matches nothing, the opposite of what was asked for.
        assert sections_requested == [None]
        assert response.json()["sections"] == []

    async def test_a_section_and_papers_both_reach_the_service(
        self, client, corpus, sections_requested
    ) -> None:
        """Both scopes are the caller's and they compose: the store intersects them
        rather than letting whichever arrived last win.

        The section is the book's and the papers are the paper's, so nothing can
        match both — which is the intersection being honest, and the reason the
        answer is zero hits rather than hits from one scope or the other.
        """
        response = await _search(
            client, section="2.2", doc_key=BOOK_KEY, paper_ids=[ARXIV_ID]
        )
        assert response.status_code == 200
        assert sections_requested == [[(corpus["book"], 2)]]
        assert response.json()["hits"] == []
        assert response.json()["scope"]["sections"] == 1
        assert response.json()["scope"]["paper_ids"] == 1


class TestAnUnmatchedName:
    async def test_it_is_422_saying_what_was_searched_for(
        self, client, sections_requested
    ) -> None:
        response = await _search(client, section="Fermi liquid", doc_key=BOOK_KEY)
        assert response.status_code == 422
        assert "Fermi liquid" in response.text
        assert BOOK_KEY in response.text

    async def test_the_error_names_the_next_call(self, client) -> None:
        """The overwhelmingly likely mistake is a title that is not in the corpus,
        and the way to see the titles is one call away."""
        response = await _search(client, section="Fermi liquid", doc_key=BOOK_KEY)
        assert f"read_sections({BOOK_KEY!r})" in response.text

    async def test_a_name_that_exists_in_another_book_is_unmatched_here(
        self, client, container, sections_requested
    ) -> None:
        """Narrowing has to be able to fail: the section exists, but not here."""
        await _seed_document(container, "another-book", "Another Book", OTHER_SECTIONS)
        response = await _search(client, section="Drude", doc_key=BOOK_KEY)
        assert response.status_code == 422
        assert "Drude" in response.text

    async def test_nothing_is_searched(self, client, sections_requested) -> None:
        """Refused before the embedding, so a typo costs no model and no vectors."""
        response = await _search(client, section="Fermi liquid")
        assert response.status_code == 422
        assert sections_requested == []

    @pytest.mark.parametrize("section", ["", "   "])
    async def test_a_blank_section_is_422_not_the_whole_corpus(
        self, client, sections_requested, section
    ) -> None:
        """Blank is not "no opinion": omitting the field is how a caller asks for
        every section, so a blank value must not quietly mean the same."""
        response = await _search(client, section=section)
        assert response.status_code == 422
        assert sections_requested == []

    async def test_a_name_matching_nothing_in_the_corpus_says_so(self, client) -> None:
        response = await _search(client, section="Fermi liquid")
        assert response.status_code == 422
        assert "in the corpus" in response.text


class TestSectionSchemaDocs:
    async def test_openapi_documents_both_fields(self, client) -> None:
        properties = (
            (await client.get("/openapi.json")).json()["components"]["schemas"][
                "SemanticSearchRequest"
            ]["properties"]
        )
        assert "2.2" in properties["section"]["description"]
        assert "422" in properties["section"]["description"]
        assert properties["section"].get("default") is None
        assert "paper_ids" in properties["doc_key"]["description"]
        assert properties["doc_key"].get("default") is None

    async def test_the_response_says_which_sections_it_matched(self, client) -> None:
        schemas = (await client.get("/openapi.json")).json()["components"]["schemas"]
        assert schemas["SemanticSearchResponse"]["properties"]["sections"]["type"] == "array"
        assert "matched" in schemas["SectionOut"]["properties"]
