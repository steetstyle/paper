"""Local documents: identity, page ranges, page provenance, structure.

The corpus stopped being arXiv-shaped partway through this project's life, and
these are the tests for the parts that changed because of it. They are written
against measurements rather than examples:

* A third of real textbooks ship without PDF bookmarks (341 entries, 174, then
  **0**), so structure has to survive without them.
* MinerU writes every heading at the same ATX depth, so ``#`` count says nothing
  about hierarchy and the numbering has to be read instead.
* MinerU's ``page_idx`` is relative to the extracted slice, not to the book:
  ``mineru -s 100 -e 104`` returned indexes 0..4 for pages 101..105.
* Embedded ``/Title`` is a LaTeX job name on one of the three books tested
  (``pethick.dvi``), so it cannot be trusted as a title.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from paper_app.clients.content.pdf_info import OutlineNode, with_page_ranges
from paper_app.domain.models import MineruOptions, PageRange, PaperMetadata, doc_slug
from paper_app.services.page_map import build_page_map, normalise
from paper_app.services.sections import (
    Section,
    _section_key,
    build_sections,
    infer_level,
    parse_markdown_headings,
)

# --------------------------------------------------------------------- identity


class TestPaperMetadataFromFile:
    def test_a_latex_job_name_is_not_a_title(self) -> None:
        """``/Title`` was ``pethick.dvi`` on a real 578-page textbook."""
        metadata = PaperMetadata.from_file(
            path=Path("icourse.pdf"), raw={"/Title": "pethick.dvi"}
        )
        assert metadata.title != "pethick.dvi"
        assert metadata.title == "Icourse"

    def test_the_slash_spelled_key_is_the_one_a_pdf_uses(self) -> None:
        """A PDF's ``/Info`` dictionary spells it ``/Title``. Looking up a plain
        ``title`` instead meant every PDF silently fell through to a title made
        from its filename, and the tests passed because they passed the wrong key.
        """
        metadata = PaperMetadata.from_file(
            path=Path("annett.pdf"), raw={"/Title": "Superconductivity, superfluids, and condensates"}
        )
        assert metadata.title == "Superconductivity, superfluids, and condensates"

    def test_a_filename_with_a_site_suffix_does_not_win_over_a_real_title(self) -> None:
        """The real case: a scraped file named ``...-685m5mne1x.pdf`` whose title
        is a clean one. The stem only *resembles* the title; it is not it."""
        path = Path("superconductivity-superfluids-and-condensates-685m5mne1x.pdf")
        metadata = PaperMetadata.from_file(
            path=path, raw={"/Title": "Superconductivity, superfluids, and condensates"}
        )
        assert metadata.title == "Superconductivity, superfluids, and condensates"
        assert "685m5mne1x" not in metadata.title

    def test_a_real_embedded_title_is_used(self) -> None:
        metadata = PaperMetadata.from_file(
            path=Path("girvin.pdf"), raw={"title": "Modern Condensed Matter Physics"}
        )
        assert metadata.title == "Modern Condensed Matter Physics"

    @pytest.mark.parametrize(
        "embedded", ["lecture.tex", "Thesis.docx", "main.aux", "book.out"]
    )
    def test_source_extensions_are_always_rejected(self, embedded: str) -> None:
        metadata = PaperMetadata.from_file(path=Path("notes.pdf"), raw={"title": embedded})
        assert metadata.title == "Notes"

    def test_a_title_that_only_restates_the_filename_is_rejected(self) -> None:
        """A build tool that writes the job name produces the stem either way."""
        metadata = PaperMetadata.from_file(
            path=Path("solid_state_basics.pdf"), raw={"title": "solid_state_basics"}
        )
        assert metadata.title == "Solid State Basics"

    def test_explicit_title_wins_over_everything(self) -> None:
        metadata = PaperMetadata.from_file(
            path=Path("icourse.pdf"), title="Superconductivity", raw={"title": "pethick.dvi"}
        )
        assert metadata.title == "Superconductivity"

    def test_a_local_document_has_no_arxiv_identity(self) -> None:
        metadata = PaperMetadata.from_file(path=Path("book.pdf"))
        assert metadata.arxiv_id is None
        assert metadata.versioned_id is None
        assert metadata.latest_versioned_id == metadata.doc_key
        assert metadata.display_id == metadata.doc_key
        assert metadata.kind == "book"

    def test_a_turkish_filename_still_yields_a_usable_handle(self) -> None:
        metadata = PaperMetadata.from_file(path=Path("Fizik 101 Dersi 3.pdf"))
        assert metadata.doc_key == "fizik-101-dersi-3"
        assert metadata.title == "Fizik 101 Dersi 3"

    def test_embedded_metadata_is_kept_verbatim(self) -> None:
        """The decision about what a title is needs the evidence it was based on."""
        metadata = PaperMetadata.from_file(
            path=Path("icourse.pdf"), raw={"title": "pethick.dvi", "/Producer": "Acrobat"}
        )
        assert metadata.raw["title"] == "pethick.dvi"


class TestDocKeyInvariant:
    def test_a_paper_keys_itself_by_its_arxiv_id(self) -> None:
        """The invariant, enforced so no caller can get it wrong."""
        metadata = PaperMetadata(
            arxiv_id="2104.00001",
            versioned_id="2104.00001v5",
            version=5,
            title="Attention Is All You Need",
            abstract="",
        )
        assert metadata.doc_key == "2104.00001"

    def test_metadata_with_no_identity_at_all_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="arxiv_id or doc_key"):
            PaperMetadata(
                arxiv_id=None,
                versioned_id=None,
                version=None,
                title="Nameless",
                abstract="",
            )

    def test_slug_folds_accents_and_punctuation(self) -> None:
        assert doc_slug("Péthick & Girvin — Süper!)") == "pethick-girvin-super"


class TestUpsertingADocumentTwice:
    """Re-ingesting a local file must refresh the row, not collide with it.

    Found the hard way. The existing-row lookup was by ``arxiv_id``; a local
    document has none, so the upsert inserted and hit the ``doc_key`` unique
    constraint — on the *second* ``--force`` run of a book, surfacing as a
    ``PendingRollbackError`` wrapping the real violation. Testing the pipeline end
    to end would have needed a PDF fixture and a full service; the defect is
    entirely in which row is looked up, so it is tested where it lives.
    """

    @staticmethod
    def _metadata(**overrides: object) -> PaperMetadata:
        base = {
            "path": Path("annett.pdf"),
            "kind": "book",
            "doc_key": "annett",
            "page_count": 140,
            "file_sha256": "a" * 64,
        }
        base.update(overrides)
        return PaperMetadata.from_file(**base)  # type: ignore[arg-type]

    async def test_the_same_document_upserts_to_one_row(self, session) -> None:
        from paper_app.db.repositories import PaperRepository  # noqa: PLC0415

        repo = PaperRepository(session)
        first = await repo.upsert(self._metadata())
        second = await repo.upsert(self._metadata(title="Superconductivity"))
        await session.commit()
        assert first.id == second.id

    async def test_a_refreshed_title_replaces_the_old_one(self, session) -> None:
        from paper_app.db.repositories import PaperRepository  # noqa: PLC0415

        repo = PaperRepository(session)
        await repo.upsert(self._metadata())
        await session.commit()
        refreshed = await repo.upsert(self._metadata(title="Superconductivity"))
        await session.commit()
        assert refreshed.title == "Superconductivity"

    async def test_a_document_is_found_by_its_doc_key(self, session) -> None:
        from paper_app.db.repositories import PaperRepository  # noqa: PLC0415

        repo = PaperRepository(session)
        await repo.upsert(self._metadata())
        await session.commit()
        found = await repo.get_by_doc_key("annett")
        assert found is not None and found.arxiv_id is None

    async def test_list_documents_sees_only_local_files(self, session) -> None:
        from paper_app.db.repositories import PaperRepository  # noqa: PLC0415

        repo = PaperRepository(session)
        await repo.upsert(self._metadata())
        await repo.upsert(
            PaperMetadata(
                arxiv_id="2104.00001",
                versioned_id="2104.00001v5",
                version=5,
                title="Attention Is All You Need",
                abstract="",
            )
        )
        await session.commit()
        documents = await repo.list_documents()
        assert [d.doc_key for d in documents] == ["annett"]
        assert await repo.list_documents(kind="book") != []
        assert await repo.list_documents(kind="thesis") == []


# ------------------------------------------------------------------ page ranges


class TestPageRange:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [("40-52", (40, 52)), ("40", (40, 40)), ("-52", (1, 52))],
    )
    def test_reads_every_accepted_form(self, text: str, expected: tuple[int, int]) -> None:
        parsed = PageRange.parse(text)
        assert parsed is not None
        assert (parsed.start, parsed.end) == expected

    def test_an_empty_value_means_the_whole_document(self) -> None:
        assert PageRange.parse("") is None
        assert PageRange.parse("   ") is None

    def test_an_open_end_is_a_sentinel_until_clamped(self) -> None:
        parsed = PageRange.parse("700-")
        assert parsed is not None
        assert parsed.end == PageRange.UNBOUNDED
        clamped = parsed.clamp(721)
        assert clamped == PageRange(700, 721)

    @pytest.mark.parametrize("text", ["0", "0-5", "-0", "9-4", "abc", "1-2-3", "--"])
    def test_nonsense_is_refused_at_the_argument(self, text: str) -> None:
        """Failing here beats an empty MinerU run and an unclear error later."""
        with pytest.raises(ValueError):
            PageRange.parse(text)

    def test_a_range_past_the_end_of_the_document_is_none(self) -> None:
        parsed = PageRange.parse("900-950")
        assert parsed is not None
        assert parsed.clamp(721) is None

    def test_count_and_text_are_the_shape_a_row_stores(self) -> None:
        span = PageRange(20, 44)
        assert span.count == 25
        assert str(span) == "20-44"


# ------------------------------------------------------------- page provenance


#: Headings the section tests use, with the block text that follows each. A
#: section is only kept if its text can be located in the blocks, so a fixture whose
#: blocks do not contain the heading produces no sections at all — which is the rule
#: working, not a fixture being decorative.
HEADING_BLOCKS = [
    {"type": "text", "page_idx": index, "text": f"{title} and the body that follows it."}
    for index, title in enumerate(
        [
            "2.3.1 The semi-classical distribution",
            "Free expansion",
            "2.4 Thermodynamic quantities",
            "2.4.1 Condensed phase",
            "2.4.2 Normal phase",
            "2.4.3 Specific heat close to Tc",
        ]
    )
]

BLOCKS = [
    {"type": "text", "page_idx": 0, "text": "Single-particle quantum states of a gas."},
    {"type": "text", "page_idx": 1, "text": "The widths depend only on the temperature."},
    {"type": "image", "page_idx": 1, "img_path": "images/a.jpg", "image_caption": []},
    {"type": "equation", "page_idx": 2, "text": "$$ E = p^2 / 2m $$"},
    {"type": "text", "page_idx": 2, "text": "Above the critical temperature the gas is normal."},
]


class TestPageMap:
    def test_pages_are_offset_by_the_extracted_slice(self) -> None:
        """Measured: ``mineru -s 100 -e 104`` reports ``page_idx`` 0..4."""
        page_map = build_page_map(BLOCKS, page_offset=101, total_pages=305)
        page, _ = page_map.page_for("Single-particle quantum states of a gas.")
        assert page == 101
        page, _ = page_map.page_for("Above the critical temperature the gas is normal.")
        assert page == 103

    def test_the_whole_document_offsets_by_one(self) -> None:
        page_map = build_page_map(BLOCKS)
        page, _ = page_map.page_for("Single-particle quantum states of a gas.")
        assert page == 1

    def test_the_cursor_only_moves_forward(self) -> None:
        page_map = build_page_map(BLOCKS)
        first, cursor = page_map.page_for("Single-particle quantum states of a gas.")
        second, cursor = page_map.page_for("The widths depend only on the temperature.", cursor)
        assert (first, second) == (1, 2)
        assert cursor > 0

    def test_a_chunk_that_starts_in_markup_is_still_found(self) -> None:
        """11 of 13 chunks matched from their first window; both misses opened
        with an image reference or a bare chapter number."""
        page_map = build_page_map(BLOCKS, page_offset=20)
        page, _ = page_map.page_for(
            "![](images/a.jpg)\n\nThe widths depend only on the temperature."
        )
        assert page == 21

    def test_text_that_is_not_there_yields_no_page_rather_than_a_guess(self) -> None:
        page_map = build_page_map(BLOCKS)
        page, cursor = page_map.page_for("A passage from a different book entirely.")
        assert page is None
        assert cursor == 0

    def test_no_blocks_means_no_map(self) -> None:
        assert not build_page_map([]).usable
        assert not build_page_map([{"type": "image", "img_path": "x.jpg"}]).usable

    def test_a_block_without_a_page_is_skipped_not_assigned_page_one(self) -> None:
        page_map = build_page_map([{"type": "text", "text": "No page here"}])
        assert not page_map.usable

    def test_normalise_folds_markup_but_not_words(self) -> None:
        # LaTeX structure is stripped from both sides, which is what makes a
        # markdown chunk match a block that wrote the same formula differently.
        assert normalise("# Heading\n\n$$\\frac{a}{b}$$ text") == "Heading fracab text"
        assert normalise("![a caption](images/x.png) rest") == "rest"
        assert normalise("[label](http://x) rest") == "label rest"


# ------------------------------------------------------------------ structure


class TestSectionLevels:
    @pytest.mark.parametrize(
        ("title", "level"),
        [
            ("2.4 Thermodynamic quantities", 2),
            ("2.3.1 The semi-classical distribution", 3),
            ("2.4.1.2 Deeper still", 4),
            ("Chapter 7 Title", 1),
            ("Appendix B Proofs", 1),
        ],
    )
    def test_depth_comes_from_the_numbering(self, title: str, level: int) -> None:
        assert infer_level(title)[0] == level

    @pytest.mark.parametrize(
        ("title", "level"),
        [
            ("(2.2) Debye Theory I", 2),
            ("(2.1) Einstein Solid", 2),
            ("2.2.1 Periodic (Born-von Karman) Boundary", 3),
        ],
    )
    def test_parenthesised_numbering_is_still_numbering(self, title: str, level: int) -> None:
        """A physics textbook writes section titles as ``(2.2) Title``. Read as
        unnumbered, a run of them chained four levels deep inside one appendix."""
        assert infer_level(title)[0] == level

    def test_an_unnumbered_heading_inherits_and_says_so(self) -> None:
        level, numbered = infer_level("Free expansion", fallback=3)
        assert (level, numbered) == (3, False)

    def test_every_heading_is_read_at_the_same_atx_depth(self) -> None:
        """The reason numbering is read at all: MinerU emits ``#`` for all."""
        markdown = "# 2.4 Thermodynamic quantities\n\ntext\n\n# 2.4.1 Condensed phase\n"
        levels = [h.level for h in parse_markdown_headings(markdown)]
        assert levels == [2, 3]


class TestBuildSections:
    OUTLINE = (
        OutlineNode(title="1 Overview", depth=0, page=10),
        OutlineNode(title="1.1 Definition", depth=1, page=10),
        OutlineNode(title="2 Electrons in Metals", depth=0, page=34),
        OutlineNode(title="2.1 Drude", depth=1, page=34),
    )

    def test_the_outline_is_the_spine_when_it_exists(self) -> None:
        sections = build_sections(outline=self.OUTLINE, document_pages=305)
        assert [s.title for s in sections] == [n.title for n in self.OUTLINE]
        assert all(s.source == "outline" for s in sections)
        # Tree depth is 0-based; a reader counts from 1.
        assert [s.level for s in sections] == [1, 2, 1, 2]

    def test_a_heading_the_outline_already_knows_is_not_repeated(self) -> None:
        markdown = "# 1 Overview\n\nbody\n\n# 2.1 Drude\n\nbody\n"
        sections = build_sections(outline=self.OUTLINE, markdown=markdown, document_pages=305)
        assert len(sections) == len(self.OUTLINE)

    def test_the_markdown_is_the_spine_when_there_is_no_outline(self) -> None:
        """One real book in three ships without bookmarks."""
        markdown = (
            "# 2.3.1 The semi-classical distribution\n\nand the body that follows it.\n\n"
            "# 2.4 Thermodynamic quantities\n\nand the body that follows it.\n\n"
            "# 2.4.1 Condensed phase\n\nand the body that follows it.\n"
        )
        sections = build_sections(
            markdown=markdown,
            page_map=build_page_map(HEADING_BLOCKS),
            document_pages=578,
        )
        assert [s.title for s in sections] == [
            "2.3.1 The semi-classical distribution",
            "2.4 Thermodynamic quantities",
            "2.4.1 Condensed phase",
        ]
        assert [s.level for s in sections] == [3, 2, 3]
        assert all(s.source == "markdown" for s in sections)

    def test_a_section_ends_where_the_next_one_at_its_level_starts(self) -> None:
        sections = build_sections(outline=self.OUTLINE, document_pages=305)
        by_title = {s.title: s for s in sections}
        # A chapter spans its own subsections rather than closing at the first one.
        assert by_title["1 Overview"].page_end == 34
        assert by_title["1.1 Definition"].page_end == 34
        # The last one has no successor, so it runs to the end of the book.
        assert by_title["2 Electrons in Metals"].page_end == 305

    def test_the_last_section_runs_to_the_end_of_the_document(self) -> None:
        """A bookmark tree only ever says where sections start."""
        sections = build_sections(outline=self.OUTLINE, document_pages=305)
        assert sections[-1].page_end == 305

    def test_without_a_page_count_the_last_range_stays_open(self) -> None:
        sections = build_sections(outline=self.OUTLINE)
        assert sections[-1].page_end is None

    def test_contents_lines_are_not_sections(self) -> None:
        """A contents page is a list of headings and the extractor reads it as
        headings. Measured: 3 real matches on one book, 0 false positives on the
        other three."""
        # Blocks carrying the four lines, so all four can be located. The two
        # contents lines and their real counterparts restate each other, which is
        # exactly what the rule looks for.
        blocks = [
            {"type": "text", "page_idx": index, "text": f"{line} and the body."}
            for index, line in enumerate(
                [
                    "1 Superconductivity 2",
                    "2 The Ginzburg-Landau model 31",
                    "1 Superconductivity",
                    "2 The Ginzburg-Landau model",
                ]
            )
        ]
        markdown = "\n\n".join(f"# {line}\n\nand the body." for line in (
            "1 Superconductivity 2",
            "2 The Ginzburg-Landau model 31",
            "1 Superconductivity",
            "2 The Ginzburg-Landau model",
        ))
        sections = build_sections(markdown=markdown, page_map=build_page_map(blocks))
        assert [s.title for s in sections] == [
            "1 Superconductivity",
            "2 The Ginzburg-Landau model",
        ]

    def test_a_section_that_merely_ends_in_a_digit_survives(self) -> None:
        blocks = [
            {"type": "text", "page_idx": 0, "text": "Chapter 1 and the body that follows it."},
            {"type": "text", "page_idx": 1, "text": "2 Pippin 2001 and the body that follows it."},
        ]
        markdown = (
            "# Chapter 1\n\nand the body that follows it.\n\n"
            "# 2 Pippin 2001\n\nand the body that follows it.\n"
        )
        sections = build_sections(markdown=markdown, page_map=build_page_map(blocks))
        assert [s.title for s in sections] == ["Chapter 1", "2 Pippin 2001"]

    def test_a_book_with_neither_source_has_no_sections(self) -> None:
        assert build_sections(markdown="just prose, no headings") == []

    def test_attached_headings_do_not_nest_inside_each_other(self) -> None:
        """Hosts come from the outline only, so a run of markdown headings cannot
        chain — a shape no book has."""
        blocks = [
            {"type": "text", "page_idx": index, "text": f"2.3.{index + 1} Subsection "
             f"{index + 1} and then the body of it."}
            for index in range(4)
        ]
        markdown = "".join(
            f"# 2.3.{n} Subsection {n}\n\nand then the body of it.\n\n"
            for n in range(1, 5)
        )
        sections = build_sections(
            outline=self.OUTLINE,
            markdown=markdown,
            # Offset so the headings land inside an outline range; one landing
            # outside every range is dropped, which is a separate behaviour.
            page_map=build_page_map(blocks, page_offset=34),
            document_pages=305,
        )
        levels = {s.title: s.level for s in sections}
        assert levels["2.3.1 Subsection 1"] == levels["2.3.4 Subsection 4"]

    def test_the_two_spellings_of_one_appendix_are_one_section(self) -> None:
        """The outline writes ``ζ(4)``, the extraction writes ``$\\zeta ( 4 )$``."""
        assert _section_key(Section(0, "2.3 Appendix to this Chapter: ζ(4)", 2)) == (
            _section_key(Section(1, "2.3 Appendix to this Chapter: $\\zeta ( 4 )$", 2))
        )

    @pytest.mark.parametrize(
        ("left", "right"),
        [
            ("2.4 Thermodynamic quantities", "Thermodynamic quantities"),
            ("2.2 Debye's Calculation", "2.2 Debye\u2019s Calculation"),
            ("1 Overview", "1. Overview"),
        ],
    )
    def test_one_heading_spelled_two_ways_is_one_heading(self, left: str, right: str) -> None:
        assert _section_key(Section(0, left, 1)) == _section_key(Section(1, right, 1))

    def test_genuinely_different_headings_stay_different(self) -> None:
        assert _section_key(Section(0, "2.2.1 Periodic Boundary Conditions", 3)) != (
            _section_key(Section(1, "2.2.2 Debye's Calculation", 3))
        )

    def test_a_heading_outside_every_outline_range_is_dropped(self) -> None:
        """The outline already describes the whole book; a heading outside it is
        an artefact, and inventing a place for it would be worse than losing it."""
        markdown = "# 9 Section From Nowhere\n\nbody\n"
        sections = build_sections(outline=self.OUTLINE, markdown=markdown, document_pages=305)
        assert not any("Nowhere" in s.title for s in sections)


class TestOutlineRanges:
    def test_a_bookmark_ends_where_the_next_one_begins(self) -> None:
        nodes = (
            OutlineNode(title="A", depth=0, page=5),
            OutlineNode(title="B", depth=1, page=9),
            OutlineNode(title="C", depth=0, page=20),
        )
        ranged = with_page_ranges(nodes)
        assert [(n.page, n.page_end) for n in ranged] == [(5, 20), (9, 20), (20, None)]

    def test_depth_is_preserved_so_the_tree_survives(self) -> None:
        nodes = (
            OutlineNode(title="A", depth=0, page=1),
            OutlineNode(title="A.1", depth=1, page=2),
            OutlineNode(title="A.1.1", depth=2, page=3),
        )
        assert [n.depth for n in with_page_ranges(nodes)] == [0, 1, 2]

    def test_an_out_of_order_bookmark_keeps_its_start(self) -> None:
        """Hand-made PDFs do this; a negative-length range would be nonsense."""
        nodes = (
            OutlineNode(title="A", depth=0, page=30),
            OutlineNode(title="B", depth=0, page=10),
        )
        ranged = with_page_ranges(nodes)
        assert ranged[0].page == 30
        assert ranged[0].page_end is None

    def test_no_outline_is_an_empty_result_not_an_error(self) -> None:
        assert with_page_ranges(()) == ()


# ------------------------------------------------------------ mineru overrides


class TestMineruOptions:
    def test_nothing_overridden_means_no_request(self) -> None:
        """The arXiv path must not allocate a request it does not need."""
        assert not MineruOptions().has_overrides
        assert MineruOptions(language="tr").has_overrides
        assert MineruOptions(formula=False).has_overrides

    def test_the_explicit_value_wins_over_the_default(self) -> None:
        merged = MineruOptions(method="ocr").merged_with(
            MineruOptions(method="auto", language="en", table=False)
        )
        assert merged.method == "ocr"
        assert merged.language == "en"
        assert merged.table is False

    def test_unset_fields_do_not_clobber_the_default(self) -> None:
        merged = MineruOptions(language="tr").merged_with(MineruOptions(table=False))
        assert merged.table is False
        assert merged.formula is None
