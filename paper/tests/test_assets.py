"""Figures, tables and equations.

Two real bugs are pinned here, both found by running the extractor against a real
arXiv rendering rather than a fixture invented to pass:

1. arXiv emits **layout tables** (``width:0.0pt``, one cell) and equation tables
   next to the data tables. Pairing a caption with a body by document order stored
   the layout tables instead — Table 1 came out as 1x1 instead of 6x24.
2. MinerU wraps display formulas in ``$$…$$`` where arXiv's HTML is bare, so the
   two spellings of one formula never matched.

The HTML fixtures below are structurally faithful to arXiv's rendering:
``<figure class="ltx_figure|ltx_table">``, a ``[n]`` refnum span, three
``ltx_bibblock``-style blocks, and captions containing ``<math>``.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest

from paper_app.clients.content.assets import (
    Assets,
    extract_assets_from_html,
    extract_assets_from_mineru,
    strip_display_delimiters,
)
from paper_app.db.asset_repository import AssetRepository, _table_shape
from paper_app.db.session import get_session_factory

PAGE_URL = "https://arxiv.org/html/1706.03762v7"

# Two captioned tables plus one *uncaptioned* layout table that appears first in
# the document — exactly the shape that made document-order pairing wrong.
HTML = """
<html><body>
<table><tr><td><span class="ltx_rule" style="width:0.0pt"></span></td></tr></table>

<figure id="S4.T1" class="ltx_table">
  <figcaption class="ltx_caption"><span class="ltx_tag ltx_tag_table">Table 1: </span>
    Maximum path lengths per layer.</figcaption>
  <table id="S4.T1.2" class="ltx_tabular">
    <thead><tr><th>Layer</th><th>Complexity</th><th>Path</th></tr></thead>
    <tbody><tr><td>Self-Attention</td><td>O(n^2 d)</td><td>O(1)</td></tr></tbody>
  </table>
</figure>

<figure id="S3.F1" class="ltx_figure">
  <img src="1706.03762v7/Figures/arch.png" width="912" height="1344" alt="caption">
  <figcaption class="ltx_caption"><span class="ltx_tag ltx_tag_figure">Figure 1: </span>
    The Transformer - model architecture.</figcaption>
</figure>

<figure id="Sx1.F3" class="ltx_figure">
  <object data="1706.03762v7/attn.svg" type="image/svg+xml"></object>
  <figcaption class="ltx_caption"><span class="ltx_tag ltx_tag_figure">Figure 3: </span>
    Attention over long distances.</figcaption>
</figure>

<figure id="S6.T2" class="ltx_table">
  <figcaption class="ltx_caption"><span class="ltx_tag ltx_tag_table">Table 2: </span>
    BLEU scores.</figcaption>
  <table class="ltx_tabular"><tr><td>a</td></tr></table>
</figure>

<p>Inline math <math alttext="h_{t}" display="inline">x</math> and
<math alttext="\\frac{QK^{T}}{\\sqrt{d_{k}}}" display="block">y</math> here.</p>
</body></html>
"""

MINERU_CONTENT_LIST = [
    {"type": "text", "text": "body"},
    {
        "type": "image",
        "img_path": "images/aaa.jpg",
        "image_caption": ["Figure 1: The Transformer - model architecture."],
        "page_idx": 2,
        "bbox": [320, 90, 673, 497],
    },
    {
        "type": "table",
        "img_path": "images/bbb.jpg",
        "table_caption": ["Table 1: Maximum path lengths per layer."],
        "table_body": "<table><tr><td>a</td></tr></table>",
        "page_idx": 5,
    },
    {
        "type": "equation",
        "img_path": "images/ccc.jpg",
        "text": "$$\n\\mathrm { A t t e n t i o n } ( Q , K , V ) = V\n$$",
        "page_idx": 3,
    },
]


@pytest.fixture
def mineru_dir(tmp_path: Path) -> Path:
    """A MinerU output layout, with the images it references."""
    run = tmp_path / "attention" / "auto"
    (run / "images").mkdir(parents=True)
    for name in ("aaa.jpg", "bbb.jpg", "ccc.jpg"):
        (run / "images" / name).write_bytes(f"fake-{name}".encode())
    (run / "attention_content_list.json").write_text(json.dumps(MINERU_CONTENT_LIST))
    return tmp_path / "attention"


@asynccontextmanager
async def session(settings) -> AsyncIterator[Any]:  # noqa: ANN401
    factory = get_session_factory(settings.database)
    async with factory() as db:
        yield db
        await db.commit()


@pytest.fixture
def repo(settings):  # noqa: ANN201
    return lambda: session(settings)


class TestHtmlExtraction:
    def test_counts(self) -> None:
        assets = extract_assets_from_html(HTML, page_url=PAGE_URL)
        assert assets.count() == {
            "figures": 2,
            "tables": 2,
            "equations": 2,
            "display_equations": 1,
        }

    def test_captions_lose_their_printed_label(self) -> None:
        assets = extract_assets_from_html(HTML, page_url=PAGE_URL)
        assert assets.figures[0].label == "Figure 1"
        assert assets.figures[0].caption == "The Transformer - model architecture."
        assert assets.tables[0].label == "Table 1"
        assert assets.tables[0].caption == "Maximum path lengths per layer."

    def test_table_body_is_the_data_table_not_the_layout_one(self) -> None:
        """Regression: document-order pairing stored the uncaptioned layout table,
        which came out as 1x1 instead of the real 2x3."""
        assets = extract_assets_from_html(HTML, page_url=PAGE_URL)
        body = assets.tables[0].body_html or ""
        assert "Self-Attention" in body
        assert _table_shape(body) == (2, 3)

    def test_layout_tables_are_not_counted_as_assets(self) -> None:
        assert len(extract_assets_from_html(HTML, page_url=PAGE_URL).tables) == 2

    def test_bitmap_and_svg_figures_both_resolve(self) -> None:
        """arXiv ships bitmaps as <img src> and vector art as <object data>."""
        assets = extract_assets_from_html(HTML, page_url=PAGE_URL)
        assert assets.figures[0].image_url == (
            "https://arxiv.org/html/1706.03762v7/Figures/arch.png"
        )
        assert assets.figures[1].image_url is not None
        assert assets.figures[1].image_url.endswith(".svg")

    def test_image_dimensions_are_kept(self) -> None:
        assets = extract_assets_from_html(HTML, page_url=PAGE_URL)
        assert (assets.figures[0].width, assets.figures[0].height) == (912, 1344)

    def test_equation_latex_comes_from_alttext(self) -> None:
        assets = extract_assets_from_html(HTML, page_url=PAGE_URL)
        assert [e.latex for e in assets.equations] == ["h_{t}", "\\frac{QK^{T}}{\\sqrt{d_{k}}}"]

    def test_display_and_inline_are_separated(self) -> None:
        assets = extract_assets_from_html(HTML, page_url=PAGE_URL)
        assert [e.is_display for e in assets.equations] == [False, True]

    def test_no_html_is_not_an_error(self) -> None:
        assert extract_assets_from_html(None).is_empty()
        assert extract_assets_from_html("").is_empty()
        assert extract_assets_from_html("<p>nothing here</p>").is_empty()

    def test_malformed_html_does_not_raise(self) -> None:
        broken = '<figure class="ltx_figure"><figcaption>Figure 1: x</figcaption>'
        assert extract_assets_from_html(broken, page_url=PAGE_URL).figures


class TestMineruExtraction:
    def test_counts(self, mineru_dir: Path) -> None:
        assets = extract_assets_from_mineru(mineru_dir)
        assert assets.count() == {
            "figures": 1,
            "tables": 1,
            "equations": 1,
            "display_equations": 1,
        }

    def test_page_and_bbox_are_kept(self, mineru_dir: Path) -> None:
        figure = extract_assets_from_mineru(mineru_dir).figures[0]
        assert figure.page_idx == 2
        assert figure.bbox == (320, 90, 673, 497)

    def test_local_images_resolve(self, mineru_dir: Path) -> None:
        figure = extract_assets_from_mineru(mineru_dir).figures[0]
        assert figure.image_path is not None
        assert Path(figure.image_path).exists()

    def test_display_delimiters_are_stripped(self, mineru_dir: Path) -> None:
        """Regression: MinerU wraps formulas in $$…, arXiv HTML does not, so the
        same formula was stored two different ways."""
        equation = extract_assets_from_mineru(mineru_dir).equations[0]
        assert equation.latex is not None
        assert not equation.latex.startswith("$$")
        assert "$$" not in equation.latex

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("$$x = y$$", "x = y"),
            ("  $$x = y$$  ", "x = y"),
            ("x = y", "x = y"),
            ("$$$$", None),
            (None, None),
        ],
    )
    def test_strip_display_delimiters(self, raw, expected) -> None:  # noqa: ANN001
        assert strip_display_delimiters(raw) == expected

    def test_absent_content_list_is_empty(self, tmp_path: Path) -> None:
        assert extract_assets_from_mineru(tmp_path).is_empty()

    def test_unreadable_content_list_is_empty(self, tmp_path: Path) -> None:
        run = tmp_path / "x"
        run.mkdir()
        (run / "x_content_list.json").write_text("{not json")
        assert extract_assets_from_mineru(tmp_path).is_empty()

    def test_missing_image_file_does_not_lose_the_asset(self, tmp_path: Path) -> None:
        run = tmp_path / "x" / "auto"
        run.mkdir(parents=True)
        (run / "x_content_list.json").write_text(
            json.dumps([{"type": "image", "img_path": "images/gone.jpg"}])
        )
        assets = extract_assets_from_mineru(tmp_path / "x")
        assert len(assets.figures) == 1
        assert assets.figures[0].image_path is None


class TestMergeAndDeduplicate:
    def test_merge_renumbers_across_both_sources(self, mineru_dir: Path) -> None:
        html = extract_assets_from_html(HTML, page_url=PAGE_URL)
        merged = html.merge(extract_assets_from_mineru(mineru_dir))
        assert [f.ordinal for f in merged.figures] == [1, 2, 3]
        assert [t.ordinal for t in merged.tables] == [1, 2, 3]

    def test_cross_source_duplicates_collapse(self, mineru_dir: Path) -> None:
        """The same figure seen by both extractors is one row, not two."""
        html = extract_assets_from_html(HTML, page_url=PAGE_URL)
        merged = html.merge(extract_assets_from_mineru(mineru_dir)).deduplicate()
        labels = [f.label for f in merged.figures]
        assert labels.count("Figure 1") == 1
        assert len(merged.figures) == 2

    def test_the_preferred_source_wins(self, mineru_dir: Path) -> None:
        """Figures prefer MinerU (it has the image bytes), equations prefer HTML
        (it reports inline math too)."""
        html = extract_assets_from_html(HTML, page_url=PAGE_URL)
        merged = html.merge(extract_assets_from_mineru(mineru_dir)).deduplicate()
        assert next(f for f in merged.figures if f.label == "Figure 1").source == "mineru"

    def test_repeated_within_one_source_is_kept(self) -> None:
        """The same inline formula twice in a paper is two real occurrences, and
        collapsing them would lose a position in the document."""
        html = """
        <p><math alttext="h_t">a</math></p><p><math alttext="h_t">b</math></p>
        """
        merged = extract_assets_from_html(html, page_url=PAGE_URL).deduplicate()
        assert len(merged.equations) == 2

    def test_equations_matching_across_sources_collapse(self, mineru_dir: Path) -> None:
        """Whitespace-insensitive, because MinerU letter-spaces its LaTeX."""
        html = (
            '<p><math alttext="\\mathrm{Attention}(Q,K,V)=V" display="block">x</math></p>'
        )
        merged = (
            extract_assets_from_html(html, page_url=PAGE_URL)
            .merge(extract_assets_from_mineru(mineru_dir))
            .deduplicate()
        )
        # MinerU's is the same formula spelled \mathrm { A t t e n t i o n } ...
        assert len(merged.equations) == 1

    def test_near_misses_are_not_merged(self, mineru_dir: Path) -> None:
        """`\\max` and `\\operatorname*{max}` are the same formula to a reader but not
        the same string, and guessing would be worse than keeping both."""
        html = '<p><math alttext="\\max(0,x)" display="block">x</math></p>'
        merged = (
            extract_assets_from_html(html, page_url=PAGE_URL)
            .merge(extract_assets_from_mineru(mineru_dir))
            .deduplicate()
        )
        assert len(merged.equations) == 2


class TestStorage:
    async def test_round_trip(self, repo) -> None:  # noqa: ANN001
        async with repo() as db:
            paper_id = await _seed_paper(db)
            assets = extract_assets_from_html(HTML, page_url=PAGE_URL)
            counts = await AssetRepository(db).replace_for_paper(paper_id, assets)

            assert counts == {"figures": 2, "tables": 2, "equations": 2}
            stored = await AssetRepository(db).count(paper_id)
            assert stored["figures"] == 2
            assert stored["display_equations"] == 1

            figures = await AssetRepository(db).figures(paper_id)
            assert figures[0].label == "Figure 1"
            assert (figures[0].image_url or "").endswith("arch.png")
            tables = await AssetRepository(db).tables(paper_id)
            assert tables[0].row_count == 2 and tables[0].column_count == 3

    async def test_reingest_replaces_rather_than_duplicates(self, repo) -> None:  # noqa: ANN001
        async with repo() as db:

            paper_id = await _seed_paper(db)
            assets = extract_assets_from_html(HTML, page_url=PAGE_URL)
            repo_assets = AssetRepository(db)
            await repo_assets.replace_for_paper(paper_id, assets)
            await repo_assets.replace_for_paper(paper_id, assets)
            assert (await repo_assets.count(paper_id))["figures"] == 2

    async def test_images_are_hashed_into_the_blob_store(
        self, repo, mineru_dir: Path, settings, tmp_path: Path
    ) -> None:  # noqa: ANN001
        from paper_app.infra.storage import LocalBlobStore

        async with repo() as db:
            paper_id = await _seed_paper(db)
            blobs = LocalBlobStore(root=tmp_path / "blobs")
            await AssetRepository(db, blobs=blobs).replace_for_paper(
                paper_id, extract_assets_from_mineru(mineru_dir)
            )
            figures = await AssetRepository(db).figures(paper_id)
            assert figures[0].image_sha256 is not None
            assert blobs.exists(figures[0].image_sha256)

    async def test_a_missing_image_does_not_lose_the_figure(self, repo) -> None:  # noqa: ANN001
        async with repo() as db:
            paper_id = await _seed_paper(db)
            assets = Assets(
                figures=extract_assets_from_html(HTML, page_url=PAGE_URL).figures,
            )
            await AssetRepository(db).replace_for_paper(paper_id, assets)
            rows = await AssetRepository(db).figures(paper_id)
            assert len(rows) == 2
            assert rows[0].image_sha256 is None

    async def test_equation_search_spans_the_corpus(self, repo) -> None:  # noqa: ANN001
        async with repo() as db:
            paper_id = await _seed_paper(db)
            await AssetRepository(db).replace_for_paper(
                paper_id, extract_assets_from_html(HTML, page_url=PAGE_URL)
            )
            repo_assets = AssetRepository(db)
            hits = await repo_assets.search_equations("sqrt")
            assert len(hits) == 1
            assert hits[0].is_display is True
            assert await repo_assets.search_equations("nothing here") == []

    async def test_equation_search_can_skip_inline(self, repo) -> None:  # noqa: ANN001
        async with repo() as db:
            paper_id = await _seed_paper(db)
            await AssetRepository(db).replace_for_paper(
                paper_id, extract_assets_from_html(HTML, page_url=PAGE_URL)
            )
            repo_assets = AssetRepository(db)
            assert len(await repo_assets.search_equations("h_")) == 1
            assert await repo_assets.search_equations("h_", display_only=True) == []


async def _seed_paper(db, arxiv_id: str = "1706.03762") -> str:  # noqa: ANN001
    from paper_app.db.repositories import PaperRepository
    from paper_app.domain.models import Author as AuthorSpec
    from paper_app.domain.models import PaperMetadata

    paper = await PaperRepository(db).upsert(
        PaperMetadata(
            arxiv_id=arxiv_id,
            versioned_id=f"{arxiv_id}v1",
            version=1,
            title=f"Paper {arxiv_id}",
            abstract="a",
            authors=(AuthorSpec(name="Ashish Vaswani"),),
            categories=("cs.CL",),
            primary_category="cs.CL",
            abs_url="u",
            pdf_url="p",
            html_url=None,
        )
    )
    await db.flush()
    return paper.id
