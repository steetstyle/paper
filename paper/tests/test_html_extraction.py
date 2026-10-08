"""HTML -> markdown conversion."""

from __future__ import annotations

import pytest

from app.clients.content.html_extractor import (
    _normalise_heading_levels,
    extract_title,
    html_to_markdown,
)

LATEXML_HTML = """<html><head>
<meta name="citation_title" content="TinyLlama">
<style>body{color:red}</style><script>alert('xss')</script>
</head><body>
<article class="ltx_document">
<h1 class="ltx_title_document">TinyLlama</h1>
<div class="ltx_authors">A. Author</div>
<h6 class="ltx_title_abstract">Abstract</h6>
<p>We present TinyLlama.</p>
<h2 class="ltx_title_section">1 Introduction</h2>
<p>Large language models.</p>
<h3 class="ltx_title_subsection">1.1 Setup</h3>
<p>Details here.</p>
<h2 class="ltx_title_section">2 Results</h2>
<p>Results here.</p>
</article>
</body></html>"""


class TestConversion:
    def test_extracts_content(self) -> None:
        markdown = html_to_markdown(LATEXML_HTML)
        assert "TinyLlama" in markdown
        assert "Large language models." in markdown
        assert "Results here." in markdown

    def test_drops_scripts_and_styles(self) -> None:
        markdown = html_to_markdown(LATEXML_HTML)
        assert "alert(" not in markdown
        assert "color:red" not in markdown

    def test_abstract_is_promoted_to_h2(self) -> None:
        """LaTeXML emits the abstract as h6; it is a top-level section."""
        markdown = html_to_markdown(LATEXML_HTML)
        assert "######" not in markdown
        assert "\n## Abstract" in markdown

    def test_outline_is_dense_and_hierarchical(self) -> None:
        markdown = html_to_markdown(LATEXML_HTML)
        headings = [line for line in markdown.splitlines() if line.startswith("#")]
        assert headings[0] == "# TinyLlama"
        assert headings[1] == "## Abstract"
        assert "## 1 Introduction" in headings
        assert "### 1.1 Setup" in headings
        assert "## 2 Results" in headings

    def test_empty_input(self) -> None:
        assert html_to_markdown("") == ""
        assert html_to_markdown("   \n ") == ""

    def test_links_are_preserved_as_markdown(self) -> None:
        markdown = html_to_markdown('<article><p>See <a href="http://x.com">the site</a>.</p></article>')
        assert "[the site](http://x.com)" in markdown


class TestHeadingNormalisation:
    def test_remaps_by_order_of_appearance(self) -> None:
        # `##` seen first becomes h1, `###` h2, `#` h3 — relative order kept.
        assert _normalise_heading_levels("## A\n\n### B\n\n# C") == "# A\n\n## B\n\n### C"

    def test_collapses_gaps(self) -> None:
        assert _normalise_heading_levels("# A\n\n#### B") == "# A\n\n## B"

    def test_maps_by_order_of_appearance(self) -> None:
        source = "# Title\n\n###### Abstract\n\n## Section"
        assert _normalise_heading_levels(source) == "# Title\n\n## Abstract\n\n### Section"

    def test_leaves_plain_text_alone(self) -> None:
        assert _normalise_heading_levels("no headings here") == "no headings here"

    def test_does_not_touch_inline_hashes(self) -> None:
        assert _normalise_heading_levels("issue #42 is closed") == "issue #42 is closed"


class TestExtractTitle:
    def test_prefers_citation_meta(self) -> None:
        assert extract_title(LATEXML_HTML) == "TinyLlama"

    def test_falls_back_to_title_tag(self) -> None:
        assert extract_title("<html><head><title>arXiv:1234.5678 Some Paper</title></head></html>") == "1234.5678 Some Paper"

    def test_returns_none_without_a_title(self) -> None:
        assert extract_title("<html><body>hi</body></html>") is None


@pytest.mark.parametrize(
    "html",
    [
        "<article><p>plain</p></article>",
        "<div><h2>Only heading</h2></div>",
    ],
)
def test_never_raises_on_partial_documents(html: str) -> None:
    assert isinstance(html_to_markdown(html), str)

#: The shape LaTeXML actually emits, copied from arXiv: a captioned table lives in
#: ``<figure class="ltx_table">``, the table itself one level down inside a ``<div>``,
#: and the ``<figcaption>`` *after* it. On 2412.13663 all eleven figures were
#: ``ltx_table`` and no table existed outside a figure — so treating figures as
#: image-only deleted every table in the paper.
TABLE_FIGURE_HTML = """<html><body><article class="ltx_document">
<h1 class="ltx_title_document">Encoder Comparison</h1>
<p>Numbers follow.</p>
<figure class="ltx_table" id="S3.T1">
  <div class="ltx_tabular_container">
    <table class="ltx_tabular"><tbody>
      <tr><th>Model</th><th>BEIR</th><th>Latency</th></tr>
      <tr><td>BERT</td><td>38.9</td><td>24.1</td></tr>
      <tr><td>ModernBERT</td><td>91.4</td><td>9.8</td></tr>
    </tbody></table>
  </div>
  <figcaption>Table 1: Retrieval scores.</figcaption>
</figure>
<p>As shown above.</p>
<figure class="ltx_figure" id="S2.F1">
  <img src="fig1.png"/>
  <figcaption>Figure 1: The architecture.</figcaption>
</figure>
</article></body></html>"""

EQUATION_FIGURE_HTML = """<html><body><article class="ltx_document">
<h1 class="ltx_title_document">With Equations</h1>
<figure class="ltx_figure" id="S1.F2">
  <img src="f.png"/>
  <table class="ltx_equation ltx_eqn_table"><tbody><tr>
    <td></td>
    <td><math alttext="E=mc^2"><mi>E</mi><mo>=</mo><mi>mc</mi></math></td>
    <td>(1)</td>
  </tr></tbody></table>
  <figcaption>Figure 2: The identity.</figcaption>
</figure>
</article></body></html>"""


class TestTablesSurviveTheirFigure:
    """A figure is an image wrapper. In LaTeXML it is also a table's address.

    The rule these tests pin: keep the table, collapse only what is genuinely an
    image. Before the fix, ``_reduce_figures_to_captions`` threw away every table
    wrapped in a figure, which on the measured paper was all of them.
    """

    def test_the_numbers_reach_the_markdown(self) -> None:
        markdown = html_to_markdown(TABLE_FIGURE_HTML)
        assert "38.9" in markdown
        assert "91.4" in markdown
        assert "24.1" in markdown

    def test_the_header_row_reaches_it_too(self) -> None:
        markdown = html_to_markdown(TABLE_FIGURE_HTML)
        for header in ("Model", "BEIR", "Latency"):
            assert header in markdown, header

    def test_it_arrives_as_a_markdown_table(self) -> None:
        """Not just the text: pipe rows are what the chunker can recognise as a
        table, and what a reader can read."""
        rows = [line for line in html_to_markdown(TABLE_FIGURE_HTML).splitlines()
                if line.count("|") >= 3]
        assert len(rows) >= 3, html_to_markdown(TABLE_FIGURE_HTML)

    def test_the_caption_is_still_there(self) -> None:
        """It was the one thing that survived before, and it is what a search for
        "the table about retrieval scores" finds."""
        assert "Table 1: Retrieval scores." in html_to_markdown(TABLE_FIGURE_HTML)

    def test_document_order_is_preserved(self) -> None:
        """LaTeXML puts the caption after the table, and so should the markdown:
        the reader meets the numbers before the sentence naming them."""
        markdown = html_to_markdown(TABLE_FIGURE_HTML)
        assert markdown.index("38.9") < markdown.index("Table 1: Retrieval scores.")

    def test_surrounding_prose_is_not_disturbed(self) -> None:
        markdown = html_to_markdown(TABLE_FIGURE_HTML)
        assert "Numbers follow." in markdown
        assert "As shown above." in markdown


class TestImageFiguresStillCollapse:
    """The fix must not undo what the original rule was for."""

    def test_an_image_becomes_its_caption(self) -> None:
        assert "Figure 1: The architecture." in html_to_markdown(TABLE_FIGURE_HTML)

    def test_the_image_itself_is_not_carried(self) -> None:
        assert "fig1.png" not in html_to_markdown(TABLE_FIGURE_HTML)

    def test_an_uncaptioned_image_is_dropped(self) -> None:
        html = ('<html><body><article><p>Before.</p>'
                '<figure><img src="x.png"/></figure><p>After.</p></article></body></html>')
        markdown = html_to_markdown(html)
        assert "x.png" not in markdown
        assert "Before." in markdown and "After." in markdown


class TestEquationsAreNotTables:
    """The reason figure handling existed at all.

    LaTeXML lays every display equation out in a three-column table. Kept as one,
    it renders as an empty-looking markdown table and "only equations" and "only
    tables" both answer nonsense. The equation is recovered from ``alttext``, so
    it must still arrive as an equation and not as pipe rows.
    """

    def test_an_equation_inside_a_figure_survives_as_an_equation(self) -> None:
        markdown = html_to_markdown(EQUATION_FIGURE_HTML)
        assert "E=mc^2" in markdown

    def test_its_layout_table_does_not_become_a_pipe_table(self) -> None:
        rows = [line for line in html_to_markdown(EQUATION_FIGURE_HTML).splitlines()
                if line.count("|") >= 2]
        assert rows == [], rows

    def test_its_caption_is_kept(self) -> None:
        assert "Figure 2: The identity." in html_to_markdown(EQUATION_FIGURE_HTML)


class TestEmptyLayoutTablesAreStillDropped:
    """A one-cell spacer renders as a table with a header and no data."""

    def test_a_cell_free_table_is_removed(self) -> None:
        html = ('<html><body><article><p>Before.</p>'
                '<table class="ltx_tabular"><tbody><tr><td></td></tr></tbody></table>'
                '<p>After.</p></article></body></html>')
        markdown = html_to_markdown(html)
        assert "| ---" not in markdown
        assert "Before." in markdown and "After." in markdown

    def test_a_real_table_with_no_caption_still_survives(self) -> None:
        """Tables are not kept *because* they are captioned. An uncaptioned one is
        still the paper's data, and this is the case a caption-driven rule would
        get wrong."""
        html = ('<html><body><article>'
                '<table class="ltx_tabular"><tbody>'
                '<tr><td>alpha</td><td>17</td></tr></tbody></table>'
                '</article></body></html>')
        assert "alpha" in html_to_markdown(html)
