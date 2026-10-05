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