"""Deciding what a chunk *is*, and fixing what the converter emits.

Two halves, because the honest fix is at both ends:

* LaTeXML lays every display equation out in a three-column ``<table>``. Left
  alone it reaches the chunker as a markdown table, and "only equations" then
  returns nothing while "only tables" returns every equation in the paper. So
  the HTML converter unwraps those tables using the ``alttext`` LaTeX.
* The classifier still has to recognise the layout, because a corpus ingested
  before that fix already holds those chunks, and because the dependency-free
  fallback parser has no LaTeXML awareness at all.
"""

from __future__ import annotations

import re

import pytest

from paper_app.clients.content.html_extractor import html_to_markdown
from paper_app.domain.enums import ChunkKind
from paper_app.services.chunk_kinds import classify_chunk, merge_run, parse_kinds

# ----------------------------------------------------------------- real corpus
# Shapes copied verbatim from arXiv 1706.03762v7 (Attention Is All You Need),
# read out of the stored chunks rather than invented.
NUMBERED_EQUATION_TABLE = """\
## 3.2.1 Scaled Dot-Product Attention

We call our particular attention "Scaled Dot-Product Attention" (Figure 2).

|  |  |  |
| --- | --- | --- |
|  | Attention(Q,K,V)=softmax(QK^{T}/sqrt(d_{k}))V |  | (1) |

The two most commonly used attention functions are additive attention [2].
"""

EQUATION_GROUP_TABLE = """\
## 3.2.2 Multi-Head Attention

Instead of performing a single attention function we use h parallel layers.

|  |  |  |  |
| --- | --- | --- | --- |
|  | MultiHead(Q,K,V)\\displaystyle\\mathrm{MultiHead}(Q,K,V) | =Concat(head1,…,headh)WO\\displaystyle=\\mathrm{Concat}(\\mathrm{head_{i}})W^{O} |  |
|  |  |  |  |
| --- | --- | --- | --- |
|  | whereheadi\\displaystyle\\text{where}~\\mathrm{head_{i}} | =Attention(...) |  |
"""

REAL_TABLE = """\
| Layer Type | Complexity per Layer | Sequential | Maximum Path Length |
| --- | --- | --- | --- |
| Self-Attention | O(n^2 * d) | Yes | O(1) |
| Recurrent | O(n * d^2) | Yes | O(n) |
"""

# A genuine table whose first *and* last columns are empty. Measured: this shape
# exists in the corpus, so a pad-cell rule without a LaTeX requirement eats it.
TABLE_WITH_EMPTY_ENDS = """\
|  | Model | EN-DE | EN-FR | Training Cost |
| --- | --- | --- | --- | --- |
| Base | Transformer | 27.3 | 38.1 | 3.3e18 |
|  | ByteNet | 23.75 |  | 1.0e18 |
"""

REFERENCE_LIST = """\
## References

Jimmy Lei Ba. Layer normalization. arXiv:1607.06450, 2016.
Dzmitry Bahdanau. Neural machine translation, 2014.
"""

APPENDIX_PROOF = """\
## Appendix C Proofs of our theorems > Proof.

First, we show that the diffusion is non-expansive.

|  |  |  |
| --- | --- | --- |
|  | Luv=-Tuv(q)\\displaystyle L^{\\widetilde{\\mathcal{F}}}\\_{uv} |  |
"""


class TestClassifyChunk:
    @pytest.mark.parametrize(
        ("text", "heading", "expected"),
        [
            (REFERENCE_LIST, None, ChunkKind.REFERENCE),
            (APPENDIX_PROOF, "Proof.", ChunkKind.EQUATION),
            (NUMBERED_EQUATION_TABLE, "3.2.1 Scaled Dot-Product Attention", ChunkKind.EQUATION),
            (EQUATION_GROUP_TABLE, "3.2.2 Multi-Head Attention", ChunkKind.EQUATION),
            (REAL_TABLE, "Complexity per Layer", ChunkKind.TABLE),
            (TABLE_WITH_EMPTY_ENDS, "Model Variations", ChunkKind.TABLE),
            (
                "Attention Is All You Need > Abstract\n\nThe dominant sequence "
                "transduction models are based on recurrent networks.",
                None,
                ChunkKind.ABSTRACT,
            ),
            (
                "Figure 1: Overall architecture of the Transformer, showing the "
                "encoder and decoder stacks.",
                "Model Architecture",
                ChunkKind.FIGURE,
            ),
            (
                "The encoder maps input to continuous representations. We use "
                "multi-head attention so heads learn different functions.",
                "Model Architecture",
                ChunkKind.BODY,
            ),
            ("```python\ndef forward(x):\n    return x\n```", "Implementation", ChunkKind.CODE),
        ],
    )
    def test_kinds(self, text: str, heading: str | None, expected: ChunkKind) -> None:
        assert classify_chunk(text, heading=heading) is expected

    def test_abstract_chunk_is_abstract_whatever_the_text(self) -> None:
        """A metadata-only chunk is the abstract by definition, not by shape."""
        from paper_app.config import ChunkingSettings
        from paper_app.services.chunker import ChunkingService

        chunker = ChunkingService(ChunkingSettings(max_tokens=80, overlap_tokens=15))
        chunks = chunker.chunk_abstract("A Paper", "| a | b |\n| --- | --- |\n| 1 | 2 |")
        assert [chunk.kind for chunk in chunks] == [ChunkKind.ABSTRACT]

    def test_heading_inside_the_text_counts(self) -> None:
        """The chunker bakes the trail into the text, so it must be read there."""
        body = "## References\n\nJimmy Lei Ba. Layer normalization, 2016.\n"
        assert classify_chunk(body) is ChunkKind.REFERENCE
        assert classify_chunk(body, heading="References") is ChunkKind.REFERENCE

    def test_a_mention_of_a_figure_stays_body(self) -> None:
        """Only a caption *at the start* counts; a mention in prose does not."""
        text = (
            "As shown in Figure 3 above, the encoder produces a sequence of "
            "continuous representations which the decoder then consumes."
        )
        assert classify_chunk(text, heading="Model Architecture") is ChunkKind.BODY

    def test_appendix_is_not_a_reference(self) -> None:
        """Regression: appendix/proof used to be labelled REFERENCE.

        Over the real corpus that mislabelled 505 chunks, which are dense with
        display equations and the opposite of a bibliography.
        """
        for heading in ("Appendix B Proofs", "Appendix", "Proof.", "Supporting Derivations"):
            assert classify_chunk("Some ordinary prose here.", heading=heading) is ChunkKind.BODY

    def test_bibliography_still_wins_over_prose_shape(self) -> None:
        """A references chunk full of pipe rows is still a references chunk."""
        text = f"{REFERENCE_LIST}\n| a | b |\n| --- | --- |\n| 1 | 2 |\n"
        assert classify_chunk(text) is ChunkKind.REFERENCE


class TestMergeRun:
    def test_empty_is_body(self) -> None:
        assert merge_run([]) is ChunkKind.BODY

    def test_most_specific_wins(self) -> None:
        run = [ChunkKind.BODY, ChunkKind.FIGURE, ChunkKind.BODY]
        assert merge_run(run) is ChunkKind.FIGURE
        assert merge_run([ChunkKind.TABLE, ChunkKind.EQUATION]) is ChunkKind.EQUATION


class TestParseKinds:
    def test_none_and_empty_are_none(self) -> None:
        assert parse_kinds(None) is None
        assert parse_kinds([]) is None
        assert parse_kinds(["  "]) is None

    def test_plural_and_abbreviation(self) -> None:
        assert parse_kinds(["equations"]) == ["equation"]
        assert parse_kinds(["eq"]) == ["equation"]
        assert parse_kinds(["figs", "tables"]) == ["figure", "table"]
        assert parse_kinds(["abstracts"]) == ["abstract"]

    def test_deduplicates_and_keeps_order(self) -> None:
        assert parse_kinds(["eq", "equation", "body"]) == ["equation", "body"]

    def test_single_string_is_accepted(self) -> None:
        assert parse_kinds("equation") == ["equation"]

    def test_unknown_kind_raises_and_names_the_set(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            parse_kinds(["equation", "eqaution"])
        message = str(excinfo.value)
        assert "eqaution" in message
        # The message must be actionable, not just "invalid input".
        for kind in ChunkKind:
            assert kind.value in message


LATEXML_EQUATION_GROUP_HTML = """
<html><body><article class="ltx_document">
<p>We use multi-head attention, defined as a concatenation of projections:</p>
<table id="S3.E2" class="ltx_equationgroup ltx_eqn_align ltx_eqn_table">
<tbody>
<tr class="ltx_equation ltx_eqn_row"><td class="ltx_eqn_cell"></td>
<td class="ltx_eqn_cell"><math alttext="\\mathrm{MultiHead}(Q,K,V)"></math></td><td></td></tr>
<tr class="ltx_equation ltx_eqn_row"><td class="ltx_eqn_cell"></td>
<td class="ltx_eqn_cell"><math alttext="=\\mathrm{Concat}(\\mathrm{head_{i}})W^{O}"></math></td><td></td></tr>
<tr class="ltx_equation ltx_eqn_row"><td class="ltx_eqn_cell"></td>
<td class="ltx_eqn_cell"><math alttext="\\text{where}~\\mathrm{head_{i}}"></math></td><td></td></tr>
</tbody></table>
<p>That is the whole computation.</p>
</article></body></html>
"""

LATEXML_FIGURE_HTML = """
<html><body><article class="ltx_document">
<p>As shown in the figure, the encoder stacks N layers.</p>
<figure id="S3.F1" class="ltx_figure">
<img src="1706.03762v7/Figures/ModalNet-21.png" alt="Refer to caption" width="912" height="1344">
<figcaption class="ltx_caption ltx_centering"><span class="ltx_tag ltx_tag_figure">Figure 1: </span>The Transformer - model architecture.</figcaption>
</figure>
<p>Figure 2 has no caption at all.</p>
<figure id="S3.F9" class="ltx_figure"><img src="x.png" alt="Refer to caption"></figure>
</article></body></html>
"""

LATEXML_EQUATION_HTML = """
<html><body><article class="ltx_document">
<p>We compute the matrix of outputs as:</p>
<table id="S3.E1" class="ltx_equation ltx_eqn_table">
<tbody><tr class="ltx_equation ltx_eqn_row ltx_align_baseline">
<td class="ltx_eqn_cell ltx_eqn_center_padleft"></td>
<td class="ltx_eqn_cell ltx_align_center">
<math id="S3.E1.m1" class="ltx_Math" alttext="\\mathrm{Attention}(Q,K,V)=\\mathrm{softmax}(\\frac{QK^{T}}{\\sqrt{d_{k}}})V" display="block"><semantics><mrow><mi>Attention</mi></mrow></semantics></math>
</td>
<td class="ltx_eqn_cell ltx_eqn_center_padright"></td>
</tr></tbody></table>
<p>That is the whole computation.</p>
<table class="ltx_tabular ltx_centering ltx_align_top"><tbody><tr><td></td></tr></tbody></table>
<table class="ltx_tabular ltx_guessed_headers"><tbody>
<tr><th>Layer Type</th><th>Complexity</th></tr>
<tr><td>Self-Attention</td><td>O(n^2 d)</td></tr>
</tbody></table>
</article></body></html>
"""


class TestHtmlEquationTables:
    def test_equation_table_becomes_display_math(self) -> None:
        markdown = html_to_markdown(LATEXML_EQUATION_HTML)
        assert "$$" in markdown
        # The LaTeX comes from `alttext`, not from scraping rendered glyphs.
        assert r"\mathrm{Attention}(Q,K,V)" in markdown

    def test_real_table_survives_as_a_table(self) -> None:
        markdown = html_to_markdown(LATEXML_EQUATION_HTML)
        assert "| Layer Type | Complexity |" in markdown

    def test_empty_spacer_table_is_dropped(self) -> None:
        """A 1x1 ltx_tabular with no text carries no data."""
        markdown = html_to_markdown(LATEXML_EQUATION_HTML)
        blocks = [line for line in markdown.splitlines() if line.strip().startswith("|")]
        # The real table contributes header, separator and one data row.
        assert len(blocks) == 3

    def test_display_math_chunk_classifies_as_equation(self) -> None:
        markdown = html_to_markdown(LATEXML_EQUATION_HTML)
        body = markdown.split("That is the whole computation.")[0]
        assert classify_chunk(body) is ChunkKind.EQUATION

    def test_empty_html_is_not_mangled(self) -> None:
        assert html_to_markdown("") == ""

    def test_an_equation_group_becomes_several_blocks(self) -> None:
        """Regression: `replace_with` detaches, so a second call on the same
        table raised `ValueError: not part of a tree`.

        Only reachable with a group of two or more `math` nodes, which the
        single-equation fixture above never produced — and every real paper has
        them (1706.03762v7 has groups of four and two).
        """
        markdown = html_to_markdown(LATEXML_EQUATION_GROUP_HTML)
        blocks = re.findall(r"\$\$\n(.+?)\n\$\$", markdown, re.S)
        assert len(blocks) == 3
        assert r"\mathrm{MultiHead}(Q,K,V)" in blocks[0]
        assert "xPAPEREQx" not in markdown


class TestFigureCaptions:
    def test_caption_text_survives(self) -> None:
        markdown = html_to_markdown(LATEXML_FIGURE_HTML)
        assert "Figure 1: The Transformer - model architecture." in markdown

    def test_the_image_itself_does_not(self) -> None:
        """No `![...]` and no alt-text boilerplate in the searchable text."""
        markdown = html_to_markdown(LATEXML_FIGURE_HTML)
        assert "![" not in markdown
        assert "Refer to caption" not in markdown

    def test_a_figure_without_a_caption_is_dropped(self) -> None:
        """An uncaptioned image has nothing searchable in it."""
        markdown = html_to_markdown(LATEXML_FIGURE_HTML)
        assert "x.png" not in markdown

    def test_a_caption_classifies_as_figure(self) -> None:
        body = html_to_markdown(LATEXML_FIGURE_HTML).split("Figure 1:")[1]
        assert classify_chunk("Paper > 3 Model Architecture\n\nFigure 1:" + body) is ChunkKind.FIGURE

    def test_a_caption_inside_a_real_chunk_is_found(self) -> None:
        """The breadcrumb comes first in every stored chunk; the caption does not."""
        chunk = (
            "Attention Is All You Need > 3 Model Architecture\n\n"
            "Figure 1: The Transformer - model architecture."
        )
        assert classify_chunk(chunk) is ChunkKind.FIGURE
