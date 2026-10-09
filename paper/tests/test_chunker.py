"""Chunking behaviour."""

from __future__ import annotations

from paper_app.config import ChunkingSettings
from paper_app.domain.enums import ContentSource
from paper_app.infra.text import approx_tokens, strip_markdown
from paper_app.services.chunker import ChunkingService

MARKDOWN = """# Introduction

Transformers rely on attention. They removed recurrence entirely from the
sequence modelling pipeline and replaced it with stacked self-attention.

## Related Work

Recurrent networks were the dominant approach. Convolutions also existed.

# Method

We describe the multi-head attention formulation here in detail.

$$a = \\text{softmax}(QK^T/\\sqrt{d})V$$

### Complexity

Self-attention is quadratic in sequence length.
"""


def build(**overrides) -> ChunkingService:
    settings = ChunkingSettings(
        max_tokens=overrides.pop("max_tokens", 60),
        overlap_tokens=overrides.pop("overlap_tokens", 10),
        min_tokens=overrides.pop("min_tokens", 5),
        respect_markdown=overrides.pop("respect_markdown", True),
    )
    return ChunkingService(settings, token_counter=approx_tokens)


def test_chunks_respect_budget() -> None:
    chunks = build().chunk(MARKDOWN, source=ContentSource.ARXIV_HTML)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.token_count <= 60


def test_chunks_carry_heading_breadcrumbs() -> None:
    chunks = build().chunk(MARKDOWN)
    headings = {chunk.heading for chunk in chunks}
    assert "Introduction" in headings
    assert "Method" in headings
    assert "Complexity" in headings

    complexity = next(c for c in chunks if c.heading == "Complexity")
    assert "Complexity" in complexity.section_path
    assert "Method" in complexity.section_path


def test_ordinals_are_dense_and_hashed() -> None:
    chunks = build().chunk(MARKDOWN)
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))
    assert all(len(c.content_hash) == 64 for c in chunks)
    assert all(c.char_end >= c.char_start for c in chunks)


def test_chunking_is_deterministic() -> None:
    service = build()
    first = service.chunk(MARKDOWN)
    second = service.chunk(MARKDOWN)
    assert [c.text for c in first] == [c.text for c in second]
    assert [c.content_hash for c in first] == [c.content_hash for c in second]


def test_oversized_paragraph_is_split() -> None:
    long_paragraph = " ".join(f"Sentence number {i} describes attention." for i in range(120))
    chunks = build(max_tokens=50, overlap_tokens=10, min_tokens=5).chunk(long_paragraph)
    assert len(chunks) > 1
    assert all(c.token_count <= 50 for c in chunks)
    joined = " ".join(c.text for c in chunks)
    assert "Sentence number 0" in joined
    assert "Sentence number 119" in joined


def test_flat_chunking_ignores_headings() -> None:
    chunks = build(respect_markdown=False).chunk(MARKDOWN)
    assert all(chunk.heading is None for chunk in chunks)


def test_empty_input_returns_no_chunks() -> None:
    assert build().chunk("") == []
    assert build().chunk("   \n  ") == []


def test_chunk_abstract() -> None:
    chunks = build().chunk_abstract("Attention Is All You Need", "We propose the Transformer.")
    assert len(chunks) == 1
    assert chunks[0].section_path == ("Abstract",)
    assert "Transformer" in chunks[0].text
    assert chunks[0].source == ContentSource.ABSTRACT_ONLY


def test_prefix_header_is_prepended() -> None:
    chunks = build().chunk(MARKDOWN, prefix_header="Attention Is All You Need")
    assert chunks
    assert all(c.text.startswith("Attention Is All You Need") for c in chunks)


def test_runt_chunks_merge_into_neighbour() -> None:
    text = "# A\n\n" + ("x " * 200) + "\n\n# B\n\n" + ("y " * 200)
    chunks = build(max_tokens=80, min_tokens=70).chunk(text)
    assert all(chunk.token_count >= 70 for chunk in chunks)


def test_strip_markdown_removes_syntax() -> None:
    markdown = "## Head\n\nSome **bold**, *italic* and `code` with [a link](http://x) and $x^2$."
    plain = strip_markdown(markdown)
    assert "##" not in plain
    assert "**" not in plain
    assert "`" not in plain
    assert "a link" in plain
    assert "http://x" not in plain