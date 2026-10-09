"""Structural code and math comparison."""

from __future__ import annotations

import pytest

from checker_app.config import PlagiarismSettings, SegmentationSettings
from checker_app.domain.enums import BlockType, MatchKind
from checker_app.domain.models import SourceDocument
from checker_app.domain.segmentation import SentenceSplitter
from checker_app.domain.text import tokenize
from checker_app.services.code_ast import CodeCompareService, normalise_math, python_signature

DOC = """# Deney

```python
def train(model, data, epochs=10):
    for epoch in range(epochs):
        loss = model.fit(data)
    return model
```

Sonuçlar aşağıdaki denklemle hesaplanmıştır:

$$L = \\frac{1}{N} \\sum_{i=1}^{N} \\left( y_i - \\hat{y}_i \\right)^2$$
"""

SOURCE = """# Eski çalışma

```python
def fit(model, data, epochs=10):
    for epoch in range(epochs):
        err = model.fit(data)
    return model
```

Denklem:

$$ L = \\frac{1}{N} \\sum_{i=1}^{N} \\left( y_i - \\hat{y}_i \\right)^2 $$
"""


def test_python_ast_ignores_variable_names() -> None:
    a = python_signature("def train(model, data):\n    return model.fit(data)\n")
    b = python_signature("def fit(estimator, dataset):\n    return estimator.fit(dataset)\n")
    assert a == b, "renaming must not change the syntax tree"


def test_python_ast_still_sees_real_differences() -> None:
    a = python_signature("def f(x):\n    return x + 1\n")
    b = python_signature("def f(x):\n    return x - 1\n")
    assert a != b


def test_unparsable_python_falls_back_to_tokens() -> None:
    signature = python_signature("def f(:\n    this is not python")
    assert signature


def test_docstrings_are_ignored_in_the_signature() -> None:
    with_doc = python_signature('def f(x):\n    """Bir açıklama."""\n    return x\n')
    without = python_signature("def f(x):\n    return x\n")
    assert with_doc == without


def test_latex_normalisation_collapses_spacing_differences() -> None:
    a = normalise_math("$$ L = \\frac{1}{N} \\sum \\left( y_i - \\hat{y}_i \\right)^2 $$")
    b = normalise_math("$$L=\\frac{1}{N}\\sum(y_i-\\hat{y}_i)^2$$")
    assert a == b


def test_latex_normalisation_keeps_different_equations_apart() -> None:
    assert normalise_math("$x^2 + y^2$") != normalise_math("$x^2 - y^2$")


def test_compare_reports_code_and_math_matches() -> None:
    settings = PlagiarismSettings()
    service = CodeCompareService(settings)
    splitter = SentenceSplitter(SegmentationSettings())

    doc = splitter.split(DOC)
    source_text = SOURCE
    source = SourceDocument(
        source_id="src-1",
        name="eski.md",
        kind="file",
        text=source_text,
        tokens=tuple(tokenize(source_text)),
    )
    source_blocks = {"src-1": service.extract_blocks(source_text, source.tokens, "tr")}
    matches = service.compare(doc.sentences, DOC, doc.tokens, [source], source_blocks)

    kinds = {m.kind for m in matches}
    assert MatchKind.CODE in kinds
    assert MatchKind.MATH in kinds
    assert all(m.ratio >= settings.near_match_ratio for m in matches)


def test_no_code_flag_returns_nothing() -> None:
    settings = PlagiarismSettings(check_code=False)
    service = CodeCompareService(settings)
    doc = SentenceSplitter(SegmentationSettings()).split(DOC)
    assert service.compare(doc.sentences, DOC, doc.tokens, [], {}) == []


def test_document_without_code_has_no_blocks() -> None:
    service = CodeCompareService(PlagiarismSettings())
    text = "Sadece düz metin içeren bir paragraf burada bulunmaktadır."
    doc = SentenceSplitter(SegmentationSettings()).split(text)
    assert service.compare(doc.sentences, text, doc.tokens, [], {}) == []


@pytest.mark.parametrize("language", ["python", "javascript", "text"])
def test_fence_language_is_detected(language: str) -> None:
    service = CodeCompareService(PlagiarismSettings())
    text = f"```{language}\nx = 1\n```\n"
    blocks = service.extract_blocks(text, tuple(tokenize(text)), "")
    assert blocks and blocks[0].kind is BlockType.CODE
    assert blocks[0].language == language
