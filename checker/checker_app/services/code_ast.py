"""Structural comparison of code and math blocks.

Text similarity punishes code twice: renaming a variable or reformatting
destroys the words while leaving the program identical. So code blocks are
compared on their **syntax tree**:

* Python: ``ast.parse`` then ``ast.dump(..., include_attributes=False)``.
  Identical dumps means an identical program, regardless of names or layout.
  When parsing fails (pseudo-code, a partial snippet) the block falls back to a
  normalised token stream.
* Other languages: token streams with strings and comments dropped, compared
  with :class:`difflib.SequenceMatcher`.

LaTeX/math is normalised before comparison (``\\left``/``\\right``, spacing
macros, ``\\frac`` and ``\\cdot`` all collapse), so the same formula written
slightly differently still matches - the common case when a paper's equations
are retyped from a previous paper.

Math and code findings are returned as :class:`~checker_app.domain.models.MatchSpan`
objects so they flow through the same per-sentence crediting as text matches.
"""

from __future__ import annotations

import ast
import difflib
import io
import re
import tokenize
from collections.abc import Sequence
from dataclasses import dataclass

from checker_app.config import PlagiarismSettings
from checker_app.domain.enums import BlockType, MatchKind
from checker_app.domain.models import MatchSpan, Sentence, SourceDocument
from checker_app.domain.text import Token
from checker_app.logging import get_logger
from checker_app.services.plagiarism import build_span

__all__ = ["CodeBlock", "CodeCompareService", "normalise_math", "python_signature"]

logger = get_logger("checker.code")

_FENCE_RE = re.compile(r"^\s*```\s*([\w+-]*)\s*$")
_MATH_INLINE_RE = re.compile(r"\$\$(.+?)\$\$|\$([^$\n]+)\$", re.DOTALL)
_MATH_SPACING_RE = re.compile(r"\\[!,;:> ]")
_MATH_WORDS_RE = re.compile(
    r"\\(?:mathrm|mathbf|mathit|text|operatorname|boldsymbol)\s*\{([^{}]*)\}"
)
_FRAC_RE = re.compile(r"\\d?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}")
_SUP_RE = re.compile(r"\^\s*\{([^{}]*)\}")
_SUB_RE = re.compile(r"_\s*\{([^{}]*)\}")
_SQRT_RE = re.compile(r"\\sqrt\s*\{([^{}]*)\}")
_WS_RE = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class CodeBlock:
    """A code or math block with its offsets in its own document."""

    index: int
    kind: BlockType
    language: str
    text: str
    char_start: int
    char_end: int
    token_count: int
    signature: str


class CodeCompareService:
    """Find code/math blocks that also appear in a reference source."""

    def __init__(self, settings: PlagiarismSettings | None = None) -> None:
        self._settings = settings or PlagiarismSettings()

    # ------------------------------------------------------------------ public
    def compare(
        self,
        sentences: Sequence[Sentence],
        doc_text: str,
        doc_tokens: Sequence[Token],
        sources: Sequence[SourceDocument],
        source_blocks: dict[str, Sequence[CodeBlock]],
    ) -> list[MatchSpan]:
        if not self._settings.check_code or not sources:
            return []
        doc_blocks = self._blocks_from_sentences(sentences, doc_tokens)
        if not doc_blocks:
            return []

        spans: list[MatchSpan] = []
        for doc_block in doc_blocks:
            for source in sources:
                for other in source_blocks.get(source.source_id, ()):
                    if other.kind is not doc_block.kind:
                        continue
                    ratio, exact = _similarity(doc_block, other)
                    if exact:
                        spans.append(
                            self._span(
                                doc_block,
                                doc_text,
                                doc_tokens,
                                source,
                                other,
                                ratio=1.0,
                                exact=True,
                            )
                        )
                        continue
                    if ratio >= self._settings.near_match_ratio:
                        spans.append(
                            self._span(
                                doc_block,
                                doc_text,
                                doc_tokens,
                                source,
                                other,
                                ratio=ratio,
                                exact=False,
                            )
                        )
        return spans

    def extract_blocks(
        self,
        text: str,
        tokens: Sequence[Token],
        language_hint: str = "",
    ) -> tuple[CodeBlock, ...]:
        """Pull code fences and math spans out of a raw text."""
        blocks: list[CodeBlock] = []
        index = 0
        fence: str | None = None
        fence_start = 0
        fence_language = ""

        for line in _lines(text):
            marker = _FENCE_RE.match(line[2])
            if fence is None:
                if marker:
                    fence = marker.group(1) or line[2].strip()
                    fence_language = marker.group(1) or language_hint
                    fence_start = line[1]
                continue
            if marker:
                body = text[fence_start : line[1]]
                blocks.append(
                    _make_block(
                        index=index,
                        kind=BlockType.CODE,
                        language=fence_language or "text",
                        text=body,
                        char_start=fence_start,
                        char_end=line[1],
                        tokens=tokens,
                    )
                )
                index += 1
                fence = None
        if fence is not None:
            body = text[fence_start:]
            blocks.append(
                _make_block(
                    index=index,
                    kind=BlockType.CODE,
                    language=fence_language or "text",
                    text=body,
                    char_start=fence_start,
                    char_end=len(text),
                    tokens=tokens,
                )
            )

        for match in _MATH_INLINE_RE.finditer(text):
            body = match.group(0)
            blocks.append(
                _make_block(
                    index=index,
                    kind=BlockType.MATH,
                    language="latex",
                    text=body,
                    char_start=match.start(),
                    char_end=match.end(),
                    tokens=tokens,
                )
            )
            index += 1
        return tuple(blocks)

    # ----------------------------------------------------------------- private
    def _blocks_from_sentences(
        self, sentences: Sequence[Sentence], tokens: Sequence[Token]
    ) -> list[CodeBlock]:
        """The splitter already isolated code/math units; reuse them."""
        blocks: list[CodeBlock] = []
        for position, sentence in enumerate(sentences):
            if sentence.location.block_type not in {BlockType.CODE, BlockType.MATH}:
                continue
            blocks.append(
                _make_block(
                    index=position,
                    kind=sentence.location.block_type,
                    language=_fence_language(sentence.text),
                    text=sentence.text,
                    char_start=sentence.location.char_start,
                    char_end=sentence.location.char_end,
                    tokens=tokens,
                )
            )
        return blocks

    def _span(
        self,
        doc_block: CodeBlock,
        doc_text: str,
        doc_tokens: Sequence[Token],
        source: SourceDocument,
        other: CodeBlock,
        *,
        ratio: float,
        exact: bool,
    ) -> MatchSpan:
        kind = MatchKind.CODE if doc_block.kind is BlockType.CODE else MatchKind.MATH
        token_index = _token_index_at(doc_tokens, doc_block.char_start)
        span = build_span(
            doc_tokens=doc_tokens[token_index : token_index + max(1, doc_block.token_count)],
            doc_text=doc_text,
            doc_start=0,
            doc_end=max(1, doc_block.token_count),
            source=source,
            src_start=_token_index_at(source.tokens, other.char_start),
            src_end=_token_index_at(source.tokens, other.char_start) + max(1, other.token_count),
            matched_words=max(1, doc_block.token_count),
            kind=kind,
            ratio=ratio,
        )
        if exact:
            logger.info(
                "tam eşleşme: %s bloğu (%s) %s içinde bulundu",
                doc_block.kind.value,
                doc_block.language,
                source.name,
            )
        return span


# --------------------------------------------------------------------- helpers
def _lines(text: str) -> list[tuple[int, int, str]]:
    out: list[tuple[int, int, str]] = []
    pos = 0
    for line in text.split("\n"):
        out.append((pos, pos + len(line), line))
        pos += len(line) + 1
    return out


def _fence_language(text: str) -> str:
    for line in text.split("\n")[:2]:
        match = _FENCE_RE.match(line.strip())
        if match:
            return match.group(1) or "text"
    return "text" if text.lstrip().startswith("```") else "latex"


def _token_index_at(tokens: Sequence[Token], char_offset: int) -> int:
    for index, token in enumerate(tokens):
        if token.char_start >= char_offset:
            return index
    return max(0, len(tokens) - 1)


def _make_block(
    *,
    index: int,
    kind: BlockType,
    language: str,
    text: str,
    char_start: int,
    char_end: int,
    tokens: Sequence[Token],
) -> CodeBlock:
    token_count = sum(1 for t in tokens if char_start <= t.char_start < char_end)
    signature = (
        python_signature(text)
        if language == "python" and kind is BlockType.CODE
        else normalise_math(text)
        if kind is BlockType.MATH
        else _token_stream(text)
    )
    return CodeBlock(
        index=index,
        kind=kind,
        language=language,
        text=text,
        char_start=char_start,
        char_end=char_end,
        token_count=token_count,
        signature=signature,
    )


class _NameNormalizer(ast.NodeTransformer):
    """Replace identifiers with canonical placeholders, in first-use order.

    Two snippets that differ only in variable names (``model``/``estimator``,
    ``train``/``fit``) must produce byte-identical trees, otherwise the AST
    comparison degenerates into a text comparison. Attribute names and keyword
    arguments are *not* normalised: ``x.fit()`` and ``x.predict()`` are
    different programs.
    """

    def __init__(self) -> None:
        self.names: dict[str, str] = {}

    def _canonical(self, name: str) -> str:
        return self.names.setdefault(name, f"v{len(self.names)}")

    def visit_Name(self, node: ast.Name) -> ast.Name:  # noqa: N802 - ast API
        node.id = self._canonical(node.id)
        return node

    def visit_arg(self, node: ast.arg) -> ast.arg:  # noqa: N802 - ast API
        node.arg = self._canonical(node.arg)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:  # noqa: N802
        node.name = self._canonical(node.name)
        self.generic_visit(node)
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AsyncFunctionDef:  # noqa: N802
        node.name = self._canonical(node.name)
        self.generic_visit(node)
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:  # noqa: N802
        node.name = self._canonical(node.name)
        self.generic_visit(node)
        return node


def python_signature(code: str) -> str:
    """Normalised ``ast.dump`` of a Python block, or a token stream if unparsable.

    Identifiers are canonicalised and docstrings are stripped: identical
    boilerplate docstrings are the least informative part of a code match.
    """
    body = _strip_fence(code)
    try:
        tree = ast.parse(body)
    except SyntaxError:
        return _token_stream(body)
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            statements = node.body
            if (
                statements
                and isinstance(statements[0], ast.Expr)
                and isinstance(statements[0].value, ast.Constant)
                and isinstance(statements[0].value.value, str)
            ):
                node.body = statements[1:] or [ast.Pass()]
    tree = _NameNormalizer().visit(tree)
    return ast.dump(tree, annotate_fields=False, include_attributes=False)


def _strip_fence(code: str) -> str:
    lines = [line for line in code.split("\n") if not line.strip().startswith("```")]
    return "\n".join(lines)


def _token_stream(code: str) -> str:
    """Language-agnostic normalised token stream of a code block."""
    body = _strip_fence(code)
    try:
        readline = io.StringIO(body).readline
        parts: list[str] = []
        for tok in tokenize.generate_tokens(readline):
            if tok.type in {
                tokenize.COMMENT,
                tokenize.NL,
                tokenize.NEWLINE,
                tokenize.INDENT,
                tokenize.DEDENT,
                tokenize.ENDMARKER,
                tokenize.ENCODING,
            }:
                continue
            parts.append(tok.string)
        return " ".join(parts)
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return _WS_RE.sub(" ", re.sub(r"//.*|#.*", "", body)).strip()


def normalise_math(latex: str) -> str:
    """Collapse LaTeX spelling differences into a comparable form."""
    text = latex.strip()
    text = re.sub(r"^\$+|\$+$", "", text).strip()
    for _ in range(3):
        text = _FRAC_RE.sub(r"(\1)/(\2)", text)
        text = _SUP_RE.sub(r"^\1", text)
        text = _SUB_RE.sub(r"_\1", text)
        text = _SQRT_RE.sub(r"sqrt(\1)", text)
    text = _MATH_WORDS_RE.sub(r"\1", text)
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("\\cdot", "*").replace("\\times", "*").replace("\\div", "/")
    text = _MATH_SPACING_RE.sub("", text)
    text = re.sub(r"\\(?:displaystyle|textstyle|small|large|quad|qquad)", "", text)
    text = _WS_RE.sub("", text)
    return text


def _similarity(left: CodeBlock, right: CodeBlock) -> tuple[float, bool]:
    if left.signature and left.signature == right.signature:
        return 1.0, True
    if not left.signature or not right.signature:
        return 0.0, False
    ratio = difflib.SequenceMatcher(None, left.signature, right.signature, autojunk=False).ratio()
    return round(ratio, 4), False
