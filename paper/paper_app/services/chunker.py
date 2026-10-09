"""Markdown-aware, token-budgeted chunking.

Design goals:

* never split a paragraph unless it exceeds the budget on its own
* keep heading breadcrumbs on every chunk so retrieved text is self-describing
* character-exact offsets back into the source document
* deterministic output so re-ingestion is reproducible
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

from paper_app.config import ChunkingSettings
from paper_app.domain.enums import ChunkKind, ContentSource
from paper_app.domain.models import TextChunk
from paper_app.infra.text import (
    approx_tokens,
    iter_paragraphs,
    parse_heading,
    sha256_text,
    split_sentences,
    strip_markdown,
)
from paper_app.services.chunk_kinds import classify_chunk

TokenCounter = Callable[[str], int]

_FENCE = "```"
_JOIN = "\n\n"


def _join(units: Sequence[str]) -> str:
    """Join chunk units into a single block, collapsing incidental whitespace."""
    return _JOIN.join(unit.strip() for unit in units if unit.strip())


class ChunkingService:
    def __init__(
        self,
        settings: ChunkingSettings | None = None,
        token_counter: TokenCounter | None = None,
    ) -> None:
        self._settings = settings or ChunkingSettings()
        self._counter = token_counter or approx_tokens

    # ------------------------------------------------------------------ public
    def count_tokens(self, text: str) -> int:
        return max(0, self._counter(text))

    def chunk(
        self,
        text: str,
        *,
        source: ContentSource = ContentSource.ARXIV_HTML,
        prefix_header: str | None = None,
        start_ordinal: int = 0,
    ) -> list[TextChunk]:
        if not text or not text.strip():
            return []
        settings = self._settings

        if settings.respect_markdown:
            units = self._split_into_sections(text)
        else:
            units = [(None, (), iter_paragraphs(text))]

        chunks: list[TextChunk] = []
        for heading, section_path, paragraphs in units:
            for body, char_start, char_end in self._pack(
                paragraphs, settings.max_tokens, settings.overlap_tokens
            ):
                content = self._compose(prefix_header, heading, body)
                token_count = self.count_tokens(strip_markdown(content))
                if token_count < settings.min_tokens and chunks:
                    # Merge a runt into the previous chunk rather than dropping it.
                    previous = chunks.pop()
                    merged_text = f"{previous.text}\n\n{body}"
                    chunks.append(
                        TextChunk(
                            ordinal=previous.ordinal,
                            text=merged_text,
                            token_count=previous.token_count + token_count,
                            char_start=previous.char_start,
                            char_end=char_end,
                            heading=previous.heading,
                            section_path=previous.section_path,
                            content_hash=sha256_text(merged_text),
                            source=source,
                        )
                    )
                    continue
                if not content.strip():
                    continue
                chunks.append(
                    TextChunk(
                        ordinal=start_ordinal + len(chunks),
                        text=content,
                        token_count=token_count,
                        char_start=char_start,
                        char_end=char_end,
                        heading=heading,
                        section_path=section_path,
                        content_hash=sha256_text(content),
                        source=source,
                        kind=classify_chunk(
                            content, heading=heading, section_path=section_path
                        ),
                    )
                )
        return chunks

    def chunk_abstract(
        self, title: str, abstract: str, *, source: ContentSource = ContentSource.ABSTRACT_ONLY
    ) -> list[TextChunk]:
        """Chunk used when only metadata (title/abstract) is available."""
        body = f"{title.strip()}\n\n{abstract.strip()}" if abstract.strip() else title.strip()
        if not body.strip():
            return []
        return [
            TextChunk(
                ordinal=0,
                text=body,
                token_count=self.count_tokens(body),
                char_start=0,
                char_end=len(body),
                heading=title.strip() or None,
                section_path=("Abstract",),
                content_hash=sha256_text(body),
                source=source,
                # By definition this is the abstract, whatever the text looks like.
                kind=ChunkKind.ABSTRACT,
            )
        ]

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def _compose(prefix_header: str | None, heading: str | None, body: str) -> str:
        header = " > ".join(filter(None, [prefix_header, heading]))
        body = body.strip()
        return f"{header}\n\n{body}" if header else body

    def _split_into_sections(
        self, text: str
    ) -> list[tuple[str | None, tuple[str, ...], list[str]]]:
        """Split on ATX headings while tracking the heading breadcrumb trail."""
        sections: list[tuple[str | None, tuple[str, ...], list[str]]] = []
        stack: list[str] = []
        current_heading: str | None = None
        current_lines: list[str] = []

        def flush() -> None:
            body = "\n".join(current_lines).strip()
            if body:
                sections.append((current_heading, tuple(stack), list(iter_paragraphs(body))))

        in_fence = False
        for line in text.splitlines():
            if line.lstrip().startswith(_FENCE):
                in_fence = not in_fence
            parsed = parse_heading(line) if (not in_fence and self._settings.respect_markdown) else None
            if parsed:
                flush()
                level, title = parsed
                while len(stack) >= level:
                    stack.pop()
                stack.append(title)
                current_heading = title
                current_lines = []
            else:
                current_lines.append(line)
        flush()

        if not sections:
            sections.append((None, (), list(iter_paragraphs(text))))
        return sections

    def _pack(
        self, paragraphs: Iterable[str], max_tokens: int, overlap_tokens: int
    ) -> list[tuple[str, int, int]]:
        """Group units into budgeted windows, carrying a short overlap forward.

        The budget is measured against the *actual* joined buffer rather than a
        running sum of estimates — estimates drift and would let chunks exceed
        ``max_tokens``.
        """
        units: list[str] = []
        for paragraph in paragraphs:
            if self.count_tokens(paragraph) <= max_tokens:
                units.append(paragraph)
            else:
                units.extend(self._split_oversized(paragraph, max_tokens))
        if not units:
            return []

        windows: list[tuple[str, int, int]] = []
        buffer: list[str] = []
        cursor = 0

        for unit in units:
            if buffer and self.count_tokens(_join(buffer + [unit])) > max_tokens:
                body = _join(buffer)
                windows.append((body, cursor, cursor + len(body)))
                cursor += len(body) + 2
                buffer = self._overlap(buffer, overlap_tokens)
                # Carried overlap may already fill the budget; drop it entirely.
                while buffer and self.count_tokens(_join(buffer + [unit])) > max_tokens:
                    buffer.pop(0)
            buffer.append(unit)

        if buffer:
            body = _join(buffer)
            windows.append((body, cursor, cursor + len(body)))
        return windows

    def _split_oversized(self, paragraph: str, max_tokens: int) -> list[str]:
        """Break a single oversized paragraph into sentence groups."""
        sentences = split_sentences(paragraph) or [paragraph]
        groups: list[str] = []
        current: list[str] = []
        for sentence in sentences:
            if current and self.count_tokens(_join(current + [sentence])) > max_tokens:
                groups.append(_join(current))
                current = []
            current.append(sentence)
        if current:
            groups.append(_join(current))
        return groups

    @staticmethod
    def _overlap(buffer: list[str], overlap_tokens: int) -> list[str]:
        """Carry the tail of the previous window so context survives the split."""
        if overlap_tokens <= 0 or len(buffer) <= 1:
            return []
        tail: list[str] = []
        for paragraph in reversed(buffer):
            candidate = [paragraph, *tail]
            if approx_tokens(_join(candidate)) > overlap_tokens:
                break
            tail.insert(0, paragraph)
        return tail