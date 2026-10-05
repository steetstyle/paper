"""BAAI BGE embeddings.

The only thing that makes BGE *retrieval* work — as opposed to plain
sentence-similarity work — is the query instruction. BGE is trained with the
query side prefixed and the passage side bare:

    query:    "Represent this sentence for searching relevant passages: <q>"
    passage:  "<p>"

Putting the instruction on the passages as well measurably hurts retrieval, so
this provider overrides :meth:`embed_query` and leaves :meth:`embed_documents`
inherited and untouched. The wording is taken from the model card for
``BAAI/bge-large-en-v1.5``, not from memory.

Dimensions are read from the loaded model rather than trusted from config:
bge-large is 1024-wide, bge-base 768, and a wrong number here becomes a pgvector
dimension mismatch at insert time instead of a warning.
"""

from __future__ import annotations

from typing import ClassVar, Final

from app.embeddings.sentence_transformers_provider import SentenceTransformerProvider

#: Verbatim from the BAAI/bge-large-en-v1.5 model card. Trailing space matters:
#: it separates the instruction from the query text the tokenizer then encodes.
BGE_QUERY_INSTRUCTION: Final[str] = (
    "Represent this sentence for searching relevant passages: "
)

#: bge-large-en-v1.5 is 1024-wide; the real width is confirmed on load anyway.
BGE_LARGE_DIMENSIONS: Final[int] = 1024


class BgeEmbeddingProvider(SentenceTransformerProvider):
    """``sentence-transformers`` with the BGE query instruction applied."""

    name: ClassVar[str] = "bge"

    def __init__(
        self,
        model: str = "BAAI/bge-large-en-v1.5",
        dimensions: int | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(
            model=model,
            dimensions=dimensions or BGE_LARGE_DIMENSIONS,
            **kwargs,  # type: ignore[arg-type]
        )

    @property
    def query_instruction(self) -> str:
        """Prefix applied to queries. Empty string means BGE-style training."""
        return BGE_QUERY_INSTRUCTION

    async def embed_query(self, text: str) -> list[float]:
        instruction = self.query_instruction
        if not instruction or not text.strip():
            return await super().embed_query(text)
        vectors = await self.embed_documents([f"{instruction}{text.strip()}"])
        return vectors[0]


__all__ = ["BGE_LARGE_DIMENSIONS", "BGE_QUERY_INSTRUCTION", "BgeEmbeddingProvider"]
