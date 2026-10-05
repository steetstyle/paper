"""Qdrant-backed store (optional dependency).

Keeps vectors outside Postgres while chunk text and metadata stay relational —
useful when the corpus outgrows a single database node.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from app.db.spaces import EmbeddingSpace
from app.db.vector_store.base import VectorFilter, VectorStore
from app.domain.models import VectorRecord
from app.logging import get_logger

logger = get_logger(__name__)

DISTANCE = {"cosine": "Cosine", "dot": "Dot", "euclid": "Euclid"}


class QdrantVectorStore(VectorStore):
    name = "qdrant"

    def __init__(
        self,
        url: str,
        space: EmbeddingSpace,
        *,
        api_key: str | None = None,
        collection_prefix: str = "paper_chunks",
        client: Any | None = None,
    ) -> None:
        super().__init__(space)
        self._url = url
        self._api_key = api_key
        self._collection = f"{collection_prefix}__{space.slug}"
        self._client = client

    def _get_client(self) -> Any:  # noqa: ANN401
        if self._client is None:
            try:
                from qdrant_client import AsyncQdrantClient  # noqa: PLC0415
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "qdrant-client not installed — pip install 'paper-app-backend[qdrant]'"
                ) from exc
            self._client = AsyncQdrantClient(url=self._url, api_key=self._api_key)
        return self._client

    async def ensure_ready(self) -> None:
        from qdrant_client.models import Distance, VectorParams  # noqa: PLC0415

        client = self._get_client()
        if not await client.collection_exists(self._collection):
            await client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(
                    size=self.space.dimensions,
                    distance=getattr(Distance, DISTANCE.get(self.space.distance, "Cosine")),
                ),
            )
            logger.info(
                "qdrant_collection_created",
                extra={
                    "collection": self._collection,
                    "space": self.space.name,
                    "dimensions": self.space.dimensions,
                },
            )

    async def upsert(self, records: Sequence[VectorRecord]) -> int:
        from qdrant_client.models import PointStruct  # noqa: PLC0415

        if not records:
            return 0
        client = self._get_client()
        points = [
            PointStruct(
                id=record.chunk_id,
                vector=record.vector,
                payload={**record.payload, "paper_id": record.paper_id, "chunk_id": record.chunk_id},
            )
            for record in records
        ]
        await client.upsert(collection_name=self._collection, points=points, wait=True)
        return len(records)

    async def search(
        self,
        vector: list[float],
        *,
        top_k: int = 10,
        filters: VectorFilter | None = None,
        min_score: float | None = None,
    ) -> list[tuple[str, float]]:
        self._check_dimensions(vector)
        filters = filters or VectorFilter()
        if filters.matches_nothing:
            # Qdrant rejects an empty MatchAny, so an explicit "match nothing"
            # has to be answered here rather than pushed into the query.
            return []
        client = self._get_client()
        query_filter = _build_qdrant_filter(filters)
        response = await client.query_points(
            collection_name=self._collection,
            query=vector,
            limit=top_k,
            query_filter=query_filter,
            with_payload=False,
            score_threshold=min_score,
        )
        return [(str(point.id), float(point.score)) for point in response.points]

    async def delete_for_papers(self, paper_ids: list[str]) -> int:
        from qdrant_client.models import FilterSelector  # noqa: PLC0415

        client = self._get_client()
        await client.delete(
            collection_name=self._collection,
            points_selector=FilterSelector(
                filter=_build_qdrant_filter(VectorFilter(paper_ids=list(paper_ids)))
            ),
        )
        return len(paper_ids)

    async def count(self) -> int:
        client = self._get_client()
        result = await client.count(collection_name=self._collection, exact=True)
        return int(result.count)

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None


def _build_qdrant_filter(filters: VectorFilter) -> Any:  # noqa: ANN401
    from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue  # noqa: PLC0415

    conditions = []
    if filters.paper_ids is not None:
        conditions.append(
            FieldCondition(key="paper_id", match=MatchAny(any=list(filters.paper_ids)))
        )
    if filters.categories is not None:
        conditions.append(
            FieldCondition(key="categories", match=MatchAny(any=list(filters.categories)))
        )
    if filters.content_kinds is not None:
        conditions.append(
            FieldCondition(key="content_kind", match=MatchAny(any=list(filters.content_kinds)))
        )
    if filters.sources is not None:
        conditions.append(FieldCondition(key="source", match=MatchAny(any=list(filters.sources))))
    for key, value in filters.extra.items():
        conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
    return Filter(must=conditions) if conditions else None