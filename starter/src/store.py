"""Persistent local Qdrant vector store with cosine search and metadata filtering."""
import uuid
from typing import Any

from qdrant_client import QdrantClient, models

from .models import Chunk

ALLOWED_FILTER_FIELDS = frozenset({"document_id", "source", "page", "year", "section_type", "topic"})


class QdrantVectorStore:
    """Disk-backed Qdrant vector store partitioned per strategy collection."""

    def __init__(self, storage_path: str, dimensions: int, strategies: tuple[str, ...] = ("A", "B", "C")) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be > 0")
        self.storage_path = storage_path
        self.dimensions = dimensions
        self.strategies = strategies
        self.client = QdrantClient(path=storage_path)

    @staticmethod
    def _collection_name(strategy: str) -> str:
        return f"day6_{strategy}"

    @staticmethod
    def _point_id(strategy: str, chunk_id: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{strategy}:{chunk_id}"))

    def create_schema(self) -> None:
        existing = {c.name for c in self.client.get_collections().collections}
        for strategy in self.strategies:
            name = self._collection_name(strategy)
            if name not in existing:
                self.client.create_collection(
                    collection_name=name,
                    vectors_config=models.VectorParams(size=self.dimensions, distance=models.Distance.COSINE),
                )

    def count(self, strategy: str) -> int:
        return self.client.count(collection_name=self._collection_name(strategy), exact=True).count

    def upsert(self, strategy: str, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError(f"chunks ({len(chunks)}) and vectors ({len(vectors)}) length mismatch")
        if not chunks:
            return

        collection = self._collection_name(strategy)
        existing = {c.name for c in self.client.get_collections().collections}
        if collection not in existing:
            self.client.create_collection(
                collection_name=collection,
                vectors_config=models.VectorParams(size=self.dimensions, distance=models.Distance.COSINE),
            )

        for vec in vectors:
            if len(vec) != self.dimensions:
                raise ValueError(f"Expected vector dimension {self.dimensions}, got {len(vec)}")

        # Remove obsolete chunks per touched document_id before writing current chunk set
        new_ids_by_doc: dict[str, set[str]] = {}
        for c in chunks:
            new_ids_by_doc.setdefault(c.document_id, set()).add(self._point_id(strategy, c.chunk_id))

        for doc_id, keep_ids in new_ids_by_doc.items():
            stale_ids: list[str] = []
            offset = None
            doc_filter = models.Filter(
                must=[models.FieldCondition(key="document_id", match=models.MatchValue(value=doc_id))]
            )
            while True:
                points, offset = self.client.scroll(
                    collection_name=collection,
                    scroll_filter=doc_filter,
                    limit=256,
                    with_payload=False,
                    with_vectors=False,
                    offset=offset,
                )
                for pt in points:
                    if str(pt.id) not in keep_ids:
                        stale_ids.append(str(pt.id))
                if offset is None:
                    break
            if stale_ids:
                self.client.delete(collection_name=collection, points_selector=models.PointIdsList(points=stale_ids))

        points_to_upsert = [
            models.PointStruct(
                id=self._point_id(strategy, chunk.chunk_id),
                vector=vec,
                payload={
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "source": chunk.source,
                    "page": chunk.page,
                    "text": chunk.text,
                    **{k: v for k, v in chunk.metadata.items() if k not in {"chunk_id", "text"}},
                },
            )
            for chunk, vec in zip(chunks, vectors)
        ]

        for i in range(0, len(points_to_upsert), 128):
            self.client.upsert(collection_name=collection, points=points_to_upsert[i : i + 128])

    def search(
        self,
        strategy: str,
        query_vector: list[float],
        top_k: int = 3,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if len(query_vector) != self.dimensions:
            raise ValueError(f"Expected query vector dim {self.dimensions}, got {len(query_vector)}")

        q_filter = None
        if filters:
            conditions = []
            for key, value in filters.items():
                if key not in ALLOWED_FILTER_FIELDS:
                    raise ValueError(f"Disallowed filter field: {key!r}")
                conditions.append(models.FieldCondition(key=key, match=models.MatchValue(value=value)))
            q_filter = models.Filter(must=conditions)

        res = self.client.query_points(
            collection_name=self._collection_name(strategy),
            query=query_vector,
            query_filter=q_filter,
            limit=top_k,
            with_payload=True,
        )

        hits: list[dict[str, Any]] = []
        for pt in res.points:
            payload = dict(pt.payload or {})
            chunk_id = payload.pop("chunk_id", str(pt.id))
            text = payload.pop("text", "")
            hits.append(
                {
                    "chunk_id": chunk_id,
                    "text": text,
                    "score": float(pt.score),
                    "metadata": payload,
                }
            )
        return hits

    def close(self) -> None:
        self.client.close()
