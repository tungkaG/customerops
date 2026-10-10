from __future__ import annotations

import json
from dataclasses import asdict
from datetime import date
from math import sqrt
from pathlib import Path

from customer_ops.retrieval.embeddings import Embedder
from customer_ops.retrieval.models import PolicyChunk, PolicyMetadata, RetrievalFilter, RetrievalHit


class LocalVectorIndex:
    """A small persistent cosine-similarity index for policy retrieval experiments."""

    def __init__(self, chunks: list[PolicyChunk], vectors: list[list[float]], embedder: Embedder) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("Each policy chunk must have exactly one vector.")
        self._chunks = chunks
        self._vectors = vectors
        self._embedder = embedder

    @classmethod
    def build(cls, chunks: list[PolicyChunk], embedder: Embedder) -> LocalVectorIndex:
        return cls(chunks, embedder.embed(chunk.text for chunk in chunks), embedder)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        records = []
        for chunk, vector in zip(self._chunks, self._vectors, strict=True):
            metadata = asdict(chunk.metadata)
            metadata["effective_date"] = chunk.metadata.effective_date.isoformat()
            records.append({"chunk_id": chunk.chunk_id, "metadata": metadata, "text": chunk.text, "vector": vector})
        path.write_text(json.dumps({"embedder": self._embedder.name, "records": records}, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path, embedder: Embedder) -> LocalVectorIndex:
        payload = json.loads(path.read_text(encoding="utf-8"))
        # Vectors from different embedding models are not comparable, even when their sizes match.
        if payload.get("embedder") != embedder.name:
            raise ValueError(
                f"{path} was built with embedder {payload.get('embedder')!r}, not {embedder.name!r}; rebuild the index."
            )
        chunks: list[PolicyChunk] = []
        vectors: list[list[float]] = []
        for record in payload["records"]:
            metadata = record["metadata"]
            chunks.append(
                PolicyChunk(
                    chunk_id=record["chunk_id"],
                    metadata=PolicyMetadata(
                        document_id=metadata["document_id"],
                        category=metadata["category"],
                        version=metadata["version"],
                        status=metadata["status"],
                        source_type=metadata["source_type"],
                        effective_date=date.fromisoformat(metadata["effective_date"]),
                        trusted=metadata["trusted"],
                    ),
                    text=record["text"],
                )
            )
            vectors.append(record["vector"])
        return cls(chunks, vectors, embedder)

    def search(
        self,
        query: str,
        *,
        retrieval_filter: RetrievalFilter | None = None,
        limit: int = 3,
        minimum_score: float = 0.1,
    ) -> list[RetrievalHit]:
        if limit <= 0:
            return []
        query_vector = self._embedder.embed([query])[0]
        hits = [
            RetrievalHit(chunk, _cosine_similarity(query_vector, vector))
            for chunk, vector in zip(self._chunks, self._vectors, strict=True)
            if retrieval_filter is None or retrieval_filter.matches(chunk)
        ]
        return sorted((hit for hit in hits if hit.score >= minimum_score), key=lambda hit: hit.score, reverse=True)[:limit]


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Query and document vectors must have the same dimensions.")
    denominator = sqrt(sum(value * value for value in left)) * sqrt(sum(value * value for value in right))
    return 0.0 if denominator == 0 else sum(a * b for a, b in zip(left, right, strict=True)) / denominator