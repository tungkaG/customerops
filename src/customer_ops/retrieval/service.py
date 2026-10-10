from __future__ import annotations

from datetime import date
from functools import lru_cache
from pathlib import Path

from customer_ops.config import settings
from customer_ops.retrieval.embeddings import Embedder, SentenceTransformerEmbedder
from customer_ops.retrieval.index import LocalVectorIndex
from customer_ops.retrieval.ingestion import ingest_policy_directory, load_source_registry
from customer_ops.retrieval.models import RetrievalFilter, RetrievalHit

DATA_DIRECTORY = Path(__file__).resolve().parents[3] / "data"


class PolicyRetriever:
    """The retriever the agent uses: only active, trusted policies in effect on the reference date."""

    def __init__(self, index: LocalVectorIndex, reference_date: date) -> None:
        self._index = index
        self._reference_date = reference_date

    @classmethod
    def from_directory(
        cls, policy_directory: Path, registry_path: Path, embedder: Embedder, reference_date: date
    ) -> PolicyRetriever:
        chunks = ingest_policy_directory(policy_directory, load_source_registry(registry_path))
        return cls(LocalVectorIndex.build(chunks, embedder), reference_date)

    def search(self, query: str, category: str | None = None, limit: int = 3) -> list[RetrievalHit]:
        retrieval_filter = RetrievalFilter(
            category=category, applicable_on=self._reference_date, trusted_only=True, active_only=True
        )
        return self._index.search(query, retrieval_filter=retrieval_filter, limit=limit)


@lru_cache(maxsize=1)
def default_policy_retriever() -> PolicyRetriever:
    return PolicyRetriever.from_directory(
        DATA_DIRECTORY / "policies",
        DATA_DIRECTORY / "policy_registry.json",
        SentenceTransformerEmbedder(settings.embedding_model),
        settings.reference_date,
    )
