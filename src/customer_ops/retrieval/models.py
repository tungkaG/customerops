from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class PolicyMetadata:
    document_id: str
    category: str
    version: str
    status: str
    source_type: str
    effective_date: date
    trusted: bool


@dataclass(frozen=True)
class PolicyChunk:
    chunk_id: str
    metadata: PolicyMetadata
    text: str


@dataclass(frozen=True)
class RetrievalFilter:
    category: str | None = None
    applicable_on: date | None = None
    trusted_only: bool = False
    active_only: bool = False

    def matches(self, chunk: PolicyChunk) -> bool:
        metadata = chunk.metadata
        if self.category is not None and metadata.category != self.category:
            return False
        if self.applicable_on is not None and metadata.effective_date > self.applicable_on:
            return False
        if self.trusted_only and not metadata.trusted:
            return False
        return not self.active_only or metadata.status == "active"


@dataclass(frozen=True)
class RetrievalHit:
    chunk: PolicyChunk
    score: float

    @property
    def citation(self) -> str:
        metadata = self.chunk.metadata
        return f"{metadata.document_id}#{self.chunk.chunk_id}@{metadata.version}"