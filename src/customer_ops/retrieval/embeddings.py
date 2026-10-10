from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from math import sqrt
from typing import Protocol


class Embedder(Protocol):
    name: str

    def embed(self, texts: Iterable[str]) -> list[list[float]]:
        """Return one normalized vector for each input text."""


class HashingEmbedder:
    """Lexical baseline: hashed word counts, which measure word overlap rather than meaning."""

    def __init__(self, dimensions: int = 1024) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self.dimensions = dimensions
        self.name = f"hashing-{dimensions}"

    def embed(self, texts: Iterable[str]) -> list[list[float]]:
        return [self._embed_text(text) for text in texts]

    def _embed_text(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            vector[index] += 1.0 if digest[4] % 2 else -1.0
        magnitude = sqrt(sum(value * value for value in vector))
        return vector if magnitude == 0 else [value / magnitude for value in vector]


class SentenceTransformerEmbedder:
    """Semantic embeddings from a sentence-transformers model, used for the main retrieval experiments."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError("Install retrieval extras with: py -m pip install '.[retrieval]'") from error
        self._model = SentenceTransformer(model_name)
        self.name = model_name

    def embed(self, texts: Iterable[str]) -> list[list[float]]:
        vectors = self._model.encode(list(texts), normalize_embeddings=True)
        return [vector.tolist() for vector in vectors]