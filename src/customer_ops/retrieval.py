from __future__ import annotations

from pathlib import Path

from sentence_transformers import SentenceTransformer, util

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
POLICY_SUFFIXES = {".md", ".txt"}


class PolicyTooLongError(ValueError):
    """Raised at load time so an oversized policy is never silently truncated."""


class PolicyIndex:
    """In-memory dense index; each policy file is one document and one entry."""

    def __init__(self, policy_dir: Path, model_name: str = DEFAULT_EMBEDDING_MODEL) -> None:
        self.model_name = model_name
        self._embedder = SentenceTransformer(model_name)
        self.documents: dict[str, str] = {}
        for path in sorted(Path(policy_dir).glob("*")):
            if path.suffix.lower() in POLICY_SUFFIXES and path.is_file():
                self.documents[path.name] = path.read_text(encoding="utf-8")
        self._check_lengths()
        self._ids = list(self.documents)
        self._embeddings = (
            self._embedder.encode(list(self.documents.values()), convert_to_tensor=True) if self._ids else None
        )

    def _check_lengths(self) -> None:
        limit = self._embedder.max_seq_length
        oversized = []
        for doc_id, text in self.documents.items():
            tokens = len(self._embedder.tokenizer(text, truncation=False)["input_ids"])
            if tokens > limit:
                oversized.append(f"{doc_id} ({tokens} tokens)")
        if oversized:
            raise PolicyTooLongError(
                f"Policies exceed the embedding model input limit of {limit} tokens: {', '.join(oversized)}. "
                "Shorten them or use an embedding model with a longer input limit."
            )

    def search(self, query: str) -> dict:
        if not self._ids:
            return {"error": "No policies are available."}
        query_embedding = self._embedder.encode(query, convert_to_tensor=True)
        hit = util.semantic_search(query_embedding, self._embeddings, top_k=1, score_function=util.cos_sim)[0][0]
        doc_id = self._ids[hit["corpus_id"]]
        return {"doc_id": doc_id, "score": round(float(hit["score"]), 4), "text": self.documents[doc_id]}
