"""
Semantic cache for RAG responses.
Caches answers by query meaning — similar questions get cached responses
without hitting the LLM again.
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np

from app.core.logging import get_logger
from app.services.embedding_service import EmbeddingService

logger = get_logger(__name__)

CACHE_PATH = Path("./data/semantic_cache.json")
SIMILARITY_THRESHOLD = 0.93  # Cosine similarity — tune up if getting wrong hits


class SemanticCache:
    """
    Caches LLM responses keyed by query embedding similarity.

    On each query:
      - Embeds the query and compares against all cached embeddings.
      - If a match above the threshold is found, returns the cached response.
      - Otherwise returns None and the caller should populate the cache after generation.
    """

    def __init__(self, embedding_service: EmbeddingService):
        self.embedding_service = embedding_service
        # Each entry: {"query": str, "embedding": list[float], "response": str}
        self._entries: list = []
        self._load()
        logger.info("semantic_cache_initialized", entries=len(self._entries))

    def get(self, query: str) -> Optional[str]:
        """Return cached response if a semantically similar query exists."""
        if not self._entries:
            return None

        query_emb = self._embed(query)
        cached_embs = np.array([e["embedding"] for e in self._entries], dtype=np.float32)
        scores = cached_embs @ query_emb  # cosine similarity (vectors are already normalized)

        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])

        if best_score >= SIMILARITY_THRESHOLD:
            logger.info(
                "semantic_cache_hit",
                score=round(best_score, 3),
                matched_query=self._entries[best_idx]["query"][:80],
            )
            return self._entries[best_idx]["response"]

        return None

    def set(self, query: str, response: str) -> None:
        """Store a query-response pair in the cache."""
        self._entries.append({
            "query": query,
            "embedding": self._embed(query).tolist(),
            "response": response,
        })
        self._save()
        logger.info("semantic_cache_stored", total_entries=len(self._entries))

    def _embed(self, text: str) -> np.ndarray:
        emb = np.array(self.embedding_service.embed_text(text), dtype=np.float32)
        emb /= np.linalg.norm(emb) + 1e-10  # normalize for cosine similarity
        return emb

    def _load(self) -> None:
        try:
            if CACHE_PATH.exists():
                with open(CACHE_PATH, "r", encoding="utf-8") as f:
                    self._entries = json.load(f)
        except Exception as e:
            logger.warning("semantic_cache_load_failed", error=str(e))
            self._entries = []

    def _save(self) -> None:
        try:
            CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump(self._entries, f, ensure_ascii=False)
        except Exception as e:
            logger.warning("semantic_cache_save_failed", error=str(e))
