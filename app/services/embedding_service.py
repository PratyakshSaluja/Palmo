"""
Embedding service for generating text embeddings.
Uses OpenAI embeddings for ingestion and query.
"""

from functools import lru_cache
from typing import List

from langchain_openai import OpenAIEmbeddings

from app.config import Settings
from app.core.exceptions import EmbeddingError
from app.core.logging import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    """
    Service for generating text embeddings using OpenAI embeddings API.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._embeddings = None
        logger.info(
            "embedding_service_initialized",
            model=settings.embedding_model_name,
        )

    @property
    def embeddings(self) -> OpenAIEmbeddings:
        """Lazy-init OpenAI embeddings client."""
        if self._embeddings is None:
            try:
                self._embeddings = OpenAIEmbeddings(
                    api_key=self.settings.openai_api_key,
                    model=self.settings.embedding_model_name,
                )
                logger.info(
                    "embedding_model_loaded",
                    model=self.settings.embedding_model_name,
                )
            except Exception as e:
                logger.error("embedding_model_load_failed", error=str(e))
                raise EmbeddingError(f"Failed to load embedding model: {e}")
        return self._embeddings

    def embed_text(self, text: str) -> List[float]:
        try:
            embedding = self.embeddings.embed_query(text)
            logger.debug("text_embedded", text_length=len(text), embedding_dim=len(embedding))
            return embedding
        except Exception as e:
            logger.error("embedding_failed", error=str(e))
            raise EmbeddingError(f"Failed to generate embedding: {e}")

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        try:
            embeddings = self.embeddings.embed_documents(texts)
            logger.info(
                "batch_embedding_completed",
                num_texts=len(texts),
                embedding_dim=len(embeddings[0]) if embeddings else 0,
            )
            return embeddings
        except Exception as e:
            logger.error("batch_embedding_failed", error=str(e), num_texts=len(texts))
            raise EmbeddingError(f"Failed to generate batch embeddings: {e}")

    def get_model_info(self) -> dict:
        return {
            "model_name": self.settings.embedding_model_name,
            "is_loaded": self._embeddings is not None,
        }


@lru_cache()
def get_embedding_service(settings: Settings) -> EmbeddingService:
    return EmbeddingService(settings)
