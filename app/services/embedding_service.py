"""
Embedding service for generating text embeddings.
Uses HuggingFace sentence-transformers for local embedding generation.
"""

from functools import lru_cache
from typing import List

from langchain_huggingface import HuggingFaceEmbeddings

from app.config import Settings
from app.core.exceptions import EmbeddingError
from app.core.logging import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    """
    Service for generating text embeddings using HuggingFace models.
    
    The embedding model is loaded once and cached for efficiency.
    """

    def __init__(self, settings: Settings):
        """
        Initialize the embedding service.
        
        Args:
            settings: Application settings.
        """
        self.settings = settings
        self._embeddings = None
        logger.info(
            "embedding_service_initialized",
            model=settings.embedding_model_name,
            device=settings.embedding_device,
        )

    @property
    def embeddings(self) -> HuggingFaceEmbeddings:
        """
        Get the HuggingFace embeddings instance.
        Lazy loads the model on first access.
        """
        if self._embeddings is None:
            try:
                self._embeddings = HuggingFaceEmbeddings(
                    model_name=self.settings.embedding_model_name,
                    model_kwargs={"device": self.settings.embedding_device},
                    encode_kwargs={"normalize_embeddings": True},
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
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed.
            
        Returns:
            Embedding vector as list of floats.
            
        Raises:
            EmbeddingError: If embedding generation fails.
        """
        try:
            embedding = self.embeddings.embed_query(text)
            logger.debug("text_embedded", text_length=len(text), embedding_dim=len(embedding))
            return embedding
        except Exception as e:
            logger.error("embedding_failed", error=str(e))
            raise EmbeddingError(f"Failed to generate embedding: {e}")

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed.
            
        Returns:
            List of embedding vectors.
            
        Raises:
            EmbeddingError: If embedding generation fails.
        """
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
        """
        Get information about the embedding model.
        
        Returns:
            Dictionary with model information.
        """
        return {
            "model_name": self.settings.embedding_model_name,
            "device": self.settings.embedding_device,
            "is_loaded": self._embeddings is not None,
        }


@lru_cache()
def get_embedding_service(settings: Settings) -> EmbeddingService:
    """
    Get cached embedding service instance.
    
    Args:
        settings: Application settings.
        
    Returns:
        Cached EmbeddingService instance.
    """
    return EmbeddingService(settings)
