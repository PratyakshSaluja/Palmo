"""
Dependency injection for FastAPI.
Provides singleton instances of services for API endpoints.
"""

from functools import lru_cache
from typing import Generator

from app.config import Settings, get_settings as _get_settings
from app.services.document_service import DocumentService
from app.services.embedding_service import EmbeddingService
from app.services.llm_service import LLMService
from app.services.rag_service import RAGService
from app.services.vector_store_service import VectorStoreService


def get_settings() -> Settings:
    """Get application settings."""
    return _get_settings()


# Service singletons
_embedding_service: EmbeddingService | None = None
_vector_store: VectorStoreService | None = None
_llm_service: LLMService | None = None
_document_service: DocumentService | None = None
_rag_service: RAGService | None = None


def get_embedding_service() -> EmbeddingService:
    """Get embedding service singleton."""
    global _embedding_service
    if _embedding_service is None:
        settings = get_settings()
        _embedding_service = EmbeddingService(settings)
    return _embedding_service


def get_vector_store() -> VectorStoreService:
    """Get vector store service singleton."""
    global _vector_store
    if _vector_store is None:
        settings = get_settings()
        embedding_service = get_embedding_service()
        _vector_store = VectorStoreService(settings, embedding_service)
    return _vector_store


def get_llm_service() -> LLMService:
    """Get LLM service singleton."""
    global _llm_service
    if _llm_service is None:
        settings = get_settings()
        _llm_service = LLMService(settings)
    return _llm_service


def get_document_service() -> DocumentService:
    """Get document service singleton."""
    global _document_service
    if _document_service is None:
        settings = get_settings()
        vector_store = get_vector_store()
        _document_service = DocumentService(settings, vector_store)
    return _document_service


def get_rag_service() -> RAGService:
    """Get RAG service singleton."""
    global _rag_service
    if _rag_service is None:
        settings = get_settings()
        vector_store = get_vector_store()
        llm_service = get_llm_service()
        _rag_service = RAGService(settings, vector_store, llm_service)
    return _rag_service


def reset_services() -> None:
    """
    Reset all service singletons.
    Useful for testing.
    """
    global _embedding_service, _vector_store, _llm_service
    global _document_service, _rag_service
    
    _embedding_service = None
    _vector_store = None
    _llm_service = None
    _document_service = None
    _rag_service = None
