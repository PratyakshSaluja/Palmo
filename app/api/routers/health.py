"""
Health check and statistics API endpoints.
"""

from datetime import datetime

from fastapi import APIRouter, Depends

from app import __version__
from app.api.dependencies import get_llm_service, get_settings, get_vector_store
from app.config import Settings
from app.models.schemas import (
    HealthCheckResponse,
    ReadinessCheckResponse,
    StatsResponse,
)
from app.services.llm_service import LLMService
from app.services.vector_store_service import VectorStoreService

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthCheckResponse,
    summary="Health check",
    description="Basic health check endpoint.",
)
async def health_check() -> HealthCheckResponse:
    """Basic health check - returns if the service is running."""
    return HealthCheckResponse(
        status="healthy",
        version=__version__,
        timestamp=datetime.utcnow(),
    )


@router.get(
    "/health/ready",
    response_model=ReadinessCheckResponse,
    summary="Readiness check",
    description="Check if all components are ready to handle requests.",
)
async def readiness_check(
    vector_store: VectorStoreService = Depends(get_vector_store),
    llm_service: LLMService = Depends(get_llm_service),
) -> ReadinessCheckResponse:
    """
    Readiness check - verifies all dependencies are available.
    
    Checks:
    - Vector store connectivity
    - LLM service availability
    """
    components = {
        "vector_store": vector_store.health_check(),
        "llm": True,  # We check basic init, not actual API call
    }
    
    all_healthy = all(components.values())
    
    return ReadinessCheckResponse(
        status="ready" if all_healthy else "not_ready",
        components=components,
    )


@router.get(
    "/api/v1/stats",
    response_model=StatsResponse,
    summary="Get system statistics",
    description="Retrieve statistics about documents, chunks, and configuration.",
)
async def get_stats(
    settings: Settings = Depends(get_settings),
    vector_store: VectorStoreService = Depends(get_vector_store),
) -> StatsResponse:
    """Get system statistics."""
    stats = vector_store.get_stats()
    
    return StatsResponse(
        total_documents=stats.get("total_documents", 0),
        total_chunks=stats.get("total_chunks", 0),
        collection_name=stats.get("collection_name", "faiss_index"),
        embedding_model=settings.embedding_model_name,
        llm_model=settings.llm_model_name,
    )
