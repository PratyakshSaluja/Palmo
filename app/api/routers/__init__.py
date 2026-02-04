"""Routers module exports."""

from app.api.routers.documents import router as documents_router
from app.api.routers.health import router as health_router
from app.api.routers.query import router as query_router

__all__ = [
    "documents_router",
    "health_router",
    "query_router",
]
