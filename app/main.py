"""
FastAPI application entry point.
Configures the application with routers, middleware, and lifecycle events.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import __version__
from app.api.routers import documents_router, health_router, query_router
from app.config import get_settings
from app.core.exceptions import BaseAppException
from app.core.logging import get_logger, setup_logging

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """
    Application lifespan manager.
    Handles startup and shutdown events.
    """
    # Startup
    settings = get_settings()
    setup_logging(debug=settings.debug)
    
    logger.info(
        "application_starting",
        version=__version__,
        debug=settings.debug,
    )
    
    # Ensure directories exist
    settings.ensure_directories()
    
    yield
    
    # Shutdown
    logger.info("application_shutting_down")


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        Configured FastAPI application instance.
    """
    settings = get_settings()
    
    app = FastAPI(
        title="University RAG Chatbot API",
        description="""
A production-ready RAG (Retrieval-Augmented Generation) chatbot API for university applications.

## Features

- 📄 **Document Management**: Upload, list, and delete PDF, DOCX, and TXT files
- 🔍 **Semantic Search**: ChromaDB-powered vector similarity search  
- 🤖 **AI-Powered Q&A**: Groq LLM with context-aware responses
- 🔄 **Streaming Responses**: Server-Sent Events for real-time answers
- 💬 **Conversation History**: Session-based chat memory

## Quick Start

1. Upload documents using `POST /api/v1/documents/upload`
2. Query the chatbot using `POST /api/v1/query`
3. Get streaming responses using `POST /api/v1/query/stream`
        """,
        version=__version__,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add exception handler
    @app.exception_handler(BaseAppException)
    async def app_exception_handler(
        request: Request,
        exc: BaseAppException,
    ) -> JSONResponse:
        """Handle custom application exceptions."""
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.to_dict(),
        )
    
    # Include routers
    app.include_router(health_router)
    app.include_router(documents_router, prefix="/api/v1")
    app.include_router(query_router, prefix="/api/v1")
    
    return app


# Create application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
    )