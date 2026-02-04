"""
Pydantic schemas for API request/response validation.
These models define the contract between the API and clients.
"""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field

from app.models.enums import DocumentStatus, FileType


# ===========================================
# Base Models
# ===========================================


class TimestampMixin(BaseModel):
    """Mixin for timestamps."""

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None


# ===========================================
# Document Models
# ===========================================


class DocumentChunkInfo(BaseModel):
    """Information about a document chunk."""

    chunk_id: str = Field(..., description="Unique chunk identifier")
    content_preview: str = Field(..., description="Preview of chunk content (first 200 chars)")
    chunk_index: int = Field(..., ge=0, description="Index of chunk within document")


class DocumentInfo(BaseModel):
    """Information about an uploaded document."""

    document_id: str = Field(..., description="Unique document identifier")
    filename: str = Field(..., description="Original filename")
    file_type: FileType = Field(..., description="Document file type")
    file_size_bytes: int = Field(..., ge=0, description="File size in bytes")
    status: DocumentStatus = Field(..., description="Processing status")
    chunk_count: int = Field(default=0, ge=0, description="Number of text chunks")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class DocumentUploadResponse(BaseModel):
    """Response after successful document upload."""

    document_id: str = Field(..., description="Assigned document ID")
    filename: str = Field(..., description="Original filename")
    file_type: FileType = Field(..., description="Detected file type")
    chunk_count: int = Field(..., ge=0, description="Number of chunks created")
    message: str = Field(default="Document uploaded and processed successfully")


class DocumentListResponse(BaseModel):
    """Response containing list of documents."""

    documents: list[DocumentInfo] = Field(default_factory=list)
    total_count: int = Field(..., ge=0, description="Total number of documents")


class DocumentDetailResponse(BaseModel):
    """Detailed document response including chunks."""

    document: DocumentInfo
    chunks: list[DocumentChunkInfo] = Field(default_factory=list)


class BulkDeleteRequest(BaseModel):
    """Request to delete multiple documents."""

    document_ids: list[str] = Field(..., min_length=1, description="List of document IDs to delete")


class BulkDeleteResponse(BaseModel):
    """Response after bulk delete operation."""

    deleted_count: int = Field(..., ge=0)
    failed_ids: list[str] = Field(default_factory=list)
    message: str


# ===========================================
# Query Models
# ===========================================


class QueryRequest(BaseModel):
    """Request for querying the RAG system."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="User's question or query",
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Session ID for conversation history",
    )
    include_sources: bool = Field(
        default=True,
        description="Include source documents in response",
    )
    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=20,
        description="Number of source documents to retrieve (overrides default)",
    )


class SourceDocument(BaseModel):
    """Source document information returned with query response."""

    document_id: str = Field(..., description="Source document ID")
    filename: str = Field(..., description="Source document filename")
    chunk_id: str = Field(..., description="Specific chunk ID")
    content: str = Field(..., description="Relevant content excerpt")
    relevance_score: float = Field(..., ge=0, le=1, description="Relevance score")


class QueryResponse(BaseModel):
    """Response from the RAG query."""

    answer: str = Field(..., description="Generated answer")
    sources: list[SourceDocument] = Field(
        default_factory=list,
        description="Source documents used to generate answer",
    )
    session_id: str = Field(..., description="Session ID for this conversation")
    query: str = Field(..., description="Original query")


class ChatMessage(BaseModel):
    """A single chat message in conversation history."""

    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ConversationHistoryResponse(BaseModel):
    """Response containing conversation history."""

    session_id: str
    messages: list[ChatMessage] = Field(default_factory=list)


# ===========================================
# Health & Stats Models
# ===========================================


class HealthCheckResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Health status: 'healthy' or 'unhealthy'")
    version: str = Field(..., description="Application version")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ReadinessCheckResponse(BaseModel):
    """Readiness check response with component status."""

    status: str = Field(..., description="Overall status")
    components: dict[str, bool] = Field(
        default_factory=dict,
        description="Individual component status",
    )


class StatsResponse(BaseModel):
    """System statistics response."""

    total_documents: int = Field(..., ge=0)
    total_chunks: int = Field(..., ge=0)
    collection_name: str
    embedding_model: str
    llm_model: str


# ===========================================
# Error Models
# ===========================================


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str = Field(..., description="Error type/name")
    message: str = Field(..., description="Error message")
    details: dict[str, Any] = Field(default_factory=dict, description="Additional error details")
