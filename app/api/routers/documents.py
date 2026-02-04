"""
Document management API endpoints.
Handles document upload, listing, retrieval, and deletion.
"""

from typing import List

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.api.dependencies import get_document_service
from app.core.exceptions import (
    BaseAppException,
    DocumentNotFoundError,
    DocumentProcessingError,
    FileTooLargeError,
    UnsupportedFileTypeError,
)
from app.models.schemas import (
    BulkDeleteRequest,
    BulkDeleteResponse,
    DocumentChunkInfo,
    DocumentDetailResponse,
    DocumentInfo,
    DocumentListResponse,
    DocumentUploadResponse,
    ErrorResponse,
)
from app.services.document_service import DocumentService

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Bad request"},
        413: {"model": ErrorResponse, "description": "File too large"},
    },
    summary="Upload a document",
    description="Upload a PDF, DOCX, or TXT document for processing and indexing.",
)
async def upload_document(
    file: UploadFile = File(..., description="Document file to upload"),
    document_service: DocumentService = Depends(get_document_service),
) -> DocumentUploadResponse:
    """
    Upload and process a document.
    
    The document will be:
    1. Validated for file type and size
    2. Parsed to extract text content
    3. Split into chunks
    4. Indexed in the vector store for semantic search
    """
    try:
        # Read file content
        content = await file.read()
        
        # Process document
        result = await document_service.upload_document(
            file_content=content,
            filename=file.filename or "unknown",
            content_type=file.content_type,
        )
        
        return result
        
    except UnsupportedFileTypeError as e:
        raise HTTPException(status_code=400, detail=e.to_dict())
    except FileTooLargeError as e:
        raise HTTPException(status_code=413, detail=e.to_dict())
    except DocumentProcessingError as e:
        raise HTTPException(status_code=400, detail=e.to_dict())
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"error": "InternalError", "message": str(e)},
        )


@router.get(
    "",
    response_model=DocumentListResponse,
    summary="List all documents",
    description="Retrieve a list of all uploaded documents with their metadata.",
)
async def list_documents(
    document_service: DocumentService = Depends(get_document_service),
) -> DocumentListResponse:
    """List all uploaded documents."""
    documents = document_service.list_documents()
    
    return DocumentListResponse(
        documents=documents,
        total_count=len(documents),
    )


@router.get(
    "/{document_id}",
    response_model=DocumentInfo,
    responses={
        404: {"model": ErrorResponse, "description": "Document not found"},
    },
    summary="Get document details",
    description="Retrieve details for a specific document by ID.",
)
async def get_document(
    document_id: str,
    document_service: DocumentService = Depends(get_document_service),
) -> DocumentInfo:
    """Get document details by ID."""
    try:
        return document_service.get_document(document_id)
    except DocumentNotFoundError as e:
        raise HTTPException(status_code=404, detail=e.to_dict())


@router.get(
    "/{document_id}/chunks",
    response_model=DocumentDetailResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Document not found"},
    },
    summary="Get document with chunks",
    description="Retrieve a document with all its text chunks.",
)
async def get_document_with_chunks(
    document_id: str,
    document_service: DocumentService = Depends(get_document_service),
) -> DocumentDetailResponse:
    """Get document details including all chunks."""
    try:
        document = document_service.get_document(document_id)
        chunks_data = document_service.get_document_chunks(document_id)
        
        chunks = [
            DocumentChunkInfo(
                chunk_id=c["chunk_id"],
                content_preview=c["content"][:200] + "..." if len(c["content"]) > 200 else c["content"],
                chunk_index=c["metadata"].get("chunk_index", 0),
            )
            for c in chunks_data
        ]
        
        return DocumentDetailResponse(document=document, chunks=chunks)
        
    except DocumentNotFoundError as e:
        raise HTTPException(status_code=404, detail=e.to_dict())


@router.delete(
    "/{document_id}",
    responses={
        200: {"description": "Document deleted successfully"},
        404: {"model": ErrorResponse, "description": "Document not found"},
    },
    summary="Delete a document",
    description="Delete a document and all its associated embeddings.",
)
async def delete_document(
    document_id: str,
    document_service: DocumentService = Depends(get_document_service),
) -> dict:
    """Delete a document by ID."""
    try:
        document_service.delete_document(document_id)
        return {"message": "Document deleted successfully", "document_id": document_id}
    except DocumentNotFoundError as e:
        raise HTTPException(status_code=404, detail=e.to_dict())


@router.post(
    "/bulk-delete",
    response_model=BulkDeleteResponse,
    summary="Delete multiple documents",
    description="Delete multiple documents at once by their IDs.",
)
async def bulk_delete_documents(
    request: BulkDeleteRequest,
    document_service: DocumentService = Depends(get_document_service),
) -> BulkDeleteResponse:
    """Delete multiple documents."""
    result = document_service.bulk_delete_documents(request.document_ids)
    
    return BulkDeleteResponse(
        deleted_count=result["deleted_count"],
        failed_ids=result["failed_ids"],
        message=f"Deleted {result['deleted_count']} documents",
    )
