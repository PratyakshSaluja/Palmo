"""
Document service for handling document upload and management.
Orchestrates document parsing, chunking, and storage in the vector store.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional
import uuid

from app.config import Settings
from app.core.exceptions import (
    DocumentNotFoundError,
    DocumentProcessingError,
    FileTooLargeError,
)
from app.core.logging import get_logger
from app.models.enums import DocumentStatus, FileType
from app.models.schemas import DocumentInfo, DocumentUploadResponse
from app.services.vector_store_service import VectorStoreService
from app.utils.document_parser import DocumentParser
from app.utils.text_splitter import TextSplitter

logger = get_logger(__name__)


class DocumentService:
    """
    Service for managing document lifecycle.
    
    Handles document upload, processing, storage, and deletion.
    """

    def __init__(
        self,
        settings: Settings,
        vector_store: VectorStoreService,
    ):
        """
        Initialize the document service.
        
        Args:
            settings: Application settings.
            vector_store: Vector store service for storing embeddings.
        """
        self.settings = settings
        self.vector_store = vector_store
        self.parser = DocumentParser(allowed_types=settings.allowed_extensions)
        self.splitter = TextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        
        # In-memory document metadata store
        # In production, this should be a database
        self._documents: Dict[str, DocumentInfo] = {}
        
        logger.info("document_service_initialized")

    async def upload_document(
        self,
        file_content: bytes,
        filename: str,
        content_type: Optional[str] = None,
    ) -> DocumentUploadResponse:
        """
        Upload and process a document.
        
        Args:
            file_content: Raw file bytes.
            filename: Original filename.
            content_type: MIME type of the file.
            
        Returns:
            DocumentUploadResponse with processing results.
            
        Raises:
            FileTooLargeError: If file exceeds size limit.
            DocumentProcessingError: If processing fails.
        """
        # Check file size
        file_size_mb = len(file_content) / (1024 * 1024)
        if file_size_mb > self.settings.max_upload_size_mb:
            raise FileTooLargeError(
                filename=filename,
                size_mb=file_size_mb,
                max_size_mb=self.settings.max_upload_size_mb,
            )
        
        # Generate document ID
        document_id = str(uuid.uuid4())
        
        logger.info(
            "processing_document",
            document_id=document_id,
            filename=filename,
            size_bytes=len(file_content),
        )
        
        try:
            # Parse document
            file_type = self.parser.get_file_type(filename)
            text_content = self.parser.parse(file_content, filename)
            
            if not text_content.strip():
                raise DocumentProcessingError(
                    filename=filename,
                    reason="No text content could be extracted from the document",
                )
            
            # Split into chunks
            chunks = self.splitter.create_chunks_with_metadata(
                text=text_content,
                document_id=document_id,
                filename=filename,
            )
            
            if not chunks:
                raise DocumentProcessingError(
                    filename=filename,
                    reason="Document could not be split into chunks",
                )
            
            # Store in vector store
            chunk_texts = [c["content"] for c in chunks]
            chunk_metadatas = [c["metadata"] for c in chunks]
            
            self.vector_store.add_documents(
                texts=chunk_texts,
                metadatas=chunk_metadatas,
                document_id=document_id,
            )
            
            # Store document metadata
            doc_info = DocumentInfo(
                document_id=document_id,
                filename=filename,
                file_type=file_type,
                file_size_bytes=len(file_content),
                status=DocumentStatus.COMPLETED,
                chunk_count=len(chunks),
                created_at=datetime.utcnow(),
                metadata={
                    "content_type": content_type,
                    "text_length": len(text_content),
                },
            )
            self._documents[document_id] = doc_info
            
            logger.info(
                "document_processed",
                document_id=document_id,
                filename=filename,
                chunks=len(chunks),
            )
            
            return DocumentUploadResponse(
                document_id=document_id,
                filename=filename,
                file_type=file_type,
                chunk_count=len(chunks),
                message="Document uploaded and processed successfully",
            )
            
        except (FileTooLargeError, DocumentProcessingError):
            raise
        except Exception as e:
            logger.error(
                "document_processing_failed",
                document_id=document_id,
                filename=filename,
                error=str(e),
            )
            raise DocumentProcessingError(filename=filename, reason=str(e))

    def get_document(self, document_id: str) -> DocumentInfo:
        """
        Get document information by ID.
        
        Args:
            document_id: Document ID to retrieve.
            
        Returns:
            DocumentInfo for the document.
            
        Raises:
            DocumentNotFoundError: If document not found.
        """
        # First check in-memory store
        if document_id in self._documents:
            return self._documents[document_id]
        
        # Check vector store
        chunks = self.vector_store.get_document_chunks(document_id)
        if not chunks.get("ids"):
            raise DocumentNotFoundError(document_id)
        
        # Reconstruct document info from chunks
        metadata = chunks.get("metadatas", [{}])[0] if chunks.get("metadatas") else {}
        doc_info = DocumentInfo(
            document_id=document_id,
            filename=metadata.get("filename", "unknown"),
            file_type=FileType.TXT,  # Default, not stored in chunks
            file_size_bytes=0,
            status=DocumentStatus.COMPLETED,
            chunk_count=len(chunks.get("ids", [])),
            created_at=datetime.utcnow(),
        )
        
        return doc_info

    def list_documents(self) -> List[DocumentInfo]:
        """
        List all documents.
        
        Returns:
            List of DocumentInfo for all documents.
        """
        # Get documents from vector store
        vector_docs = self.vector_store.list_documents()
        
        # Merge with in-memory store
        documents = []
        seen_ids = set()
        
        for doc in vector_docs:
            doc_id = doc["document_id"]
            seen_ids.add(doc_id)
            
            if doc_id in self._documents:
                documents.append(self._documents[doc_id])
            else:
                # Create basic info from vector store data
                doc_info = DocumentInfo(
                    document_id=doc_id,
                    filename=doc.get("filename", "unknown"),
                    file_type=FileType.TXT,
                    file_size_bytes=0,
                    status=DocumentStatus.COMPLETED,
                    chunk_count=doc.get("chunk_count", 0),
                    created_at=datetime.utcnow(),
                )
                documents.append(doc_info)
        
        return documents

    def delete_document(self, document_id: str) -> bool:
        """
        Delete a document and its embeddings.
        
        Args:
            document_id: Document ID to delete.
            
        Returns:
            True if deleted successfully.
            
        Raises:
            DocumentNotFoundError: If document not found.
        """
        # Check if document exists
        try:
            self.get_document(document_id)
        except DocumentNotFoundError:
            raise
        
        # Delete from vector store
        deleted_count = self.vector_store.delete_document(document_id)
        
        # Remove from in-memory store
        if document_id in self._documents:
            del self._documents[document_id]
        
        logger.info(
            "document_deleted",
            document_id=document_id,
            chunks_deleted=deleted_count,
        )
        
        return True

    def bulk_delete_documents(self, document_ids: List[str]) -> Dict[str, Any]:
        """
        Delete multiple documents.
        
        Args:
            document_ids: List of document IDs to delete.
            
        Returns:
            Dictionary with deleted_count and failed_ids.
        """
        deleted_count = 0
        failed_ids = []
        
        for doc_id in document_ids:
            try:
                self.delete_document(doc_id)
                deleted_count += 1
            except DocumentNotFoundError:
                failed_ids.append(doc_id)
            except Exception as e:
                logger.error(
                    "bulk_delete_document_failed",
                    document_id=doc_id,
                    error=str(e),
                )
                failed_ids.append(doc_id)
        
        return {
            "deleted_count": deleted_count,
            "failed_ids": failed_ids,
        }

    def get_document_chunks(self, document_id: str) -> List[Dict[str, Any]]:
        """
        Get all chunks for a document.
        
        Args:
            document_id: Document ID.
            
        Returns:
            List of chunk dictionaries.
            
        Raises:
            DocumentNotFoundError: If document not found.
        """
        chunks = self.vector_store.get_document_chunks(document_id)
        
        if not chunks.get("ids"):
            raise DocumentNotFoundError(document_id)
        
        result = []
        for i, chunk_id in enumerate(chunks["ids"]):
            result.append({
                "chunk_id": chunk_id,
                "content": chunks["documents"][i] if chunks["documents"] else "",
                "metadata": chunks["metadatas"][i] if chunks["metadatas"] else {},
            })
        
        return result
