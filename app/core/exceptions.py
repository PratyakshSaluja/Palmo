"""
Custom exception classes for the RAG chatbot application.
Provides clear error handling with HTTP status codes for API responses.
"""

from typing import Any, Optional


class BaseAppException(Exception):
    """
    Base exception class for the application.
    All custom exceptions should inherit from this class.
    """

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)

    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary for API response."""
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "details": self.details,
        }


class DocumentNotFoundError(BaseAppException):
    """Raised when a document is not found in the system."""

    def __init__(self, document_id: str) -> None:
        super().__init__(
            message=f"Document with ID '{document_id}' not found",
            status_code=404,
            details={"document_id": document_id},
        )


class DocumentProcessingError(BaseAppException):
    """Raised when document processing fails."""

    def __init__(self, filename: str, reason: str) -> None:
        super().__init__(
            message=f"Failed to process document '{filename}': {reason}",
            status_code=400,
            details={"filename": filename, "reason": reason},
        )


class UnsupportedFileTypeError(BaseAppException):
    """Raised when an unsupported file type is uploaded."""

    def __init__(self, filename: str, file_type: str, allowed_types: list[str]) -> None:
        super().__init__(
            message=f"Unsupported file type '{file_type}' for file '{filename}'",
            status_code=400,
            details={
                "filename": filename,
                "file_type": file_type,
                "allowed_types": allowed_types,
            },
        )


class EmbeddingError(BaseAppException):
    """Raised when embedding generation fails."""

    def __init__(self, reason: str) -> None:
        super().__init__(
            message=f"Embedding generation failed: {reason}",
            status_code=500,
            details={"reason": reason},
        )


class VectorStoreError(BaseAppException):
    """Raised when vector store operations fail."""

    def __init__(self, operation: str, reason: str) -> None:
        super().__init__(
            message=f"Vector store {operation} failed: {reason}",
            status_code=500,
            details={"operation": operation, "reason": reason},
        )


class LLMError(BaseAppException):
    """Raised when LLM operations fail."""

    def __init__(self, reason: str) -> None:
        super().__init__(
            message=f"LLM operation failed: {reason}",
            status_code=500,
            details={"reason": reason},
        )


class FileTooLargeError(BaseAppException):
    """Raised when uploaded file exceeds size limit."""

    def __init__(self, filename: str, size_mb: float, max_size_mb: int) -> None:
        super().__init__(
            message=f"File '{filename}' ({size_mb:.1f}MB) exceeds maximum size ({max_size_mb}MB)",
            status_code=413,
            details={
                "filename": filename,
                "size_mb": size_mb,
                "max_size_mb": max_size_mb,
            },
        )


class QueryError(BaseAppException):
    """Raised when query processing fails."""

    def __init__(self, reason: str) -> None:
        super().__init__(
            message=f"Query processing failed: {reason}",
            status_code=400,
            details={"reason": reason},
        )
