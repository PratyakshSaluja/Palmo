"""
Enumeration types for the RAG chatbot application.
"""

from enum import Enum


class FileType(str, Enum):
    """Supported file types for document upload."""

    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"

    @classmethod
    def from_extension(cls, extension: str) -> "FileType":
        """
        Get FileType from file extension string.
        
        Args:
            extension: File extension without dot (e.g., 'pdf').
            
        Returns:
            Corresponding FileType enum value.
            
        Raises:
            ValueError: If extension is not supported.
        """
        ext_lower = extension.lower().lstrip(".")
        try:
            return cls(ext_lower)
        except ValueError:
            raise ValueError(f"Unsupported file extension: {extension}")

    @classmethod
    def get_mime_types(cls) -> dict[str, "FileType"]:
        """Get mapping of MIME types to FileType."""
        return {
            "application/pdf": cls.PDF,
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": cls.DOCX,
            "text/plain": cls.TXT,
        }


class DocumentStatus(str, Enum):
    """Document processing status."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
