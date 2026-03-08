"""
Text splitter for chunking documents into smaller pieces.
Uses LangChain's RecursiveCharacterTextSplitter for intelligent splitting.
"""

from typing import Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.logging import get_logger

logger = get_logger(__name__)


class TextSplitter:
    """
    Text splitter for breaking documents into chunks.
    
    Uses recursive character splitting which attempts to split on paragraph
    boundaries, sentences, and words in that order of preference.
    """

    # Default separators in priority order
    DEFAULT_SEPARATORS = [
        "\n\n",  # Paragraph breaks
        "\n",    # Line breaks
        ". ",    # Sentence endings
        "! ",    # Exclamation endings
        "? ",    # Question endings
        "; ",    # Semicolon
        ", ",    # Comma
        " ",     # Word boundaries
        "",      # Last resort: character by character
    ]

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[list[str]] = None,
    ):
        """
        Initialize the text splitter.
        
        Args:
            chunk_size: Maximum size of each chunk in characters.
            chunk_overlap: Number of characters to overlap between chunks.
            separators: Custom list of separators to use for splitting.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or self.DEFAULT_SEPARATORS
        
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=self.separators,
            length_function=len,
            is_separator_regex=False,
        )
        
        logger.info(
            "text_splitter_initialized",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def split_text(self, text: str) -> list[str]:
        """
        Split text into chunks.
        
        Args:
            text: Text content to split.
            
        Returns:
            List of text chunks.
        """
        if not text or not text.strip():
            logger.warning("empty_text_provided")
            return []
            
        # Clean the text
        cleaned_text = self._clean_text(text)
        
        # Split into chunks
        chunks = self._splitter.split_text(cleaned_text)
        
        # Filter out empty chunks
        chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
        
        logger.info(
            "text_split_completed",
            original_length=len(text),
            num_chunks=len(chunks),
            avg_chunk_size=sum(len(c) for c in chunks) // len(chunks) if chunks else 0,
        )
        
        return chunks

    def _clean_text(self, text: str) -> str:
        """
        Clean text by removing excessive whitespace.
        
        Args:
            text: Raw text to clean.
            
        Returns:
            Cleaned text.
        """
        # Replace multiple newlines with double newlines
        import re
        text = re.sub(r"\n{3,}", "\n\n", text)
        
        # Replace multiple spaces with single space
        text = re.sub(r" {2,}", " ", text)
        
        # Strip leading/trailing whitespace from each line
        lines = [line.strip() for line in text.split("\n")]
        text = "\n".join(lines)
        
        return text.strip()

    def create_chunks_with_metadata(
        self,
        text: str,
        document_id: str,
        filename: str,
    ) -> list[dict]:
        """
        Split text and create chunk dictionaries with metadata.
        
        Args:
            text: Text content to split.
            document_id: ID of the source document.
            filename: Name of the source file.
            
        Returns:
            List of chunk dictionaries with content and metadata.
        """
        chunks = self.split_text(text)
        
        chunk_dicts = []
        for i, chunk in enumerate(chunks):
            chunk_dict = {
                "content": chunk,
                "metadata": {
                    "document_id": document_id,
                    "filename": filename,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                },
            }
            chunk_dicts.append(chunk_dict)
            
        return chunk_dicts