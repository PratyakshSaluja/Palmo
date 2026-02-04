"""
Tests for service layer.
"""

import pytest

from app.utils.text_splitter import TextSplitter


class TestTextSplitter:
    """Tests for the TextSplitter utility."""
    
    def test_split_text_basic(self):
        """Test basic text splitting."""
        splitter = TextSplitter(chunk_size=100, chunk_overlap=20)
        
        text = "This is a test document. " * 20
        chunks = splitter.split_text(text)
        
        assert len(chunks) > 0
        assert all(len(chunk) <= 100 + 50 for chunk in chunks)  # Allow some margin
    
    def test_split_empty_text(self):
        """Test splitting empty text."""
        splitter = TextSplitter()
        
        chunks = splitter.split_text("")
        assert chunks == []
        
        chunks = splitter.split_text("   ")
        assert chunks == []
    
    def test_split_short_text(self):
        """Test splitting text shorter than chunk size."""
        splitter = TextSplitter(chunk_size=1000)
        
        text = "Short text."
        chunks = splitter.split_text(text)
        
        assert len(chunks) == 1
        assert chunks[0] == "Short text."
    
    def test_create_chunks_with_metadata(self):
        """Test creating chunks with metadata."""
        splitter = TextSplitter(chunk_size=50, chunk_overlap=10)
        
        text = "This is a longer test document that should be split into multiple chunks for testing."
        chunks = splitter.create_chunks_with_metadata(
            text=text,
            document_id="doc-123",
            filename="test.txt",
        )
        
        assert len(chunks) > 0
        for i, chunk in enumerate(chunks):
            assert "content" in chunk
            assert "metadata" in chunk
            assert chunk["metadata"]["document_id"] == "doc-123"
            assert chunk["metadata"]["filename"] == "test.txt"
            assert chunk["metadata"]["chunk_index"] == i
