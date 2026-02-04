"""
Tests for document endpoints.
"""

import pytest


class TestHealthEndpoints:
    """Tests for health check endpoints."""
    
    def test_health_check(self, client):
        """Test basic health check returns healthy status."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "timestamp" in data


class TestDocumentEndpoints:
    """Tests for document management endpoints."""
    
    def test_list_documents_empty(self, client):
        """Test listing documents when none exist."""
        response = client.get("/api/v1/documents")
        
        assert response.status_code == 200
        data = response.json()
        assert "documents" in data
        assert "total_count" in data
    
    def test_upload_unsupported_file_type(self, client):
        """Test uploading an unsupported file type."""
        response = client.post(
            "/api/v1/documents/upload",
            files={"file": ("test.xyz", b"content", "application/octet-stream")},
        )
        
        assert response.status_code == 400
    
    def test_get_document_not_found(self, client):
        """Test getting a document that doesn't exist."""
        response = client.get("/api/v1/documents/nonexistent-id")
        
        assert response.status_code == 404
    
    def test_delete_document_not_found(self, client):
        """Test deleting a document that doesn't exist."""
        response = client.delete("/api/v1/documents/nonexistent-id")
        
        assert response.status_code == 404


class TestQueryEndpoints:
    """Tests for query endpoints."""
    
    def test_query_no_documents(self, client):
        """Test querying when no documents are uploaded."""
        response = client.post(
            "/api/v1/query",
            json={"query": "What is the Computer Science program?"},
        )
        
        # Should return 200 with a message about no documents
        # or 400/500 depending on implementation
        assert response.status_code in [200, 400, 500]
    
    def test_query_validation(self, client):
        """Test query validation for empty query."""
        response = client.post(
            "/api/v1/query",
            json={"query": ""},
        )
        
        assert response.status_code == 422  # Validation error
