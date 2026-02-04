"""
Test configuration and fixtures.
"""

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import reset_services
from app.main import app


@pytest.fixture(scope="function")
def client():
    """Create a test client for the FastAPI application."""
    # Reset services before each test
    reset_services()
    
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def sample_pdf_content() -> bytes:
    """Generate sample PDF-like content for testing."""
    # This is a minimal PDF structure for testing
    # In real tests, you'd use a proper test PDF file
    return b"%PDF-1.4\nTest content for PDF parsing tests."


@pytest.fixture
def sample_txt_content() -> bytes:
    """Generate sample text content for testing."""
    return b"""
    University RAG Chatbot Documentation
    
    This is a test document for the university RAG chatbot.
    It contains information about courses, programs, and campus facilities.
    
    Computer Science Program:
    - Introduction to Programming
    - Data Structures and Algorithms
    - Machine Learning
    - Artificial Intelligence
    
    Campus Facilities:
    - Main Library
    - Student Center
    - Research Labs
    """
