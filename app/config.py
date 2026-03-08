"""
Configuration management using Pydantic Settings.
Loads configuration from environment variables and .env file.
"""

from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ===========================================
    # LLM Configuration
    # ===========================================
    groq_api_key: str = Field(..., description="Groq API key for LLM access")
    llm_model_name: str = Field(
        default="llama3-70b-8192",
        description="Groq model name to use",
    )
    llm_temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="LLM temperature for response generation",
    )
    llm_max_tokens: int = Field(
        default=2048,
        ge=100,
        le=8192,
        description="Maximum tokens in LLM response",
    )

    # ===========================================
    # Embedding Configuration
    # ===========================================
    embedding_model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="HuggingFace embedding model name",
    )
    embedding_device: str = Field(
        default="cpu",
        description="Device for embedding computation (cpu/cuda)",
    )

    # ===========================================
    # Vector Store Configuration (FAISS)
    # ===========================================
    chroma_persist_dir: Path = Field(
        default=Path("./data/faiss_index"),
        description="FAISS index persistence directory",
    )
    chroma_collection_name: str = Field(
        default="university_docs",
        description="Collection name (for compatibility)",
    )

    # ===========================================
    # Document Processing Configuration
    # ===========================================
    chunk_size: int = Field(
        default=1000,
        ge=100,
        le=4000,
        description="Text chunk size for splitting",
    )
    chunk_overlap: int = Field(
        default=200,
        ge=0,
        le=500,
        description="Overlap between text chunks",
    )
    max_upload_size_mb: int = Field(
        default=50,
        ge=1,
        le=100,
        description="Maximum upload file size in MB",
    )
    upload_dir: Path = Field(
        default=Path("./data/uploads"),
        description="Directory for temporary file uploads",
    )
    allowed_extensions: List[str] = Field(
        default=["pdf", "docx", "txt"],
        description="Allowed file extensions for upload",
    )

    # ===========================================
    # API Configuration
    # ===========================================
    api_host: str = Field(default="0.0.0.0", description="API host address")
    api_port: int = Field(default=8000, ge=1, le=65535, description="API port")
    debug: bool = Field(default=False, description="Enable debug mode")
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080"],
        description="Allowed CORS origins",
    )

    # ===========================================
    # RAG Configuration
    # ===========================================
    retriever_k: int = Field(
        default=6,
        ge=1,
        le=20,
        description="Number of documents to retrieve",
    )

    @field_validator("chroma_persist_dir", "upload_dir", mode="before")
    @classmethod
    def ensure_path(cls, v):
        """Convert string to Path if needed."""
        if isinstance(v, str):
            return Path(v)
        return v

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        """Parse CORS origins from string or list."""
        if isinstance(v, str):
            # Handle JSON-formatted string from env
            import json
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                # Fallback to comma-separated
                return [origin.strip() for origin in v.split(",")]
        return v

    def ensure_directories(self) -> None:
        """Ensure required directories exist."""
        self.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    Uses LRU cache to avoid reloading settings on every request.
    """
    settings = Settings()
    settings.ensure_directories()
    return settings
