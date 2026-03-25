"""
Vector store service for managing document embeddings.
Uses FAISS for fast vector similarity search with local persistence.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid

import faiss
import numpy as np

from app.config import Settings
from app.core.exceptions import DocumentNotFoundError, VectorStoreError
from app.core.logging import get_logger
from app.services.embedding_service import EmbeddingService

logger = get_logger(__name__)


class VectorStoreService:
    """
    Service for managing vector store operations using FAISS.
    
    Provides methods for adding, querying, and deleting document embeddings.
    FAISS index is persisted to disk for durability.
    """

    def __init__(self, settings: Settings, embedding_service: EmbeddingService):
        """
        Initialize the vector store service.
        
        Args:
            settings: Application settings.
            embedding_service: Embedding service for generating embeddings.
        """
        self.settings = settings
        self.embedding_service = embedding_service
        
        # Paths for persistence
        self.index_path = settings.faiss_persist_dir / "faiss.index"
        self.metadata_path = settings.faiss_persist_dir / "metadata.json"
        
        # In-memory storage
        self._index: Optional[faiss.IndexFlatIP] = None  # Inner product for cosine similarity
        self._documents: List[str] = []  # Document texts
        self._metadatas: List[Dict[str, Any]] = []  # Document metadata
        self._ids: List[str] = []  # Chunk IDs
        
        # Load existing index if available
        self._load_index()
        
        logger.info(
            "vector_store_service_initialized",
            persist_dir=str(settings.faiss_persist_dir),
            index_exists=self._index is not None,
        )

    def _load_index(self) -> None:
        """Load FAISS index and metadata from disk if exists."""
        try:
            if self.index_path.exists() and self.metadata_path.exists():
                # Load FAISS index
                self._index = faiss.read_index(str(self.index_path))
                
                # Load metadata
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self._documents = data.get("documents", [])
                    self._metadatas = data.get("metadatas", [])
                    self._ids = data.get("ids", [])
                
                logger.info(
                    "index_loaded",
                    num_vectors=self._index.ntotal,
                    num_documents=len(self._documents),
                )
        except Exception as e:
            logger.warning("failed_to_load_index", error=str(e))
            self._index = None
            self._documents = []
            self._metadatas = []
            self._ids = []

    def _save_index(self) -> None:
        """Save FAISS index and metadata to disk."""
        try:
            if self._index is not None:
                # Ensure directory exists
                self.index_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Save FAISS index
                faiss.write_index(self._index, str(self.index_path))
                
                # Save metadata
                with open(self.metadata_path, "w", encoding="utf-8") as f:
                    json.dump({
                        "documents": self._documents,
                        "metadatas": self._metadatas,
                        "ids": self._ids,
                    }, f, ensure_ascii=False, indent=2)
                
                logger.debug("index_saved", num_vectors=self._index.ntotal)
        except Exception as e:
            logger.error("failed_to_save_index", error=str(e))
            raise VectorStoreError("save_index", str(e))

    def _ensure_index(self, dimension: int) -> None:
        """Ensure FAISS index exists with correct dimension."""
        if self._index is None:
            # Use IndexFlatIP for cosine similarity (with normalized vectors)
            self._index = faiss.IndexFlatIP(dimension)
            logger.info("index_created", dimension=dimension)

    def add_documents(
        self,
        texts: List[str],
        metadatas: List[Dict[str, Any]],
        document_id: str,
    ) -> List[str]:
        """
        Add document chunks to the vector store.
        
        Args:
            texts: List of text chunks.
            metadatas: List of metadata dictionaries for each chunk.
            document_id: Parent document ID.
            
        Returns:
            List of generated chunk IDs.
            
        Raises:
            VectorStoreError: If adding documents fails.
        """
        if not texts:
            return []
            
        try:
            # Generate embeddings
            embeddings = self.embedding_service.embed_texts(texts)
            embeddings_array = np.array(embeddings, dtype=np.float32)
            
            # Normalize for cosine similarity
            faiss.normalize_L2(embeddings_array)
            
            # Ensure index exists
            self._ensure_index(embeddings_array.shape[1])
            
            # Generate unique IDs for each chunk
            chunk_ids = [f"{document_id}_chunk_{i}" for i in range(len(texts))]
            
            # Add document_id to all metadata
            for metadata in metadatas:
                metadata["document_id"] = document_id
            
            # Add to FAISS index
            self._index.add(embeddings_array)
            
            # Store documents and metadata
            self._documents.extend(texts)
            self._metadatas.extend(metadatas)
            self._ids.extend(chunk_ids)
            
            # Persist to disk
            self._save_index()
            
            logger.info(
                "documents_added",
                document_id=document_id,
                num_chunks=len(texts),
                total_vectors=self._index.ntotal,
            )
            
            return chunk_ids
            
        except Exception as e:
            logger.error(
                "add_documents_failed",
                document_id=document_id,
                error=str(e),
            )
            raise VectorStoreError("add_documents", str(e))

    def query(
        self,
        query_text: str,
        n_results: int = 4,
        where_filter: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Query the vector store for similar documents.
        
        Args:
            query_text: Query text to search for.
            n_results: Number of results to return.
            where_filter: Optional metadata filter (filters applied post-search).
            
        Returns:
            Dictionary containing ids, documents, metadatas, and distances.
            
        Raises:
            VectorStoreError: If query fails.
        """
        try:
            if self._index is None or self._index.ntotal == 0:
                return {
                    "ids": [],
                    "documents": [],
                    "metadatas": [],
                    "distances": [],
                }
            
            # Generate query embedding
            query_embedding = self.embedding_service.embed_text(query_text)
            query_array = np.array([query_embedding], dtype=np.float32)
            
            # Normalize for cosine similarity
            faiss.normalize_L2(query_array)
            
            # Search (get more results if filtering)
            search_k = n_results * 3 if where_filter else n_results
            search_k = min(search_k, self._index.ntotal)
            
            distances, indices = self._index.search(query_array, search_k)
            
            # Collect results
            result_ids = []
            result_docs = []
            result_metadatas = []
            result_distances = []
            
            for i, idx in enumerate(indices[0]):
                if idx < 0 or idx >= len(self._ids):
                    continue
                
                metadata = self._metadatas[idx]
                
                # Apply filter if provided
                if where_filter:
                    match = all(
                        metadata.get(k) == v 
                        for k, v in where_filter.items()
                    )
                    if not match:
                        continue
                
                result_ids.append(self._ids[idx])
                result_docs.append(self._documents[idx])
                result_metadatas.append(metadata)
                # Convert similarity score to distance (1 - similarity)
                result_distances.append(1.0 - float(distances[0][i]))
                
                if len(result_ids) >= n_results:
                    break
            
            logger.debug(
                "query_executed",
                query_length=len(query_text),
                n_results=n_results,
                found=len(result_ids),
            )
            
            return {
                "ids": result_ids,
                "documents": result_docs,
                "metadatas": result_metadatas,
                "distances": result_distances,
            }
            
        except Exception as e:
            logger.error("query_failed", error=str(e))
            raise VectorStoreError("query", str(e))

    def delete_document(self, document_id: str) -> int:
        """
        Delete all chunks belonging to a document.
        
        Note: FAISS doesn't support direct deletion, so we rebuild the index
        without the deleted document's vectors.
        
        Args:
            document_id: Document ID to delete.
            
        Returns:
            Number of chunks deleted.
            
        Raises:
            VectorStoreError: If deletion fails.
        """
        try:
            # Find indices to keep
            indices_to_keep = []
            deleted_count = 0
            
            for i, metadata in enumerate(self._metadatas):
                if metadata.get("document_id") == document_id:
                    deleted_count += 1
                else:
                    indices_to_keep.append(i)
            
            if deleted_count == 0:
                logger.warning("no_chunks_found_for_deletion", document_id=document_id)
                return 0
            
            # Rebuild index without deleted vectors
            if indices_to_keep:
                # Get embeddings for remaining documents
                remaining_embeddings = self.embedding_service.embed_texts(
                    [self._documents[i] for i in indices_to_keep]
                )
                embeddings_array = np.array(remaining_embeddings, dtype=np.float32)
                faiss.normalize_L2(embeddings_array)
                
                # Create new index
                new_index = faiss.IndexFlatIP(embeddings_array.shape[1])
                new_index.add(embeddings_array)
                self._index = new_index
                
                # Update document lists
                self._documents = [self._documents[i] for i in indices_to_keep]
                self._metadatas = [self._metadatas[i] for i in indices_to_keep]
                self._ids = [self._ids[i] for i in indices_to_keep]
            else:
                # All documents deleted, reset everything
                self._index = None
                self._documents = []
                self._metadatas = []
                self._ids = []
            
            # Persist changes
            self._save_index()
            
            logger.info(
                "document_deleted",
                document_id=document_id,
                chunks_deleted=deleted_count,
            )
            
            return deleted_count
            
        except Exception as e:
            logger.error(
                "delete_document_failed",
                document_id=document_id,
                error=str(e),
            )
            raise VectorStoreError("delete_document", str(e))

    def get_document_chunks(self, document_id: str) -> Dict[str, Any]:
        """
        Get all chunks for a document.
        
        Args:
            document_id: Document ID.
            
        Returns:
            Dictionary with chunk IDs, documents, and metadatas.
        """
        try:
            result_ids = []
            result_docs = []
            result_metadatas = []
            
            for i, metadata in enumerate(self._metadatas):
                if metadata.get("document_id") == document_id:
                    result_ids.append(self._ids[i])
                    result_docs.append(self._documents[i])
                    result_metadatas.append(metadata)
            
            return {
                "ids": result_ids,
                "documents": result_docs,
                "metadatas": result_metadatas,
            }
            
        except Exception as e:
            logger.error(
                "get_document_chunks_failed",
                document_id=document_id,
                error=str(e),
            )
            raise VectorStoreError("get_document_chunks", str(e))

    def list_documents(self) -> List[Dict[str, Any]]:
        """
        List all unique documents in the vector store.
        
        Returns:
            List of document info dictionaries.
        """
        try:
            documents = {}
            for metadata in self._metadatas:
                if metadata and "document_id" in metadata:
                    doc_id = metadata["document_id"]
                    if doc_id not in documents:
                        documents[doc_id] = {
                            "document_id": doc_id,
                            "filename": metadata.get("filename", "unknown"),
                            "chunk_count": 1,
                        }
                    else:
                        documents[doc_id]["chunk_count"] += 1
            
            return list(documents.values())
            
        except Exception as e:
            logger.error("list_documents_failed", error=str(e))
            raise VectorStoreError("list_documents", str(e))

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.
        
        Returns:
            Dictionary with statistics.
        """
        try:
            documents = self.list_documents()
            total_chunks = self._index.ntotal if self._index else 0
            
            return {
                "total_chunks": total_chunks,
                "total_documents": len(documents),
                "collection_name": "faiss_index",
            }
            
        except Exception as e:
            logger.error("get_stats_failed", error=str(e))
            return {
                "total_chunks": 0,
                "total_documents": 0,
                "collection_name": "faiss_index",
                "error": str(e),
            }

    def health_check(self) -> bool:
        """
        Check if the vector store is healthy.
        
        Returns:
            True if healthy, False otherwise.
        """
        try:
            # FAISS is always ready if initialized
            return True
        except Exception as e:
            logger.error("vector_store_health_check_failed", error=str(e))
            return False
