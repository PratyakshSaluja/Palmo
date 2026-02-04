"""
RAG service for retrieval-augmented generation.
Orchestrates the retrieval and generation pipeline.
"""

from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple
import uuid

from app.config import Settings
from app.core.exceptions import QueryError
from app.core.logging import get_logger
from app.models.schemas import QueryResponse, SourceDocument
from app.services.llm_service import LLMService
from app.services.vector_store_service import VectorStoreService

logger = get_logger(__name__)


class RAGService:
    """
    Service for RAG (Retrieval-Augmented Generation) operations.
    
    Combines vector search with LLM generation for context-aware responses.
    """

    def __init__(
        self,
        settings: Settings,
        vector_store: VectorStoreService,
        llm_service: LLMService,
    ):
        """
        Initialize the RAG service.
        
        Args:
            settings: Application settings.
            vector_store: Vector store service for retrieval.
            llm_service: LLM service for generation.
        """
        self.settings = settings
        self.vector_store = vector_store
        self.llm_service = llm_service
        
        # Session-based conversation history
        # In production, use Redis or database
        self._sessions: Dict[str, List[Tuple[str, str]]] = {}
        
        logger.info("rag_service_initialized")

    def query(
        self,
        query: str,
        session_id: Optional[str] = None,
        include_sources: bool = True,
        top_k: Optional[int] = None,
    ) -> QueryResponse:
        """
        Process a query using RAG.
        
        Args:
            query: User's question.
            session_id: Optional session ID for conversation history.
            include_sources: Whether to include source documents.
            top_k: Number of documents to retrieve (overrides default).
            
        Returns:
            QueryResponse with answer and sources.
            
        Raises:
            QueryError: If query processing fails.
        """
        # Generate or use provided session ID
        if not session_id:
            session_id = str(uuid.uuid4())
        
        logger.info(
            "processing_query",
            session_id=session_id,
            query_length=len(query),
        )
        
        try:
            # Retrieve relevant documents
            k = top_k or self.settings.retriever_k
            logger.info(f"retrieving_documents", k=k, top_k=top_k, setting_k=self.settings.retriever_k)
            
            search_results = self.vector_store.query(
                query_text=query,
                n_results=k,
            )
            
            num_docs = len(search_results.get("documents", []))
            logger.info("retrieved_documents_count", count=num_docs, ids=search_results.get("ids", []))
            
            # Check if we have any results
            if not search_results.get("documents"):
                logger.warning("no_documents_found", query=query[:100])
                return QueryResponse(
                    answer="I don't have any documents to answer your question. Please upload some documents first.",
                    sources=[],
                    session_id=session_id,
                    query=query,
                )
            
            # Build context from retrieved documents
            context = self._build_context(search_results)
            
            # Get conversation history
            chat_history = self._sessions.get(session_id, [])
            
            # Generate response
            answer = self.llm_service.generate_response(
                query=query,
                context=context,
                chat_history=chat_history,
            )
            
            # Update conversation history
            self._update_history(session_id, query, answer)
            
            # Build source documents list
            sources = []
            if include_sources:
                sources = self._extract_sources(search_results)
            
            logger.info(
                "query_completed",
                session_id=session_id,
                sources_count=len(sources),
                answer_length=len(answer),
            )
            
            return QueryResponse(
                answer=answer,
                sources=sources,
                session_id=session_id,
                query=query,
            )
            
        except Exception as e:
            logger.error("query_failed", error=str(e))
            raise QueryError(str(e))

    async def query_stream(
        self,
        query: str,
        session_id: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Process a query with streaming response.
        
        Args:
            query: User's question.
            session_id: Optional session ID for conversation history.
            top_k: Number of documents to retrieve.
            
        Yields:
            Response chunks as they are generated.
            
        Raises:
            QueryError: If query processing fails.
        """
        # Generate or use provided session ID
        if not session_id:
            session_id = str(uuid.uuid4())
        
        try:
            # Get conversation history first to potentially expand query
            chat_history = self._sessions.get(session_id, [])
            
            # Expand vague follow-up queries with context from previous turn
            search_query = query
            vague_patterns = ["tell me more", "more about", "explain more", "what about", "and", "also"]
            if chat_history and any(p in query.lower() for p in vague_patterns):
                # Add previous query topic to search
                last_query = chat_history[-1][0]
                search_query = f"{last_query} {query}"
                logger.info("query_expanded", original=query, expanded=search_query)
            
            # Retrieve relevant documents
            k = top_k or self.settings.retriever_k
            search_results = self.vector_store.query(
                query_text=search_query,
                n_results=k,
            )
            
            # Check if we have any results
            if not search_results.get("documents"):
                yield "I don't have any documents to answer your question. Please upload some documents first."
                return
            
            # Build context
            context = self._build_context(search_results)
            
            # Log session info (chat_history already retrieved above)
            logger.info("streaming_query", session_id=session_id, history_length=len(chat_history), 
                       history_preview=chat_history[-2:] if chat_history else [])
            
            # Stream response
            full_response = []
            async for chunk in self.llm_service.generate_response_stream(
                query=query,
                context=context,
                chat_history=chat_history,
            ):
                full_response.append(chunk)
                yield chunk
            
            # Update conversation history
            answer = "".join(full_response)
            self._update_history(session_id, query, answer)
            
        except Exception as e:
            logger.error("streaming_query_failed", error=str(e))
            raise QueryError(str(e))

    def _build_context(self, search_results: Dict[str, Any]) -> str:
        """
        Build context string from search results.
        
        Args:
            search_results: Results from vector search.
            
        Returns:
            Formatted context string.
        """
        documents = search_results.get("documents", [])
        
        # Just join the document texts without any formatting
        # This prevents the LLM from outputting source tags
        return "\n\n".join(documents)

    def _extract_sources(self, search_results: Dict[str, Any]) -> List[SourceDocument]:
        """
        Extract source document information from search results.
        
        Args:
            search_results: Results from vector search.
            
        Returns:
            List of SourceDocument objects.
        """
        sources = []
        
        ids = search_results.get("ids", [])
        documents = search_results.get("documents", [])
        metadatas = search_results.get("metadatas", [])
        distances = search_results.get("distances", [])
        
        for i, chunk_id in enumerate(ids):
            metadata = metadatas[i] if i < len(metadatas) else {}
            distance = distances[i] if i < len(distances) else 1.0
            
            # Convert distance to relevance score (cosine similarity)
            # ChromaDB returns distance, lower is better
            relevance_score = max(0.0, min(1.0, 1.0 - distance))
            
            sources.append(SourceDocument(
                document_id=metadata.get("document_id", "unknown"),
                filename=metadata.get("filename", "unknown"),
                chunk_id=chunk_id,
                content=documents[i][:500] if i < len(documents) else "",  # Truncate
                relevance_score=relevance_score,
            ))
        
        return sources

    def _update_history(self, session_id: str, query: str, answer: str) -> None:
        """
        Update conversation history for a session.
        
        Args:
            session_id: Session identifier.
            query: User's question.
            answer: Generated answer.
        """
        if session_id not in self._sessions:
            self._sessions[session_id] = []
        
        self._sessions[session_id].append((query, answer))
        
        # Keep only last 10 exchanges per session
        if len(self._sessions[session_id]) > 10:
            self._sessions[session_id] = self._sessions[session_id][-10:]

    def get_session_history(self, session_id: str) -> List[Dict[str, str]]:
        """
        Get conversation history for a session.
        
        Args:
            session_id: Session identifier.
            
        Returns:
            List of message dictionaries.
        """
        history = self._sessions.get(session_id, [])
        
        messages = []
        for query, answer in history:
            messages.append({"role": "user", "content": query})
            messages.append({"role": "assistant", "content": answer})
        
        return messages

    def clear_session(self, session_id: str) -> bool:
        """
        Clear conversation history for a session.
        
        Args:
            session_id: Session identifier.
            
        Returns:
            True if session existed and was cleared.
        """
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False
