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
from app.services.semantic_cache import SemanticCache
from app.services.vector_store_service import VectorStoreService

logger = get_logger(__name__)

# Chunks with distance above this threshold are too dissimilar to be useful

# Patterns that indicate a vague follow-up needing query expansion
VAGUE_PATTERNS = ["tell me more", "more about", "explain more", "what about"]

# Patterns that indicate a meta-question about the conversation itself
HISTORY_PATTERNS = [
    "previous question", "last question", "what did i ask",
    "what was my question", "my last query", "what did you say",
    "previous answer", "last answer", "what did you just",
]


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
        self.settings = settings
        self.vector_store = vector_store
        self.llm_service = llm_service
        self.semantic_cache = SemanticCache(vector_store.embedding_service)

        # Session-based conversation history
        # In production, use Redis or database
        self._sessions: Dict[str, List[Tuple[str, str]]] = {}

        logger.info("rag_service_initialized")

    async def query(
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
        if not session_id:
            session_id = str(uuid.uuid4())

        logger.info(
            "processing_query",
            session_id=session_id,
            query_length=len(query),
        )

        try:
            # Get conversation history
            chat_history = self._sessions.get(session_id, [])

            # Short-circuit: answer meta/history questions without hitting vector store
            if self._is_history_query(query):
                return self._answer_from_history(query, chat_history, session_id)

            # Check semantic cache before retrieval + LLM call
            cached = self.semantic_cache.get(query)
            if cached is not None:
                self._update_history(session_id, query, cached)
                return QueryResponse(
                    answer=cached,
                    sources=[],
                    session_id=session_id,
                    query=query,
                )

            # Expand vague follow-up queries using previous turn's topic
            search_query = self._expand_query(query, chat_history)

            # Retrieve top-k documents
            k = top_k or self.settings.retriever_k
            logger.info("retrieving_documents", k=k)

            search_results = self.vector_store.query(
                query_text=search_query,
                n_results=k,
            )

            num_docs = len(search_results.get("documents", []))
            logger.info("retrieved_documents_count", count=num_docs)

            if not search_results.get("documents"):
                logger.warning("no_relevant_documents_found", query=query[:100])
                return QueryResponse(
                    answer="I don't have any relevant information to answer your question.",
                    sources=[],
                    session_id=session_id,
                    query=query,
                )

            # Build context from retrieved documents
            context = self._build_context(search_results)

            # Generate response (non-blocking — generate_response is now async)
            answer = await self.llm_service.generate_response(
                query=query,
                context=context,
                chat_history=chat_history,
            )

            # Store in semantic cache for future similar queries
            self.semantic_cache.set(query, answer)

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
        if not session_id:
            session_id = str(uuid.uuid4())

        try:
            # Get conversation history
            chat_history = self._sessions.get(session_id, [])

            # Short-circuit: answer meta/history questions without hitting vector store
            if self._is_history_query(query):
                result = self._answer_from_history(query, chat_history, session_id)
                yield result.answer
                return

            # Expand vague follow-up queries using previous turn's topic
            search_query = self._expand_query(query, chat_history)

            # Retrieve top-k documents
            k = top_k or self.settings.retriever_k
            search_results = self.vector_store.query(
                query_text=search_query,
                n_results=k,
            )

            if not search_results.get("documents"):
                yield "I don't have any relevant information to answer your question."
                return

            # Build context
            context = self._build_context(search_results)

            logger.info(
                "streaming_query",
                session_id=session_id,
                history_length=len(chat_history),
            )

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

    def _expand_query(self, query: str, chat_history: List[Tuple[str, str]]) -> str:
        """
        Expand vague follow-up queries using the previous turn's topic.

        Args:
            query: Current user query.
            chat_history: List of (user_query, assistant_answer) tuples.

        Returns:
            Expanded query string, or original query if no expansion needed.
        """
        if chat_history and any(p in query.lower() for p in VAGUE_PATTERNS):
            last_query = chat_history[-1][0]
            expanded = f"{last_query} {query}"
            logger.info("query_expanded", original=query, expanded=expanded)
            return expanded
        return query


    def _build_context(self, search_results: Dict[str, Any]) -> str:
        """
        Build context string from search results, skipping near-duplicate chunks.

        Adjacent chunks share ~200 chars of overlap; skip any chunk whose words
        overlap >60% with an already-included chunk.
        """
        documents = search_results.get("documents", [])
        seen: List[str] = []
        deduped: List[str] = []
        for doc in documents:
            doc_words = set(doc.split())
            if not any(
                len(doc_words & set(prev.split())) / max(len(doc_words), 1) > 0.6
                for prev in seen
            ):
                deduped.append(doc)
                seen.append(doc)
        return "\n\n".join(deduped)

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
            relevance_score = max(0.0, min(1.0, 1.0 - distance))

            sources.append(SourceDocument(
                document_id=metadata.get("document_id", "unknown"),
                filename=metadata.get("filename", "unknown"),
                chunk_id=chunk_id,
                content=documents[i][:500] if i < len(documents) else "",
                relevance_score=relevance_score,
            ))

        return sources

    def _is_history_query(self, query: str) -> bool:
        """Return True if the query is asking about the conversation history."""
        q = query.lower()
        return any(p in q for p in HISTORY_PATTERNS)

    def _answer_from_history(
        self,
        query: str,
        chat_history: List[Tuple[str, str]],
        session_id: str,
    ) -> QueryResponse:
        """Answer a meta/history question directly from chat_history."""
        if not chat_history:
            answer = "There is no previous conversation in this session."
        else:
            last_user, _ = chat_history[-1]
            answer = f"Your last question was: \"{last_user}\""
        logger.info("answered_from_history", session_id=session_id)
        return QueryResponse(
            answer=answer,
            sources=[],
            session_id=session_id,
            query=query,
        )

    def _update_history(self, session_id: str, query: str, answer: str) -> None:
        """Update conversation history for a session."""
        if session_id not in self._sessions:
            self._sessions[session_id] = []

        self._sessions[session_id].append((query, answer))

        # Keep only last 10 exchanges per session
        if len(self._sessions[session_id]) > 10:
            self._sessions[session_id] = self._sessions[session_id][-10:]

    def get_session_history(self, session_id: str) -> List[Dict[str, str]]:
        """Get conversation history for a session."""
        history = self._sessions.get(session_id, [])

        messages = []
        for query, answer in history:
            messages.append({"role": "user", "content": query})
            messages.append({"role": "assistant", "content": answer})

        return messages

    def clear_session(self, session_id: str) -> bool:
        """Clear conversation history for a session."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False
