"""
Query API endpoints.
Handles RAG queries and conversation history.
"""

from typing import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.api.dependencies import get_rag_service
from app.core.exceptions import QueryError
from app.models.schemas import (
    ConversationHistoryResponse,
    ErrorResponse,
    QueryRequest,
    QueryResponse,
)
from app.services.rag_service import RAGService

router = APIRouter(prefix="/query", tags=["Query"])


@router.post(
    "",
    response_model=QueryResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Bad request"},
        500: {"model": ErrorResponse, "description": "Internal error"},
    },
    summary="Query the chatbot",
    description="Ask a question and get an answer based on the uploaded documents.",
)
async def query(
    request: QueryRequest,
    rag_service: RAGService = Depends(get_rag_service),
) -> QueryResponse:
    """
    Process a query using RAG.
    
    The query will:
    1. Search for relevant document chunks
    2. Use them as context for the LLM
    3. Generate a contextual answer
    4. Return the answer with source documents
    """
    try:
        response = await rag_service.query(
            query=request.query,
            session_id=request.session_id,
            include_sources=request.include_sources,
            top_k=request.top_k,
        )
        return response
        
    except QueryError as e:
        raise HTTPException(status_code=400, detail=e.to_dict())
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"error": "InternalError", "message": str(e)},
        )


@router.post(
    "/stream",
    responses={
        400: {"model": ErrorResponse, "description": "Bad request"},
        500: {"model": ErrorResponse, "description": "Internal error"},
    },
    summary="Stream query response",
    description="Ask a question and receive a streaming response using Server-Sent Events.",
)
async def query_stream(
    request: QueryRequest,
    rag_service: RAGService = Depends(get_rag_service),
) -> StreamingResponse:
    """
    Process a query with streaming response.
    
    Returns a Server-Sent Events stream with response chunks.
    """
    async def generate() -> AsyncGenerator[str, None]:
        try:
            async for chunk in rag_service.query_stream(
                query=request.query,
                session_id=request.session_id,
                top_k=request.top_k,
            ):
                # Format as SSE
                yield f"data: {chunk}\n\n"
            
            # Send done signal
            yield "data: [DONE]\n\n"
            
        except QueryError as e:
            yield f"data: Error: {e.message}\n\n"
        except Exception as e:
            yield f"data: Error: {str(e)}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get(
    "/history/{session_id}",
    response_model=ConversationHistoryResponse,
    summary="Get conversation history",
    description="Retrieve the conversation history for a session.",
)
async def get_conversation_history(
    session_id: str,
    rag_service: RAGService = Depends(get_rag_service),
) -> ConversationHistoryResponse:
    """Get conversation history for a session."""
    messages = rag_service.get_session_history(session_id)
    
    return ConversationHistoryResponse(
        session_id=session_id,
        messages=[
            {"role": msg["role"], "content": msg["content"]}
            for msg in messages
        ],
    )


@router.delete(
    "/history/{session_id}",
    summary="Clear conversation history",
    description="Clear the conversation history for a session.",
)
async def clear_conversation_history(
    session_id: str,
    rag_service: RAGService = Depends(get_rag_service),
) -> dict:
    """Clear conversation history for a session."""
    cleared = rag_service.clear_session(session_id)

    if cleared:
        return {"message": "Conversation history cleared", "session_id": session_id}
    else:
        return {"message": "No history found for session", "session_id": session_id}
