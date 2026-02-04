"""
LLM service for interacting with language models.
Uses Groq for fast inference with Llama models.
"""

from typing import AsyncGenerator, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from app.config import Settings
from app.core.exceptions import LLMError
from app.core.logging import get_logger

logger = get_logger(__name__)


# System prompt for the RAG chatbot
DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant. Answer questions using the provided context.
RULES:
- For greetings like "hi" or "hello": respond with "Hello! How can I help you today?"
- Give SHORT, direct answers
- If you don't know: say "I don't have that information"
"""

# Context template - sent as a separate user message  
CONTEXT_TEMPLATE = """CONTEXT:
---
{context}
---

QUESTION: {query}

Answer briefly:"""


class LLMService:
    """
    Service for interacting with language models.
    
    Provides methods for generating responses using the Groq API.
    """

    def __init__(
        self,
        settings: Settings,
        system_prompt: Optional[str] = None,
    ):
        """
        Initialize the LLM service.
        
        Args:
            settings: Application settings.
            system_prompt: Custom system prompt (optional).
        """
        self.settings = settings
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self._llm = None
        
        logger.info(
            "llm_service_initialized",
            model=settings.llm_model_name,
            temperature=settings.llm_temperature,
        )

    @property
    def llm(self) -> ChatGroq:
        """Get or create the LLM instance."""
        if self._llm is None:
            try:
                self._llm = ChatGroq(
                    api_key=self.settings.groq_api_key,
                    model_name=self.settings.llm_model_name,
                    temperature=self.settings.llm_temperature,
                    max_tokens=self.settings.llm_max_tokens,
                )
                logger.info("llm_model_loaded", model=self.settings.llm_model_name)
            except Exception as e:
                logger.error("llm_model_load_failed", error=str(e))
                raise LLMError(f"Failed to initialize LLM: {e}")
        return self._llm

    def generate_response(
        self,
        query: str,
        context: str,
        chat_history: Optional[List[tuple]] = None,
    ) -> str:
        """
        Generate a response based on query and context.
        
        Args:
            query: User's question.
            context: Retrieved context from documents.
            chat_history: Optional list of (query, response) tuples.
            
        Returns:
            Generated response string.
            
        Raises:
            LLMError: If generation fails.
        """
        try:
            # Build messages
            messages = []
            
            # Add system message (instructions only, no context)
            messages.append(SystemMessage(content=self.system_prompt))
            
            # Add chat history if provided
            if chat_history:
                for user_msg, ai_msg in chat_history[-5:]:
                    messages.append(HumanMessage(content=user_msg))
                    messages.append(AIMessage(content=ai_msg))
            
            # Add current query WITH context as a user message
            user_content = CONTEXT_TEMPLATE.format(context=context, query=query)
            messages.append(HumanMessage(content=user_content))
            
            # Generate response
            response = self.llm.invoke(messages)
            
            logger.info(
                "response_generated",
                query_length=len(query),
                context_length=len(context),
                response_length=len(response.content),
            )
            
            return response.content
            
        except Exception as e:
            logger.error("generation_failed", error=str(e))
            raise LLMError(f"Failed to generate response: {e}")

    async def generate_response_stream(
        self,
        query: str,
        context: str,
        chat_history: Optional[List[tuple]] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Generate a streaming response.
        
        Args:
            query: User's question.
            context: Retrieved context from documents.
            chat_history: Optional list of (query, response) tuples.
            
        Yields:
            Response chunks as they are generated.
            
        Raises:
            LLMError: If generation fails.
        """
        try:
            # Build messages
            messages = []
            
            # Add system message (instructions only, no context)
            messages.append(SystemMessage(content=self.system_prompt))
            
            # Add chat history if provided
            if chat_history:
                for user_msg, ai_msg in chat_history[-5:]:
                    messages.append(HumanMessage(content=user_msg))
                    messages.append(AIMessage(content=ai_msg))
            
            # Add current query WITH context as a user message
            user_content = CONTEXT_TEMPLATE.format(context=context, query=query)
            messages.append(HumanMessage(content=user_content))
            
            # Stream response
            async for chunk in self.llm.astream(messages):
                if chunk.content:
                    yield chunk.content
                    
        except Exception as e:
            logger.error("streaming_generation_failed", error=str(e))
            raise LLMError(f"Failed to generate streaming response: {e}")

    def get_model_info(self) -> dict:
        """
        Get information about the LLM model.
        
        Returns:
            Dictionary with model information.
        """
        return {
            "model_name": self.settings.llm_model_name,
            "temperature": self.settings.llm_temperature,
            "max_tokens": self.settings.llm_max_tokens,
            "is_loaded": self._llm is not None,
        }

    def health_check(self) -> bool:
        """
        Check if the LLM service is healthy.
        
        Returns:
            True if healthy, False otherwise.
        """
        try:
            # Try a simple generation
            test_response = self.llm.invoke([
                HumanMessage(content="Say 'OK' if you can hear me.")
            ])
            return "OK" in test_response.content.upper()
        except Exception as e:
            logger.error("llm_health_check_failed", error=str(e))
            return False
