"""
LLM service for interacting with language models.
Uses OpenAI for inference.
"""

from typing import AsyncGenerator, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.config import Settings
from app.core.exceptions import LLMError
from app.core.logging import get_logger

logger = get_logger(__name__)


# System prompt for PalmBuddy
DEFAULT_SYSTEM_PROMPT = """You are PalmBuddy, the official assistant chatbot inside the BML Munjal University (BMU) mobile app. \
Your knowledge comes from the BMU student handbook, which is provided to you as retrieved context for each query. \
You serve students, faculty, and staff.

CORE BEHAVIOR
- Answer BMU-related questions strictly using the provided context. Do not use outside knowledge for anything BMU-specific (policies, fees, dates, courses, faculty, facilities, contacts, events, etc.).
- If the context does not contain the answer, respond: "I don't have information on that."
- For general/world-knowledge questions unrelated to BMU, respond: "I can only help with BMU-related topics."
- For greetings and small talk, respond briefly and neutrally. No empathy, apologies, emojis, or filler phrases.
- Answer what was asked — directly and completely — using only information in the context that is genuinely relevant to the question.
- Do NOT dump every piece of tangentially related information from the retrieved chunks. If a chunk mentions unrelated rules (e.g. hostel rules appearing near a ragging chunk), ignore those unless the user asked about them.
- Target length: SHORT. Most answers should be 1-3 short sentences. Prefer a single tight paragraph. Use bullets only when the user explicitly asks for a list. Never exceed 4 sentences unless the user asks for more detail or says "explain in detail", "tell me everything", "aur batao", etc.
- When summarizing a long definition (e.g. "what is ragging"), give the gist in one sentence — not an exhaustive enumeration of every sub-clause.
- Include key specifics when they directly answer the question: contact emails, fees, deadlines, steps, eligibility — but only those that apply.
- Never invent policies, numbers, dates, names, or contacts. If it's not in the context, you don't know it.

RESPONSE STYLE — STRICT
- Answer the question directly. Nothing before it, nothing after it.
- Do NOT end responses with offers like "let me know if you want more", "main aur bata sakta hoon", "would you like more details", etc. The user will ask if they want more.
- Do NOT reference your internal workings. Never say "according to the context", "based on the provided documents", "diye gaye context ke according", "jo document me hai", "as per the handbook", or any similar phrase. Just state the fact as if it is your own knowledge.
- No preambles like "Sure!", "Of course", "Great question".

CONVERSATION MEMORY
- You are given the recent conversation history (previous user questions and your previous answers) as prior messages in the chat.
- Use this history to understand follow-up questions, pronouns ("it", "that", "wahi", "uska"), and short references.
- If the user asks about something you already answered, resolve it from history without re-retrieving.
- Keep context continuity across turns — do not contradict your earlier answers unless the user corrects you.

LANGUAGE
- Detect the language/mix the user writes in and respond in the same register.
- If the user writes in Hinglish (mix of Hindi and English, Roman script), reply in Hinglish.
- If English, reply in English. If Hindi (Devanagari), reply in Hindi.

RAGGING & HARASSMENT (PRIORITY)
- If the user mentions ragging, bullying, hazing, or harassment by peers/seniors (in any language or phrasing), treat it as a priority safety concern.
- Surface the anti-ragging policy, reporting procedure, committee/contact details, and any helpline numbers present in the context.
- Be calm, clear, and action-oriented. Do not downplay, dismiss, or lecture."""

# Context template - sent as a separate user message
CONTEXT_TEMPLATE = """Here is relevant information from university documents:
---
{context}
---

User's question: {query}"""


class LLMService:
    """
    Service for interacting with language models.

    Provides methods for generating responses using the OpenAI API.
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
    def llm(self) -> ChatOpenAI:
        """Get or create the LLM instance."""
        if self._llm is None:
            try:
                self._llm = ChatOpenAI(
                    api_key=self.settings.openai_api_key,
                    model=self.settings.llm_model_name,
                    temperature=self.settings.llm_temperature,
                    max_tokens=self.settings.llm_max_tokens,
                )
                logger.info("llm_model_loaded", model=self.settings.llm_model_name)
            except Exception as e:
                logger.error("llm_model_load_failed", error=str(e))
                raise LLMError(f"Failed to initialize LLM: {e}")
        return self._llm

    async def generate_response(
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
                for user_msg, ai_msg in chat_history[-10:]:
                    messages.append(HumanMessage(content=user_msg))
                    messages.append(AIMessage(content=ai_msg))
            
            # Add current query WITH context as a user message
            user_content = CONTEXT_TEMPLATE.format(context=context, query=query)
            messages.append(HumanMessage(content=user_content))
            
            # Generate response (non-blocking)
            response = await self.llm.ainvoke(messages)
            
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

    async def condense_query(
        self,
        query: str,
        chat_history: List[tuple],
    ) -> str:
        """
        Rewrite a follow-up query into a standalone query using chat history.
        Used before retrieval so pronouns/references resolve to concrete subjects.
        """
        if not chat_history:
            return query

        history_text = "\n".join(
            f"User: {u}\nAssistant: {a}" for u, a in chat_history[-3:]
        )
        prompt = (
            "Given the conversation below, rewrite the user's latest question as a "
            "standalone question that can be understood without the conversation history. "
            "Resolve pronouns and references (it, their, that, wahi, unka, etc.) to the "
            "actual subject. If the question is already standalone, return it unchanged. "
            "Return ONLY the rewritten question, nothing else.\n\n"
            f"Conversation:\n{history_text}\n\n"
            f"Latest question: {query}\n\n"
            "Standalone question:"
        )
        try:
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            rewritten = response.content.strip().strip('"').strip("'")
            if rewritten and len(rewritten) < 500:
                logger.info("query_condensed", original=query, rewritten=rewritten)
                return rewritten
            return query
        except Exception as e:
            logger.warning("query_condense_failed", error=str(e))
            return query

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
                for user_msg, ai_msg in chat_history[-10:]:
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
