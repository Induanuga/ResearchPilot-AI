"""
Research Chat Agent — RAG-based conversational Q&A over retrieved papers.
Supports follow-up questions with memory and source citations.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from loguru import logger

from backend.agents.base_agent import BaseAgent
from backend.models import ChatMessage, ChatResponse, ResearchState
from backend.services.vector_store import VectorStoreService


class ChatAgent(BaseAgent):
    """
    Handles follow-up questions about the research using RAG.
    Retrieves relevant document chunks and generates grounded answers
    with proper source attribution.
    """

    name = "ChatAgent"

    def __init__(self) -> None:
        super().__init__()
        self.vs = VectorStoreService()

    def run(self, state: ResearchState) -> ResearchState:
        """Not used in the main pipeline; chat is handled via chat_query()."""
        return state

    def chat_query(
        self,
        session_id: str,
        question: str,
        chat_history: Optional[List[ChatMessage]] = None,
        topic: str = "",
    ) -> ChatResponse:
        """
        Answer a follow-up question using RAG over the session's vector store.

        Args:
            session_id: The research session identifier.
            question: User's question.
            chat_history: Prior conversation messages.
            topic: Original research topic for context.

        Returns:
            ChatResponse with answer and source citations.
        """
        logger.info(f"[ChatAgent] Processing question: '{question[:80]}'")

        # Retrieve relevant context
        relevant_docs = self.vs.retrieve_relevant_documents(
            session_id=session_id,
            query=question,
            k=5,
        )

        if not relevant_docs:
            return ChatResponse(
                answer="I don't have enough context to answer this question. "
                       "Please ensure the research session has been completed first.",
                sources=[],
                session_id=session_id,
            )

        # Build context for the LLM
        context_blocks = self._format_context(relevant_docs)
        history_text = self._format_history(chat_history or [])
        sources = self._extract_sources(relevant_docs)

        answer = self._generate_answer(
            question=question,
            context=context_blocks,
            history=history_text,
            topic=topic,
        )

        return ChatResponse(
            answer=answer,
            sources=sources,
            session_id=session_id,
        )

    # ------------------------------------------------------------------

    def _format_context(self, docs: List[Dict[str, Any]]) -> str:
        blocks = []
        for i, doc in enumerate(docs, 1):
            meta = doc.get("metadata", {})
            title = meta.get("title", "Unknown Paper")
            authors = meta.get("authors", "")
            year = meta.get("year", "")
            content = doc.get("content", "")
            blocks.append(
                f"[Source {i}] Title: {title}\n"
                f"Authors: {authors} ({year})\n"
                f"Content: {content[:800]}"
            )
        return "\n\n".join(blocks)

    def _format_history(self, history: List[ChatMessage]) -> str:
        if not history:
            return ""
        lines = []
        for msg in history[-6:]:  # Last 3 exchanges
            role = "Human" if msg.role == "user" else "Assistant"
            lines.append(f"{role}: {msg.content[:400]}")
        return "\n".join(lines)

    def _extract_sources(self, docs: List[Dict[str, Any]]) -> List[str]:
        sources = []
        seen = set()
        for doc in docs:
            meta = doc.get("metadata", {})
            title = meta.get("title", "")
            url = meta.get("url", "")
            year = meta.get("year", "")
            if title and title not in seen:
                seen.add(title)
                sources.append(f"{title} ({year}) — {url}")
        return sources

    def _generate_answer(
        self,
        question: str,
        context: str,
        history: str,
        topic: str,
    ) -> str:
        history_section = f"\nConversation History:\n{history}\n" if history else ""

        prompt = f"""You are an expert research assistant specialising in "{topic}". 
Answer questions accurately based ONLY on the provided research paper context.
Always cite your sources using [Source N] notation.

{history_section}
Research Context:
{context}

Question: {question}

Instructions:
- Answer based solely on the provided context
- Cite sources using [Source N] for every claim
- If comparing papers, be specific about differences
- If the answer is not in the context, say so clearly
- Keep the answer focused and well-structured
- For methodology questions, be technical and precise

Answer:"""

        return self._call_llm(prompt)
