"""
Retrieval Agent — builds FAISS vector index and enables semantic search.
"""

from __future__ import annotations

from loguru import logger

from backend.agents.base_agent import BaseAgent
from backend.models import ResearchState
from backend.services.vector_store import VectorStoreService


class RetrievalAgent(BaseAgent):
    """
    Converts paper abstracts to embeddings, builds the FAISS vector store,
    and performs an initial retrieval pass to populate state.retrieved_docs.
    """

    name = "RetrievalAgent"

    def __init__(self) -> None:
        super().__init__()
        self.vs = VectorStoreService()

    def run(self, state: ResearchState) -> ResearchState:
        start = self._log_start(state, f"Building vector store for {len(state.papers)} papers")

        if not state.papers:
            self._log_failure(state, start, ValueError("No papers to index"))
            return state

        try:
            path, embedding_time = self.vs.build_vector_store(
                session_id=state.session_id,
                papers=state.papers,
            )
            state.vectorstore_path = path

            if state.metrics:
                state.metrics.embedding_time = embedding_time

            # Initial retrieval using the research topic
            retrieved = self.vs.retrieve_relevant_documents(
                session_id=state.session_id,
                query=state.research_topic,
                k=min(len(state.papers), 10),
            )
            state.retrieved_docs = retrieved

            logger.info(
                f"Vector store at {path} | embedded in {embedding_time:.2f}s | "
                f"retrieved {len(retrieved)} docs"
            )

        except Exception as e:
            self._log_failure(state, start, e)
            return state

        self._log_success(
            state,
            start,
            f"Vector store ready. {len(state.retrieved_docs)} docs retrieved.",
        )
        return state
