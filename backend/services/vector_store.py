"""
FAISS vector store service.
Builds, persists, and queries a local FAISS index for semantic retrieval.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Tuple

from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from loguru import logger

from backend.config import get_settings
from backend.models import PaperMetadata
from backend.services.embedding_service import EmbeddingService


class VectorStoreService:
    """
    Manages FAISS vector indexes per research session.
    Each session gets its own sub-directory under the configured vectorstore path.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.embedding_service = EmbeddingService()
        self._stores: Dict[str, FAISS] = {}  # session_id → FAISS store

    def _session_path(self, session_id: str) -> str:
        path = os.path.join(self.settings.vectorstore_path, session_id)
        os.makedirs(path, exist_ok=True)
        return path

    def _papers_to_documents(self, papers: List[PaperMetadata]) -> List[Document]:
        """Convert PaperMetadata list to LangChain Document list."""
        docs: List[Document] = []
        for paper in papers:
            # Combine title + abstract for richer embedding
            content = f"Title: {paper.title}\n\nAbstract: {paper.abstract}"

            metadata = {
                "paper_id": paper.paper_id,
                "title": paper.title,
                "authors": ", ".join(paper.authors[:5]),
                "year": paper.year,
                "url": paper.url,
                "categories": ", ".join(paper.categories[:3]),
            }
            docs.append(Document(page_content=content, metadata=metadata))
        return docs

    def build_vector_store(
        self, session_id: str, papers: List[PaperMetadata]
    ) -> Tuple[str, float]:
        """
        Build a FAISS index from paper metadata and persist it.

        Args:
            session_id: Unique session identifier.
            papers: List of PaperMetadata to index.

        Returns:
            Tuple of (vectorstore_path, embedding_time_seconds).
        """
        logger.info(f"Building vector store for session {session_id} with {len(papers)} papers")

        start = time.time()
        documents = self._papers_to_documents(papers)
        embeddings = self.embedding_service.get_langchain_embeddings()

        store = FAISS.from_documents(documents, embeddings)
        embedding_time = time.time() - start

        # Persist
        path = self._session_path(session_id)
        store.save_local(path)
        self._stores[session_id] = store

        logger.info(f"Vector store built and saved to {path} in {embedding_time:.2f}s")
        return path, embedding_time

    def load_vector_store(self, session_id: str) -> Optional[FAISS]:
        """
        Load an existing FAISS index from disk.

        Args:
            session_id: Session identifier.

        Returns:
            FAISS store or None if not found.
        """
        if session_id in self._stores:
            return self._stores[session_id]

        path = self._session_path(session_id)
        index_file = os.path.join(path, "index.faiss")

        if not os.path.exists(index_file):
            logger.warning(f"No vector store found for session {session_id}")
            return None

        embeddings = self.embedding_service.get_langchain_embeddings()
        store = FAISS.load_local(
            path,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        self._stores[session_id] = store
        logger.info(f"Vector store loaded from {path}")
        return store

    def retrieve_relevant_documents(
        self,
        session_id: str,
        query: str,
        k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top-k documents relevant to the query.

        Args:
            session_id: Session identifier.
            query: Search query string.
            k: Number of results to return.

        Returns:
            List of document dicts with content and metadata.
        """
        store = self.load_vector_store(session_id)
        if store is None:
            logger.error(f"Cannot retrieve — no vector store for session {session_id}")
            return []

        results = store.similarity_search_with_score(query, k=k)

        docs = []
        for doc, score in results:
            docs.append(
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "relevance_score": float(1 - score),  # FAISS returns L2 distance
                }
            )

        logger.info(f"Retrieved {len(docs)} documents for query: '{query[:60]}'")
        return docs

    def search_similar_papers(
        self,
        session_id: str,
        query: str,
        k: int = 3,
    ) -> List[Dict[str, Any]]:
        """Alias for retrieve_relevant_documents with smaller k, for chat context."""
        return self.retrieve_relevant_documents(session_id, query, k=k)

    def get_all_documents(self, session_id: str) -> List[Dict[str, Any]]:
        """Retrieve all documents from the store (for full-context tasks)."""
        store = self.load_vector_store(session_id)
        if store is None:
            return []

        # FAISS does not have a native "get all", so we do a broad search
        results = store.similarity_search("research paper abstract methodology results", k=50)
        return [{"content": doc.page_content, "metadata": doc.metadata} for doc in results]
