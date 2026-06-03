"""
Embedding service using HuggingFace sentence-transformers.
Provides lazy-loaded, cached embedding model.
"""

from __future__ import annotations

import time
from typing import List

from langchain_community.embeddings import HuggingFaceEmbeddings
from loguru import logger

from backend.config import get_settings


class EmbeddingService:
    """
    Singleton-style service for generating text embeddings.
    Uses sentence-transformers/all-MiniLM-L6-v2 by default.
    """

    _instance: "EmbeddingService | None" = None
    _model: HuggingFaceEmbeddings | None = None

    def __new__(cls) -> "EmbeddingService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        self.settings = get_settings()

    @property
    def model(self) -> HuggingFaceEmbeddings:
        """Lazy-load the embedding model."""
        if self._model is None:
            logger.info(f"Loading embedding model: {self.settings.embedding_model}")
            start = time.time()
            self._model = HuggingFaceEmbeddings(
                model_name=self.settings.embedding_model,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
            elapsed = time.time() - start
            logger.info(f"Embedding model loaded in {elapsed:.2f}s")
        return self._model

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: Input text strings.

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []

        logger.debug(f"Generating embeddings for {len(texts)} texts")
        start = time.time()
        embeddings = self.model.embed_documents(texts)
        elapsed = time.time() - start
        logger.debug(f"Embeddings generated in {elapsed:.2f}s")
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """
        Generate an embedding for a single query string.

        Args:
            text: Query string.

        Returns:
            Embedding vector.
        """
        return self.model.embed_query(text)

    def get_langchain_embeddings(self) -> HuggingFaceEmbeddings:
        """Return the raw LangChain embeddings object (for FAISS etc.)."""
        return self.model
