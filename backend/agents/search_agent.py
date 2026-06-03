"""
Search Agent — queries ArXiv and stores paper metadata in SQLite.
"""

from __future__ import annotations

import time

from loguru import logger

from backend.agents.base_agent import BaseAgent
from backend.models import ResearchDepth, ResearchState
from backend.services.arxiv_service import ArxivService
from backend.services.database import DatabaseService


_DEPTH_PER_QUERY: dict[ResearchDepth, int] = {
    ResearchDepth.BASIC: 5,
    ResearchDepth.STANDARD: 7,
    ResearchDepth.DEEP: 10,
}


class SearchAgent(BaseAgent):
    """
    Executes ArXiv searches using the planner's keywords/queries,
    deduplicates results, and persists paper metadata.
    """

    name = "SearchAgent"

    def __init__(self) -> None:
        super().__init__()
        self.arxiv = ArxivService()
        self.db = DatabaseService()

    def run(self, state: ResearchState) -> ResearchState:
        start = self._log_start(state, "Searching ArXiv for relevant papers")
        retrieval_start = time.time()

        if state.research_plan is None:
            self._log_failure(state, start, ValueError("No research plan found"))
            return state

        per_query = _DEPTH_PER_QUERY[state.depth]
        queries = state.research_plan.search_queries

        # Cap total papers at num_papers
        total_max = state.num_papers

        try:
            papers = self.arxiv.search_multiple_queries(
                queries=queries,
                max_per_query=per_query,
                total_max=total_max,
            )

            if not papers:
                # Fallback: search with raw topic
                logger.warning("No results from plan queries, falling back to topic search")
                papers = self.arxiv.search_papers(
                    query=state.research_topic,
                    max_results=total_max,
                )

            state.papers = papers

            # Persist to SQLite
            self.db.save_papers(
                session_id=state.session_id,
                papers=[p.model_dump() for p in papers],
            )

            retrieval_time = time.time() - retrieval_start
            if state.metrics:
                state.metrics.retrieval_time = retrieval_time
                state.metrics.number_of_papers = len(papers)

            logger.info(f"Search complete: {len(papers)} papers retrieved")

        except Exception as e:
            self._log_failure(state, start, e)
            return state

        self._log_success(state, start, f"Found {len(state.papers)} papers")
        return state
