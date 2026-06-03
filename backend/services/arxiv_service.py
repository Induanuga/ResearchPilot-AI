"""
ArXiv API service — search and retrieve paper metadata.
Uses the `arxiv` Python library with retry logic.
"""

from __future__ import annotations

import time
from typing import List, Optional

import arxiv
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from backend.config import get_settings
from backend.models import PaperMetadata


class ArxivService:
    """Handles all interactions with the ArXiv API."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.client = arxiv.Client(
            page_size=self.settings.arxiv_max_results,
            delay_seconds=1.0,
            num_retries=3,
        )

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def search_papers(
        self,
        query: str,
        max_results: int = 10,
        sort_by: arxiv.SortCriterion = arxiv.SortCriterion.Relevance,
    ) -> List[PaperMetadata]:
        """
        Search ArXiv for papers matching the query.

        Args:
            query: Search query string.
            max_results: Maximum number of results to return.
            sort_by: Sorting criterion.

        Returns:
            List of PaperMetadata objects.
        """
        logger.info(f"Searching ArXiv for: '{query}' (max={max_results})")

        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=sort_by,
            sort_order=arxiv.SortOrder.Descending,
        )

        papers: List[PaperMetadata] = []
        seen_ids: set[str] = set()

        for result in self.client.results(search):
            paper_id = result.entry_id.split("/")[-1]

            # Deduplicate
            if paper_id in seen_ids:
                continue
            seen_ids.add(paper_id)

            # Extract year safely
            year = result.published.year if result.published else 0

            metadata = PaperMetadata(
                paper_id=paper_id,
                title=result.title.strip().replace("\n", " "),
                authors=[a.name for a in result.authors],
                year=year,
                abstract=result.summary.strip().replace("\n", " "),
                url=result.entry_id,
                pdf_url=result.pdf_url,
                categories=result.categories,
                published_date=result.published.strftime("%Y-%m-%d") if result.published else "",
                journal_ref=result.journal_ref or "",
                doi=result.doi or "",
            )
            papers.append(metadata)
            logger.debug(f"Found paper: {metadata.title[:80]}...")

        logger.info(f"Retrieved {len(papers)} papers from ArXiv")
        return papers

    def search_multiple_queries(
        self,
        queries: List[str],
        max_per_query: int = 5,
        total_max: Optional[int] = None,
    ) -> List[PaperMetadata]:
        """
        Execute multiple search queries and merge/deduplicate results.

        Args:
            queries: List of search query strings.
            max_per_query: Max results per query.
            total_max: Overall result cap.

        Returns:
            Deduplicated list of PaperMetadata.
        """
        all_papers: List[PaperMetadata] = []
        seen_ids: set[str] = set()

        for query in queries:
            try:
                papers = self.search_papers(query, max_results=max_per_query)
                for paper in papers:
                    if paper.paper_id not in seen_ids:
                        seen_ids.add(paper.paper_id)
                        all_papers.append(paper)
                time.sleep(0.5)  # Respect rate limits
            except Exception as e:
                logger.warning(f"Query '{query}' failed: {e}")
                continue

        if total_max:
            all_papers = all_papers[:total_max]

        logger.info(f"Total unique papers after multi-query search: {len(all_papers)}")
        return all_papers

    def get_paper_by_id(self, arxiv_id: str) -> Optional[PaperMetadata]:
        """Fetch a single paper by its ArXiv ID."""
        try:
            search = arxiv.Search(id_list=[arxiv_id])
            for result in self.client.results(search):
                year = result.published.year if result.published else 0
                return PaperMetadata(
                    paper_id=arxiv_id,
                    title=result.title.strip().replace("\n", " "),
                    authors=[a.name for a in result.authors],
                    year=year,
                    abstract=result.summary.strip().replace("\n", " "),
                    url=result.entry_id,
                    pdf_url=result.pdf_url,
                    categories=result.categories,
                    published_date=result.published.strftime("%Y-%m-%d") if result.published else "",
                    journal_ref=result.journal_ref or "",
                    doi=result.doi or "",
                )
        except Exception as e:
            logger.error(f"Failed to fetch paper {arxiv_id}: {e}")
            return None
