"""
Summarization Agent — generates structured summaries for each paper.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from backend.agents.base_agent import BaseAgent
from backend.models import PaperMetadata, PaperSummary, ResearchState
from backend.services.database import DatabaseService


class SummarizerAgent(BaseAgent):
    """
    Generates a structured JSON summary for each retrieved paper.
    Includes: summary, key findings, methodology, limitations, future work.
    """

    name = "SummarizerAgent"

    def __init__(self) -> None:
        super().__init__()
        self.db = DatabaseService()

    def run(self, state: ResearchState) -> ResearchState:
        start = self._log_start(state, f"Summarizing {len(state.papers)} papers")

        if not state.papers:
            self._log_failure(state, start, ValueError("No papers to summarize"))
            return state

        summaries: List[PaperSummary] = []

        for idx, paper in enumerate(state.papers):
            logger.info(f"Summarizing paper {idx + 1}/{len(state.papers)}: {paper.title[:60]}")
            try:
                summary = self._summarize_paper(paper, state.research_topic)
                summaries.append(summary)
            except Exception as e:
                logger.warning(f"Summary failed for {paper.paper_id}: {e}")
                summaries.append(self._fallback_summary(paper))

        state.summaries = summaries

        # Persist
        self.db.save_summaries(
            session_id=state.session_id,
            summaries=[s.model_dump() for s in summaries],
        )

        self._log_success(state, start, f"Summarized {len(summaries)} papers")
        return state

    # ------------------------------------------------------------------

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=5), reraise=True)
    def _summarize_paper(self, paper: PaperMetadata, topic: str) -> PaperSummary:
        authors_str = ", ".join(paper.authors[:5])
        if len(paper.authors) > 5:
            authors_str += f" et al."

        prompt = f"""You are a research analyst. Analyse this paper and return ONLY a valid JSON object.

Research Context: "{topic}"

Paper Details:
- Title: {paper.title}
- Authors: {authors_str}
- Year: {paper.year}
- Categories: {', '.join(paper.categories[:3])}
- Abstract: {paper.abstract[:3000]}

Return this EXACT JSON (no markdown, no extra text):
{{
  "summary": "2-3 sentence comprehensive summary of the paper",
  "key_findings": ["finding 1", "finding 2", "finding 3", "finding 4"],
  "methodology": "1-2 sentence description of the methodology/approach used",
  "limitations": ["limitation 1", "limitation 2", "limitation 3"],
  "future_work": ["direction 1", "direction 2"],
  "relevance_score": 0.85
}}

Guidelines:
- summary: concise but informative; mention the main contribution
- key_findings: concrete, specific findings with numbers/results where available
- methodology: describe the technical approach (e.g., transformer-based, reinforcement learning, etc.)
- limitations: honest assessment from abstract/context
- future_work: logical next steps
- relevance_score: 0.0-1.0 float indicating relevance to "{topic}" """

        raw = self._call_llm(prompt)
        return self._parse_summary(raw, paper)

    def _parse_summary(self, raw: str, paper: PaperMetadata) -> PaperSummary:
        raw = re.sub(r"```(?:json)?", "", raw).strip()
        try:
            data: Dict[str, Any] = json.loads(raw)
            return PaperSummary(
                paper_id=paper.paper_id,
                title=paper.title,
                summary=data.get("summary", ""),
                key_findings=data.get("key_findings", []),
                methodology=data.get("methodology", ""),
                limitations=data.get("limitations", []),
                future_work=data.get("future_work", []),
                relevance_score=float(data.get("relevance_score", 0.5)),
            )
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Parse error for {paper.paper_id}: {e}")
            return self._fallback_summary(paper)

    def _fallback_summary(self, paper: PaperMetadata) -> PaperSummary:
        return PaperSummary(
            paper_id=paper.paper_id,
            title=paper.title,
            summary=paper.abstract[:500],
            key_findings=["See abstract for details"],
            methodology="See abstract",
            limitations=["Not extracted"],
            future_work=["Not extracted"],
            relevance_score=0.5,
        )
