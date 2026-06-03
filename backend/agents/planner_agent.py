"""
Planner Agent — expands user query into structured research plan.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict

from loguru import logger

from backend.agents.base_agent import BaseAgent
from backend.models import EvaluationMetrics, ResearchDepth, ResearchPlan, ResearchState


_DEPTH_KEYWORD_COUNT = {
    ResearchDepth.BASIC: 4,
    ResearchDepth.STANDARD: 6,
    ResearchDepth.DEEP: 10,
}

_DEPTH_QUERY_COUNT = {
    ResearchDepth.BASIC: 2,
    ResearchDepth.STANDARD: 3,
    ResearchDepth.DEEP: 5,
}


class PlannerAgent(BaseAgent):
    """
    Decomposes a research topic into keywords, search queries,
    and a structured research plan.
    """

    name = "PlannerAgent"

    def run(self, state: ResearchState) -> ResearchState:
        start = self._log_start(state, f"Planning research for: '{state.research_topic}'")

        # Initialise metrics
        state.metrics = EvaluationMetrics(session_id=state.session_id)

        try:
            plan = self._generate_plan(state.research_topic, state.depth)
            state.research_plan = plan
            logger.info(f"Plan generated: {len(plan.keywords)} keywords, {len(plan.search_queries)} queries")
        except Exception as e:
            self._log_failure(state, start, e)
            # Fallback plan
            state.research_plan = self._fallback_plan(state.research_topic)

        self._log_success(state, start, f"Plan ready — {len(state.research_plan.keywords)} keywords")
        return state

    # ------------------------------------------------------------------

    def _generate_plan(self, topic: str, depth: ResearchDepth) -> ResearchPlan:
        num_keywords = _DEPTH_KEYWORD_COUNT[depth]
        num_queries = _DEPTH_QUERY_COUNT[depth]

        prompt = f"""You are a senior research scientist. Analyse the research topic and produce a structured JSON research plan.

Topic: "{topic}"
Depth: {depth.value}

Produce EXACTLY this JSON (no extra text, no markdown fences):
{{
  "keywords": ["{num_keywords} distinct search keywords/phrases relevant to the topic"],
  "research_plan": ["step 1 description", "step 2 description", "step 3 description", "step 4 description"],
  "research_domains": ["domain1", "domain2", "domain3"],
  "search_queries": ["{num_queries} arxiv-friendly search query strings"]
}}

Guidelines:
- keywords: specific, diverse, arxiv-searchable terms
- research_plan: 4 high-level steps (background → methods → evaluation → applications)
- research_domains: academic disciplines (e.g., Machine Learning, Robotics, NLP)
- search_queries: combine keywords for maximum recall; use arxiv query syntax where helpful

Return only valid JSON."""

        raw = self._call_llm(prompt)
        return self._parse_plan(raw, topic)

    def _parse_plan(self, raw: str, topic: str) -> ResearchPlan:
        """Parse LLM output into ResearchPlan, with fallback on parse failure."""
        # Strip markdown fences
        raw = re.sub(r"```(?:json)?", "", raw).strip()

        try:
            data: Dict[str, Any] = json.loads(raw)
            return ResearchPlan(
                keywords=data.get("keywords", [topic]),
                research_plan=data.get("research_plan", []),
                research_domains=data.get("research_domains", []),
                search_queries=data.get("search_queries", [topic]),
            )
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse failed: {e}. Attempting extraction.")
            return self._extract_plan_fallback(raw, topic)

    def _extract_plan_fallback(self, raw: str, topic: str) -> ResearchPlan:
        """Extract plan data from malformed JSON using regex."""
        keywords = re.findall(r'"([^"]{5,80})"', raw)[:8]
        if not keywords:
            keywords = [topic, f"{topic} survey", f"{topic} methods"]
        return ResearchPlan(
            keywords=keywords,
            research_plan=[
                "Survey existing literature",
                "Analyse methodologies",
                "Compare experimental results",
                "Identify research gaps",
            ],
            research_domains=["Computer Science", "Artificial Intelligence"],
            search_queries=[topic, f"{topic} survey", f"{topic} deep learning"],
        )

    def _fallback_plan(self, topic: str) -> ResearchPlan:
        """Hardcoded fallback used when LLM call fails."""
        words = topic.split()
        return ResearchPlan(
            keywords=[topic] + words[:3],
            research_plan=[
                "Literature survey",
                "Methodology analysis",
                "Results comparison",
                "Gap identification",
            ],
            research_domains=["Computer Science"],
            search_queries=[topic, f"{topic} review"],
        )
