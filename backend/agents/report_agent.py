"""
Report Generation Agent — synthesises all agent outputs into a structured
Markdown report and generates a PDF export.
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from typing import Any, Dict, List

from loguru import logger

from backend.agents.base_agent import BaseAgent
from backend.models import ResearchReport, ResearchState
from backend.services.database import DatabaseService


class ReportAgent(BaseAgent):
    """
    Combines summaries, citations, and retrieved documents into a
    comprehensive, citation-aware literature review report.
    """

    name = "ReportAgent"

    def __init__(self) -> None:
        super().__init__()
        self.db = DatabaseService()

    def run(self, state: ResearchState) -> ResearchState:
        start_time = self._log_start(state, "Generating research report")
        report_start = time.time()

        try:
            report = self._build_report(state)
            state.report = report

            # Persist
            version = self.db.save_report(
                session_id=state.session_id,
                topic=state.research_topic,
                markdown_content=report.markdown_content,
                metadata={
                    "num_papers": len(state.papers),
                    "num_citations": len(state.citations),
                    "depth": state.depth.value,
                },
            )
            logger.info(f"Report v{version} saved for session {state.session_id}")

            report_time = time.time() - report_start
            if state.metrics:
                state.metrics.report_generation_time = report_time

        except Exception as e:
            self._log_failure(state, start_time, e)
            return state

        self._log_success(state, start_time, "Report generation complete")
        return state

    # ------------------------------------------------------------------

    def _build_report(self, state: ResearchState) -> ResearchReport:
        """Orchestrate LLM calls to build each report section."""
        papers = state.papers
        summaries = state.summaries
        citations = state.citations
        topic = state.research_topic

        # Build citation lookup
        citation_map: Dict[str, Any] = {c.paper_id: c for c in citations}
        summary_map: Dict[str, Any] = {s.paper_id: s for s in summaries}

        logger.info("Generating executive summary...")
        executive_summary = self._gen_executive_summary(topic, summaries)

        logger.info("Generating research overview...")
        research_overview = self._gen_research_overview(topic, summaries)

        logger.info("Generating cross-paper insights...")
        cross_insights = self._gen_cross_paper_insights(topic, summaries)

        logger.info("Generating method comparison...")
        method_comparison = self._gen_method_comparison(topic, summaries)

        logger.info("Generating trends & limitations...")
        trends = self._gen_trends(topic, summaries)
        limitations = self._gen_limitations(topic, summaries)

        logger.info("Generating future directions...")
        future_directions = self._gen_future_directions(topic, summaries)

        # Build key papers section
        key_papers = self._build_key_papers(papers, summary_map, citation_map)

        # Build references list (APA)
        references = [citation_map[pid].apa for pid in citation_map if citation_map[pid].apa]

        # Assemble full Markdown
        markdown = self._assemble_markdown(
            topic=topic,
            executive_summary=executive_summary,
            research_overview=research_overview,
            key_papers=key_papers,
            cross_insights=cross_insights,
            method_comparison=method_comparison,
            trends=trends,
            limitations=limitations,
            future_directions=future_directions,
            references=references,
            state=state,
        )

        return ResearchReport(
            session_id=state.session_id,
            topic=topic,
            executive_summary=executive_summary,
            research_overview=research_overview,
            key_papers=key_papers,
            cross_paper_insights=cross_insights,
            method_comparison=method_comparison,
            trends=trends,
            limitations=limitations,
            future_directions=future_directions,
            references=references,
            markdown_content=markdown,
        )

    # ------------------------------------------------------------------
    # Section generators
    # ------------------------------------------------------------------

    def _gen_executive_summary(self, topic: str, summaries: List[Any]) -> str:
        summaries_text = "\n".join(
            f"- [{s.title[:60]}]: {s.summary[:300]}" for s in summaries[:8]
        )
        prompt = f"""You are a senior research analyst writing a concise executive summary.

Topic: "{topic}"
Number of papers analysed: {len(summaries)}

Paper Summaries:
{summaries_text}

Write a 3-4 paragraph executive summary covering:
1. The scope and importance of this research area
2. Major findings and trends across the papers
3. Key methodological approaches used
4. Practical implications

Be concise, authoritative, and citation-aware. Write in professional academic style."""
        return self._call_llm(prompt)

    def _gen_research_overview(self, topic: str, summaries: List[Any]) -> str:
        context = "\n".join(
            f"Paper {i+1}: {s.title} — {s.methodology}" for i, s in enumerate(summaries[:10])
        )
        prompt = f"""Write a comprehensive research overview for the topic: "{topic}"

Papers analysed:
{context}

Write 2-3 paragraphs covering:
1. Historical context and evolution of this research area
2. Current state of research 
3. Main sub-fields and research directions represented in these papers"""
        return self._call_llm(prompt)

    def _gen_cross_paper_insights(self, topic: str, summaries: List[Any]) -> str:
        findings_text = "\n".join(
            f"Paper '{s.title[:50]}' findings: {'; '.join(s.key_findings[:3])}"
            for s in summaries[:10]
        )
        prompt = f"""Analyse these research papers on "{topic}" and identify cross-cutting insights.

Key findings across papers:
{findings_text}

Write 2-3 paragraphs identifying:
1. Common themes and patterns across papers
2. Contradictions or conflicting findings
3. Convergent evidence for major claims
4. Gaps that multiple papers acknowledge"""
        return self._call_llm(prompt)

    def _gen_method_comparison(self, topic: str, summaries: List[Any]) -> str:
        methods_text = "\n".join(
            f"- {s.title[:50]}: {s.methodology}" for s in summaries[:10]
        )
        prompt = f"""Compare the methodologies used across these papers on "{topic}".

Methodologies:
{methods_text}

Write a structured comparison covering:
1. Dominant methodological approaches
2. Unique or novel methods
3. Experimental setups and evaluation metrics
4. Strengths and weaknesses of each approach"""
        return self._call_llm(prompt)

    def _gen_trends(self, topic: str, summaries: List[Any]) -> str:
        years = sorted({s.paper_id[:4] for s in summaries if s.paper_id[:4].isdigit()})
        context = "\n".join(f"- {s.title[:60]} ({s.paper_id[:8]})" for s in summaries[:10])
        prompt = f"""Identify research trends in "{topic}" based on these papers:
{context}

Write 2 paragraphs about:
1. Temporal trends (how the field has evolved)
2. Emerging techniques or paradigm shifts"""
        return self._call_llm(prompt)

    def _gen_limitations(self, topic: str, summaries: List[Any]) -> str:
        all_limitations = []
        for s in summaries:
            all_limitations.extend(s.limitations[:2])
        limitations_text = "\n".join(f"- {lim}" for lim in all_limitations[:20])
        prompt = f"""Synthesise the limitations identified across papers on "{topic}":

{limitations_text}

Write 1-2 paragraphs summarising:
1. Common limitations shared across papers
2. Dataset or evaluation biases
3. Generalisation challenges"""
        return self._call_llm(prompt)

    def _gen_future_directions(self, topic: str, summaries: List[Any]) -> str:
        all_future = []
        for s in summaries:
            all_future.extend(s.future_work[:2])
        future_text = "\n".join(f"- {f}" for f in all_future[:20])
        prompt = f"""Based on these future work suggestions from papers on "{topic}":

{future_text}

Write 2 paragraphs about the most promising future research directions, 
grouping related suggestions and highlighting the most impactful opportunities."""
        return self._call_llm(prompt)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_key_papers(
        self,
        papers: List[Any],
        summary_map: Dict[str, Any],
        citation_map: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        key_papers = []
        for paper in papers[:10]:
            summary = summary_map.get(paper.paper_id)
            citation = citation_map.get(paper.paper_id)
            key_papers.append(
                {
                    "paper_id": paper.paper_id,
                    "title": paper.title,
                    "authors": paper.authors[:3],
                    "year": paper.year,
                    "url": paper.url,
                    "summary": summary.summary if summary else paper.abstract[:300],
                    "key_findings": summary.key_findings if summary else [],
                    "apa_citation": citation.apa if citation else "",
                }
            )
        return key_papers

    def _assemble_markdown(
        self,
        topic: str,
        executive_summary: str,
        research_overview: str,
        key_papers: List[Dict[str, Any]],
        cross_insights: str,
        method_comparison: str,
        trends: str,
        limitations: str,
        future_directions: str,
        references: List[str],
        state: ResearchState,
    ) -> str:
        from backend.config import get_settings
        settings = get_settings()

        generated_at = datetime.utcnow().strftime("%B %d, %Y at %H:%M UTC")
        num_papers = len(state.papers)
        depth = state.depth.value.title()

        # Build key papers section
        papers_md = ""
        for i, p in enumerate(key_papers, 1):
            authors_str = ", ".join(p["authors"][:3])
            if len(p.get("authors", [])) > 3:
                authors_str += " et al."
            findings_md = "\n".join(f"  - {f}" for f in p.get("key_findings", [])[:4])
            papers_md += f"""
### {i}. {p['title']}

**Authors:** {authors_str} | **Year:** {p['year']}  
**Link:** [{p['url']}]({p['url']})

**Summary:** {p['summary']}

**Key Findings:**
{findings_md}

**Citation:** {p.get('apa_citation', '')}

---
"""

        # Build references section
        references_md = "\n".join(f"{i+1}. {ref}" for i, ref in enumerate(references))

        # Metrics
        metrics = state.metrics
        metrics_md = ""
        if metrics:
            metrics_md = f"""
| Metric | Value |
|--------|-------|
| Papers Retrieved | {metrics.number_of_papers} |
| Retrieval Time | {metrics.retrieval_time:.2f}s |
| Embedding Time | {metrics.embedding_time:.2f}s |
| Report Generation | {metrics.report_generation_time:.2f}s |
| Citations Generated | {metrics.citation_count} |
"""

        return f"""# ResearchPilot AI — Research Report

**Topic:** {topic}  
**Research Depth:** {depth}  
**Papers Analysed:** {num_papers}  
**Generated:** {generated_at}  
**Session ID:** `{state.session_id}`

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Research Overview](#research-overview)
3. [Key Papers](#key-papers)
4. [Cross-Paper Insights](#cross-paper-insights)
5. [Method Comparison](#method-comparison)
6. [Research Trends](#research-trends)
7. [Limitations](#limitations)
8. [Future Research Directions](#future-research-directions)
9. [References](#references)
10. [Session Metrics](#session-metrics)

---

## Executive Summary

{executive_summary}

---

## Research Overview

{research_overview}

---

## Key Papers

{papers_md}

---

## Cross-Paper Insights

{cross_insights}

---

## Method Comparison

{method_comparison}

---

## Research Trends

{trends}

---

## Limitations

{limitations}

---

## Future Research Directions

{future_directions}

---

## References

{references_md}

---

## Session Metrics

{metrics_md}

---

*Generated by {settings.app_name} v{settings.app_version} — Powered by Groq & LangGraph*
"""
