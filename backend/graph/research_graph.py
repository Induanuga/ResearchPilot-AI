"""
LangGraph research pipeline.
Uses TypedDict + Annotated reducers for langgraph 0.3.x compatibility.
"""

from __future__ import annotations

import uuid
from typing import Annotated, Any, Dict, List, Optional

from langgraph.graph import END, StateGraph
from loguru import logger

from backend.agents.citation_agent import CitationAgent
from backend.agents.planner_agent import PlannerAgent
from backend.agents.report_agent import ReportAgent
from backend.agents.retrieval_agent import RetrievalAgent
from backend.agents.search_agent import SearchAgent
from backend.agents.summarizer_agent import SummarizerAgent
from backend.models import ResearchDepth, ResearchState
from backend.services.database import DatabaseService


# ---------------------------------------------------------------------------
# Singletons (created once, reused across requests)
# ---------------------------------------------------------------------------

_planner   = PlannerAgent()
_searcher  = SearchAgent()
_retriever = RetrievalAgent()
_summarizer = SummarizerAgent()
_citation  = CitationAgent()
_reporter  = ReportAgent()
_db        = DatabaseService()


# ---------------------------------------------------------------------------
# Helper: serialise / deserialise ResearchState through the graph
#
# LangGraph 0.3.x passes state as a plain dict between nodes.
# We wrap each node to convert dict → ResearchState → run → dict.
# ---------------------------------------------------------------------------

def _wrap(agent_run_fn):
    """Wrap an agent's run() so it accepts and returns a plain dict."""
    def node(state_dict: Dict[str, Any]) -> Dict[str, Any]:
        state = ResearchState(**state_dict)
        state = agent_run_fn(state)
        return state.model_dump()
    node.__name__ = agent_run_fn.__self__.__class__.__name__
    return node


def _run_planner(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    state = ResearchState(**state_dict)
    state = _planner.run(state)
    return state.model_dump()


def _run_searcher(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    state = ResearchState(**state_dict)
    state = _searcher.run(state)
    return state.model_dump()


def _run_retriever(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    state = ResearchState(**state_dict)
    state = _retriever.run(state)
    return state.model_dump()


def _run_summarizer(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    state = ResearchState(**state_dict)
    state = _summarizer.run(state)
    return state.model_dump()


def _run_citation(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    state = ResearchState(**state_dict)
    state = _citation.run(state)
    return state.model_dump()


def _run_reporter(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    state = ResearchState(**state_dict)
    state = _reporter.run(state)
    return state.model_dump()


# ---------------------------------------------------------------------------
# Conditional edge: abort early if no papers found
# ---------------------------------------------------------------------------

def _check_papers(state_dict: Dict[str, Any]) -> str:
    papers = state_dict.get("papers", [])
    if not papers:
        logger.warning("No papers found — terminating pipeline early")
        return "end"
    return "continue"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_research_graph():
    """Build and compile the LangGraph StateGraph."""

    # Use plain dict schema — each node receives and returns a full state dict.
    graph = StateGraph(dict)

    graph.add_node("planner",    _run_planner)
    graph.add_node("searcher",   _run_searcher)
    graph.add_node("retriever",  _run_retriever)
    graph.add_node("summarizer", _run_summarizer)
    graph.add_node("citation",   _run_citation)
    graph.add_node("reporter",   _run_reporter)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "searcher")

    graph.add_conditional_edges(
        "searcher",
        _check_papers,
        {"continue": "retriever", "end": END},
    )

    graph.add_edge("retriever",  "summarizer")
    graph.add_edge("summarizer", "citation")
    graph.add_edge("citation",   "reporter")
    graph.add_edge("reporter",   END)

    return graph.compile()


# ---------------------------------------------------------------------------
# High-level runner
# ---------------------------------------------------------------------------

class ResearchPipeline:
    """Orchestrates the full research pipeline via LangGraph."""

    def __init__(self) -> None:
        self.graph = build_research_graph()
        self.db    = DatabaseService()

    def run(
        self,
        topic:      str,
        depth:      ResearchDepth = ResearchDepth.STANDARD,
        num_papers: int            = 10,
        session_id: Optional[str]  = None,
    ) -> ResearchState:
        """
        Execute the full research pipeline.

        Args:
            topic:      Research topic string.
            depth:      Research depth level.
            num_papers: Number of papers to retrieve.
            session_id: Existing session ID or None to create a new one.

        Returns:
            Final ResearchState after all agents complete.
        """
        if session_id is None:
            session_id = str(uuid.uuid4())

        logger.info(
            f"Starting research pipeline | session={session_id} | topic='{topic}'"
        )

        # Persist session record
        self.db.create_session(
            session_id=session_id,
            topic=topic,
            depth=depth.value,
            num_papers=num_papers,
        )
        self.db.update_session_status(session_id, "running")

        # Build initial state dict
        initial_state = ResearchState(
            session_id=session_id,
            research_topic=topic,
            depth=depth,
            num_papers=num_papers,
        ).model_dump()

        try:
            final_dict  = self.graph.invoke(initial_state)
            final_state = ResearchState(**final_dict)

            # Persist agent logs
            for log in final_state.agent_logs:
                self.db.save_agent_log(
                    session_id=session_id,
                    agent_name=log.agent_name,
                    status=str(log.status),
                    message=log.message,
                    duration=log.duration_seconds or 0.0,
                    token_usage=log.token_usage or {},
                )

            # Persist metrics
            if final_state.metrics:
                self.db.save_metrics(
                    session_id=session_id,
                    metrics=final_state.metrics.model_dump(),
                )

            self.db.update_session_status(session_id, "completed")
            logger.info(f"Pipeline complete for session {session_id}")
            return final_state

        except Exception as e:
            logger.error(f"Pipeline failed for session {session_id}: {e}")
            self.db.update_session_status(session_id, "failed")
            raise
