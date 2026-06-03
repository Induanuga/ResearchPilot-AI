"""
Shared data models (Pydantic schemas) used across the application.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class ResearchDepth(str, Enum):
    BASIC = "basic"
    STANDARD = "standard"
    DEEP = "deep"


class AgentStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class CitationFormat(str, Enum):
    APA = "apa"
    MLA = "mla"
    IEEE = "ieee"
    BIBTEX = "bibtex"


# ---------------------------------------------------------------------------
# Paper models
# ---------------------------------------------------------------------------


class PaperMetadata(BaseModel):
    """Metadata for a single research paper."""

    paper_id: str
    title: str
    authors: List[str]
    year: int
    abstract: str
    url: str
    pdf_url: Optional[str] = None
    categories: List[str] = Field(default_factory=list)
    published_date: Optional[str] = None
    journal_ref: Optional[str] = None
    doi: Optional[str] = None


class PaperSummary(BaseModel):
    """AI-generated summary of a paper."""

    paper_id: str
    title: str
    summary: str
    key_findings: List[str] = Field(default_factory=list)
    methodology: str = ""
    limitations: List[str] = Field(default_factory=list)
    future_work: List[str] = Field(default_factory=list)
    relevance_score: float = 0.0


class PaperCitation(BaseModel):
    """Citation strings for a paper in multiple formats."""

    paper_id: str
    title: str
    apa: str = ""
    mla: str = ""
    ieee: str = ""
    bibtex: str = ""


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class ResearchRequest(BaseModel):
    """Incoming research request from the user."""

    topic: str = Field(..., min_length=3, max_length=500)
    depth: ResearchDepth = ResearchDepth.STANDARD
    num_papers: int = Field(default=10, ge=3, le=30)
    session_id: Optional[str] = None


class ChatMessage(BaseModel):
    """A single chat message."""

    role: str  # "user" | "assistant"
    content: str
    sources: List[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ChatRequest(BaseModel):
    """Incoming chat request."""

    session_id: str
    question: str = Field(..., min_length=1, max_length=2000)
    chat_history: List[ChatMessage] = Field(default_factory=list)


class ChatResponse(BaseModel):
    """Response to a chat request."""

    answer: str
    sources: List[str] = Field(default_factory=list)
    session_id: str


# ---------------------------------------------------------------------------
# Agent state models
# ---------------------------------------------------------------------------


class AgentLog(BaseModel):
    """Execution log entry for an agent."""

    agent_name: str
    status: AgentStatus
    message: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    duration_seconds: Optional[float] = None
    token_usage: Optional[Dict[str, int]] = None

    model_config = {"use_enum_values": True}


class ResearchPlan(BaseModel):
    """Output of the Planner Agent."""

    keywords: List[str]
    research_plan: List[str]
    research_domains: List[str]
    search_queries: List[str]


class ResearchReport(BaseModel):
    """Final structured research report."""

    session_id: str
    topic: str
    executive_summary: str
    research_overview: str
    key_papers: List[Dict[str, Any]] = Field(default_factory=list)
    cross_paper_insights: str
    method_comparison: str
    trends: str
    limitations: str
    future_directions: str
    references: List[str] = Field(default_factory=list)
    markdown_content: str = ""
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class EvaluationMetrics(BaseModel):
    """Performance metrics for a research session."""

    session_id: str
    retrieval_time: float = 0.0
    embedding_time: float = 0.0
    report_generation_time: float = 0.0
    number_of_papers: int = 0
    citation_count: int = 0
    total_tokens_used: int = 0
    agent_durations: Dict[str, float] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Session state (used by LangGraph StateGraph)
# ---------------------------------------------------------------------------


class ResearchState(BaseModel):
    """Full state object passed through the LangGraph pipeline."""

    # Input
    session_id: str
    research_topic: str
    depth: ResearchDepth = ResearchDepth.STANDARD
    num_papers: int = 10

    # Planner outputs
    research_plan: Optional[ResearchPlan] = None

    # Search outputs
    papers: List[PaperMetadata] = Field(default_factory=list)

    # Retrieval outputs
    vectorstore_path: Optional[str] = None
    retrieved_docs: List[Dict[str, Any]] = Field(default_factory=list)

    # Summaries
    summaries: List[PaperSummary] = Field(default_factory=list)

    # Citations
    citations: List[PaperCitation] = Field(default_factory=list)

    # Final report
    report: Optional[ResearchReport] = None

    # Chat
    chat_history: List[ChatMessage] = Field(default_factory=list)

    # Agent logs & metrics
    agent_logs: List[AgentLog] = Field(default_factory=list)
    metrics: Optional[EvaluationMetrics] = None

    # Error tracking
    errors: List[str] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True
