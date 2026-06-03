"""
FastAPI application entry point.
Provides REST API for the ResearchPilot AI backend.
"""

from __future__ import annotations

import asyncio
import uuid
import warnings

# Suppress known deprecation noise from third-party libs
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langgraph")

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
from loguru import logger
from pydantic import BaseModel

from backend.agents.chat_agent import ChatAgent
from backend.config import get_settings
from backend.graph.research_graph import ResearchPipeline
from backend.models import (
    ChatRequest,
    ChatResponse,
    ResearchDepth,
    ResearchRequest,
    ResearchState,
)
from backend.services.database import DatabaseService
from backend.services.pdf_service import PDFService

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

settings = get_settings()

app = FastAPI(
    title="ResearchPilot AI",
    description="Multi-Agent Research Assistant powered by Groq LLM & LangGraph",
    version=settings.app_version,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Singleton services
_db = DatabaseService()
_pdf = PDFService()
_chat = ChatAgent()
_executor = ThreadPoolExecutor(max_workers=4)

# In-memory session status cache (for real-time progress)
_session_progress: Dict[str, Dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class ResearchStatusResponse(BaseModel):
    session_id: str
    status: str
    topic: str
    num_papers: int
    agent_logs: List[Dict[str, Any]] = []
    metrics: Optional[Dict[str, Any]] = None
    created_at: str


class ReportResponse(BaseModel):
    session_id: str
    topic: str
    markdown_content: str
    version: int
    created_at: str


class SessionListItem(BaseModel):
    id: str
    topic: str
    status: str
    created_at: str


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    """Redirect root to the interactive API docs."""
    return RedirectResponse(url="/api/docs")


@app.get("/health", tags=["System"])
async def health_check() -> Dict[str, str]:
    return {
        "status": "healthy",
        "app": settings.app_name,
        "version": settings.app_version,
        "timestamp": datetime.utcnow().isoformat(),
    }


# ---------------------------------------------------------------------------
# Research pipeline
# ---------------------------------------------------------------------------


@app.post("/api/research", tags=["Research"], response_model=Dict[str, str])
async def start_research(
    request: ResearchRequest,
    background_tasks: BackgroundTasks,
) -> Dict[str, str]:
    """
    Kick off a new research session in the background.
    Returns the session_id immediately.
    """
    session_id = request.session_id or str(uuid.uuid4())

    _session_progress[session_id] = {
        "status": "running",
        "topic": request.topic,
        "depth": request.depth.value,
        "num_papers": request.num_papers,
        "started_at": datetime.utcnow().isoformat(),
    }

    background_tasks.add_task(
        _run_pipeline_background,
        session_id,
        request.topic,
        request.depth,
        request.num_papers,
    )

    return {"session_id": session_id, "status": "running"}


def _run_pipeline_background(
    session_id: str,
    topic: str,
    depth: ResearchDepth,
    num_papers: int,
) -> None:
    """Execute the full research pipeline (runs in background thread)."""
    try:
        pipeline = ResearchPipeline()
        final_state = pipeline.run(
            topic=topic,
            depth=depth,
            num_papers=num_papers,
            session_id=session_id,
        )
        _session_progress[session_id]["status"] = "completed"
        _session_progress[session_id]["num_papers_found"] = len(final_state.papers)
        logger.info(f"Background pipeline done: {session_id}")
    except Exception as e:
        _session_progress[session_id]["status"] = "failed"
        _session_progress[session_id]["error"] = str(e)
        logger.error(f"Background pipeline failed {session_id}: {e}")


@app.post("/api/research/sync", tags=["Research"])
async def run_research_sync(request: ResearchRequest) -> Dict[str, Any]:
    """
    Run the full research pipeline synchronously and return results.
    Use for smaller requests (basic depth, few papers).
    """
    session_id = request.session_id or str(uuid.uuid4())

    loop = asyncio.get_event_loop()
    try:
        pipeline = ResearchPipeline()
        final_state = await loop.run_in_executor(
            _executor,
            lambda: pipeline.run(
                topic=request.topic,
                depth=request.depth,
                num_papers=request.num_papers,
                session_id=session_id,
            ),
        )

        return {
            "session_id": session_id,
            "status": "completed",
            "num_papers": len(final_state.papers),
            "has_report": final_state.report is not None,
            "errors": final_state.errors,
        }

    except Exception as e:
        logger.error(f"Sync pipeline failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Session status
# ---------------------------------------------------------------------------


@app.get("/api/session/{session_id}/status", tags=["Session"])
async def get_session_status(session_id: str) -> Dict[str, Any]:
    """Get real-time status for a research session."""
    # Check in-memory progress first
    progress = _session_progress.get(session_id, {})

    # Supplement with DB data
    db_info = _db.get_session_info(session_id)
    agent_logs = _db.get_agent_logs(session_id)
    metrics = _db.get_metrics(session_id)

    if not db_info and not progress:
        raise HTTPException(status_code=404, detail="Session not found")

    return {
        "session_id": session_id,
        "status": db_info.get("status", progress.get("status", "unknown")) if db_info else progress.get("status", "unknown"),
        "topic": db_info.get("topic", progress.get("topic", "")) if db_info else progress.get("topic", ""),
        "num_papers": db_info.get("num_papers", 0) if db_info else 0,
        "depth": db_info.get("depth", "") if db_info else "",
        "agent_logs": agent_logs,
        "metrics": metrics,
        "created_at": db_info.get("created_at", "") if db_info else "",
        "progress": progress,
    }


@app.get("/api/sessions", tags=["Session"])
async def list_sessions() -> List[Dict[str, Any]]:
    """List all research sessions."""
    return _db.list_sessions()


# ---------------------------------------------------------------------------
# Report endpoints
# ---------------------------------------------------------------------------


@app.get("/api/session/{session_id}/report", tags=["Report"])
async def get_report(session_id: str) -> Dict[str, Any]:
    """Get the latest report for a session."""
    report = _db.get_latest_report(session_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found for this session")
    return report


@app.get("/api/session/{session_id}/report/versions", tags=["Report"])
async def get_report_versions(session_id: str) -> List[Dict[str, Any]]:
    """List all report versions for a session."""
    return _db.get_report_versions(session_id)


@app.get("/api/session/{session_id}/report/pdf", tags=["Report"])
async def download_report_pdf(session_id: str) -> StreamingResponse:
    """Download the report as a PDF file."""
    report = _db.get_latest_report(session_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")

    try:
        pdf_bytes = _pdf.generate_pdf_bytes(report["markdown_content"])
        return StreamingResponse(
            iter([pdf_bytes]),
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename=report_{session_id[:8]}.pdf"
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(e)}")


@app.get("/api/session/{session_id}/report/markdown", tags=["Report"])
async def download_report_markdown(session_id: str) -> Response:
    """Download the report as a Markdown file."""
    report = _db.get_latest_report(session_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")

    return Response(
        content=report["markdown_content"],
        media_type="text/markdown",
        headers={
            "Content-Disposition": f"attachment; filename=report_{session_id[:8]}.md"
        },
    )


# ---------------------------------------------------------------------------
# Papers, summaries, citations
# ---------------------------------------------------------------------------


@app.get("/api/session/{session_id}/papers", tags=["Papers"])
async def get_papers(session_id: str) -> List[Dict[str, Any]]:
    """Get all papers for a session."""
    return _db.get_papers(session_id)


@app.get("/api/session/{session_id}/summaries", tags=["Papers"])
async def get_summaries(session_id: str) -> List[Dict[str, Any]]:
    """Get paper summaries for a session."""
    return _db.get_summaries(session_id)


@app.get("/api/session/{session_id}/citations", tags=["Papers"])
async def get_citations(session_id: str) -> List[Dict[str, Any]]:
    """Get citations for a session."""
    return _db.get_citations(session_id)


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------


@app.post("/api/session/{session_id}/chat", tags=["Chat"])
async def research_chat(session_id: str, request: ChatRequest) -> ChatResponse:
    """
    Ask a follow-up question about the research using RAG.
    """
    # Verify session exists
    db_info = _db.get_session_info(session_id)
    if not db_info:
        raise HTTPException(status_code=404, detail="Session not found")

    loop = asyncio.get_event_loop()
    try:
        response = await loop.run_in_executor(
            _executor,
            lambda: _chat.chat_query(
                session_id=session_id,
                question=request.question,
                chat_history=request.chat_history,
                topic=db_info.get("topic", ""),
            ),
        )
        return response
    except Exception as e:
        logger.error(f"Chat failed for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@app.get("/api/session/{session_id}/metrics", tags=["Metrics"])
async def get_metrics(session_id: str) -> Dict[str, Any]:
    """Get evaluation metrics for a session."""
    metrics = _db.get_metrics(session_id)
    if not metrics:
        raise HTTPException(status_code=404, detail="Metrics not found")
    return metrics


# ---------------------------------------------------------------------------
# Dev runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host=settings.backend_host,
        port=settings.backend_port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )
