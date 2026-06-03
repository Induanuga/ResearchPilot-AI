"""
SQLite database service using SQLAlchemy.
Stores sessions, papers, summaries, citations, and reports.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from backend.config import get_settings


# ---------------------------------------------------------------------------
# ORM Models
# ---------------------------------------------------------------------------


class Base(DeclarativeBase):
    pass


class ResearchSessionDB(Base):
    __tablename__ = "research_sessions"

    id = Column(String, primary_key=True)
    topic = Column(String, nullable=False)
    depth = Column(String, default="standard")
    num_papers = Column(Integer, default=10)
    status = Column(String, default="running")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    metadata_json = Column(Text, default="{}")


class PaperDB(Base):
    __tablename__ = "papers"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, nullable=False, index=True)
    paper_id = Column(String, nullable=False)
    title = Column(String)
    authors = Column(Text)  # JSON list
    year = Column(Integer)
    abstract = Column(Text)
    url = Column(String)
    pdf_url = Column(String)
    categories = Column(Text)  # JSON list
    published_date = Column(String)
    journal_ref = Column(String)
    doi = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)


class SummaryDB(Base):
    __tablename__ = "summaries"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, nullable=False, index=True)
    paper_id = Column(String, nullable=False)
    summary = Column(Text)
    key_findings = Column(Text)  # JSON list
    methodology = Column(Text)
    limitations = Column(Text)  # JSON list
    future_work = Column(Text)  # JSON list
    relevance_score = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)


class CitationDB(Base):
    __tablename__ = "citations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, nullable=False, index=True)
    paper_id = Column(String, nullable=False)
    apa = Column(Text)
    mla = Column(Text)
    ieee = Column(Text)
    bibtex = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)


class ReportDB(Base):
    __tablename__ = "reports"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, nullable=False, index=True)
    version = Column(Integer, default=1)
    topic = Column(String)
    markdown_content = Column(Text)
    metadata_json = Column(Text, default="{}")
    created_at = Column(DateTime, default=datetime.utcnow)


class AgentLogDB(Base):
    __tablename__ = "agent_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, nullable=False, index=True)
    agent_name = Column(String)
    status = Column(String)
    message = Column(Text)
    duration_seconds = Column(Float)
    token_usage = Column(Text)  # JSON
    timestamp = Column(DateTime, default=datetime.utcnow)


class MetricsDB(Base):
    __tablename__ = "metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, nullable=False, unique=True, index=True)
    retrieval_time = Column(Float, default=0.0)
    embedding_time = Column(Float, default=0.0)
    report_generation_time = Column(Float, default=0.0)
    number_of_papers = Column(Integer, default=0)
    citation_count = Column(Integer, default=0)
    total_tokens_used = Column(Integer, default=0)
    agent_durations = Column(Text, default="{}")
    created_at = Column(DateTime, default=datetime.utcnow)


# ---------------------------------------------------------------------------
# Database Service
# ---------------------------------------------------------------------------


class DatabaseService:
    """Handles all SQLite database operations."""

    _instance: "DatabaseService | None" = None

    def __new__(cls) -> "DatabaseService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self.settings = get_settings()
        self.engine = create_engine(
            self.settings.database_url,
            connect_args={"check_same_thread": False},
            echo=False,
        )
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        Base.metadata.create_all(bind=self.engine)
        self._initialized = True
        logger.info("Database initialized")

    def get_session(self) -> Session:
        return self.SessionLocal()

    # ------------------------------------------------------------------
    # Research Sessions
    # ------------------------------------------------------------------

    def create_session(self, session_id: str, topic: str, depth: str, num_papers: int) -> None:
        with self.get_session() as db:
            record = ResearchSessionDB(
                id=session_id,
                topic=topic,
                depth=depth,
                num_papers=num_papers,
            )
            db.add(record)
            db.commit()
            logger.debug(f"Created session: {session_id}")

    def update_session_status(self, session_id: str, status: str) -> None:
        with self.get_session() as db:
            db.execute(
                text("UPDATE research_sessions SET status=:s, updated_at=:u WHERE id=:id"),
                {"s": status, "u": datetime.utcnow(), "id": session_id},
            )
            db.commit()

    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        with self.get_session() as db:
            row = db.query(ResearchSessionDB).filter_by(id=session_id).first()
            if not row:
                return None
            return {
                "id": row.id,
                "topic": row.topic,
                "depth": row.depth,
                "num_papers": row.num_papers,
                "status": row.status,
                "created_at": row.created_at.isoformat(),
            }

    def list_sessions(self) -> List[Dict[str, Any]]:
        with self.get_session() as db:
            rows = db.query(ResearchSessionDB).order_by(ResearchSessionDB.created_at.desc()).all()
            return [
                {
                    "id": r.id,
                    "topic": r.topic,
                    "status": r.status,
                    "created_at": r.created_at.isoformat(),
                }
                for r in rows
            ]

    # ------------------------------------------------------------------
    # Papers
    # ------------------------------------------------------------------

    def save_papers(self, session_id: str, papers: List[Dict[str, Any]]) -> None:
        with self.get_session() as db:
            for p in papers:
                record = PaperDB(
                    session_id=session_id,
                    paper_id=p.get("paper_id", ""),
                    title=p.get("title", ""),
                    authors=json.dumps(p.get("authors", [])),
                    year=p.get("year", 0),
                    abstract=p.get("abstract", ""),
                    url=p.get("url", ""),
                    pdf_url=p.get("pdf_url", ""),
                    categories=json.dumps(p.get("categories", [])),
                    published_date=p.get("published_date", ""),
                    journal_ref=p.get("journal_ref", ""),
                    doi=p.get("doi", ""),
                )
                db.add(record)
            db.commit()
            logger.debug(f"Saved {len(papers)} papers for session {session_id}")

    def get_papers(self, session_id: str) -> List[Dict[str, Any]]:
        with self.get_session() as db:
            rows = db.query(PaperDB).filter_by(session_id=session_id).all()
            return [
                {
                    "paper_id": r.paper_id,
                    "title": r.title,
                    "authors": json.loads(r.authors or "[]"),
                    "year": r.year,
                    "abstract": r.abstract,
                    "url": r.url,
                    "pdf_url": r.pdf_url,
                    "categories": json.loads(r.categories or "[]"),
                    "published_date": r.published_date,
                }
                for r in rows
            ]

    # ------------------------------------------------------------------
    # Summaries
    # ------------------------------------------------------------------

    def save_summaries(self, session_id: str, summaries: List[Dict[str, Any]]) -> None:
        with self.get_session() as db:
            for s in summaries:
                record = SummaryDB(
                    session_id=session_id,
                    paper_id=s.get("paper_id", ""),
                    summary=s.get("summary", ""),
                    key_findings=json.dumps(s.get("key_findings", [])),
                    methodology=s.get("methodology", ""),
                    limitations=json.dumps(s.get("limitations", [])),
                    future_work=json.dumps(s.get("future_work", [])),
                    relevance_score=s.get("relevance_score", 0.0),
                )
                db.add(record)
            db.commit()

    def get_summaries(self, session_id: str) -> List[Dict[str, Any]]:
        with self.get_session() as db:
            rows = db.query(SummaryDB).filter_by(session_id=session_id).all()
            return [
                {
                    "paper_id": r.paper_id,
                    "summary": r.summary,
                    "key_findings": json.loads(r.key_findings or "[]"),
                    "methodology": r.methodology,
                    "limitations": json.loads(r.limitations or "[]"),
                    "future_work": json.loads(r.future_work or "[]"),
                }
                for r in rows
            ]

    # ------------------------------------------------------------------
    # Citations
    # ------------------------------------------------------------------

    def save_citations(self, session_id: str, citations: List[Dict[str, Any]]) -> None:
        with self.get_session() as db:
            for c in citations:
                record = CitationDB(
                    session_id=session_id,
                    paper_id=c.get("paper_id", ""),
                    apa=c.get("apa", ""),
                    mla=c.get("mla", ""),
                    ieee=c.get("ieee", ""),
                    bibtex=c.get("bibtex", ""),
                )
                db.add(record)
            db.commit()

    def get_citations(self, session_id: str) -> List[Dict[str, Any]]:
        with self.get_session() as db:
            rows = db.query(CitationDB).filter_by(session_id=session_id).all()
            return [
                {
                    "paper_id": r.paper_id,
                    "apa": r.apa,
                    "mla": r.mla,
                    "ieee": r.ieee,
                    "bibtex": r.bibtex,
                }
                for r in rows
            ]

    # ------------------------------------------------------------------
    # Reports
    # ------------------------------------------------------------------

    def save_report(
        self,
        session_id: str,
        topic: str,
        markdown_content: str,
        metadata: Dict[str, Any],
    ) -> int:
        with self.get_session() as db:
            # Version increment
            existing = db.query(ReportDB).filter_by(session_id=session_id).count()
            version = existing + 1

            record = ReportDB(
                session_id=session_id,
                version=version,
                topic=topic,
                markdown_content=markdown_content,
                metadata_json=json.dumps(metadata),
            )
            db.add(record)
            db.commit()
            logger.debug(f"Saved report v{version} for session {session_id}")
            return version

    def get_latest_report(self, session_id: str) -> Optional[Dict[str, Any]]:
        with self.get_session() as db:
            row = (
                db.query(ReportDB)
                .filter_by(session_id=session_id)
                .order_by(ReportDB.version.desc())
                .first()
            )
            if not row:
                return None
            return {
                "session_id": row.session_id,
                "version": row.version,
                "topic": row.topic,
                "markdown_content": row.markdown_content,
                "metadata": json.loads(row.metadata_json or "{}"),
                "created_at": row.created_at.isoformat(),
            }

    def get_report_versions(self, session_id: str) -> List[Dict[str, Any]]:
        with self.get_session() as db:
            rows = (
                db.query(ReportDB)
                .filter_by(session_id=session_id)
                .order_by(ReportDB.version.desc())
                .all()
            )
            return [
                {"version": r.version, "created_at": r.created_at.isoformat()}
                for r in rows
            ]

    # ------------------------------------------------------------------
    # Agent Logs
    # ------------------------------------------------------------------

    def save_agent_log(
        self,
        session_id: str,
        agent_name: str,
        status: str,
        message: str,
        duration: float = 0.0,
        token_usage: Optional[Dict[str, int]] = None,
    ) -> None:
        with self.get_session() as db:
            record = AgentLogDB(
                session_id=session_id,
                agent_name=agent_name,
                status=status,
                message=message,
                duration_seconds=duration,
                token_usage=json.dumps(token_usage or {}),
            )
            db.add(record)
            db.commit()

    def get_agent_logs(self, session_id: str) -> List[Dict[str, Any]]:
        with self.get_session() as db:
            rows = (
                db.query(AgentLogDB)
                .filter_by(session_id=session_id)
                .order_by(AgentLogDB.timestamp)
                .all()
            )
            return [
                {
                    "agent_name": r.agent_name,
                    "status": r.status,
                    "message": r.message,
                    "duration_seconds": r.duration_seconds,
                    "token_usage": json.loads(r.token_usage or "{}"),
                    "timestamp": r.timestamp.isoformat(),
                }
                for r in rows
            ]

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def save_metrics(self, session_id: str, metrics: Dict[str, Any]) -> None:
        with self.get_session() as db:
            existing = db.query(MetricsDB).filter_by(session_id=session_id).first()
            if existing:
                for key, value in metrics.items():
                    if key == "agent_durations":
                        setattr(existing, key, json.dumps(value))
                    elif hasattr(existing, key):
                        setattr(existing, key, value)
            else:
                record = MetricsDB(
                    session_id=session_id,
                    retrieval_time=metrics.get("retrieval_time", 0.0),
                    embedding_time=metrics.get("embedding_time", 0.0),
                    report_generation_time=metrics.get("report_generation_time", 0.0),
                    number_of_papers=metrics.get("number_of_papers", 0),
                    citation_count=metrics.get("citation_count", 0),
                    total_tokens_used=metrics.get("total_tokens_used", 0),
                    agent_durations=json.dumps(metrics.get("agent_durations", {})),
                )
                db.add(record)
            db.commit()

    def get_metrics(self, session_id: str) -> Optional[Dict[str, Any]]:
        with self.get_session() as db:
            row = db.query(MetricsDB).filter_by(session_id=session_id).first()
            if not row:
                return None
            return {
                "session_id": row.session_id,
                "retrieval_time": row.retrieval_time,
                "embedding_time": row.embedding_time,
                "report_generation_time": row.report_generation_time,
                "number_of_papers": row.number_of_papers,
                "citation_count": row.citation_count,
                "total_tokens_used": row.total_tokens_used,
                "agent_durations": json.loads(row.agent_durations or "{}"),
            }
