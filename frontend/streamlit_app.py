"""
ResearchPilot AI — Streamlit Multi-Page Frontend
Pages:
  1. Research       – topic input + configuration
  2. Agent Dashboard – real-time execution progress
  3. Report         – view, download PDF / Markdown
  4. Research Chat  – RAG-based follow-up Q&A
  5. History        – past sessions
"""
from __future__ import annotations

import io
import json
import os
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests
import streamlit as st

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ResearchPilot AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Constants ────────────────────────────────────────────────────────────────
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
POLL_INTERVAL = 3  # seconds between status polls

AGENT_ORDER = [
    "PlannerAgent",
    "SearchAgent",
    "RetrievalAgent",
    "SummarizerAgent",
    "CitationAgent",
    "ReportAgent",
]

AGENT_ICONS = {
    "PlannerAgent":     "🗺️",
    "SearchAgent":      "🔍",
    "RetrievalAgent":   "📦",
    "SummarizerAgent":  "📝",
    "CitationAgent":    "📚",
    "ReportAgent":      "📄",
}

STATUS_COLOR = {
    "pending":   "#94a3b8",
    "running":   "#f59e0b",
    "completed": "#22c55e",
    "failed":    "#ef4444",
    "skipped":   "#a855f7",
}

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ── */
[data-testid="stAppViewContainer"] { background: #0f172a; }
[data-testid="stSidebar"] { background: #1e293b; }
h1, h2, h3, h4 { color: #e2e8f0 !important; }
p, li, label, span { color: #cbd5e1; }

/* ── Sidebar brand ── */
.brand-title {
    font-size: 1.6rem; font-weight: 800;
    background: linear-gradient(135deg, #3b82f6, #8b5cf6);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0;
}
.brand-sub { font-size: 0.78rem; color: #64748b; margin-top: 0; }

/* ── Cards ── */
.card {
    background: #1e293b; border: 1px solid #334155;
    border-radius: 12px; padding: 1.2rem 1.4rem; margin-bottom: 1rem;
}
.card-title {
    font-size: 0.9rem; font-weight: 700; color: #94a3b8;
    text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem;
}
.metric-value { font-size: 2rem; font-weight: 800; color: #3b82f6; }
.metric-label { font-size: 0.8rem; color: #64748b; }

/* ── Agent status badge ── */
.badge {
    display: inline-block; padding: 2px 10px; border-radius: 999px;
    font-size: 0.72rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.04em;
}

/* ── Agent row ── */
.agent-row {
    display: flex; align-items: center; gap: 12px;
    background: #0f172a; border: 1px solid #1e293b;
    border-radius: 10px; padding: 0.75rem 1rem; margin-bottom: 0.5rem;
}
.agent-icon { font-size: 1.4rem; }
.agent-name { font-weight: 700; color: #e2e8f0; flex: 1; }
.agent-msg  { font-size: 0.8rem; color: #64748b; }

/* ── Chat bubbles ── */
.bubble-user {
    background: #1d4ed8; color: #e0f2fe;
    border-radius: 18px 18px 4px 18px;
    padding: 10px 16px; margin: 6px 0; max-width: 80%; margin-left: auto;
    font-size: 0.93rem;
}
.bubble-assistant {
    background: #1e293b; color: #e2e8f0;
    border: 1px solid #334155;
    border-radius: 18px 18px 18px 4px;
    padding: 10px 16px; margin: 6px 0; max-width: 90%;
    font-size: 0.93rem;
}
.source-chip {
    display: inline-block; background: #0f172a; border: 1px solid #334155;
    border-radius: 6px; padding: 2px 8px; font-size: 0.72rem;
    color: #94a3b8; margin: 2px 2px 0 0;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #3b82f6, #6366f1) !important;
    color: white !important; border: none !important;
    border-radius: 8px !important; font-weight: 600 !important;
    padding: 0.5rem 1.5rem !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #2563eb, #4f46e5) !important;
    transform: translateY(-1px); box-shadow: 0 4px 12px rgba(59,130,246,0.4) !important;
}

/* ── Markdown report area ── */
.report-area {
    background: #1e293b; border: 1px solid #334155;
    border-radius: 10px; padding: 1.5rem; font-family: 'Georgia', serif;
    color: #e2e8f0; line-height: 1.7;
}

/* ── Progress bar label ── */
.stProgress > div > div > div { background: linear-gradient(90deg, #3b82f6, #8b5cf6) !important; }
</style>
""", unsafe_allow_html=True)

# ─── Session State Init ───────────────────────────────────────────────────────
def _init_state() -> None:
    defaults: Dict[str, Any] = {
        "page": "research",
        "session_id": None,
        "research_status": "idle",   # idle | running | completed | failed
        "poll_start": None,
        "agent_logs": [],
        "metrics": {},
        "report_md": "",
        "papers": [],
        "summaries": [],
        "citations": [],
        "chat_history": [],
        "topic": "",
        "num_papers": 10,
        "depth": "standard",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ─── Backend helpers ─────────────────────────────────────────────────────────
def _api(method: str, path: str, **kwargs) -> Optional[Dict[str, Any]]:
    try:
        r = requests.request(method, f"{BACKEND_URL}{path}", timeout=60, **kwargs)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("⚠️ Cannot reach backend. Is it running on port 8000?")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"API error {e.response.status_code}: {e.response.text[:200]}")
        return None
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return None

def _api_raw(method: str, path: str, **kwargs) -> Optional[requests.Response]:
    try:
        r = requests.request(method, f"{BACKEND_URL}{path}", timeout=120, **kwargs)
        r.raise_for_status()
        return r
    except Exception:
        return None

def _check_backend() -> bool:
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False

# ─── Sidebar ─────────────────────────────────────────────────────────────────
def render_sidebar() -> None:
    with st.sidebar:
        st.markdown('<p class="brand-title">ResearchPilot AI</p>', unsafe_allow_html=True)
        st.markdown('<p class="brand-sub">Multi-Agent Research Assistant</p>', unsafe_allow_html=True)
        st.markdown("---")

        # Backend health
        healthy = _check_backend()
        dot = "🟢" if healthy else "🔴"
        st.markdown(f"{dot} Backend: {'Online' if healthy else 'Offline'}")
        st.markdown("---")

        # Navigation
        st.markdown("**Navigation**")
        pages = {
            "🧪 Research":        "research",
            "⚙️ Agent Dashboard": "dashboard",
            "📄 Report":          "report",
            "💬 Research Chat":   "chat",
            "🕐 History":         "history",
        }
        for label, key in pages.items():
            active = st.session_state.page == key
            style = "background:#1d4ed8;border-radius:6px;padding:4px 8px;" if active else "padding:4px 8px;"
            if st.button(label, key=f"nav_{key}", use_container_width=True):
                st.session_state.page = key
                st.rerun()

        st.markdown("---")

        # Active session info
        if st.session_state.session_id:
            st.markdown("**Active Session**")
            sid = st.session_state.session_id
            st.code(sid[:16] + "...", language=None)
            st.caption(f"Topic: {st.session_state.topic[:40]}...")
            status = str(st.session_state.research_status).lower()
            color = STATUS_COLOR.get(status, "#94a3b8")
            st.markdown(
                f'<span class="badge" style="background:{color}22;color:{color};">{status}</span>',
                unsafe_allow_html=True,
            )

            if st.button("🗑️ Clear Session", use_container_width=True):
                for k in ["session_id", "research_status", "agent_logs", "metrics",
                          "report_md", "papers", "summaries", "citations", "chat_history"]:
                    st.session_state[k] = [] if isinstance(st.session_state[k], list) else \
                                          {} if isinstance(st.session_state[k], dict) else \
                                          None if k == "session_id" else \
                                          "" if k in ["report_md", "topic"] else "idle"
                st.rerun()

        st.markdown("---")
        st.caption("Powered by Groq LLM & LangGraph")


# ─── Page 1 : Research Input ─────────────────────────────────────────────────
def page_research() -> None:
    st.markdown("## 🧪 Start New Research")
    st.markdown("Configure your research parameters and launch the multi-agent pipeline.")

    col_left, col_right = st.columns([2, 1], gap="large")

    with col_left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Research Topic</div>', unsafe_allow_html=True)
        topic = st.text_area(
            "Enter your research topic",
            value=st.session_state.get("topic", ""),
            placeholder="e.g. Multi-agent systems in healthcare AI",
            height=100,
            label_visibility="collapsed",
        )
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Configuration</div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            depth = st.selectbox(
                "Research Depth",
                options=["basic", "standard", "deep"],
                index=["basic", "standard", "deep"].index(st.session_state.get("depth", "standard")),
                format_func=lambda x: {"basic": "⚡ Basic", "standard": "🔬 Standard", "deep": "🧬 Deep"}[x],
            )
        with c2:
            num_papers = st.selectbox(
                "Number of Papers",
                options=[5, 10, 15, 20],
                index=[5, 10, 15, 20].index(
                    st.session_state.get("num_papers", 10)
                    if st.session_state.get("num_papers", 10) in [5, 10, 15, 20] else 10
                ),
            )
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        # Depth info
        depth_info = {
            "basic":    ("⚡ Basic", "2 search queries\n4 keywords\nFast results"),
            "standard": ("🔬 Standard", "3 search queries\n6 keywords\nBalanced depth"),
            "deep":     ("🧬 Deep", "5 search queries\n10 keywords\nMaximum coverage"),
        }
        selected_depth = depth if topic else st.session_state.get("depth", "standard")
        d_title, d_desc = depth_info.get(selected_depth, depth_info["standard"])
        st.markdown(f"""
        <div class="card">
            <div class="card-title">Depth Profile</div>
            <div style="font-size:1.5rem;margin-bottom:8px;">{d_title}</div>
            <pre style="color:#94a3b8;font-size:0.82rem;font-family:monospace;">{d_desc}</pre>
        </div>
        """, unsafe_allow_html=True)

        # Pipeline preview
        st.markdown("""
        <div class="card">
            <div class="card-title">Pipeline</div>
            <div style="font-size:0.82rem;color:#94a3b8;line-height:1.8;">
            🗺️ Planner → 🔍 Search<br>
            → 📦 Retrieval → 📝 Summarize<br>
            → 📚 Citations → 📄 Report
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Launch button
    col_btn, col_msg = st.columns([1, 3])
    with col_btn:
        launch = st.button("🚀 Generate Research Report", use_container_width=True,
                           disabled=(not topic.strip()))

    if launch:
        if not topic.strip():
            st.warning("Please enter a research topic.")
            return

        session_id = str(uuid.uuid4())
        st.session_state.session_id   = session_id
        st.session_state.topic        = topic.strip()
        st.session_state.depth        = depth
        st.session_state.num_papers   = num_papers
        st.session_state.research_status = "running"
        st.session_state.agent_logs   = []
        st.session_state.metrics      = {}
        st.session_state.report_md    = ""
        st.session_state.papers       = []
        st.session_state.chat_history = []
        st.session_state.poll_start   = time.time()

        payload = {
            "topic":      topic.strip(),
            "depth":      depth,
            "num_papers": num_papers,
            "session_id": session_id,
        }
        result = _api("POST", "/api/research", json=payload)
        if result:
            st.success(f"✅ Research pipeline started! Session: `{session_id[:16]}...`")
            st.session_state.page = "dashboard"
            st.rerun()


# ─── Page 2 : Agent Dashboard ────────────────────────────────────────────────
def _agent_status_from_logs(logs: List[Dict]) -> Dict[str, str]:
    """Derive latest status per agent from log list."""
    status: Dict[str, str] = {a: "pending" for a in AGENT_ORDER}
    for log in logs:
        name = log.get("agent_name", "")
        raw  = log.get("status", "")
        # Handle enum serialization: could be "COMPLETED", {"value":"completed"}, etc.
        if isinstance(raw, dict):
            s = str(raw.get("value", "")).lower()
        else:
            s = str(raw).lower()
        if name in status and s:
            status[name] = s
    return status

def _progress_from_status(status_map: Dict[str, str]) -> float:
    done = sum(1 for s in status_map.values() if s.lower() == "completed")
    return done / max(len(AGENT_ORDER), 1)

def page_dashboard() -> None:
    sid = st.session_state.get("session_id")
    if not sid:
        st.info("No active session. Start a research task first.")
        if st.button("Go to Research"):
            st.session_state.page = "research"
            st.rerun()
        return

    st.markdown("## ⚙️ Agent Execution Dashboard")
    st.caption(f"Session: `{sid}`  |  Topic: *{st.session_state.topic}*")

    # Poll backend
    data = _api("GET", f"/api/session/{sid}/status")
    if data:
        raw_status = data.get("status", "running")
        # Normalize: could be "COMPLETED", {"value":"completed"}, etc.
        if isinstance(raw_status, dict):
            raw_status = raw_status.get("value", "running")
        st.session_state.research_status = str(raw_status).lower()
        st.session_state.agent_logs      = data.get("agent_logs", [])
        st.session_state.metrics         = data.get("metrics") or {}

    status      = st.session_state.research_status.lower() if st.session_state.research_status else "running"
    logs        = st.session_state.agent_logs
    agent_stat  = _agent_status_from_logs(logs)
    progress    = _progress_from_status(agent_stat)

    # ── Overall progress bar ──
    st.markdown("### Overall Progress")
    col_prog, col_stat = st.columns([3, 1])
    with col_prog:
        st.progress(progress)
    with col_stat:
        color = STATUS_COLOR.get(status, "#94a3b8")
        st.markdown(
            f'<span class="badge" style="background:{color}22;color:{color};">{status}</span>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Agent cards ──
    st.markdown("### Agent Status")
    for agent in AGENT_ORDER:
        s     = agent_stat.get(agent, "pending").lower()
        icon  = AGENT_ICONS.get(agent, "🤖")
        color = STATUS_COLOR.get(s, "#94a3b8")

        # Find latest log message for this agent
        msg = ""
        dur = ""
        for log in reversed(logs):
            if log.get("agent_name") == agent:
                msg = log.get("message", "")[:100]
                d   = log.get("duration_seconds")
                if d:
                    dur = f"{d:.1f}s"
                break

        spinner = "⏳ " if s == "running" else ""
        st.markdown(f"""
        <div class="agent-row">
            <span class="agent-icon">{icon}</span>
            <span class="agent-name">{spinner}{agent}</span>
            <span class="agent-msg">{msg}</span>
            <span class="badge" style="background:{color}22;color:{color};">{s} {dur}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Metrics panel ──
    metrics = st.session_state.metrics or {}
    if metrics:
        st.markdown("### 📊 Session Metrics")
        m1, m2, m3, m4, m5 = st.columns(5)
        def _metric_card(col, label, value):
            col.markdown(f"""
            <div class="card" style="text-align:center;">
                <div class="metric-value">{value}</div>
                <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)

        _metric_card(m1, "Papers",        metrics.get("number_of_papers", "—"))
        _metric_card(m2, "Citations",     metrics.get("citation_count", "—"))
        _metric_card(m3, "Retrieval (s)", f"{metrics.get('retrieval_time', 0):.1f}")
        _metric_card(m4, "Embedding (s)", f"{metrics.get('embedding_time', 0):.1f}")
        _metric_card(m5, "Report (s)",    f"{metrics.get('report_generation_time', 0):.1f}")

    # ── Auto-refresh while running ──
    if status == "running":
        time.sleep(POLL_INTERVAL)
        st.rerun()

    # ── Navigation when done ──
    if status == "completed":
        st.success("✅ Research pipeline completed successfully!")
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("📄 View Report", use_container_width=True):
                st.session_state.page = "report"
                st.rerun()
        with c2:
            if st.button("💬 Research Chat", use_container_width=True):
                st.session_state.page = "chat"
                st.rerun()
        with c3:
            if st.button("🧪 New Research", use_container_width=True):
                st.session_state.page = "research"
                st.rerun()

    if status == "failed":
        st.error("❌ Pipeline failed. Check logs above for details.")


# ─── Page 3 : Report ─────────────────────────────────────────────────────────
def page_report() -> None:
    sid = st.session_state.get("session_id")
    if not sid:
        st.info("No active session.")
        return

    st.markdown("## 📄 Research Report")
    st.caption(f"Session: `{sid}`  |  Topic: *{st.session_state.topic}*")

    # Load report
    if not st.session_state.report_md:
        with st.spinner("Loading report..."):
            data = _api("GET", f"/api/session/{sid}/report")
            if data:
                st.session_state.report_md = data.get("markdown_content", "")
                st.session_state.report_meta = data

    if not st.session_state.report_md:
        st.warning("Report not ready yet. Complete the research pipeline first.")
        if st.button("⚙️ Go to Dashboard"):
            st.session_state.page = "dashboard"
            st.rerun()
        return

    md   = st.session_state.report_md
    meta = st.session_state.get("report_meta", {})

    # ── Download buttons ──
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        r_pdf = _api_raw("GET", f"/api/session/{sid}/report/pdf")
        if r_pdf:
            st.download_button(
                "⬇️ Download PDF",
                data=r_pdf.content,
                file_name=f"research_report_{sid[:8]}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
    with col2:
        st.download_button(
            "⬇️ Download Markdown",
            data=md.encode("utf-8"),
            file_name=f"research_report_{sid[:8]}.md",
            mime="text/markdown",
            use_container_width=True,
        )

    st.markdown("---")

    # ── Report versions ──
    with st.expander("📋 Report Version History"):
        versions = _api("GET", f"/api/session/{sid}/report/versions") or []
        if versions:
            for v in versions:
                st.markdown(f"- **v{v['version']}** — {v['created_at']}")
        else:
            st.caption("No version history available.")

    # ── Report content tabs ──
    tab_preview, tab_raw, tab_papers, tab_citations = st.tabs(
        ["📖 Preview", "🔤 Raw Markdown", "📚 Papers", "🏷️ Citations"]
    )

    with tab_preview:
        st.markdown(f'<div class="report-area">', unsafe_allow_html=True)
        # Render sections individually for better formatting
        sections = md.split("\n---\n")
        for section in sections:
            st.markdown(section)
            st.markdown("---")
        st.markdown('</div>', unsafe_allow_html=True)

    with tab_raw:
        st.code(md, language="markdown")

    with tab_papers:
        papers = _api("GET", f"/api/session/{sid}/papers") or []
        summaries_data = _api("GET", f"/api/session/{sid}/summaries") or []
        summary_map = {s["paper_id"]: s for s in summaries_data}

        st.markdown(f"**{len(papers)} papers retrieved**")
        for i, paper in enumerate(papers, 1):
            with st.expander(f"{i}. {paper.get('title', 'Untitled')[:80]} ({paper.get('year', '?')})"):
                st.markdown(f"**Authors:** {', '.join(paper.get('authors', [])[:5])}")
                st.markdown(f"**URL:** [{paper.get('url', '')}]({paper.get('url', '')})")
                st.markdown(f"**Categories:** {', '.join(paper.get('categories', []))}")
                st.markdown("**Abstract:**")
                st.markdown(paper.get("abstract", "")[:600] + "...")

                summ = summary_map.get(paper.get("paper_id", ""))
                if summ:
                    st.markdown("**Key Findings:**")
                    for f_item in summ.get("key_findings", [])[:4]:
                        st.markdown(f"- {f_item}")

    with tab_citations:
        citations = _api("GET", f"/api/session/{sid}/citations") or []
        if not citations:
            st.info("No citations found.")
        else:
            fmt = st.selectbox("Format", ["apa", "mla", "ieee", "bibtex"])
            bibtex_all = ""
            for c in citations:
                title = c.get("title", c.get("paper_id", ""))
                with st.expander(f"📌 {title[:70]}"):
                    val = c.get(fmt, "")
                    if fmt == "bibtex":
                        st.code(val, language="bibtex")
                    else:
                        st.markdown(f"> {val}")
                if fmt == "bibtex":
                    bibtex_all += c.get("bibtex", "") + "\n\n"

            if fmt == "bibtex" and bibtex_all:
                st.download_button(
                    "⬇️ Download All BibTeX",
                    data=bibtex_all.encode(),
                    file_name=f"references_{sid[:8]}.bib",
                    mime="text/plain",
                )


# ─── Page 4 : Research Chat ──────────────────────────────────────────────────
def page_chat() -> None:
    sid = st.session_state.get("session_id")
    if not sid:
        st.info("No active session. Please complete a research task first.")
        return

    st.markdown("## 💬 Research Chat")
    st.caption(
        f"Ask follow-up questions about the retrieved papers. "
        f"Session: `{sid}`"
    )

    # Check session status
    status = st.session_state.research_status
    if status not in ("completed",):
        st.warning(f"Research pipeline status: **{status}**. Chat is available after completion.")
        if st.button("⚙️ Go to Dashboard"):
            st.session_state.page = "dashboard"
            st.rerun()
        return

    # Example questions
    with st.expander("💡 Example questions"):
        examples = [
            "What are the common limitations across all papers?",
            "Which paper introduced the most novel methodology?",
            "Compare the evaluation metrics used in these papers.",
            "What datasets were most commonly used?",
            "What are the main future research directions suggested?",
        ]
        for ex in examples:
            if st.button(ex, key=f"ex_{ex[:20]}"):
                st.session_state._chat_prefill = ex
                st.rerun()

    st.markdown("---")

    # Chat history display
    history = st.session_state.chat_history
    chat_container = st.container()
    with chat_container:
        if not history:
            st.markdown(
                '<div style="text-align:center;color:#475569;padding:2rem;">'
                '🔬 Ask a question about the research papers above</div>',
                unsafe_allow_html=True,
            )
        for msg in history:
            role    = msg.get("role", "user")
            content = msg.get("content", "")
            sources = msg.get("sources", [])
            if role == "user":
                st.markdown(
                    f'<div class="bubble-user">{content}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="bubble-assistant">{content}</div>',
                    unsafe_allow_html=True,
                )
                if sources:
                    chips = "".join(
                        f'<span class="source-chip">📌 {s[:60]}</span>'
                        for s in sources[:4]
                    )
                    st.markdown(
                        f'<div style="margin-top:4px;">{chips}</div>',
                        unsafe_allow_html=True,
                    )

    st.markdown("---")

    # Input
    prefill = st.session_state.pop("_chat_prefill", "")
    with st.form("chat_form", clear_on_submit=True):
        cols = st.columns([6, 1])
        with cols[0]:
            question = st.text_input(
                "Your question",
                value=prefill,
                placeholder="e.g. What methods were used in paper 3?",
                label_visibility="collapsed",
            )
        with cols[1]:
            submit = st.form_submit_button("Send ➤", use_container_width=True)

    if submit and question.strip():
        # Append user message
        st.session_state.chat_history.append({
            "role": "user",
            "content": question.strip(),
            "sources": [],
        })

        with st.spinner("Thinking..."):
            payload = {
                "session_id":   sid,
                "question":     question.strip(),
                "chat_history": st.session_state.chat_history[:-1],
            }
            resp = _api("POST", f"/api/session/{sid}/chat", json=payload)

        if resp:
            st.session_state.chat_history.append({
                "role":    "assistant",
                "content": resp.get("answer", "No answer returned."),
                "sources": resp.get("sources", []),
            })
        else:
            st.session_state.chat_history.append({
                "role":    "assistant",
                "content": "⚠️ Failed to get a response. Please try again.",
                "sources": [],
            })
        st.rerun()

    # Clear chat
    if history:
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()


# ─── Page 5 : History ────────────────────────────────────────────────────────
def page_history() -> None:
    st.markdown("## 🕐 Research History")
    st.caption("Browse and resume past research sessions.")

    sessions = _api("GET", "/api/sessions") or []

    if not sessions:
        st.info("No research sessions found yet. Start your first research task!")
        if st.button("🧪 Start Research"):
            st.session_state.page = "research"
            st.rerun()
        return

    # Search/filter
    search = st.text_input("🔍 Filter by topic", placeholder="Search sessions...")
    if search:
        sessions = [s for s in sessions if search.lower() in s.get("topic", "").lower()]

    st.markdown(f"**{len(sessions)} sessions found**")
    st.markdown("---")

    for sess in sessions:
        sid    = sess.get("id", "")
        topic  = sess.get("topic", "Unknown")
        status = str(sess.get("status", "unknown")).lower()
        ts     = sess.get("created_at", "")
        color  = STATUS_COLOR.get(status, "#94a3b8")

        c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
        with c1:
            st.markdown(
                f"**{topic[:60]}**  \n"
                f"<small style='color:#64748b;'>{sid[:16]}... | {ts[:19]}</small>",
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f'<span class="badge" style="background:{color}22;color:{color};">{status}</span>',
                unsafe_allow_html=True,
            )
        with c3:
            if status == "completed":
                if st.button("📄 Report", key=f"rep_{sid}"):
                    st.session_state.session_id      = sid
                    st.session_state.topic           = topic
                    st.session_state.research_status = status
                    st.session_state.report_md       = ""
                    st.session_state.page            = "report"
                    st.rerun()
        with c4:
            if status == "completed":
                if st.button("💬 Chat", key=f"chat_{sid}"):
                    st.session_state.session_id      = sid
                    st.session_state.topic           = topic
                    st.session_state.research_status = status
                    st.session_state.chat_history    = []
                    st.session_state.page            = "chat"
                    st.rerun()
        st.markdown("---")


# ─── Router ──────────────────────────────────────────────────────────────────
def main() -> None:
    render_sidebar()

    page = st.session_state.get("page", "research")

    if page == "research":
        page_research()
    elif page == "dashboard":
        page_dashboard()
    elif page == "report":
        page_report()
    elif page == "chat":
        page_chat()
    elif page == "history":
        page_history()
    else:
        page_research()


if __name__ == "__main__":
    main()
