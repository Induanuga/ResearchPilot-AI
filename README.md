# ResearchPilot AI (Multi-Agent Research Assistant)

AI research system powered by Groq (Llama 3.3 70B), LangGraph, FAISS, and HuggingFace Embeddings. Accepts a research topic and autonomously retrieves papers, builds a knowledge base, generates summaries, citations, and a full literature review, then enables follow-up Q&A via RAG.


## Features

| Feature | Description |
|---|---|
| **Planner Agent** | Expands topic into keywords, domains, and search queries |
| **Search Agent** | Queries ArXiv API, deduplicates and ranks papers |
| **Retrieval Agent** | Builds FAISS vector index from paper abstracts |
| **Summarizer Agent** | Generates structured JSON summaries (findings, methodology, limitations) |
| **Citation Agent** | Produces APA, MLA, IEEE, BibTeX for every paper |
| **Report Agent** | Synthesises a full literature review with 9 sections |
| **Chat Agent** | RAG-based Q&A with source attribution |
| **SQLite Persistence** | Sessions, papers, summaries, citations, reports stored locally |
| **PDF & Markdown Export** | Download formatted reports |
| **Metrics Dashboard** | Retrieval time, embedding time, token usage, paper count |


## Architecture

```
User Query
    │
    ▼
┌──────────────────────────────────────────────────────┐
│                  LangGraph StateGraph                │
│                                                      │
│  ┌──────────┐   ┌──────────┐   ┌─────────────┐       │
│  │ Planner  │──▶│  Search  │──▶│  Retrieval │       │
│  │  Agent   │   │  Agent   │   │    Agent    │       │
│  └──────────┘   └──────────┘   └─────────────┘       │
│       │              │               │               │
│  Research Plan   ArXiv API       FAISS Index         │
│                  + SQLite        + Embeddings        │
│                                      │               │
│  ┌─────────────┐   ┌──────────┐   ┌──▼──────────┐    │
│  │   Report    │◀──│ Citation │◀──│ Summarizer │    │
│  │    Agent    │   │  Agent   │   │    Agent    │    │
│  └─────────────┘   └──────────┘   └─────────────┘    │
│        │                                             │
│    Markdown + PDF                                    │
└──────────────────────────────────────────────────────┘
    │
    ▼
┌───────────────────────────┐
│   Research Chat Agent     │  ◀── RAG over FAISS
│   (Follow-up Q&A)         │
└───────────────────────────┘
```


## Project Structure

```
research-assistant/
├── backend/
│   ├── agents/
│   │   ├── base_agent.py          # Abstract base with LLM + logging
│   │   ├── planner_agent.py       # Query expansion + research plan
│   │   ├── search_agent.py        # ArXiv search + SQLite storage
│   │   ├── retrieval_agent.py     # FAISS index construction
│   │   ├── summarizer_agent.py    # Per-paper structured summaries
│   │   ├── citation_agent.py      # APA / MLA / IEEE / BibTeX
│   │   ├── report_agent.py        # Full literature review generation
│   │   └── chat_agent.py          # RAG-based conversational Q&A
│   ├── graph/
│   │   └── research_graph.py      # LangGraph StateGraph pipeline
│   ├── services/
│   │   ├── arxiv_service.py       # ArXiv API wrapper with retry
│   │   ├── embedding_service.py   # HuggingFace sentence-transformers
│   │   ├── vector_store.py        # FAISS per-session vector store
│   │   ├── database.py            # SQLAlchemy + SQLite ORM
│   │   ├── pdf_service.py         # ReportLab PDF generation
│   │   └── cache_service.py       # diskcache TTL caching
│   ├── config.py                  # Pydantic settings
│   ├── models.py                  # Shared Pydantic schemas
│   └── main.py                    # FastAPI app + endpoints
├── frontend/
│   └── streamlit_app.py           # 5-page Streamlit UI
├── data/
│   ├── vectorstore/               # FAISS indexes (per session)
│   ├── reports/                   # PDF + Markdown exports
│   ├── papers/                    # Paper cache
│   └── cache/                     # diskcache entries
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── run_backend.py
└── run_frontend.py
```


## Quick Start

### 1. Prerequisites

- Python 3.11+
- Groq API key — get one free at [console.groq.com/keys](https://console.groq.com/keys)

### 2. Clone & Install

```bash
git clone <repo-url>
cd research-assistant

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Configure

```bash
cp .env.example .env
# Edit .env and set your GROQ_API_KEY
```

### 4. Run (two terminals)

**Terminal 1 — Backend:**
```bash
python run_backend.py
# API available at http://localhost:8000
# Docs at http://localhost:8000/api/docs
```

**Terminal 2 — Frontend:**
```bash
python run_frontend.py
# UI available at http://localhost:8501
```


## Docker

```bash
# Copy and configure env
cp .env.example .env
# Set GROQ_API_KEY in .env

# Build & run both services
docker-compose up --build

# Backend: http://localhost:8000
# Frontend: http://localhost:8501
```


## Streamlit UI Pages

| Page | Description |
|------|-------------|
|  **Research** | Enter topic, select depth (Basic/Standard/Deep) and paper count |
|  **Agent Dashboard** | Live progress — agent status badges, metrics cards, auto-refresh |
|  **Report** | View full report, preview tabs, download PDF/Markdown, citation manager |
|  **Research Chat** | RAG Q&A with source chips, conversation memory, example questions |
|  **History** | Browse past sessions, reports or chat |


## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/api/research` | Start async research pipeline |
| `POST` | `/api/research/sync` | Run pipeline synchronously |
| `GET` | `/api/session/{id}/status` | Live status + agent logs |
| `GET` | `/api/sessions` | List all sessions |
| `GET` | `/api/session/{id}/report` | Get latest report |
| `GET` | `/api/session/{id}/report/pdf` | Download PDF |
| `GET` | `/api/session/{id}/report/markdown` | Download Markdown |
| `GET` | `/api/session/{id}/report/versions` | Report version history |
| `GET` | `/api/session/{id}/papers` | List papers |
| `GET` | `/api/session/{id}/summaries` | Get summaries |
| `GET` | `/api/session/{id}/citations` | Get citations |
| `POST` | `/api/session/{id}/chat` | RAG chat query |
| `GET` | `/api/session/{id}/metrics` | Session metrics |

Full interactive docs: `http://localhost:8000/api/docs`


## Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | Groq — Llama 3.3 70B Versatile |
| Agent Framework | LangGraph + LangChain |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector DB | FAISS (CPU) |
| Paper Retrieval | ArXiv API |
| Backend | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Database | SQLite (SQLAlchemy ORM) |
| PDF Export | ReportLab |
| Caching | diskcache |


## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | *(required)* | Groq API key — [console.groq.com/keys](https://console.groq.com/keys) |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Model identifier |
| `GROQ_TEMPERATURE` | `0.3` | Generation temperature |
| `GROQ_MAX_TOKENS` | `8192` | Max output tokens |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace model |
| `DATABASE_URL` | `sqlite:///./data/research_pilot.db` | SQLite path |
| `VECTORSTORE_PATH` | `./data/vectorstore` | FAISS index directory |
| `REPORTS_PATH` | `./data/reports` | PDF/MD output |
| `ARXIV_MAX_RESULTS` | `20` | Max papers per query |
| `BACKEND_HOST` | `0.0.0.0` | FastAPI host |
| `BACKEND_PORT` | `8000` | FastAPI port |




## LangGraph Pipeline State

```python
ResearchState {
    session_id, research_topic, depth, num_papers,
    research_plan,       # PlannerAgent output
    papers,              # SearchAgent output (list of PaperMetadata)
    vectorstore_path,    # RetrievalAgent output
    retrieved_docs,      # FAISS top-k results
    summaries,           # SummarizerAgent output
    citations,           # CitationAgent output
    report,              # ReportAgent output
    chat_history,        # ChatAgent memory
    agent_logs,          # Execution trace
    metrics,             # Performance metrics
    errors               # Error accumulator
}
```



## Report Sections

Every generated report contains:

1. **Executive Summary** — high-level overview of findings
2. **Research Overview** — field context and history
3. **Key Papers** — per-paper summaries with citations
4. **Cross-Paper Insights** — patterns and contradictions across papers
5. **Method Comparison** — side-by-side methodology analysis
6. **Research Trends** — temporal evolution of the field
7. **Limitations** — aggregated limitations across papers
8. **Future Directions** — promising next steps
9. **References** — full APA bibliography


