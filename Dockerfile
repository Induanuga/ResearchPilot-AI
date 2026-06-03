# ── ResearchPilot AI — Docker Image ──────────────────────────────────────────
# Multi-stage build: deps → runtime

FROM python:3.11-slim AS base

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Dependency stage ──────────────────────────────────────────────────────────
FROM base AS deps

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM deps AS runtime

# Copy application code
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY .env.example .env.example

# Create data directories
RUN mkdir -p data/vectorstore data/reports data/papers data/cache

# Create non-root user
RUN useradd -m -u 1001 appuser && chown -R appuser:appuser /app
USER appuser

# Expose ports
EXPOSE 8000 8501

# Default: run backend (override in docker-compose for frontend)
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
