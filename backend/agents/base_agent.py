"""
Base agent class providing shared LLM access, logging, and timing.
All specialized agents inherit from this class.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from langchain_groq import ChatGroq
from loguru import logger

from backend.config import get_settings
from backend.models import AgentLog, AgentStatus, ResearchState


class BaseAgent(ABC):
    """Abstract base for all ResearchPilot agents."""

    name: str = "BaseAgent"

    def __init__(self) -> None:
        self.settings = get_settings()
        self._llm: Optional[ChatGroq] = None
        self.total_tokens: int = 0

    @property
    def llm(self) -> ChatGroq:
        """Lazy-load the Groq LLM client."""
        if self._llm is None:
            self._llm = ChatGroq(
                model=self.settings.groq_model,
                groq_api_key=self.settings.groq_api_key,
                temperature=self.settings.groq_temperature,
                max_tokens=self.settings.groq_max_tokens,
            )
        return self._llm

    def _log_start(self, state: ResearchState, message: str = "") -> float:
        """Record agent start and return start timestamp."""
        msg = message or f"{self.name} started"
        logger.info(f"[{self.name}] {msg}")
        log = AgentLog(agent_name=self.name, status=AgentStatus.RUNNING, message=msg)
        state.agent_logs.append(log)
        return time.time()

    def _log_success(
        self,
        state: ResearchState,
        start_time: float,
        message: str = "",
        token_usage: Optional[Dict[str, int]] = None,
    ) -> float:
        """Record successful agent completion. Returns duration in seconds."""
        duration = time.time() - start_time
        msg = message or f"{self.name} completed in {duration:.2f}s"
        logger.info(f"[{self.name}] {msg}")
        log = AgentLog(
            agent_name=self.name,
            status=AgentStatus.COMPLETED,
            message=msg,
            duration_seconds=duration,
            token_usage=token_usage,
        )
        state.agent_logs.append(log)

        # Update metrics
        if state.metrics:
            state.metrics.agent_durations[self.name] = duration

        return duration

    def _log_failure(
        self,
        state: ResearchState,
        start_time: float,
        error: Exception,
    ) -> None:
        """Record agent failure."""
        duration = time.time() - start_time
        msg = f"{self.name} failed after {duration:.2f}s: {str(error)}"
        logger.error(f"[{self.name}] {msg}")
        log = AgentLog(
            agent_name=self.name,
            status=AgentStatus.FAILED,
            message=msg,
            duration_seconds=duration,
        )
        state.agent_logs.append(log)
        state.errors.append(msg)

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM and return the text response."""
        from langchain_core.messages import HumanMessage

        response = self.llm.invoke([HumanMessage(content=prompt)])
        content = response.content
        # Track token usage if available
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            usage = response.usage_metadata
            self.total_tokens += getattr(usage, "total_tokens", 0)
        return content

    @abstractmethod
    def run(self, state: ResearchState) -> ResearchState:
        """Execute the agent logic and return the updated state."""
        ...
