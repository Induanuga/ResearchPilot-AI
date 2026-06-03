"""
Application configuration using Pydantic Settings.
Loads from environment variables and .env file.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from functools import lru_cache
import os


class Settings(BaseSettings):
    """Central application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # App
    app_name: str = Field(default="ResearchPilot AI")
    app_version: str = Field(default="1.0.0")
    debug: bool = Field(default=False)
    log_level: str = Field(default="INFO")

    # Backend
    backend_host: str = Field(default="0.0.0.0")
    backend_port: int = Field(default=8000)

    # Groq LLM
    groq_api_key: str = Field(default="")
    groq_model: str = Field(default="llama-3.3-70b-versatile")
    groq_temperature: float = Field(default=0.3)
    groq_max_tokens: int = Field(default=8192)

    # Database
    database_url: str = Field(default="sqlite:///./data/research_pilot.db")

    # Paths
    vectorstore_path: str = Field(default="./data/vectorstore")
    reports_path: str = Field(default="./data/reports")
    papers_path: str = Field(default="./data/papers")
    cache_path: str = Field(default="./data/cache")

    # Embedding
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")

    # ArXiv
    arxiv_max_results: int = Field(default=20)
    arxiv_timeout: int = Field(default=30)

    # Cache
    cache_ttl: int = Field(default=3600)

    def ensure_directories(self) -> None:
        """Create required directories if they don't exist."""
        dirs = [
            self.vectorstore_path,
            self.reports_path,
            self.papers_path,
            self.cache_path,
            "./data",
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)


@lru_cache()
def get_settings() -> Settings:
    """Return cached settings instance."""
    settings = Settings()
    settings.ensure_directories()
    return settings
