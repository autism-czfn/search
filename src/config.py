from __future__ import annotations
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    database_url: str
    default_result_limit: int = 10
    max_result_limit: int = 50
    log_level: str = "INFO"
    ncbi_api_key: str | None = None

    # User data DB (mzhu_test_ prefix, written by collect, read-only from search)
    user_database_url: str | None = None

    # Base URL of the collect service — used to persist weekly summaries
    collect_base_url: str = "https://localhost:18001"

    # Translation API for multilingual search (P7)
    translation_api: str = "google"  # "google" | "deepl"
    translation_api_key: str | None = None

    # Search routing & fallback (P9)
    live_search_enabled: bool = True
    live_search_min_local_results: int = 3
    live_search_min_local_score: float = 0.4
    live_search_max_sources: int = 23
    live_search_timeout_sec: float = 5.0

    # Agent subprocess timeout (P-SRC-8)
    agent_timeout_seconds: int = 60

    # Redis for safety state persistence (P-SRC-6b / SAFETY_EXPANDED_MODE)
    # If unavailable: safety flag is disabled, HYBRID mode is used instead.
    redis_url: str | None = "redis://localhost:6379/0"


settings = Settings()
