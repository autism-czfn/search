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


settings = Settings()
