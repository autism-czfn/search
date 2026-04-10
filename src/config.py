from __future__ import annotations
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    database_url: str
    default_result_limit: int = 10
    max_result_limit: int = 50
    log_level: str = "INFO"
    ncbi_api_key: str | None = None


settings = Settings()
