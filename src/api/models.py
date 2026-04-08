from __future__ import annotations
"""
Pydantic request / response schemas for the autism-search API.
Column names match the actual crawled_items table (verified 2026-04-08).
"""

from datetime import datetime
from typing import Any
from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    # --- identity ---
    id: int
    external_id: str | None = None
    source: str
    surface_key: str

    # --- content ---
    title: str
    url: str
    description: str | None = None
    content_body: str | None = None

    # --- authorship ---
    author: str | None = None
    authors_json: list[Any] | None = None   # [{family, given, orcid}, ...]
    journal: str | None = None
    open_access: bool | None = None
    doi: str | None = None

    # --- timestamps ---
    published_at: datetime | None = None
    collected_at: datetime

    # --- metadata ---
    lang: str | None = None
    engagement: dict[str, Any] | None = None  # {comments, upvotes, shares, ...}

    # --- scores (added by search service, not in DB) ---
    semantic_score: float = Field(
        0.0, description="Normalised cosine similarity 0–1; 0 if not in semantic results"
    )
    keyword_score: float = Field(
        0.0, description="Normalised ts_rank 0–1; 0 if not in keyword results"
    )
    combined_score: float = Field(
        0.0, description="Final rerank score used for ordering"
    )


class SearchResponse(BaseModel):
    results: list[SearchResult]
    total: int = Field(description="Number of results returned (≤ limit)")
    search_mode: str = Field(description='"hybrid" | "keyword_only"')
    search_time_ms: int
    summary: str | None = Field(
        None, description="LLM-generated summary of top results; null if unavailable"
    )
    llm_time_ms: int | None = Field(
        None, description="Time taken to generate summary in ms; null if unavailable"
    )


class StatsResponse(BaseModel):
    total_items: int
    embedded_items: int
    items_by_source: dict[str, int]
    last_collected_at: datetime | None = None
    last_embedded_at: datetime | None = None


class HealthResponse(BaseModel):
    status: str   # "ok" | "error"
    db: str       # "connected" | "unreachable"
