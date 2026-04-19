from __future__ import annotations
"""
Pydantic request / response schemas for the autism-search API.
Column names match the actual crawled_items table (verified 2026-04-08).
"""

from datetime import datetime
from typing import Any
from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    model_config = {"extra": "ignore"}  # tolerate extra fields from DB rows / cache

    # --- identity (id also serves as chunk_id for traceability) ---
    id: int
    external_id: str | None = None
    source: str
    surface_key: str
    source_id: str | None = Field(
        None, description="Registry source_id mapped from surface_key (P3 traceability)"
    )

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

    # --- source registry metadata (added by search service via hybrid.py) ---
    # API CONTRACT (P-SRC-1): source_name is canonical (mapped from organization_name in DB)
    source_name: str | None = Field(
        None, description="Human-readable source org name from registry (canonical API field)"
    )
    # chunk_id: required by P1.2 evidence panel — aliased to id for local results
    chunk_id: int | None = Field(
        None, description="chunk_id for GET /api/evidence/{chunk_id}; null for live results"
    )
    authority_tier: int | None = Field(
        None, description="1 (official), 2 (academic), 3 (nonprofit), null (unknown)"
    )
    audience_type: str | None = Field(
        None, description='"parent_facing" | "clinician_facing" | "mixed" | null'
    )
    is_live_result: bool = Field(
        False, description="True if this result came from live search (P7), not local DB"
    )

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
    agent_iterations: int | None = Field(
        None,
        description=(
            "1 if the enhanced claude -p agent ran successfully (internal loop "
            "count is opaque and not exposed). "
            "null if the enhanced agent was not used (fallback to single-shot "
            "summarize())."
        ),
    )
    confidence_level: str | None = Field(
        None,
        description=(
            '"strong_pattern" | "possible_pattern" | "insufficient_data" | null. '
            "Set by /api/insights pattern endpoint only; always null on /api/search."
        ),
    )
    safety_flag: bool = Field(
        False, description="True if a safety concern was detected in the query."
    )
    live_fallback_triggered: bool = Field(
        False, description="True if fallback retrieval was triggered due to insufficient local results."
    )
    intent_type: str | None = Field(
        None,
        description="Classified intent: BEHAVIORAL | MEDICAL | SAFETY | GENERAL",
    )
    safety_level: str | None = Field(
        None,
        description="Safety level: LOW | MEDIUM | HIGH",
    )


class CacheStats(BaseModel):
    cache_size: int = 0
    oldest_entry: datetime | None = None


class StatsResponse(BaseModel):
    total_items: int
    embedded_items: int
    items_by_source: dict[str, int]
    last_collected_at: datetime | None = None
    last_embedded_at: datetime | None = None
    evidence_cache: CacheStats | None = None


class HealthResponse(BaseModel):
    status: str   # "ok" | "error"
    db: str       # "connected" | "unreachable"


# ── Analytics response models ─────────────────────────────────────────────────

class TriggerCount(BaseModel):
    trigger: str
    count: int
    pct: float
    is_safety: bool = False
    raw_signals: list[str] = Field(default_factory=list)


class OutcomeCount(BaseModel):
    outcome: str
    count: int
    pct: float


class PatternEntry(BaseModel):
    trigger: str
    outcome: str
    co_occurrence_count: int
    co_occurrence_pct: float
    total_trigger_events: int
    confidence_level: str
    sample_count: int


class InterventionEffectiveness(BaseModel):
    intervention_id: str
    suggestion_text: str
    started_at: str
    meltdown_rate_before: float
    meltdown_rate_after: float
    delta: float
    observation_days_before: int
    observation_days_after: int
    confidence_level: str


class DailyCheckAverages(BaseModel):
    sleep_quality: float | None = None
    mood: float | None = None
    sensory_sensitivity: float | None = None
    appetite: float | None = None
    social_tolerance: float | None = None
    meltdown_count: float | None = None
    routine_adherence: float | None = None
    communication_ease: float | None = None
    physical_activity: float | None = None
    caregiver_rating: float | None = None


class LowSleepCorrelation(BaseModel):
    low_sleep_days: int
    meltdown_log_avg_low_sleep: float | None = None
    meltdown_log_avg_other_days: float | None = None
    delta: float | None = None


class DailyCheckTrends(BaseModel):
    coverage_days: int
    total_days: int
    averages: DailyCheckAverages
    low_sleep_meltdown_correlation: LowSleepCorrelation


class WeeklyCheckTrend(BaseModel):
    week_start: str
    sleep_quality_avg: float | None = None
    mood_avg: float | None = None
    caregiver_rating_avg: float | None = None


class DailyCheckSummary(BaseModel):
    coverage_days: int
    total_days: int
    averages: DailyCheckAverages
    weekly_trends: list[WeeklyCheckTrend]


class InsightsResponse(BaseModel):
    top_triggers: list[TriggerCount]
    top_outcomes: list[OutcomeCount]
    patterns: list[PatternEntry]
    intervention_effectiveness: list[InterventionEffectiveness]
    log_count: int
    date_range: dict[str, Any]
    daily_check_trends: DailyCheckTrends
    generated_at: str = ""
    cached: bool = False


class WeeklySummaryResponse(BaseModel):
    week_start: str
    week_end: str
    stats: dict[str, Any]
    summary_text: str
    generated_at: str
    cached: bool


class EventFrequency(BaseModel):
    total: int
    per_week: list[dict[str, Any]]


class ClinicianReportResponse(BaseModel):
    date_range: dict[str, Any]
    event_frequency: EventFrequency
    top_triggers: list[TriggerCount]
    top_outcomes: list[OutcomeCount]
    patterns: list[PatternEntry]
    intervention_outcomes: list[InterventionEffectiveness]
    daily_check_summary: DailyCheckSummary | None = None
    key_concerns_text: str | None
    generated_at: str
    cached: bool = False


# ── Source registry response models (P1) ─────────────────────────────────────

class SourceListItem(BaseModel):
    source_id: str
    source_name: str = Field(description="Human-readable source name (canonical API field)")
    authority_tier: int | None = None
    source_type: str
    audience_type: str
    publication_type: str | None = None
    language: str
    country: str | None = None
    domain: str
    is_active: bool
    access_mode: str = Field(description="How content is accessed: 'crawl' | 'api' | 'live_search'")


class SourceListResponse(BaseModel):
    sources: list[SourceListItem]
    total: int


# ── Evidence traceability response (P3) ──────────────────────────────────────

class EvidenceResponse(BaseModel):
    chunk_id: int
    source_id: str | None = None
    source_domain: str | None = None
    source_name: str | None = Field(None, description="Human-readable source name (canonical API field)")
    authority_tier: int | None = None
    audience_type: str | None = None
    page_title: str
    page_url: str
    full_text: str | None = None
    snippet: str
    published_at: datetime | None = None
    collected_at: datetime


# ── Curated evidence models (P8) ─────────────────────────────────────────────

class EvidenceCard(BaseModel):
    source_title: str
    source_name: str = Field(description="Human-readable source name (canonical API field)")
    publication_type: str | None = None
    link: str | None = None
    summary: str
    confidence_tag: str | None = None  # "high" | "medium" | None
    is_live_result: bool = False


class PatternEvidenceResponse(BaseModel):
    trigger: str
    outcome: str | None = None
    evidence: list[EvidenceCard]
    generated_at: str


class InsightRecommendation(BaseModel):
    text: str


class PatternWithEvidence(BaseModel):
    trigger: str
    outcome: str
    co_occurrence_count: int
    co_occurrence_pct: float
    total_trigger_events: int
    confidence_level: str
    sample_count: int
    raw_signals: list[str] = Field(default_factory=list)
    evidence: list[EvidenceCard]
    recommendations: list[InsightRecommendation]
    live_sites_searched: int = Field(0, description="Number of websites queried during live search for this pattern")


class InsightWithEvidenceResponse(BaseModel):
    top_triggers: list[TriggerCount]
    top_outcomes: list[OutcomeCount]
    patterns: list[PatternWithEvidence]
    intervention_effectiveness: list[InterventionEffectiveness]
    log_count: int
    date_range: dict[str, Any]
    daily_check_trends: DailyCheckTrends
    generated_at: str = ""
    cached: bool = False


# ── Webhook models (Collect P3.1) ──────────────────────────────────────────

class TriggerEventPayload(BaseModel):
    event_type: str = Field(description='"safety_alert" | "high_severity" | "new_trigger"')
    child_id: str = "default"
    trigger: str
    severity: int | None = None
    tags: list[str] = []
    logged_at: str | None = None


class TriggerEventResponse(BaseModel):
    status: str
    event_type: str
    search_triggered: bool
    results_cached: int = 0


# ── Safety webhook models (P-SRC-6b — canonical endpoint) ─────────────────

class SafetyWebhookPayload(BaseModel):
    """Canonical safety webhook payload from collect service (P-SRC-6b)."""
    event_id: str = Field(description="Unique event ID (UUID from collect)")
    child_id: str = Field(description="Child identifier")
    trigger_type: str = Field(description='"self_harm" | "violence" | "abuse" | "elopement" | "aggression" | "emergency"')
    severity: int | None = Field(None, description="Integer severity 1-5")
    raw_text: str | None = Field(None, description="Original caregiver language preserved")
    normalized_intent: str | None = Field(None, description="Collect's classified intent label")
    timestamp: str | None = Field(None, description="ISO 8601 timestamp")
    source: str = Field("collect", description='Always "collect"')


class SafetyWebhookResponse(BaseModel):
    status: str
    event_id: str
    child_id: str
    trigger_type: str
    safety_flag_set: bool
    results_cached: int = 0


# ── Safety search response extras ─────────────────────────────────────────

class SafetyExtras(BaseModel):
    """Extra fields returned in SAFETY_EXPANDED_MODE for self_harm intent."""
    cross_source_consensus: str | None = Field(
        None, description="LLM-generated summary across sources; null if LLM failed"
    )
    cross_source_consensus_fallback: str | None = Field(
        None, description='"Insufficient model response" if LLM failed'
    )
    key_clinical_guidance: str | None = Field(
        None, description="Bullet points of actionable guidance; null if LLM failed"
    )
    key_clinical_guidance_fallback: str | None = Field(
        None, description='"Insufficient model response" if LLM failed'
    )
    urgent_help_section: str = Field(
        description="When to seek urgent help — ALWAYS present (hardcoded fallback if LLM fails)"
    )


class SafetySearchResponse(BaseModel):
    """Response model for SAFETY_EXPANDED_MODE searches."""
    results: list[SearchResult]
    total: int
    search_mode: str = "safety_expanded"
    search_time_ms: int
    summary: str | None = None
    fallback_message: str | None = None
    llm_time_ms: int | None = None
    agent_iterations: int | None = None
    # Safety fields — always present in safety mode
    safety_flag: bool = True
    safety_incomplete: bool = Field(
        False, description="True if diversity constraints could not be met after Phase 1+2"
    )
    intent_type: str | None = None
    safety_level: str | None = None
    live_fallback_triggered: bool = True
    # Extras — present for self_harm intent
    extras: SafetyExtras | None = None
