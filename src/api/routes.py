from __future__ import annotations
"""
FastAPI route handlers for autism-search.

Endpoints:
  GET  /api/search             — hybrid / keyword-only search (blocking)
  GET  /api/search/stream      — same search via Server-Sent Events (streaming)
  GET  /api/stats              — crawled_items statistics
  GET  /api/health             — liveness + DB connectivity check
  GET  /api/sources            — list active sources from registry (P1)
  GET  /api/evidence/{id}      — evidence traceability detail (P3)
  GET  /api/insights/evidence  — curated evidence per pattern (P8)
  GET  /api/insights/full      — insights + evidence + recommendations (P8)
  POST /api/safety-webhook     — safety webhook receiver (Collect P3.1 / P-SRC-6b)
"""

import json
import logging
import time
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse

from ..config import settings
from ..db import get_pool
from ..user_db import get_user_pool
from ..embedder import embed_query
from ..search.keyword import keyword_search
from ..search.semantic import semantic_search
from ..search.hybrid import merge_and_rerank
from ..llm.agent import run_agent
from ..llm.summarize import summarize
from ..safety import check_safety
from ..search.intent_classifier import classify_intent
from ..search.local_qualifier import qualify_local_results
from ..analytics.patterns import fetch_log_context, fetch_insights
from ..analytics.summary import get_weekly_summary
from ..analytics.clinician import get_clinician_report
from ..sources.registry import get_registry
from ..search.cache import normalize_query, compute_cache_key, extract_trigger_key, check_cache, store_cache
from ..search.trigger_policy import should_search, fetch_and_invalidate_trigger_cache
from ..search.ranking import compute_search_score
from ..search.multilingual import run_multilingual_search
from ..search.live_fallback import determine_route, run_live_search
from ..search.query_transform import transform_query
from ..evidence.search import fetch_curated_evidence, generate_recommendations
from ..safety_state import set_safety_flag, get_safety_flag
from .models import (
    SearchResponse, SearchResult, StatsResponse, HealthResponse,
    InsightsResponse, WeeklySummaryResponse, ClinicianReportResponse,
    SourceListResponse, SourceListItem,
    EvidenceResponse, EvidenceCard,
    PatternEvidenceResponse, InsightWithEvidenceResponse,
    PatternWithEvidence,
    TriggerEventPayload, TriggerEventResponse,
    SafetyWebhookPayload, SafetyWebhookResponse,
    ChatSearchRequest, ChatSearchResult, ChatSearchResponse,
)
from .stream import search_stream_generator

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api")


# ── /api/search ──────────────────────────────────────────────────────────────

@router.get("/search", response_model=SearchResponse)
async def search(
    q: str = Query(..., min_length=1, description="Search query text"),
    limit: int | None = Query(
        default=None,
        ge=1,
        description="Max results to return",
    ),
    source: str | None = Query(default=None, description="Filter by source (e.g. reddit, pubmed)"),
    days: int | None = Query(default=None, ge=1, description="Only items published within N days"),
    audience: str | None = Query(default=None, description="Filter by audience: parent | clinician"),
    lang: str = Query(default="en", description="Preferred language for results"),
    refresh: bool = Query(default=False, description="Bypass evidence cache"),
    child_id: str = Query(default="default", description="Child identifier for Redis safety flag lookup"),
    pool=Depends(get_pool),
    user_pool=Depends(get_user_pool),
):
    from datetime import date as date_type

    effective_limit = min(
        limit if limit is not None else settings.default_result_limit,
        settings.max_result_limit,
    )
    fetch_limit = effective_limit * 2   # fetch more before reranking

    t0 = time.monotonic()
    log.info("search START q=%r limit=%s source=%s days=%s lang=%s audience=%s child_id=%s",
             q, effective_limit, source, days, lang, audience, child_id)

    # ── Intent classification (add_safety.txt [1]) ─────────────────────────
    intent = classify_intent(q)
    log.info("search INTENT type=%s safety=%s conf=%.2f rule=%s",
             intent.intent_type, intent.safety_level, intent.confidence, intent.matched_rule)

    # ── P-SRC-6b: Check Redis safety flag for this child ───────────────────
    redis_safety_flag = await get_safety_flag(child_id)
    if redis_safety_flag:
        log.info("search REDIS_SAFETY_FLAG child_id=%s trigger=%s",
                 child_id, redis_safety_flag.get("trigger"))

    # ── P-SRC-6: Route to SAFETY_EXPANDED_MODE if applicable ───────────────
    # Must check BEFORE cache — safety mode bypasses cache entirely
    from ..search.live_fallback import determine_route as _determine_route
    early_route = _determine_route([], q, intent=intent, safety_flag=redis_safety_flag)
    if early_route == "SAFETY_EXPANDED_MODE":
        log.info("search ROUTING to SAFETY_EXPANDED_MODE (intent=%s redis_flag=%s)",
                 intent.intent_type, bool(redis_safety_flag))
        from ..search.safety_expanded import run_safety_expanded_search
        from .models import SafetySearchResponse, SafetyExtras
        safety_result = await run_safety_expanded_search(
            query=q, intent_type=intent.intent_type, child_id=child_id
        )
        elapsed_ms = int((time.monotonic() - t0) * 1000)

        # Build extras model if present
        extras_model = None
        extras_dict = safety_result.get("extras")
        if extras_dict is not None:
            extras_model = SafetyExtras(
                cross_source_consensus=extras_dict.get("cross_source_consensus"),
                cross_source_consensus_fallback=extras_dict.get("cross_source_consensus_fallback"),
                key_clinical_guidance=extras_dict.get("key_clinical_guidance"),
                key_clinical_guidance_fallback=extras_dict.get("key_clinical_guidance_fallback"),
                urgent_help_section=extras_dict.get("urgent_help_section", "If in immediate danger, call 911. Mental health crisis line: 988 (US)."),
            )

        raw_results = safety_result.get("results", [])
        results = []
        for item in raw_results[:effective_limit]:
            try:
                results.append(SearchResult.model_validate(item))
            except Exception as e:
                log.debug("safety_expanded: skip invalid result: %s", e)

        return SafetySearchResponse(
            results=results,
            total=len(results),
            search_mode="safety_expanded",
            search_time_ms=elapsed_ms,
            safety_flag=True,
            safety_incomplete=safety_result.get("safety_incomplete", False),
            intent_type=intent.intent_type,
            safety_level=intent.safety_level,
            live_fallback_triggered=True,
            extras=extras_model,
        )

    # ── P-SRC-9: Invalidate evidence cache for triggers with new log activity ──
    if user_pool is not None:
        await fetch_and_invalidate_trigger_cache(
            user_pool, settings.collect_base_url, child_id=child_id,
        )

    # ── P5: Trigger policy — decide if we should search or use cache ────────
    run_search = True
    cached_response = None
    trigger_reason = "no_cache"
    if user_pool is not None:
        run_search, trigger_reason, cached_response = await should_search(
            user_pool, q, language=lang, force=refresh, intent=intent,
        )
        log.info("search TRIGGER_POLICY run=%s reason=%s", run_search, trigger_reason)

    if not run_search and cached_response:
        # Return cached results (skip search + LLM)
        cached_results = cached_response.get("retrieval_results", [])
        cached_evidence = cached_response.get("extracted_evidence") or {}
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        # Cached items were serialized via json.dumps(default=str) — use
        # model_validate to handle string dates and missing optional fields
        results = []
        for item in cached_results[:effective_limit]:
            try:
                results.append(SearchResult.model_validate(item))
            except Exception:
                log.debug("Skipping invalid cached result: %s", item.get("id"))
        return SearchResponse(
            results=results,
            total=min(len(cached_results), effective_limit),
            search_mode="cached",
            search_time_ms=elapsed_ms,
            summary=cached_evidence.get("summary"),
            llm_time_ms=None,
            safety_flag=check_safety(q, intent=intent),
            intent_type=intent.intent_type,
            safety_level=intent.safety_level,
        )

    # ── Step 1: embed query ─────────────────────────────────────────────────
    embedding = await embed_query(q)
    t1 = time.monotonic()
    log.info("search EMBED done=%s elapsed=%dms", embedding is not None, int((t1 - t0) * 1000))

    # ── Step 2: semantic search ─────────────────────────────────────────────
    sem_results: list[dict] = []
    if embedding is not None:
        sem_results = await semantic_search(pool, embedding, fetch_limit, source, days,
                                            official_only=True)
    t2 = time.monotonic()
    log.info("search SEMANTIC rows=%d elapsed=%dms", len(sem_results), int((t2 - t1) * 1000))

    # ── Step 3: keyword search ──────────────────────────────────────────────
    kw_results = await keyword_search(pool, q, fetch_limit, source, days, official_only=True)
    t3 = time.monotonic()
    log.info("search KEYWORD rows=%d elapsed=%dms", len(kw_results), int((t3 - t2) * 1000))

    # ── P7: multilingual search (translate query, search local DB) ───────
    live_results: list[dict] = []
    if lang == "all":
        live_results = await run_multilingual_search(q, pool)
        log.info("search MULTILINGUAL rows=%d lang=all", len(live_results))
    elif lang and lang != "en":
        live_results = await run_multilingual_search(q, pool, target_langs=[lang])
        log.info("search MULTILINGUAL rows=%d lang=%s", len(live_results), lang)

    if not sem_results and not kw_results and not live_results:
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        log.info("search DONE no results total=%dms", elapsed_ms)
        return SearchResponse(
            results=[],
            total=0,
            search_mode="keyword_only",
            search_time_ms=elapsed_ms,
            summary=None,
            llm_time_ms=None,
            safety_flag=check_safety(q, intent=intent),
            intent_type=intent.intent_type,
            safety_level=intent.safety_level,
        )

    # ── Step 4: initial merge + 5-factor rerank (P4) ──────────────────────────
    cross_lingual = bool(live_results)
    target_lang = lang if lang != "en" else None
    all_kw = kw_results + live_results
    merged, mode = merge_and_rerank(
        sem_results, all_kw, top_n=effective_limit * 2,
        query_text=q, user_lang=lang,
        cross_lingual=cross_lingual, target_lang=target_lang,
        official_only=True,
    )
    t4 = time.monotonic()
    log.info("search MERGE mode=%s results=%d elapsed=%dms", mode, len(merged), int((t4 - t3) * 1000))

    # ── P9: routing engine — decide if live search is needed ────────────────
    # Note: SAFETY_EXPANDED_MODE was already handled above and branched out early.
    # This call only handles HYBRID / LIVE_ONLY / LOCAL_ONLY for normal queries.
    route = determine_route(merged, q, intent=intent)
    log.info("search ROUTE=%s local=%d intent=%s/%s", route, len(merged),
             intent.intent_type, intent.safety_level)

    fallback_results: list[dict] = []
    if route in ("HYBRID", "LIVE_ONLY"):
        fallback_results, _ = await run_live_search(q, merged)
        log.info("search LIVE rows=%d", len(fallback_results))
        # Re-merge with live search results (re-applies 5-factor ranking)
        all_kw = kw_results + live_results + fallback_results
        merged, mode = merge_and_rerank(
            sem_results, all_kw, top_n=effective_limit * 2,
            query_text=q, user_lang=lang,
            cross_lingual=cross_lingual, target_lang=target_lang,
            official_only=True,
        )

    # ── P6: audience filtering ──────────────────────────────────────────────
    if audience:
        audience_key = "parent_facing" if audience == "parent" else "clinician_facing"
        filtered = [
            item for item in merged
            if item.get("audience_type") in (audience_key, "mixed", None)
        ]
        if filtered:
            merged = filtered
        else:
            log.warning("Audience filter '%s' removed all results — returning unfiltered", audience)
    merged = merged[:effective_limit]

    elapsed_ms = int((t4 - t0) * 1000)

    # ── Step 5b: safety detection (uses intent classifier) ─────────────────
    safety_flag = check_safety(q, intent=intent)
    log.info("search SAFETY flag=%s q=%r", safety_flag, q)

    # ── Local qualification gate (add_safety.txt [4]) ──────────────────────
    # For safety=HIGH, filter out local results that don't meet quality bar.
    # Live results are always kept. Non-safety queries bypass the gate.
    if safety_flag and fallback_results:
        local_items = [r for r in merged if not r.get("is_live_result")]
        qual = qualify_local_results(local_items, safety_level=intent.safety_level)
        log.info("search LOCAL_QUAL include=%s reason=%s rel=%.2f qual=%.2f rec=%.2f cov=%d",
                 qual.include_local, qual.reason, qual.best_relevance,
                 qual.best_quality, qual.best_recency, qual.relevant_count)
        if not qual.include_local:
            # Remove local results, keep only live results
            merged = [r for r in merged if r.get("is_live_result")]
            log.info("search LOCAL_QUAL: excluded local results, keeping %d live results", len(merged))

    # ── Step 5c: personalization context ────────────────────────────────────
    log_context: str | None = None
    if user_pool is not None:
        try:
            log_context = await fetch_log_context(user_pool)
        except Exception as e:
            log.warning("search: failed to fetch log context: %s", e)

    # ── Step 5: run enhanced agent ──────────────────────────────────────────
    t_llm = time.monotonic()
    agent_iterations: int | None = None
    summary_text: str | None = None

    agent_summary, agent_iters = await run_agent(
        query=q,
        initial_results=merged[:5],
        pool=pool,
        fetch_limit=fetch_limit,
        log_context=log_context,
    )
    llm_ms = int((time.monotonic() - t_llm) * 1000)

    if agent_summary is not None:
        summary_text = agent_summary
        agent_iterations = agent_iters
        log.info("search LLM agent OK elapsed=%dms", llm_ms)
    else:
        log.info("search LLM agent failed/skipped — falling back to summarize()")
        summary_text = await summarize(q, merged[:5], log_context)
        llm_ms = int((time.monotonic() - t_llm) * 1000)
        log.info("search LLM summarize ok=%s elapsed=%dms", summary_text is not None, llm_ms)

    # ── P2: store in cache (user DB) ──────────────────────────────────────
    if user_pool is not None:
        normalized = normalize_query(q)
        cache_key = compute_cache_key(normalized, lang, date_type.today())
        trigger_key = extract_trigger_key(q)
        await store_cache(
            user_pool, cache_key, trigger_key, lang, date_type.today(), q,
            retrieval_results=merged[:effective_limit],
            extracted_evidence={"summary": summary_text} if summary_text else None,
            ttl_hours=24,
        )

    log.info(
        "search COMPLETE q=%r mode=%s results=%d search=%dms llm=%dms safety=%s cached=new",
        q, mode, len(merged), elapsed_ms, llm_ms, safety_flag,
    )

    return SearchResponse(
        results=[SearchResult.model_validate(item) for item in merged],
        total=len(merged),
        search_mode=mode,
        search_time_ms=elapsed_ms,
        summary=summary_text,
        llm_time_ms=llm_ms if summary_text is not None else None,
        agent_iterations=agent_iterations,
        safety_flag=safety_flag,
        live_fallback_triggered=bool(fallback_results),
        intent_type=intent.intent_type,
        safety_level=intent.safety_level,
    )


# ── /api/search/stream ───────────────────────────────────────────────────────

@router.get("/search/stream")
async def search_stream(
    q: str = Query(..., min_length=1, description="Search query text"),
    limit: int | None = Query(default=None, ge=1, description="Max results to return"),
    source: str | None = Query(default=None, description="Filter by source (e.g. reddit, pubmed)"),
    days: int | None = Query(default=None, ge=1, description="Only items published within N days"),
    audience: str | None = Query(default=None, description="Filter by audience: parent | clinician"),
    lang: str = Query(default="en", description="Preferred language for results"),
    child_id: str = Query(default="default", description="Child identifier for Redis safety flag lookup"),
    pool=Depends(get_pool),
    user_pool=Depends(get_user_pool),
):
    """
    Streaming variant of /api/search — returns Server-Sent Events.

    Normal event sequence:
      metadata × 1     (safety_flag — emitted immediately before any LLM content)
      stage    × 4-5   (embedding → semantic → keyword → merge → agent)
      results  × 1     (source cards — sent BEFORE the LLM runs)
      agent_activity × 0-N
      summary  × 0-1
      done     × 1

    SAFETY_EXPANDED_MODE event sequence (P-SRC-6):
      metadata × 1     (safety_flag: true)
      stage    × 1-2   (safety_fan_out → safety_merge)
      results  × 1     (source cards)
      safety_extras × 0-1 (cross_source_consensus, key_clinical_guidance, urgent_help_section)
      done     × 1

    Falls back through two tiers: streaming agent → single-shot summarize(),
    so the UI always receives a summary event unless the database itself
    is unavailable.
    """
    effective_limit = min(
        limit if limit is not None else settings.default_result_limit,
        settings.max_result_limit,
    )
    log.info(
        "search_stream START q=%r limit=%s source=%s days=%s child_id=%s",
        q, effective_limit, source, days, child_id,
    )
    return StreamingResponse(
        search_stream_generator(q, effective_limit, source, days, pool, user_pool,
                                child_id=child_id, lang=lang),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "Connection":        "keep-alive",
            "X-Accel-Buffering": "no",   # disable nginx / proxy response buffering
        },
    )


# ── /api/chat-search ─────────────────────────────────────────────────────────

@router.post("/chat-search", response_model=ChatSearchResponse)
async def chat_search(
    body: ChatSearchRequest,
    pool=Depends(get_pool),
):
    """
    POST /api/chat-search — simplified search endpoint for collect's chat pipeline.

    Runs the same hybrid search pipeline as /api/search but:
      - Accepts a JSON body (not query params)
      - Skips LLM summary, cache, and personalization
      - Returns a simplified result schema: {title, url, snippet, source, score}
      - Audience filtering active in Phase 1 (same logic as /api/search)
      - On any unhandled exception, returns results: [] (graceful degradation)
    """
    effective_limit = min(body.limit, settings.max_result_limit)
    fetch_limit = effective_limit * 2

    t0 = time.monotonic()
    log.info("chat_search START q=%r limit=%s audience=%s", body.query, effective_limit, body.audience)

    sites_attempted = 0
    try:
        # Step 0: transform the raw user question into focused search keywords.
        # "how to deal with food pattern?" → "autism food selectivity eating mealtime behavior"
        # Falls back to body.query on any failure (CLI absent, timeout, etc.)
        search_query = await transform_query(body.query)
        if search_query != body.query:
            log.info("chat_search TRANSFORM %r → %r", body.query, search_query)

        # Step 1: intent classification
        intent = classify_intent(search_query)

        # Step 2: embed query
        embedding = await embed_query(search_query)

        # Step 3: semantic search
        sem_results: list[dict] = []
        if embedding is not None:
            sem_results = await semantic_search(pool, embedding, fetch_limit, None, None,
                                                official_only=True)

        # Step 4: keyword search
        kw_results = await keyword_search(pool, search_query, fetch_limit, None, None,
                                          official_only=True)

        # Step 5: merge and rerank
        merged, mode = merge_and_rerank(
            sem_results, kw_results,
            top_n=fetch_limit,
            query_text=search_query,
            official_only=True,
        )

        # Steps 6-7: route decision + live search
        # Always run live search so sites_attempted is always reported in the response.
        # For LOCAL_ONLY the live results are discarded (local DB was sufficient),
        # but we still want to show the user how many sources were checked.
        route = determine_route(merged, search_query, intent=intent)
        log.info("chat_search ROUTE=%s local=%d", route, len(merged))

        fallback_results, sites_attempted = await run_live_search(search_query, merged)
        if route in ("HYBRID", "LIVE_ONLY", "SAFETY_EXPANDED_MODE") and fallback_results:
            # Live results have already been content-verified in site_search.py,
            # so relax official_only here to allow content-verified tier-2 sources
            # (EuropePMC, Semantic Scholar, ClinicalTrials, etc.) to surface when
            # they are the best available answer for the query.
            merged, mode = merge_and_rerank(
                sem_results, kw_results + fallback_results,
                top_n=fetch_limit,
                query_text=search_query,
                official_only=False,
            )

        # Step 8: audience filtering (production-tested logic from /api/search)
        if body.audience:
            audience_key = "parent_facing" if body.audience == "parent" else "clinician_facing"
            filtered = [
                item for item in merged
                if item.get("audience_type") in (audience_key, "mixed", None)
            ]
            if filtered:
                merged = filtered
            else:
                log.warning(
                    "chat_search: audience filter '%s' removed all results — returning unfiltered",
                    body.audience,
                )

        # Step 9: drop zero-score LOCAL DB results only — live results (id < 0)
        # are content-verified at fetch time and always kept so the scoring system
        # can rank them. Zero-score local results are provably off-topic (trigger
        # terms absent from title/description/body).
        scored = [r for r in merged if r.get("combined_score", 0) > 0 or (r.get("id") or 0) < 0]
        if scored:
            merged = scored
        else:
            log.warning("chat_search: all results scored 0 — returning unfiltered for q=%r (transformed=%r)", body.query, search_query)

        # Step 10: truncate to requested limit
        merged = merged[:effective_limit]

    except Exception as exc:
        log.warning("chat_search FAILED q=%r — returning empty results: %s", body.query, exc)
        return ChatSearchResponse(results=[])

    elapsed_ms = int((time.monotonic() - t0) * 1000)
    log.info("chat_search DONE q=%r mode=%s results=%d elapsed=%dms",
             body.query, mode, len(merged), elapsed_ms)

    # Step 10: build simplified response
    results: list[ChatSearchResult] = []
    for item in merged:
        snippet = item.get("description") or ""
        if not snippet and item.get("content_body"):
            snippet = item["content_body"][:300]
        results.append(ChatSearchResult(
            title=item.get("title", ""),
            url=item.get("url", ""),
            snippet=snippet or None,
            source=item.get("source_name"),
            score=float(item.get("combined_score", 0.0)),
        ))

    return ChatSearchResponse(results=results, sites_attempted=sites_attempted, search_query=search_query)


# ── /api/stats ───────────────────────────────────────────────────────────────

@router.get("/stats", response_model=StatsResponse)
async def stats(pool=Depends(get_pool), user_pool=Depends(get_user_pool)):
    try:
        async with pool.acquire() as conn:
            total = await conn.fetchval("SELECT COUNT(*) FROM crawled_items")
            embedded = await conn.fetchval(
                "SELECT COUNT(*) FROM crawled_items WHERE embedding IS NOT NULL"
            )
            by_source_rows = await conn.fetch(
                "SELECT source, COUNT(*) AS n FROM crawled_items GROUP BY source ORDER BY n DESC"
            )
            last_collected = await conn.fetchval(
                "SELECT MAX(collected_at) FROM crawled_items"
            )
            last_embedded = await conn.fetchval(
                "SELECT MAX(embedded_at) FROM crawled_items WHERE embedded_at IS NOT NULL"
            )
    except Exception as e:
        log.error("Stats DB error: %s", e)
        raise HTTPException(status_code=503, detail={"error": "database unavailable", "detail": str(e)})

    # P2.4: evidence cache stats (user DB, silently skip if unavailable)
    from .models import CacheStats
    cache_stats = None
    if user_pool is not None:
        try:
            async with user_pool.acquire() as conn:
                cache_size = await conn.fetchval(
                    "SELECT COUNT(*) FROM evidence_cache"
                )
                oldest = await conn.fetchval(
                    "SELECT MIN(created_at) FROM evidence_cache"
                )
                cache_stats = CacheStats(cache_size=cache_size or 0, oldest_entry=oldest)
        except Exception as e:
            log.debug("Cache stats unavailable (table may not exist): %s", e)

    return StatsResponse(
        total_items=total,
        embedded_items=embedded,
        items_by_source={r["source"]: r["n"] for r in by_source_rows},
        last_collected_at=last_collected,
        last_embedded_at=last_embedded,
        evidence_cache=cache_stats,
    )


# ── /api/health ───────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse)
async def health(pool=Depends(get_pool)):
    try:
        async with pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        return HealthResponse(status="ok", db="connected")
    except Exception as e:
        log.error("Health check DB error: %s", e)
        raise HTTPException(status_code=503, detail={"status": "error", "db": "unreachable"})


# ── /api/insights ─────────────────────────────────────────────────────────────

@router.get("/insights", response_model=InsightsResponse)
async def insights(
    days: int = Query(default=30, ge=1, description="Lookback window in days"),
    refresh: bool = Query(default=False, description="Bypass cache and recompute"),
    user_pool=Depends(get_user_pool),
):
    """
    Deterministic analytics over mzhu_test_logs.
    No LLM — all values are SQL aggregates.
    Returns top triggers, top outcomes, co-occurrence patterns with
    confidence levels, and intervention effectiveness (meltdown rates
    before/after each adopted intervention).
    Cached for 1 hour; pass ?refresh=true to force recomputation.
    """
    if user_pool is None:
        raise HTTPException(status_code=503, detail="User database not configured (USER_DATABASE_URL missing)")
    try:
        return await fetch_insights(user_pool, days, refresh=refresh)
    except Exception as e:
        log.error("insights error: %s", e, exc_info=True)
        raise HTTPException(status_code=503, detail={"error": "insights query failed", "detail": str(e)})


# ── /api/weekly-summary ───────────────────────────────────────────────────────

@router.get("/weekly-summary", response_model=WeeklySummaryResponse)
async def weekly_summary(user_pool=Depends(get_user_pool)):
    """
    Generates (or returns cached) weekly summary for the current week.
    Cached for 24 hours in mzhu_test_summaries.
    LLM narrates pre-computed stats — does not invent numbers.
    """
    if user_pool is None:
        raise HTTPException(status_code=503, detail="User database not configured (USER_DATABASE_URL missing)")
    try:
        return await get_weekly_summary(user_pool, settings.collect_base_url)
    except Exception as e:
        log.error("weekly-summary error: %s", e, exc_info=True)
        raise HTTPException(status_code=503, detail={"error": "weekly summary failed", "detail": str(e)})


# ── /api/clinician-report ─────────────────────────────────────────────────────

@router.get("/clinician-report", response_model=ClinicianReportResponse)
async def clinician_report(
    days: int = Query(default=90, ge=1, description="Lookback window in days (default 3 months)"),
    refresh: bool = Query(default=False, description="Bypass cache and regenerate report"),
    pool=Depends(get_pool),
    user_pool=Depends(get_user_pool),
):
    """
    Structured report for a medical appointment.
    Deterministic stats + LLM-narrated 'Key concerns' section grounded
    in the stats and top 3 evidence results for the most frequent trigger.
    Cached for 2 hours; pass ?refresh=true to force regeneration.
    """
    if user_pool is None:
        raise HTTPException(status_code=503, detail="User database not configured (USER_DATABASE_URL missing)")
    try:
        return await get_clinician_report(user_pool, pool, days, refresh=refresh)
    except Exception as e:
        log.error("clinician-report error: %s", e, exc_info=True)
        raise HTTPException(status_code=503, detail={"error": "clinician report failed", "detail": str(e)})


# ── /api/sources (P1) ───────────────────────────────────────────────────────

@router.get("/sources", response_model=SourceListResponse)
async def list_sources():
    """List all active sources from the source registry."""
    registry = get_registry()
    active = registry.get_active_sources()
    items = [
        SourceListItem(
            source_id=s.source_id,
            source_name=s.organization_name,
            authority_tier=s.authority_tier,
            source_type=s.source_type,
            audience_type=s.audience_type,
            publication_type=s.publication_type,
            language=s.language,
            country=s.country,
            domain=s.domain,
            is_active=s.is_active,
            access_mode=s.access_mode,
        )
        for s in active
    ]
    return SourceListResponse(sources=items, total=len(items))


# ── /api/evidence/{chunk_id} (P3) ───────────────────────────────────────────

@router.get("/evidence/{chunk_id}", response_model=EvidenceResponse)
async def get_evidence(chunk_id: int, pool=Depends(get_pool)):
    """
    Return full evidence detail for a search result by its chunk_id (= crawled_items.id).

    For live_search results (id == -1), this endpoint cannot serve them —
    the UI should link directly to the original URL instead.
    """
    if chunk_id < 0:
        raise HTTPException(status_code=400, detail="Live search results have no local evidence record. Use the original URL.")

    try:
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM crawled_items WHERE id = $1", chunk_id
            )
    except Exception as e:
        log.error("evidence lookup DB error: %s", e)
        raise HTTPException(status_code=503, detail="database unavailable")

    if row is None:
        raise HTTPException(status_code=404, detail=f"No evidence found for chunk_id={chunk_id}")

    row = dict(row)  # asyncpg.Record → dict (supports .get())
    registry = get_registry()
    source_key = row.get("source") or ""
    entry = registry.get_source_by_key(source_key)

    content = row.get("content_body") or row.get("description") or ""
    snippet = content[:500] if content else ""

    return EvidenceResponse(
        chunk_id=chunk_id,
        source_id=entry.source_id if entry else None,
        source_domain=entry.domain if entry else None,
        source_name=entry.organization_name if entry else None,
        authority_tier=entry.authority_tier if entry else None,
        audience_type=entry.audience_type if entry else None,
        page_title=row.get("title") or "(untitled)",
        page_url=row.get("url") or "",
        full_text=row.get("content_body"),
        snippet=snippet,
        published_at=row.get("published_at"),
        collected_at=row["collected_at"],
    )


# ── /api/insights/evidence (P8) ─────────────────────────────────────────────

@router.get("/insights/evidence", response_model=PatternEvidenceResponse)
async def insights_evidence(
    trigger: str = Query(..., description="Trigger to find evidence for"),
    outcome: str | None = Query(None, description="Optional outcome"),
    limit: int = Query(5, ge=1, le=10),
    pool=Depends(get_pool),
):
    """
    Curated evidence cards for a specific trigger/outcome pattern.
    Called by UI per-pattern in the Insights tab.
    """
    query = f"{trigger} autism"
    if outcome:
        query += f" {outcome}"
    cards, _ = await fetch_curated_evidence(pool, query, limit)
    return PatternEvidenceResponse(
        trigger=trigger,
        outcome=outcome,
        evidence=cards,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


# ── /api/insights/full (P8) ─────────────────────────────────────────────────

async def _get_cached_insights_full(user_pool, days: int) -> dict | None:
    """Return cached full insights if no new log has been submitted since generation."""
    try:
        async with user_pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT response_json, generated_at FROM mzhu_test_insights_full_cache
                WHERE days = $1
                ORDER BY generated_at DESC LIMIT 1
                """,
                days,
            )
            if row is None:
                return None
            last_log = await conn.fetchval(
                "SELECT MAX(logged_at) FROM mzhu_test_logs WHERE NOT voided"
            )
        if last_log is not None and last_log > row["generated_at"]:
            return None
        data = row["response_json"]
        return json.loads(data) if isinstance(data, str) else data
    except Exception as e:
        log.warning("insights_full cache read failed: %s", e)
        return None


async def _persist_insights_full(user_pool, days: int, response: dict) -> None:
    try:
        async with user_pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO mzhu_test_insights_full_cache (days, response_json) VALUES ($1, $2::jsonb)",
                days, json.dumps(response, default=str),
            )
            await conn.execute(
                "DELETE FROM mzhu_test_insights_full_cache WHERE generated_at < now() - interval '24 hours'",
            )
    except Exception as e:
        log.warning("insights_full cache write failed: %s", e)


@router.get("/insights/full", response_model=InsightWithEvidenceResponse)
async def insights_full(
    days: int = Query(default=30, ge=1),
    refresh: bool = Query(default=False),
    pool=Depends(get_pool),
    user_pool=Depends(get_user_pool),
):
    """
    Combined insights + evidence + LLM recommendations in a single call.
    Enriches the top 3 patterns with curated evidence and recommendations.
    Cached for 2 hours (separate from base insights cache).
    """
    if user_pool is None:
        raise HTTPException(status_code=503, detail="User database not configured")

    if not refresh:
        cached = await _get_cached_insights_full(user_pool, days)
        if cached:
            return {**cached, "cached": True}

    try:
        # Fetch base insights
        base = await fetch_insights(user_pool, days, refresh=refresh)

        # Build raw_signals lookup from top_triggers
        from ..search.live_fallback import _SAFETY_TRIGGERS
        top_triggers = base.get("top_triggers", [])
        raw_signals_by_trigger = {}
        for tt in top_triggers:
            trig = tt.get("trigger", "")
            sigs = tt.get("raw_signals") or []
            if sigs:
                raw_signals_by_trigger[trig] = sigs

        # Enrich top 3 patterns + any safety-critical patterns with evidence
        patterns_enriched = []
        patterns = base.get("patterns", [])
        enriched_count = 0

        for i, p in enumerate(patterns):
            trigger = p.get("trigger", "")
            outcome = p.get("outcome", "")

            evidence: list[EvidenceCard] = []
            recommendations = []

            is_safety = trigger in _SAFETY_TRIGGERS or outcome in _SAFETY_TRIGGERS
            should_enrich = enriched_count < 3 or is_safety

            live_sites_searched = 0
            if should_enrich:
                query = f"{trigger} {outcome} autism"
                evidence, live_sites_searched = await fetch_curated_evidence(pool, query, limit=3)
                recommendations = await generate_recommendations(p, evidence)
                enriched_count += 1

            patterns_enriched.append(PatternWithEvidence(
                trigger=trigger,
                outcome=outcome,
                co_occurrence_count=p.get("co_occurrence_count", 0),
                co_occurrence_pct=p.get("co_occurrence_pct", 0.0),
                total_trigger_events=p.get("total_trigger_events", 0),
                confidence_level=p.get("confidence_level", "insufficient_data"),
                sample_count=p.get("sample_count", 0),
                raw_signals=raw_signals_by_trigger.get(trigger, []),
                evidence=evidence,
                recommendations=recommendations,
                live_sites_searched=live_sites_searched,
            ))

        # Surface safety triggers that appear in top_triggers but have
        # no co-occurrence patterns (e.g. aggression logged without an
        # outcome).  These still need live search enrichment.
        pattern_triggers = {p.get("trigger", "") for p in patterns}
        top_triggers = base.get("top_triggers", [])
        for tt in top_triggers:
            trig = tt.get("trigger", "")
            if trig in _SAFETY_TRIGGERS and trig not in pattern_triggers:
                query = f"{trig} autism"
                evidence, live_sites_searched = await fetch_curated_evidence(pool, query, limit=3)
                synthetic_pattern = {
                    "trigger": trig,
                    "outcome": "",
                    "sample_count": tt.get("count", 0),
                    "co_occurrence_pct": 0,
                }
                recommendations = await generate_recommendations(synthetic_pattern, evidence)
                patterns_enriched.append(PatternWithEvidence(
                    trigger=trig,
                    outcome="",
                    co_occurrence_count=0,
                    co_occurrence_pct=0.0,
                    total_trigger_events=tt.get("count", 0),
                    confidence_level="insufficient_data",
                    sample_count=tt.get("count", 0),
                    raw_signals=raw_signals_by_trigger.get(trig, []),
                    evidence=evidence,
                    recommendations=recommendations,
                    live_sites_searched=live_sites_searched,
                ))

        response = InsightWithEvidenceResponse(
            top_triggers=base.get("top_triggers", []),
            top_outcomes=base.get("top_outcomes", []),
            patterns=patterns_enriched,
            intervention_effectiveness=base.get("intervention_effectiveness", []),
            log_count=base.get("log_count", 0),
            date_range=base.get("date_range", {}),
            daily_check_trends=base.get("daily_check_trends", {}),
            generated_at=datetime.now(timezone.utc).isoformat(),
            cached=False,
        )
        await _persist_insights_full(user_pool, days, response.dict())
        return response

    except HTTPException:
        raise
    except Exception as e:
        log.error("insights/full error: %s", e, exc_info=True)
        raise HTTPException(status_code=503, detail={"error": "insights full failed", "detail": str(e)})


# ── /api/safety-webhook (Collect P3.1 / P-SRC-6b) ───────────────────────────

@router.post("/safety-webhook", response_model=SafetyWebhookResponse)
async def receive_safety_webhook(
    payload: SafetyWebhookPayload,
    pool=Depends(get_pool),
    user_pool=Depends(get_user_pool),
):
    """
    Receive safety webhook from the collect service (P-SRC-6b).

    Canonical endpoint: POST /api/safety-webhook
    Called when a safety-critical log is created (e.g. self_harm, severity >= 4).
    Sets the Redis safety flag for the child so that subsequent search requests
    activate SAFETY_EXPANDED_MODE.
    Also runs a proactive search and caches results for instant retrieval.
    """
    from datetime import date as date_type

    log.info("safety_webhook RECEIVED event_id=%s child_id=%s trigger=%s severity=%s",
             payload.event_id, payload.child_id, payload.trigger_type, payload.severity)

    # ── P-SRC-6b: Set Redis safety flag for this child ─────────────────────
    await set_safety_flag(payload.child_id, payload.trigger_type)
    log.info("safety_webhook REDIS_SET child_id=%s trigger=%s", payload.child_id, payload.trigger_type)

    # Build a search query from the trigger
    query = f"{payload.trigger_type.replace('_', ' ')} autism"

    # Classify intent (will detect safety)
    intent = classify_intent(query)
    log.info("safety_webhook INTENT type=%s safety=%s", intent.intent_type, intent.safety_level)

    # Run the search pipeline (same as /api/search but without LLM summary)
    t0 = time.monotonic()

    embedding = await embed_query(query)
    sem_results: list[dict] = []
    if embedding is not None:
        sem_results = await semantic_search(pool, embedding, 20, official_only=True)

    kw_results = await keyword_search(pool, query, 20, official_only=True)

    merged, mode = merge_and_rerank(sem_results, kw_results, top_n=20, query_text=query,
                                    official_only=True)

    # Force live search for safety events
    route = determine_route(merged, query, intent=intent)
    fallback_results: list[dict] = []
    if route in ("HYBRID", "LIVE_ONLY"):
        fallback_results, _ = await run_live_search(query, merged)
        if fallback_results:
            all_kw = kw_results + fallback_results
            merged, mode = merge_and_rerank(sem_results, all_kw, top_n=20, query_text=query,
                                            official_only=True)

    elapsed_ms = int((time.monotonic() - t0) * 1000)
    results_count = len(merged)

    # Cache proactively so next user search is instant
    cached_count = 0
    if user_pool is not None and merged:
        try:
            normalized = normalize_query(query)
            cache_key = compute_cache_key(normalized, "en", date_type.today())
            trigger_key = extract_trigger_key(query)
            await store_cache(
                user_pool, cache_key, trigger_key, "en", date_type.today(), query,
                retrieval_results=merged[:10],
                extracted_evidence=None,
                ttl_hours=24,
            )
            cached_count = min(len(merged), 10)
            log.info("safety_webhook CACHED key=%s results=%d", cache_key[:16], cached_count)
        except Exception as e:
            log.warning("safety_webhook cache failed: %s", e)

    log.info(
        "safety_webhook DONE event_id=%s trigger=%s results=%d live=%d elapsed=%dms cached=%d",
        payload.event_id, payload.trigger_type, results_count,
        len(fallback_results), elapsed_ms, cached_count,
    )

    return SafetyWebhookResponse(
        status="ok",
        event_id=payload.event_id,
        child_id=payload.child_id,
        trigger_type=payload.trigger_type,
        safety_flag_set=True,
        results_cached=cached_count,
    )


# ── /api/webhooks/trigger-event (legacy alias — kept for backward compat) ───

@router.post("/webhooks/trigger-event", response_model=TriggerEventResponse)
async def receive_trigger_event(
    payload: TriggerEventPayload,
    pool=Depends(get_pool),
    user_pool=Depends(get_user_pool),
):
    """
    Legacy endpoint — kept for backward compatibility.
    New code should use POST /api/safety-webhook instead.
    """
    from datetime import date as date_type

    log.info("webhook (legacy) RECEIVED event=%s trigger=%s severity=%s",
             payload.event_type, payload.trigger, payload.severity)

    # Build a search query from the trigger
    query = f"{payload.trigger.replace('_', ' ')} autism"

    # Classify intent (will detect safety)
    intent = classify_intent(query)

    # Run the search pipeline (same as /api/search but without LLM summary)
    t0 = time.monotonic()

    embedding = await embed_query(query)
    sem_results: list[dict] = []
    if embedding is not None:
        sem_results = await semantic_search(pool, embedding, 20, official_only=True)

    kw_results = await keyword_search(pool, query, 20, official_only=True)

    merged, mode = merge_and_rerank(sem_results, kw_results, top_n=20, query_text=query,
                                    official_only=True)

    # Force live search for safety events
    route = determine_route(merged, query, intent=intent)
    fallback_results: list[dict] = []
    if route in ("HYBRID", "LIVE_ONLY"):
        fallback_results, _ = await run_live_search(query, merged)
        if fallback_results:
            all_kw = kw_results + fallback_results
            merged, mode = merge_and_rerank(sem_results, all_kw, top_n=20, query_text=query,
                                            official_only=True)

    elapsed_ms = int((time.monotonic() - t0) * 1000)
    results_count = len(merged)

    # Cache proactively so next user search is instant
    cached_count = 0
    if user_pool is not None and merged:
        try:
            normalized = normalize_query(query)
            cache_key = compute_cache_key(normalized, "en", date_type.today())
            trigger_key = extract_trigger_key(query)
            await store_cache(
                user_pool, cache_key, trigger_key, "en", date_type.today(), query,
                retrieval_results=merged[:10],
                extracted_evidence=None,
                ttl_hours=24,
            )
            cached_count = min(len(merged), 10)
            log.info("webhook CACHED key=%s results=%d", cache_key[:16], cached_count)
        except Exception as e:
            log.warning("webhook cache failed: %s", e)

    log.info("webhook DONE event=%s trigger=%s results=%d live=%d elapsed=%dms cached=%d",
             payload.event_type, payload.trigger, results_count,
             len(fallback_results), elapsed_ms, cached_count)

    return TriggerEventResponse(
        status="ok",
        event_type=payload.event_type,
        search_triggered=True,
        results_cached=cached_count,
    )
