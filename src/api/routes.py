from __future__ import annotations
"""
FastAPI route handlers for autism-search.

Endpoints:
  GET /api/search        — hybrid / keyword-only search (blocking)
  GET /api/search/stream — same search via Server-Sent Events (streaming)
  GET /api/stats         — crawled_items statistics
  GET /api/health        — liveness + DB connectivity check
"""

import logging
import time
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse

from ..config import settings
from ..db import get_pool
from ..embedder import embed_query
from ..search.keyword import keyword_search
from ..search.semantic import semantic_search
from ..search.hybrid import merge_and_rerank
from ..llm.agent import run_agent
from ..llm.summarize import summarize
from .models import SearchResponse, SearchResult, StatsResponse, HealthResponse
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
    pool=Depends(get_pool),
):
    effective_limit = min(
        limit if limit is not None else settings.default_result_limit,
        settings.max_result_limit,
    )
    fetch_limit = effective_limit * 2   # fetch more before reranking

    t0 = time.monotonic()
    log.info("search START q=%r limit=%s source=%s days=%s", q, effective_limit, source, days)

    # Step 1: try to embed the query (may return None → keyword-only mode)
    embedding = await embed_query(q)
    t1 = time.monotonic()
    log.info("search EMBED done=%s elapsed=%dms", embedding is not None, int((t1 - t0) * 1000))

    # Step 2: semantic search (skipped if no embedding)
    sem_results: list[dict] = []
    if embedding is not None:
        sem_results = await semantic_search(pool, embedding, fetch_limit, source, days)
    t2 = time.monotonic()
    log.info("search SEMANTIC rows=%d elapsed=%dms", len(sem_results), int((t2 - t1) * 1000))

    # Step 3: keyword search (always runs)
    kw_results = await keyword_search(pool, q, fetch_limit, source, days)
    t3 = time.monotonic()
    log.info("search KEYWORD rows=%d elapsed=%dms", len(kw_results), int((t3 - t2) * 1000))

    if not sem_results and not kw_results:
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        log.info("search DONE no results total=%dms", elapsed_ms)
        return SearchResponse(
            results=[],
            total=0,
            search_mode="keyword_only",
            search_time_ms=elapsed_ms,
            summary=None,
            llm_time_ms=None,
        )

    # Step 4: merge + rerank
    merged, mode = merge_and_rerank(sem_results, kw_results, top_n=effective_limit)
    t4 = time.monotonic()
    log.info("search MERGE mode=%s results=%d elapsed=%dms", mode, len(merged), int((t4 - t3) * 1000))

    elapsed_ms = int((t4 - t0) * 1000)

    # Step 5: run enhanced agent (claude -p with Read + Bash tools)
    t_llm = time.monotonic()
    agent_iterations: int | None = None
    summary: str | None = None

    agent_summary, agent_iters = await run_agent(
        query=q,
        initial_results=merged[:5],
        pool=pool,
        fetch_limit=fetch_limit,
    )
    llm_ms = int((time.monotonic() - t_llm) * 1000)

    if agent_summary is not None:
        summary = agent_summary
        agent_iterations = agent_iters
        log.info("search LLM agent OK elapsed=%dms", llm_ms)
    else:
        # Fallback to single-shot claude -p summarize()
        log.info("search LLM agent failed/skipped — falling back to summarize()")
        summary = await summarize(q, merged[:5])
        llm_ms = int((time.monotonic() - t_llm) * 1000)
        log.info("search LLM summarize ok=%s elapsed=%dms", summary is not None, llm_ms)

    log.info(
        "search COMPLETE q=%r mode=%s results=%d search=%dms llm=%dms",
        q, mode, len(merged), elapsed_ms, llm_ms,
    )

    return SearchResponse(
        results=[SearchResult(**item) for item in merged],
        total=len(merged),
        search_mode=mode,
        search_time_ms=elapsed_ms,
        summary=summary,
        llm_time_ms=llm_ms if summary is not None else None,
        agent_iterations=agent_iterations,
    )


# ── /api/search/stream ───────────────────────────────────────────────────────

@router.get("/search/stream")
async def search_stream(
    q: str = Query(..., min_length=1, description="Search query text"),
    limit: int | None = Query(default=None, ge=1, description="Max results to return"),
    source: str | None = Query(default=None, description="Filter by source (e.g. reddit, pubmed)"),
    days: int | None = Query(default=None, ge=1, description="Only items published within N days"),
    pool=Depends(get_pool),
):
    """
    Streaming variant of /api/search — returns Server-Sent Events.

    Event sequence:
      stage (×4-5) → results → agent_activity (×0-N) → summary → done

    The `results` event arrives before the LLM runs, so the browser can
    render source cards immediately while the agent composes the answer.
    Falls back through two tiers: streaming agent → single-shot summarize(),
    so the UI always receives a summary event unless the database itself
    is unavailable.
    """
    effective_limit = min(
        limit if limit is not None else settings.default_result_limit,
        settings.max_result_limit,
    )
    log.info(
        "search_stream START q=%r limit=%s source=%s days=%s",
        q, effective_limit, source, days,
    )
    return StreamingResponse(
        search_stream_generator(q, effective_limit, source, days, pool),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "Connection":        "keep-alive",
            "X-Accel-Buffering": "no",   # disable nginx / proxy response buffering
        },
    )


# ── /api/stats ───────────────────────────────────────────────────────────────

@router.get("/stats", response_model=StatsResponse)
async def stats(pool=Depends(get_pool)):
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

    return StatsResponse(
        total_items=total,
        embedded_items=embedded,
        items_by_source={r["source"]: r["n"] for r in by_source_rows},
        last_collected_at=last_collected,
        last_embedded_at=last_embedded,
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
