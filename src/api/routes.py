from __future__ import annotations
"""
FastAPI route handlers for autism-search.

Endpoints:
  GET /api/search   — hybrid / keyword-only search
  GET /api/stats    — crawled_items statistics
  GET /api/health   — liveness + DB connectivity check
"""

import logging
import time
from fastapi import APIRouter, Depends, HTTPException, Query

from ..config import settings
from ..db import get_pool
from ..embedder import embed_query
from ..search.keyword import keyword_search
from ..search.semantic import semantic_search
from ..search.hybrid import merge_and_rerank
from ..llm.summarize import summarize
from .models import SearchResponse, SearchResult, StatsResponse, HealthResponse

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

    # Step 1: try to embed the query (may return None → keyword-only mode)
    embedding = await embed_query(q)

    # Step 2: semantic search (skipped if no embedding)
    sem_results: list[dict] = []
    if embedding is not None:
        sem_results = await semantic_search(pool, embedding, fetch_limit, source, days)

    # Step 3: keyword search (always runs)
    kw_results = await keyword_search(pool, q, fetch_limit, source, days)

    if not sem_results and not kw_results:
        elapsed_ms = int((time.monotonic() - t0) * 1000)
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

    elapsed_ms = int((time.monotonic() - t0) * 1000)

    # Step 5: summarize top 5 results via claude -p
    t_llm = time.monotonic()
    summary = await summarize(q, merged[:5])
    llm_ms = int((time.monotonic() - t_llm) * 1000) if summary is not None else None

    log.info(
        "search q=%r mode=%s results=%d time=%dms llm=%s",
        q, mode, len(merged), elapsed_ms,
        f"{llm_ms}ms" if llm_ms is not None else "unavailable",
    )

    return SearchResponse(
        results=[SearchResult(**item) for item in merged],
        total=len(merged),
        search_mode=mode,
        search_time_ms=elapsed_ms,
        summary=summary,
        llm_time_ms=llm_ms,
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
