from __future__ import annotations
"""
SSE generator for GET /api/search/stream.

Emits one SSE event per pipeline stage so the browser can update its UI
progressively, without waiting 30-60 s for a single JSON response.

Event sequence:
  stage          × 4-5   (embedding → semantic → keyword → merge → agent)
  results        × 1     (source cards — sent BEFORE the LLM runs)
  agent_activity × 0-N   (one per tool call the agent makes)
  summary        × 0-1   (LLM answer; absent only if DB is down)
  done           × 1     (always the final event)

The generator never raises — it always closes with a `done` event so the
browser can tidy up its EventSource.  Two-tier LLM fallback:
  Tier 1: run_agent_stream() — streaming agent with live activity events
  Tier 2: summarize()        — single-shot claude -p (if streaming fails)
"""

import json
import logging
import time
from typing import AsyncGenerator

from ..embedder import embed_query
from ..search.keyword import keyword_search
from ..search.semantic import semantic_search
from ..search.hybrid import merge_and_rerank
from ..llm.agent_stream import run_agent_stream
from ..llm.summarize import summarize
from .models import SearchResult

log = logging.getLogger(__name__)


# ── SSE wire-format helper ─────────────────────────────────────────────────

def _sse(event: str, payload: dict) -> str:
    """
    Serialise a single SSE message.
    `default=str` ensures datetime / Decimal objects don't raise.
    """
    data = json.dumps(payload, default=str)
    return f"event: {event}\ndata: {data}\n\n"


# ── Main SSE generator ─────────────────────────────────────────────────────

async def search_stream_generator(
    q: str,
    limit: int,
    source: str | None,
    days: int | None,
    pool,
) -> AsyncGenerator[str, None]:
    """
    Full SSE pipeline for /api/search/stream.
    Yields SSE-formatted strings; FastAPI's StreamingResponse writes them
    to the HTTP response as each one is produced.
    Guaranteed to always yield a final `done` event, even on unexpected errors.
    """
    t0 = time.monotonic()

    def elapsed() -> int:
        return int((time.monotonic() - t0) * 1000)

    fetch_limit = limit * 2   # fetch extra before reranking

    try:
        # ── stage: embedding ──────────────────────────────────────────────────
        yield _sse("stage", {
            "stage":      "embedding",
            "message":    "Embedding query…",
            "elapsed_ms": elapsed(),
        })
        embedding = await embed_query(q)

        # ── stage: semantic ───────────────────────────────────────────────────
        yield _sse("stage", {
            "stage":      "semantic",
            "message":    "Running semantic search…" if embedding else "Keyword-only mode (embedding unavailable)",
            "elapsed_ms": elapsed(),
        })
        sem_results: list[dict] = []
        if embedding is not None:
            sem_results = await semantic_search(pool, embedding, fetch_limit, source, days)

        # ── stage: keyword ────────────────────────────────────────────────────
        yield _sse("stage", {
            "stage":      "keyword",
            "message":    "Running keyword search…",
            "elapsed_ms": elapsed(),
        })
        kw_results = await keyword_search(pool, q, fetch_limit, source, days)

        # ── early exit: no results ────────────────────────────────────────────
        if not sem_results and not kw_results:
            yield _sse("results", {
                "results":        [],
                "total":          0,
                "search_mode":    "keyword_only",
                "search_time_ms": elapsed(),
            })
            yield _sse("done", {
                "agent_iterations": None,
                "llm_time_ms":      None,
                "total_time_ms":    elapsed(),
            })
            return

        # ── stage: merge ──────────────────────────────────────────────────────
        yield _sse("stage", {
            "stage":      "merge",
            "message":    "Merging and ranking results…",
            "elapsed_ms": elapsed(),
        })
        merged, mode = merge_and_rerank(sem_results, kw_results, top_n=limit)
        search_time = elapsed()
        log.info("stream MERGE mode=%s results=%d elapsed=%dms", mode, len(merged), search_time)

        # ── results event (before LLM — lets the UI render cards immediately) ─
        serialised: list[dict] = []
        for item in merged:
            try:
                serialised.append(SearchResult(**item).model_dump(mode="json"))
            except Exception:
                pass   # skip any malformed rows rather than aborting the stream

        yield _sse("results", {
            "results":        serialised,
            "total":          len(merged),
            "search_mode":    mode,
            "search_time_ms": search_time,
        })

        # ── stage: agent ──────────────────────────────────────────────────────
        yield _sse("stage", {
            "stage":      "agent",
            "message":    "Agent is reading results and composing answer…",
            "elapsed_ms": elapsed(),
        })

        # ── LLM phase — streaming agent with fallback ─────────────────────────
        t_llm          = time.monotonic()
        summary: str | None  = None
        agent_iterations: int | None = None
        llm_ms: int | None   = None

        async for event_type, payload in run_agent_stream(
            query=q,
            initial_results=merged[:5],
            pool=pool,
            fetch_limit=fetch_limit,
        ):
            if event_type == "agent_activity":
                yield _sse("agent_activity", payload)

            elif event_type == "summary":
                summary = payload["text"]
                yield _sse("summary", payload)

            elif event_type == "done_agent":
                agent_iterations = payload.get("agent_iterations")
                llm_ms           = payload.get("llm_ms")

            elif event_type == "error":
                # Streaming agent failed — emit a "thinking" activity so the UI
                # shows something, then fall through to the summarize() fallback.
                log.info("stream agent_stream error=%r — falling back to summarize()", payload.get("message"))
                yield _sse("agent_activity", {
                    "type":    "thinking",
                    "message": "Switching to fallback summariser…",
                    "detail":  payload.get("message"),
                })
                break   # exit the async-for; summary is still None → triggers fallback

        # ── fallback: only when streaming produced no summary ─────────────────
        # Checking `summary is None` (not a stream_failed flag) ensures we never
        # emit a second summary event if streaming already delivered one.
        if summary is None:
            log.info("stream FALLBACK to summarize()")
            summary          = await summarize(q, merged[:5])
            agent_iterations = None
            llm_ms           = int((time.monotonic() - t_llm) * 1000)

            if summary:
                yield _sse("summary", {"text": summary})
            # If summary is still None (both tiers failed), the UI will show
            # the amber "Summary unavailable" notice from the results event alone.

        # ── done — always the final event ─────────────────────────────────────
        total_ms = elapsed()
        if llm_ms is None:
            llm_ms = int((time.monotonic() - t_llm) * 1000)

        log.info(
            "stream COMPLETE q=%r mode=%s results=%d search=%dms llm=%dms total=%dms",
            q, mode, len(merged), search_time, llm_ms, total_ms,
        )
        yield _sse("done", {
            "agent_iterations": agent_iterations,
            "llm_time_ms":      llm_ms,
            "total_time_ms":    total_ms,
        })

    except Exception as exc:
        log.error(
            "search_stream_generator UNHANDLED q=%r error=%s",
            q, exc, exc_info=True,
        )
        try:
            yield _sse("error", {"message": "Internal search error"})
            yield _sse("done", {
                "agent_iterations": None,
                "llm_time_ms":      None,
                "total_time_ms":    elapsed(),
            })
        except Exception:
            pass   # client disconnected before we could send the error event
