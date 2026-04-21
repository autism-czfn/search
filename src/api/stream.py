from __future__ import annotations
"""
SSE generator for GET /api/search/stream.

Emits one SSE event per pipeline stage so the browser can update its UI
progressively, without waiting 30-60 s for a single JSON response.

Event sequence (normal mode):
  metadata       × 1     (safety_flag — emitted first, before any LLM content)
  stage          × 4-5   (embedding → semantic → keyword → merge → agent)
  results        × 1     (source cards — sent BEFORE the LLM runs)
  agent_activity × 0-N   (one per tool call the agent makes)
  summary        × 0-1   (LLM answer; absent only if DB is down)
  done           × 1     (always the final event)

Event sequence (SAFETY_EXPANDED_MODE — P-SRC-6):
  metadata       × 1     (safety_flag: true — emitted first)
  stage          × 1-2   (safety_fan_out → safety_merge)
  results        × 1     (source cards — sent BEFORE the LLM runs)
  safety_extras  × 0-1   (cross_source_consensus, key_clinical_guidance, urgent_help_section)
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
from ..safety import check_safety
from ..analytics.patterns import fetch_log_context
from ..safety_state import get_safety_flag
from ..search.live_fallback import determine_route, _SAFETY_EXPANDED_INTENTS
from ..search.intent_classifier import classify_intent
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
    user_pool=None,
    child_id: str = "default",
    lang: str = "en",
) -> AsyncGenerator[str, None]:
    """
    Full SSE pipeline for /api/search/stream.
    Yields SSE-formatted strings; FastAPI's StreamingResponse writes them
    to the HTTP response as each one is produced.
    Guaranteed to always yield a final `done` event, even on unexpected errors.

    In SAFETY_EXPANDED_MODE: emits metadata(safety_flag:true), stage events,
    results, safety_extras, done — bypasses normal LLM agent pipeline.
    """
    t0 = time.monotonic()

    def elapsed() -> int:
        return int((time.monotonic() - t0) * 1000)

    fetch_limit = limit * 2   # fetch extra before reranking

    try:
        # ── Intent classification + Redis safety flag check ───────────────────
        intent = classify_intent(q)
        redis_safety_flag = await get_safety_flag(child_id)

        # ── P-SRC-6: Check for SAFETY_EXPANDED_MODE before anything else ─────
        is_safety_expanded = (
            (intent.intent_type and intent.intent_type.lower() in _SAFETY_EXPANDED_INTENTS)
            or redis_safety_flag is not None
        )

        # ── metadata event (first — safety_flag before any LLM content) ───────
        safety_flag_val = is_safety_expanded or check_safety(q, intent=intent)
        yield _sse("metadata", {"safety_flag": safety_flag_val})

        # ── SAFETY_EXPANDED_MODE branch ───────────────────────────────────────
        if is_safety_expanded:
            log.info("stream SAFETY_EXPANDED_MODE query=%r intent=%s redis_flag=%s",
                     q, intent.intent_type, bool(redis_safety_flag))
            yield _sse("stage", {
                "stage":      "safety_fan_out",
                "message":    "Running safety-expanded search across all trusted authorities…",
                "elapsed_ms": elapsed(),
            })

            from ..search.safety_expanded import run_safety_expanded_search
            safety_result = await run_safety_expanded_search(
                query=q, intent_type=intent.intent_type, child_id=child_id
            )

            yield _sse("stage", {
                "stage":      "safety_merge",
                "message":    "Merging and ranking safety results…",
                "elapsed_ms": elapsed(),
            })

            raw_results = safety_result.get("results", [])
            serialised: list[dict] = []
            for item in raw_results[:limit]:
                try:
                    serialised.append(SearchResult(**item).model_dump(mode="json"))
                except Exception:
                    pass

            yield _sse("results", {
                "results":        serialised,
                "total":          len(serialised),
                "search_mode":    "safety_expanded",
                "search_time_ms": elapsed(),
                "safety_flag":    True,
                "safety_incomplete": safety_result.get("safety_incomplete", False),
            })

            # ── safety_extras event ───────────────────────────────────────────
            extras = safety_result.get("extras")
            if extras is not None:
                yield _sse("safety_extras", extras)

            yield _sse("done", {
                "agent_iterations": None,
                "llm_time_ms":      elapsed(),
                "total_time_ms":    elapsed(),
            })
            return

        # ── Normal pipeline (non-safety) ─────────────────────────────────────

        # ── fetch personalization context ─────────────────────────────────────
        log_context: str | None = None
        if user_pool is not None:
            try:
                log_context = await fetch_log_context(user_pool)
            except Exception as e:
                log.warning("stream: failed to fetch log context: %s", e)

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
            sem_results = await semantic_search(pool, embedding, fetch_limit, source, days,
                                                official_only=True)

        # ── stage: keyword ────────────────────────────────────────────────────
        yield _sse("stage", {
            "stage":      "keyword",
            "message":    "Running keyword search…",
            "elapsed_ms": elapsed(),
        })
        kw_results = await keyword_search(pool, q, fetch_limit, source, days,
                                          official_only=True)

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
        merged, mode = merge_and_rerank(sem_results, kw_results, top_n=limit,
                                        official_only=True)
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
            log_context=log_context,
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
            summary          = await summarize(q, merged[:5], log_context)
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
