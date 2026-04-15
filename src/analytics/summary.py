from __future__ import annotations
"""
Weekly summary generator for /api/weekly-summary.

Step 1: Compute deterministic stats from mzhu_test_logs (SQL).
Step 2: Check mzhu_test_summaries cache (< 24h → return cached).
Step 3: claude -p narrates the stats in plain sentences.
Step 4: Persist via collect POST /summaries.
"""

import asyncio
import json
import logging
import os
from collections import Counter
from datetime import date, datetime, timedelta, timezone

import httpx

log = logging.getLogger(__name__)

NARRATE_TIMEOUT = 30   # seconds for claude -p


# ── Step 1: compute weekly stats ─────────────────────────────────────────────

async def _compute_weekly_stats(pool, week_start: date, week_end: date) -> dict:
    ws = datetime(week_start.year, week_start.month, week_start.day, tzinfo=timezone.utc)
    we = datetime(week_end.year,   week_end.month,   week_end.day,   tzinfo=timezone.utc) \
         + timedelta(days=1)   # exclusive upper bound (end of Sunday)

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT triggers, outcome
            FROM mzhu_test_logs
            WHERE logged_at >= $1 AND logged_at < $2 AND NOT voided
            """,
            ws, we,
        )

        adopted_rows = await conn.fetch(
            """
            SELECT suggestion_text FROM mzhu_test_interventions
            WHERE started_at >= $1 AND started_at < $2 AND NOT voided
            """,
            ws, we,
        )

    trigger_counts: Counter = Counter()
    outcome_counts: Counter = Counter()
    for row in rows:
        for t in (row["triggers"] or []):
            trigger_counts[t] += 1
        outcome_counts[row["outcome"]] += 1

    top_triggers = [
        {"trigger": t, "count": c} for t, c in trigger_counts.most_common(3)
    ]
    top_outcomes = [
        {"outcome": o, "count": c} for o, c in outcome_counts.most_common(3)
    ]

    return {
        "event_count":            len(rows),
        "top_triggers":           top_triggers,
        "top_outcomes":           top_outcomes,
        "interventions_adopted":  [r["suggestion_text"] for r in adopted_rows],
    }


# ── Step 2: cache check ───────────────────────────────────────────────────────

async def _get_cached_summary(pool, week_start: date) -> dict | None:
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT summary_text, stats_json, generated_at
            FROM mzhu_test_summaries
            WHERE week_start = $1
            """,
            week_start,
        )
    if row is None:
        return None

    generated_at: datetime = row["generated_at"]
    if generated_at.tzinfo is None:
        generated_at = generated_at.replace(tzinfo=timezone.utc)
    age_hours = (datetime.now(timezone.utc) - generated_at).total_seconds() / 3600
    if age_hours >= 24:
        return None

    return {
        "summary_text": row["summary_text"],
        "stats_json":   row["stats_json"],
        "generated_at": generated_at,
    }


# ── Step 3: LLM narration ─────────────────────────────────────────────────────

async def _narrate(stats: dict) -> str | None:
    stats_json = json.dumps(stats, indent=2, default=str)
    prompt = (
        "Narrate the following weekly stats in 3–5 plain sentences for a parent. "
        "Do not invent numbers. Be warm and practical.\n\n"
        f"Stats:\n{stats_json}"
    )
    try:
        env = os.environ.copy()
        env.pop("CLAUDECODE", None)
        proc = await asyncio.create_subprocess_exec(
            "claude", "-p", prompt,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=NARRATE_TIMEOUT)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            log.warning("weekly-summary narration timed out after %ds", NARRATE_TIMEOUT)
            return None
        if proc.returncode != 0:
            log.warning("weekly-summary narration FAIL claude exit=%d", proc.returncode)
            return None
        output = stdout.decode(errors="replace").strip()
        if not output:
            log.warning("weekly-summary narration FAIL empty output")
            return None
        log.info("weekly-summary narration OK chars=%d", len(output))
        return output
    except FileNotFoundError:
        log.warning("weekly-summary narration: claude CLI not found")
        return None
    except Exception as e:
        log.warning("weekly-summary narration error: %s", e)
        return None


# ── Step 4: persist via collect ───────────────────────────────────────────────

async def _persist(collect_base_url: str, week_start: date, summary_text: str, stats: dict) -> None:
    try:
        async with httpx.AsyncClient(verify=False, timeout=10) as client:
            resp = await client.post(
                f"{collect_base_url}/summaries",
                json={
                    "week_start":    str(week_start),
                    "summary_text":  summary_text,
                    "stats_json":    stats,
                },
            )
            resp.raise_for_status()
        log.info("weekly-summary persisted to collect for week_start=%s", week_start)
    except Exception as e:
        log.warning("weekly-summary: failed to persist to collect: %s — summary still returned", e)


# ── Public entry point ────────────────────────────────────────────────────────

async def get_weekly_summary(user_pool, collect_base_url: str) -> dict:
    today      = date.today()
    week_start = today - timedelta(days=today.weekday())   # Monday
    week_end   = week_start + timedelta(days=6)            # Sunday

    stats = await _compute_weekly_stats(user_pool, week_start, week_end)

    cached = await _get_cached_summary(user_pool, week_start)
    if cached:
        log.info("weekly-summary CACHE HIT week_start=%s", week_start)
        return {
            "week_start":    str(week_start),
            "week_end":      str(week_end),
            "stats":         cached["stats_json"],
            "summary_text":  cached["summary_text"],
            "generated_at":  cached["generated_at"].isoformat(),
            "cached":        True,
        }

    log.info("weekly-summary CACHE MISS — generating for week_start=%s", week_start)
    summary_text = await _narrate(stats)
    if summary_text is None:
        summary_text = "Summary generation unavailable at this time."

    generated_at = datetime.now(timezone.utc)
    await _persist(collect_base_url, week_start, summary_text, stats)

    return {
        "week_start":    str(week_start),
        "week_end":      str(week_end),
        "stats":         stats,
        "summary_text":  summary_text,
        "generated_at":  generated_at.isoformat(),
        "cached":        False,
    }
