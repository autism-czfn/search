from __future__ import annotations
"""
Clinician report generator for /api/clinician-report.

Step 1: Compute deterministic stats from mzhu_test_logs (SQL).
Step 2: claude -p narrates a "Key concerns" section, grounded in the stats
        and top 3 evidence results for the most frequent trigger.
"""

import asyncio
import json
import logging
import os
from collections import Counter
from datetime import datetime, timedelta, timezone

from .patterns import _pattern_confidence, _intervention_confidence
from .daily_checks import fetch_check_averages, fetch_weekly_check_trends

log = logging.getLogger(__name__)

NARRATE_TIMEOUT = 60   # longer than weekly summary — includes evidence


# ── Step 1: compute clinician stats ──────────────────────────────────────────

async def _compute_stats(user_pool, days: int) -> dict:
    async with user_pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT triggers, outcome, logged_at
            FROM mzhu_test_logs
            WHERE logged_at >= now() - $1 * interval '1 day' AND NOT voided
            ORDER BY logged_at
            """,
            days,
        )

        date_row = await conn.fetchrow(
            """
            SELECT MIN(logged_at)::date AS from_date,
                   MAX(logged_at)::date AS to_date
            FROM mzhu_test_logs
            WHERE logged_at >= now() - $1 * interval '1 day' AND NOT voided
            """,
            days,
        )

        interventions = await conn.fetch(
            """
            SELECT id, suggestion_text, started_at
            FROM mzhu_test_interventions
            WHERE status = 'adopted' AND NOT voided AND started_at IS NOT NULL
            """,
        )

    # events per week (for chart data)
    week_buckets: dict[str, int] = {}
    trigger_counts: Counter = Counter()
    outcome_counts: Counter = Counter()
    for row in rows:
        # bucket by ISO week
        dt: datetime = row["logged_at"]
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        week_key = dt.strftime("%Y-W%W")
        week_buckets[week_key] = week_buckets.get(week_key, 0) + 1
        for t in (row["triggers"] or []):
            trigger_counts[t] += 1
        if row["outcome"] is not None:
            outcome_counts[row["outcome"]] += 1

    total_triggers = sum(trigger_counts.values()) or 1
    total_outcomes = sum(outcome_counts.values()) or 1

    top_triggers = [
        {
            "trigger": t,
            "count":   c,
            "pct":     round(c / total_triggers, 2),
        }
        for t, c in trigger_counts.most_common(10)
    ]
    top_outcomes = [
        {
            "outcome": o,
            "count":   c,
            "pct":     round(c / total_outcomes, 2),
        }
        for o, c in outcome_counts.most_common()
    ]

    # co-occurrence patterns (Python — reuses same row data)
    pair_counts: Counter = Counter()
    trigger_totals: Counter = Counter()
    for row in rows:
        if row["outcome"] is None:
            continue
        for t in (row["triggers"] or []):
            trigger_totals[t] += 1
            pair_counts[(t, row["outcome"])] += 1

    patterns = []
    for (trigger, outcome), co_count in pair_counts.most_common():
        total = trigger_totals[trigger]
        co_pct = co_count / total
        patterns.append({
            "trigger":               trigger,
            "outcome":               outcome,
            "co_occurrence_count":   co_count,
            "co_occurrence_pct":     round(co_pct, 3),
            "total_trigger_events":  total,
            "confidence_level":      _pattern_confidence(co_pct, total),
            "sample_count":          total,
        })

    # intervention outcomes
    now_utc = datetime.now(timezone.utc)
    intervention_outcomes = []
    async with user_pool.acquire() as conn:
        for intv in interventions:
            started_at: datetime = intv["started_at"]
            if started_at.tzinfo is None:
                started_at = started_at.replace(tzinfo=timezone.utc)
            days_since = (now_utc - started_at).days
            window = min(14, max(1, days_since))
            before_start = started_at - timedelta(days=window)
            after_end    = started_at + timedelta(days=window)

            mb = await conn.fetchval(
                """
                SELECT COUNT(*) FROM mzhu_test_logs
                WHERE outcome = 'meltdown' AND NOT voided
                  AND logged_at >= $1 AND logged_at < $2
                """,
                before_start, started_at,
            )
            ma = await conn.fetchval(
                """
                SELECT COUNT(*) FROM mzhu_test_logs
                WHERE outcome = 'meltdown' AND NOT voided
                  AND logged_at >= $1 AND logged_at < $2
                """,
                started_at, after_end,
            )

            rate_before = round(mb / window, 3)
            rate_after  = round(ma / window, 3)

            intervention_outcomes.append({
                "intervention_id":      str(intv["id"]),
                "suggestion_text":      intv["suggestion_text"],
                "started_at":           started_at.date().isoformat(),
                "meltdown_rate_before": rate_before,
                "meltdown_rate_after":  rate_after,
                "delta":                round(rate_after - rate_before, 3),
                "observation_days_before": window,
                "observation_days_after":  window,
                "confidence_level":     _intervention_confidence(window),
            })

    # Daily check-in trends over the full lookback window
    from datetime import date as _date
    check_from = _date.today() - timedelta(days=days)
    check_to   = _date.today()
    check_data     = await fetch_check_averages(user_pool, check_from, check_to)
    weekly_trends  = await fetch_weekly_check_trends(user_pool, check_from, check_to)

    return {
        "date_range": {
            "from": str(date_row["from_date"]) if date_row and date_row["from_date"] else None,
            "to":   str(date_row["to_date"])   if date_row and date_row["to_date"]   else None,
        },
        "event_frequency": {
            "total":    len(rows),
            "per_week": [{"week": k, "count": v} for k, v in sorted(week_buckets.items())],
        },
        "top_triggers":          top_triggers,
        "top_outcomes":          top_outcomes,
        "patterns":              patterns,
        "intervention_outcomes": intervention_outcomes,
        "daily_check_summary": {
            "coverage_days": check_data["coverage_days"],
            "total_days":    check_data["total_days"],
            "averages":      check_data["averages"],
            "weekly_trends": weekly_trends,
        },
    }


# ── Step 2: evidence lookup + LLM narration ───────────────────────────────────

async def _search_evidence(evidence_pool, query: str, limit: int = 3) -> list[dict]:
    """Search evidence DB for the most frequent trigger."""
    try:
        from ..embedder import embed_query
        from ..search.semantic import semantic_search
        from ..search.keyword import keyword_search
        from ..search.hybrid import merge_and_rerank

        embedding = await embed_query(query)
        sem = await semantic_search(evidence_pool, embedding, limit * 2, None, None) \
              if embedding else []
        kw  = await keyword_search(evidence_pool, query, limit * 2, None, None)
        merged, _ = merge_and_rerank(sem, kw, top_n=limit)
        return merged
    except Exception as e:
        log.warning("clinician evidence search failed: %s", e)
        return []


async def _narrate_key_concerns(stats: dict, evidence: list[dict]) -> str | None:
    evidence_text = ""
    for i, r in enumerate(evidence, 1):
        title  = r.get("title", "")
        source = r.get("source", "")
        desc   = (r.get("description") or "")[:200]
        evidence_text += f"[{i}] {title} ({source})\n    {desc}\n\n"

    stats_json = json.dumps(
        {k: v for k, v in stats.items() if k != "event_frequency"},
        indent=2, default=str,
    )

    prompt = (
        "You are writing a clinical report section for a medical appointment about a child with autism.\n\n"
        "Based ONLY on the stats and sources below, write a 'Key concerns' section in 3–5 sentences. "
        "Do not state anything not supported by the stats or the provided sources. "
        "Use clinical but parent-accessible language.\n\n"
        f"Stats:\n{stats_json}\n\n"
        f"Sources:\n{evidence_text}"
        "Key concerns:"
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
            log.warning("clinician report narration timed out after %ds", NARRATE_TIMEOUT)
            return None
        if proc.returncode != 0:
            log.warning("clinician report narration FAIL claude exit=%d", proc.returncode)
            return None
        output = stdout.decode(errors="replace").strip()
        if not output:
            log.warning("clinician report narration FAIL empty output")
            return None
        log.info("clinician report narration OK chars=%d", len(output))
        return output
    except FileNotFoundError:
        log.warning("clinician report: claude CLI not found")
        return None
    except Exception as e:
        log.warning("clinician report narration error: %s", e)
        return None


# ── Cache helpers ─────────────────────────────────────────────────────────────

async def _get_cached_report(user_pool, ttl_hours: int = 2) -> dict | None:
    try:
        async with user_pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT report_json FROM mzhu_test_clinician_cache
                WHERE generated_at > now() - make_interval(hours => $1)
                ORDER BY generated_at DESC LIMIT 1
                """,
                ttl_hours,
            )
        if row:
            data = row["report_json"]
            return json.loads(data) if isinstance(data, str) else data
        return None
    except Exception as e:
        log.warning("clinician cache read failed: %s", e)
        return None


async def _persist_report(user_pool, report: dict) -> None:
    try:
        async with user_pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO mzhu_test_clinician_cache (report_json) VALUES ($1::jsonb)",
                json.dumps(report, default=str),
            )
            await conn.execute(
                "DELETE FROM mzhu_test_clinician_cache WHERE generated_at < now() - interval '24 hours'",
            )
    except Exception as e:
        log.warning("clinician cache write failed: %s", e)


# ── Public entry point ────────────────────────────────────────────────────────

async def get_clinician_report(user_pool, evidence_pool, days: int, refresh: bool = False) -> dict:
    if not refresh:
        cached = await _get_cached_report(user_pool)
        if cached:
            return {**cached, "cached": True}

    stats = await _compute_stats(user_pool, days)

    top_trigger = (
        stats["top_triggers"][0]["trigger"] if stats["top_triggers"] else None
    )
    evidence = await _search_evidence(evidence_pool, top_trigger) if top_trigger else []

    key_concerns = await _narrate_key_concerns(stats, evidence)

    report = {
        **stats,
        "key_concerns_text": key_concerns,
        "generated_at":      datetime.now(timezone.utc).isoformat(),
        "cached":            False,
    }
    await _persist_report(user_pool, report)
    return report
