from __future__ import annotations
"""
Deterministic SQL aggregates for /api/insights and personalization context.
No LLM involved — all values are SQL/Python computations.
"""

import json
import logging
from collections import Counter
from datetime import date, datetime, timedelta, timezone

from .daily_checks import fetch_check_averages, fetch_low_sleep_correlation

log = logging.getLogger(__name__)


# ── Confidence level thresholds ───────────────────────────────────────────────

def _pattern_confidence(co_pct: float, sample_count: int) -> str:
    if co_pct >= 0.80 and sample_count >= 20:
        return "strong_pattern"
    if co_pct >= 0.50 and sample_count >= 10:
        return "possible_pattern"
    return "insufficient_data"


def _intervention_confidence(window: int) -> str:
    """
    Minimal confidence signal for intervention effectiveness.
    A full 14-day window on both sides = possible_pattern.
    Anything less = insufficient_data (intervention too recent).
    """
    return "possible_pattern" if window >= 7 else "insufficient_data"


# ── Personalization context for LLM prompt injection ─────────────────────────

async def fetch_log_context(pool) -> str | None:
    """
    Fetch last 30 days of logs and compute a compact context string for
    LLM personalization.  Returns None when there are fewer than 5 entries
    (insufficient data — skip injection).
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT triggers, outcome
            FROM mzhu_test_logs
            WHERE logged_at >= now() - interval '30 days'
              AND NOT voided
            ORDER BY logged_at DESC
            LIMIT 200
            """,
        )
        if len(rows) < 5:
            return None

        active_interventions = await conn.fetch(
            """
            SELECT suggestion_text
            FROM mzhu_test_interventions
            WHERE status = 'adopted' AND NOT voided
            """,
        )

    trigger_counts: Counter = Counter()
    outcome_counts: Counter = Counter()
    for row in rows:
        for t in (row["triggers"] or []):
            trigger_counts[t] += 1
        outcome_counts[row["outcome"]] += 1

    context = {
        "total_events": len(rows),
        "top_triggers": [
            {"trigger": t, "count": c} for t, c in trigger_counts.most_common(3)
        ],
        "top_outcomes": [
            {"outcome": o, "count": c} for o, c in outcome_counts.most_common(3)
        ],
        "active_interventions": [r["suggestion_text"] for r in active_interventions],
    }

    # Step 2b: inject last 7 days of daily check-in averages (5 key ratings)
    # Skip if fewer than 3 days have check-ins — insufficient data.
    try:
        today     = date.today()
        check_data = await fetch_check_averages(pool, today - timedelta(days=7), today)
        if check_data["coverage_days"] >= 3:
            avgs = check_data["averages"]
            def _fmt(v: float | None) -> str:
                return f"{v:.1f}" if v is not None else "n/a"
            context["check_in_averages_7d"] = (
                f"sleep={_fmt(avgs['sleep_quality'])}, "
                f"mood={_fmt(avgs['mood'])}, "
                f"sensory={_fmt(avgs['sensory_sensitivity'])}, "
                f"meltdown_count_avg={_fmt(avgs['meltdown_count'])}, "
                f"caregiver_rating={_fmt(avgs['caregiver_rating'])} "
                f"(N={check_data['coverage_days']} days)"
            )
    except Exception as e:
        log.warning("fetch_log_context: check-in averages failed: %s", e)

    return json.dumps(context)


# ── Insights ──────────────────────────────────────────────────────────────────

async def _get_cached_insights(pool, days: int, ttl_hours: int = 1) -> dict | None:
    try:
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT insights_json FROM mzhu_test_insights_cache
                WHERE days = $1
                  AND generated_at > now() - make_interval(hours => $2)
                ORDER BY generated_at DESC LIMIT 1
                """,
                days, ttl_hours,
            )
        if row:
            data = row["insights_json"]
            return json.loads(data) if isinstance(data, str) else data
        return None
    except Exception as e:
        log.warning("insights cache read failed: %s", e)
        return None


async def _persist_insights(pool, days: int, insights: dict) -> None:
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO mzhu_test_insights_cache (days, insights_json) VALUES ($1, $2::jsonb)",
                days, json.dumps(insights, default=str),
            )
            await conn.execute(
                "DELETE FROM mzhu_test_insights_cache WHERE generated_at < now() - interval '24 hours'",
            )
    except Exception as e:
        log.warning("insights cache write failed: %s", e)


async def fetch_insights(pool, days: int, refresh: bool = False) -> dict:
    """Compute all /api/insights data deterministically from SQL, with 1h cache."""
    if not refresh:
        cached = await _get_cached_insights(pool, days)
        if cached:
            return {**cached, "cached": True}

    async with pool.acquire() as conn:

        # ── log count + date range ────────────────────────────────────────────
        log_count = await conn.fetchval(
            """
            SELECT COUNT(*) FROM mzhu_test_logs
            WHERE logged_at >= now() - $1 * interval '1 day' AND NOT voided
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

        # ── top triggers (with raw_signals aggregation) ────────────────────
        trigger_rows = await conn.fetch(
            """
            WITH trigger_expanded AS (
                SELECT id, unnest(triggers) AS trigger
                FROM mzhu_test_logs
                WHERE logged_at >= now() - $1 * interval '1 day' AND NOT voided
            ),
            signal_expanded AS (
                SELECT id, unnest(triggers) AS trigger, unnest(raw_signals) AS raw_signal
                FROM mzhu_test_logs
                WHERE logged_at >= now() - $1 * interval '1 day' AND NOT voided
                  AND array_length(raw_signals, 1) > 0
            )
            SELECT t.trigger,
                   COUNT(DISTINCT t.id) AS cnt,
                   (SELECT array_agg(DISTINCT s.raw_signal)
                    FROM signal_expanded s
                    WHERE s.id IN (SELECT te.id FROM trigger_expanded te WHERE te.trigger = t.trigger)
                      AND s.raw_signal IS NOT NULL
                      AND s.raw_signal != t.trigger
                   ) AS raw_signals
            FROM trigger_expanded t
            GROUP BY t.trigger
            ORDER BY cnt DESC
            LIMIT 20
            """,
            days,
        )

        # ── top outcomes ──────────────────────────────────────────────────────
        outcome_rows = await conn.fetch(
            """
            SELECT outcome, COUNT(*) AS cnt
            FROM mzhu_test_logs
            WHERE logged_at >= now() - $1 * interval '1 day' AND NOT voided
              AND outcome IS NOT NULL
            GROUP BY outcome
            ORDER BY cnt DESC
            """,
            days,
        )

        # ── co-occurrence patterns ────────────────────────────────────────────
        pattern_rows = await conn.fetch(
            """
            WITH expanded AS (
                SELECT unnest(triggers) AS trigger, outcome
                FROM mzhu_test_logs
                WHERE logged_at >= now() - $1 * interval '1 day' AND NOT voided
                  AND outcome IS NOT NULL
            ),
            trigger_totals AS (
                SELECT trigger, COUNT(*)::int AS total
                FROM expanded GROUP BY trigger
            ),
            pairs AS (
                SELECT trigger, outcome, COUNT(*)::int AS co_count
                FROM expanded GROUP BY trigger, outcome
            )
            SELECT
                p.trigger,
                p.outcome,
                p.co_count AS co_occurrence_count,
                t.total    AS total_trigger_events,
                p.co_count::float / t.total AS co_occurrence_pct
            FROM pairs p
            JOIN trigger_totals t ON p.trigger = t.trigger
            ORDER BY p.co_count DESC
            """,
            days,
        )

        # ── intervention effectiveness ─────────────────────────────────────────
        interventions = await conn.fetch(
            """
            SELECT id, suggestion_text, started_at
            FROM mzhu_test_interventions
            WHERE status = 'adopted' AND NOT voided AND started_at IS NOT NULL
            """,
        )

        effectiveness = []
        now_utc = datetime.now(timezone.utc)

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

            effectiveness.append({
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

    # ── assemble response ─────────────────────────────────────────────────────
    total_triggers = sum(r["cnt"] for r in trigger_rows) or 1
    total_outcomes = sum(r["cnt"] for r in outcome_rows) or 1

    # Safety triggers are boosted to top regardless of frequency —
    # one "self_harm" or "suicide" matters more than 50 "noise" events.
    _SAFETY_TRIGGERS = {"self_harm", "aggression", "elopement", "suicide"}

    top_triggers = [
        {
            "trigger": r["trigger"],
            "count":   r["cnt"],
            "pct":     round(r["cnt"] / total_triggers, 2),
            "is_safety": r["trigger"] in _SAFETY_TRIGGERS,
            "raw_signals": r["raw_signals"] or [],
        }
        for r in trigger_rows
    ]
    top_triggers.sort(key=lambda t: (not t["is_safety"], -t["count"]))

    top_outcomes = [
        {
            "outcome": r["outcome"],
            "count":   r["cnt"],
            "pct":     round(r["cnt"] / total_outcomes, 2),
        }
        for r in outcome_rows
    ]

    patterns = [
        {
            "trigger":               r["trigger"],
            "outcome":               r["outcome"],
            "co_occurrence_count":   r["co_occurrence_count"],
            "co_occurrence_pct":     round(r["co_occurrence_pct"], 3),
            "total_trigger_events":  r["total_trigger_events"],
            "confidence_level":      _pattern_confidence(
                                         r["co_occurrence_pct"],
                                         r["total_trigger_events"],
                                     ),
            "sample_count":          r["total_trigger_events"],
        }
        for r in pattern_rows
    ]
    # Boost safety-related patterns to top
    patterns.sort(key=lambda p: (p["trigger"] not in _SAFETY_TRIGGERS, -p["co_occurrence_count"]))

    # Daily check-in trends (SQL aggregates over mzhu_test_daily_checks)
    from_date = date.today() - timedelta(days=days)
    to_date   = date.today()
    check_data  = await fetch_check_averages(pool, from_date, to_date)
    correlation = await fetch_low_sleep_correlation(pool, from_date, to_date)

    result = {
        "top_triggers":              top_triggers,
        "top_outcomes":              top_outcomes,
        "patterns":                  patterns,
        "intervention_effectiveness": effectiveness,
        "log_count":                 log_count,
        "date_range": {
            "from": str(date_row["from_date"]) if date_row and date_row["from_date"] else None,
            "to":   str(date_row["to_date"])   if date_row and date_row["to_date"]   else None,
        },
        "daily_check_trends": {
            **check_data,
            "low_sleep_meltdown_correlation": correlation,
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cached":       False,
    }
    await _persist_insights(pool, days, result)
    return result
