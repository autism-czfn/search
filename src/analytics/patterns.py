from __future__ import annotations
"""
Deterministic SQL aggregates for /api/insights and personalization context.
No LLM involved — all values are SQL/Python computations.
"""

import json
import logging
from collections import Counter
from datetime import datetime, timedelta, timezone

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
    return json.dumps(context)


# ── Insights ──────────────────────────────────────────────────────────────────

async def fetch_insights(pool, days: int) -> dict:
    """Compute all /api/insights data deterministically from SQL."""
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

        # ── top triggers ──────────────────────────────────────────────────────
        trigger_rows = await conn.fetch(
            """
            SELECT unnest(triggers) AS trigger, COUNT(*) AS cnt
            FROM mzhu_test_logs
            WHERE logged_at >= now() - $1 * interval '1 day' AND NOT voided
            GROUP BY trigger
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

    top_triggers = [
        {
            "trigger": r["trigger"],
            "count":   r["cnt"],
            "pct":     round(r["cnt"] / total_triggers, 2),
        }
        for r in trigger_rows
    ]

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

    return {
        "top_triggers":              top_triggers,
        "top_outcomes":              top_outcomes,
        "patterns":                  patterns,
        "intervention_effectiveness": effectiveness,
        "log_count":                 log_count,
        "date_range": {
            "from": str(date_row["from_date"]) if date_row and date_row["from_date"] else None,
            "to":   str(date_row["to_date"])   if date_row and date_row["to_date"]   else None,
        },
    }
