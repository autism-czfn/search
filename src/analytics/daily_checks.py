from __future__ import annotations
"""
SQL aggregates over mzhu_test_daily_checks.
All functions are deterministic — no LLM.

IMPORTANT: All JSONB casts use (ratings->>'key')::numeric form.
           Never use ratings->>'key'::numeric (wrong precedence — casts the key string).
"""

import logging
from datetime import date

log = logging.getLogger(__name__)

ACTIVITY_KEYS = [
    "sleep_quality", "mood", "sensory_sensitivity", "appetite",
    "social_tolerance", "meltdown_count", "routine_adherence",
    "communication_ease", "physical_activity", "caregiver_rating",
]


def _round(val) -> float | None:
    return round(float(val), 2) if val is not None else None


async def fetch_check_averages(pool, from_date: date, to_date: date) -> dict:
    """
    Returns per-activity averages and coverage for the date range [from_date, to_date].

    Return schema:
      {
        "coverage_days": int,
        "total_days":    int,
        "averages": {
          "sleep_quality": float | null, "mood": float | null,
          "sensory_sensitivity": float | null, "appetite": float | null,
          "social_tolerance": float | null, "meltdown_count": float | null,
          "routine_adherence": float | null, "communication_ease": float | null,
          "physical_activity": float | null, "caregiver_rating": float | null
        }
      }
    All averages are null when coverage_days == 0.
    """
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT
                COUNT(*) AS coverage_days,
                AVG((ratings->>'sleep_quality')::numeric)       AS sleep_quality,
                AVG((ratings->>'mood')::numeric)                AS mood,
                AVG((ratings->>'sensory_sensitivity')::numeric) AS sensory_sensitivity,
                AVG((ratings->>'appetite')::numeric)            AS appetite,
                AVG((ratings->>'social_tolerance')::numeric)    AS social_tolerance,
                AVG((ratings->>'meltdown_count')::numeric)      AS meltdown_count,
                AVG((ratings->>'routine_adherence')::numeric)   AS routine_adherence,
                AVG((ratings->>'communication_ease')::numeric)  AS communication_ease,
                AVG((ratings->>'physical_activity')::numeric)   AS physical_activity,
                AVG((ratings->>'caregiver_rating')::numeric)    AS caregiver_rating
            FROM mzhu_test_daily_checks
            WHERE check_date BETWEEN $1 AND $2
            """,
            from_date, to_date,
        )

    total_days = (to_date - from_date).days + 1
    coverage = int(row["coverage_days"] or 0)

    return {
        "coverage_days": coverage,
        "total_days":    total_days,
        "averages":      {key: _round(row[key]) for key in ACTIVITY_KEYS},
    }


async def fetch_weekly_check_trends(pool, from_date: date, to_date: date) -> list[dict]:
    """
    Returns week-by-week averages for sleep_quality, mood, and caregiver_rating.

    Return schema: list of {
      "week_start":           str (ISO date, Monday),
      "sleep_quality_avg":    float | null,
      "mood_avg":             float | null,
      "caregiver_rating_avg": float | null
    }
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT
                date_trunc('week', check_date::timestamptz)::date AS week_start,
                AVG((ratings->>'sleep_quality')::numeric)         AS sleep_quality_avg,
                AVG((ratings->>'mood')::numeric)                  AS mood_avg,
                AVG((ratings->>'caregiver_rating')::numeric)      AS caregiver_rating_avg
            FROM mzhu_test_daily_checks
            WHERE check_date BETWEEN $1 AND $2
            GROUP BY week_start
            ORDER BY week_start
            """,
            from_date, to_date,
        )

    return [
        {
            "week_start":           str(row["week_start"]),
            "sleep_quality_avg":    _round(row["sleep_quality_avg"]),
            "mood_avg":             _round(row["mood_avg"]),
            "caregiver_rating_avg": _round(row["caregiver_rating_avg"]),
        }
        for row in rows
    ]


async def fetch_low_sleep_correlation(pool, from_date: date, to_date: date) -> dict:
    """
    Compares avg meltdown log events per day on days where sleep_quality <= 2
    vs. all other check-in days in the window.

    Unit: avg meltdown log events per day (NOT a proportion — can exceed 1.0).
    Null for avg/delta fields when the relevant day count is 0 (no division by zero).

    Return schema: {
      "low_sleep_days":              int,
      "meltdown_log_avg_low_sleep":  float | null,   # null if low_sleep_days == 0
      "meltdown_log_avg_other_days": float | null,   # null if other_days == 0
      "delta":                       float | null    # avg_low - avg_other; null if either is null
    }
    """
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            WITH low_sleep AS (
                SELECT check_date,
                       (ratings->>'sleep_quality')::int AS sq
                FROM mzhu_test_daily_checks
                WHERE check_date BETWEEN $1 AND $2
            ),
            meltdown_logs AS (
                SELECT logged_at::date AS log_date, COUNT(*) AS cnt
                FROM mzhu_test_logs
                WHERE outcome ILIKE '%meltdown%'
                  AND NOT voided
                  AND logged_at::date BETWEEN $1 AND $2
                GROUP BY log_date
            ),
            daily AS (
                SELECT
                    ls.check_date,
                    ls.sq <= 2                   AS is_low_sleep,
                    COALESCE(ml.cnt, 0)          AS meltdown_cnt
                FROM low_sleep ls
                LEFT JOIN meltdown_logs ml ON ml.log_date = ls.check_date
            )
            SELECT
                COUNT(*)         FILTER (WHERE is_low_sleep)       AS low_sleep_days,
                COUNT(*)         FILTER (WHERE NOT is_low_sleep)   AS other_days,
                SUM(meltdown_cnt) FILTER (WHERE is_low_sleep)      AS meltdown_low,
                SUM(meltdown_cnt) FILTER (WHERE NOT is_low_sleep)  AS meltdown_other
            FROM daily
            """,
            from_date, to_date,
        )

    low_sleep_days = int(row["low_sleep_days"] or 0)
    other_days     = int(row["other_days"]     or 0)
    meltdown_low   = int(row["meltdown_low"]   or 0)
    meltdown_other = int(row["meltdown_other"] or 0)

    # NULLIF pattern: return None (not a SQL error) when denominator is 0
    avg_low   = round(meltdown_low   / low_sleep_days, 3) if low_sleep_days > 0 else None
    avg_other = round(meltdown_other / other_days,     3) if other_days     > 0 else None
    delta     = round(avg_low - avg_other, 3) \
                if (avg_low is not None and avg_other is not None) else None

    return {
        "low_sleep_days":              low_sleep_days,
        "meltdown_log_avg_low_sleep":  avg_low,
        "meltdown_log_avg_other_days": avg_other,
        "delta":                       delta,
    }
