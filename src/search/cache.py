from __future__ import annotations
"""
Trigger-level evidence caching (P2).

Hybrid cache key strategy:
  Primary key: normalized_query_hash (SHA-256)
  Secondary dimensions: trigger_key (nullable), language, time_bucket (daily)

Cache stored in evidence_cache table (must be created via migration).
"""

import hashlib
import json
import logging
import re
from datetime import date, datetime

log = logging.getLogger(__name__)

# Known trigger vocabulary (stopgap until Collect P1 provides canonical list)
_TRIGGER_VOCAB = {
    "noise", "transition", "poor_sleep", "sleep", "crowd", "routine_change",
    "sensory_overload", "sensory", "school_stress", "school", "food",
    "social", "screens", "meltdown", "anxiety", "aggression", "stimming",
    "communication", "change", "separation",
}

# Aliases for trigger extraction
_TRIGGER_ALIASES = {
    "loud noise": "noise", "bad sleep": "poor_sleep", "no sleep": "poor_sleep",
    "routine change": "routine_change", "sensory overload": "sensory_overload",
    "school stress": "school_stress", "screen time": "screens",
}


def normalize_query(query_text: str) -> str:
    """Normalize a query for cache key computation.

    Lowercase, strip punctuation, collapse whitespace, sort tokens alphabetically.
    """
    text = query_text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = sorted(text.split())
    return " ".join(tokens)


def compute_cache_key(normalized_query: str, language: str, day: date) -> str:
    """SHA-256 hash of normalized_query + language + date."""
    raw = f"{normalized_query}|{language}|{day.isoformat()}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def extract_trigger_key(query_text: str) -> str | None:
    """Extract a trigger key from free-text query using keyword matching.

    Returns the first matching trigger from the vocabulary, or None.
    Used for secondary cache indexing and invalidation, not as primary key.
    """
    text = query_text.lower()

    # Check multi-word aliases first
    for alias, trigger in _TRIGGER_ALIASES.items():
        if alias in text:
            return trigger

    # Check single-word vocabulary
    words = set(re.sub(r"[^\w\s]", " ", text).split())
    for trigger in _TRIGGER_VOCAB:
        if trigger in words:
            return trigger

    return None


async def check_cache(
    pool,
    query_hash: str,
    language: str,
    time_bucket: date,
) -> dict | None:
    """Check if cached evidence exists and is not expired.

    Returns the cached evidence dict or None on miss/expiry/error.
    """
    try:
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT extracted_evidence, retrieval_results
                FROM evidence_cache
                WHERE query_hash = $1 AND language = $2 AND time_bucket = $3
                  AND expires_at > now()
                ORDER BY created_at DESC LIMIT 1
                """,
                query_hash, language, time_bucket,
            )
        if row:
            evidence = row["extracted_evidence"]
            results = row["retrieval_results"]
            return {
                "extracted_evidence": json.loads(evidence) if isinstance(evidence, str) else evidence,
                "retrieval_results": json.loads(results) if isinstance(results, str) else results,
            }
        return None
    except Exception as e:
        log.warning("Cache check failed (table may not exist): %s", e)
        return None


async def store_cache(
    pool,
    query_hash: str,
    trigger_key: str | None,
    language: str,
    time_bucket: date,
    query_text: str,
    retrieval_results: list[dict],
    extracted_evidence: dict | None,
    ttl_hours: int = 24,
) -> None:
    """Store evidence in cache. Silently fails if table doesn't exist."""
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO evidence_cache
                    (query_hash, trigger_key, language, time_bucket, query_text,
                     retrieval_results, extracted_evidence, created_at, expires_at)
                VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7::jsonb, now(),
                        now() + make_interval(hours => $8))
                ON CONFLICT (query_hash, language, time_bucket) DO UPDATE SET
                    trigger_key = EXCLUDED.trigger_key,
                    query_text = EXCLUDED.query_text,
                    retrieval_results = EXCLUDED.retrieval_results,
                    extracted_evidence = EXCLUDED.extracted_evidence,
                    created_at = now(),
                    expires_at = now() + make_interval(hours => $8)
                """,
                query_hash,
                trigger_key,
                language,
                time_bucket,
                query_text,
                json.dumps(retrieval_results, default=str),
                json.dumps(extracted_evidence, default=str) if extracted_evidence else None,
                ttl_hours,
            )
    except Exception as e:
        log.warning("Cache store failed (table may not exist): %s", e)


async def invalidate_by_trigger(pool, trigger_key: str) -> int:
    """Delete all cache entries matching a trigger key. Returns count deleted."""
    try:
        async with pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM evidence_cache WHERE trigger_key = $1",
                trigger_key,
            )
            count = int(result.split()[-1]) if result else 0
            if count:
                log.info("Cache invalidated %d entries for trigger=%s", count, trigger_key)
            return count
    except Exception as e:
        log.warning("Cache invalidation failed: %s", e)
        return 0
