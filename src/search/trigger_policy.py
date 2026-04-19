from __future__ import annotations
"""
Search trigger policy (P5 + add_safety.txt): decides whether to run
evidence search for a given trigger/query, or skip because cached
evidence is strong enough.

Uses the intent classifier for safety detection instead of hardcoded
keyword sets. Safety queries always bypass cache.
"""

import logging
from datetime import date, datetime, timezone

import httpx

from .cache import check_cache, compute_cache_key, normalize_query, extract_trigger_key, invalidate_by_trigger
from .intent_classifier import IntentResult

log = logging.getLogger(__name__)

# Module-level dict storing last-seen `last_seen` datetime per trigger_key.
# Resets on service restart — acceptable per spec.
_trigger_last_seen: dict[str, datetime] = {}

# Minimum cache freshness threshold — if cached evidence has fewer than this
# many results, re-search even if cache is valid
_MIN_CACHED_RESULTS = 2


async def fetch_and_invalidate_trigger_cache(
    user_pool,
    collect_base_url: str,
    child_id: str = "default",
) -> None:
    """Fetch /logs/trigger-signals from collect and invalidate stale evidence cache entries.

    For each trigger signal returned, compares `last_seen` against the last-known
    value stored in `_trigger_last_seen`. If it has advanced (new log events), any
    evidence_cache rows keyed to that trigger are deleted so the next search fetches
    fresh results.

    Runs silently — any fetch or DB failure is logged at WARNING and the pipeline
    continues unaffected.

    If the response does not include `last_seen` per trigger, invalidation is skipped.
    """
    try:
        async with httpx.AsyncClient(verify=False, timeout=5.0) as client:
            resp = await client.get(
                f"{collect_base_url}/logs/trigger-signals",
                params={"child_id": child_id},
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        log.warning("fetch_and_invalidate_trigger_cache: could not fetch trigger-signals: %s", e)
        return

    if not isinstance(data, dict):
        log.warning(
            "fetch_and_invalidate_trigger_cache: unexpected response schema "
            "(expected dict, got %s) — skipping invalidation",
            type(data).__name__,
        )
        return

    trigger_signals = data.get("trigger_signals")
    if not trigger_signals or not isinstance(trigger_signals, list):
        return

    for signal in trigger_signals:
        trigger_key = signal.get("trigger")
        last_seen_raw = signal.get("last_seen")

        if not trigger_key or last_seen_raw is None:
            # Skip if response doesn't include the expected fields
            continue

        try:
            if isinstance(last_seen_raw, str):
                last_seen = datetime.fromisoformat(last_seen_raw.replace("Z", "+00:00"))
            elif isinstance(last_seen_raw, datetime):
                last_seen = last_seen_raw
            else:
                continue

            # Normalise to UTC-aware
            if last_seen.tzinfo is None:
                last_seen = last_seen.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            continue

        prev = _trigger_last_seen.get(trigger_key)

        if prev is not None and last_seen > prev:
            # New log events recorded for this trigger — invalidate stale cache
            log.debug(
                "trigger cache invalidation: trigger=%s last_seen advanced %s → %s",
                trigger_key, prev.isoformat(), last_seen.isoformat(),
            )
            await invalidate_by_trigger(user_pool, trigger_key)

        # Always update the remembered timestamp
        _trigger_last_seen[trigger_key] = last_seen


async def should_search(
    pool,
    query: str,
    language: str = "en",
    force: bool = False,
    intent: IntentResult | None = None,
) -> tuple[bool, str, dict | None]:
    """Decide whether to run evidence search or return cached results.

    Args:
        pool: database connection pool
        query: user query text
        language: target language
        force: bypass cache entirely
        intent: IntentResult from classify_intent(); used for safety detection

    Returns:
        (should_run_search, reason, cached_evidence_or_none)

    Reasons:
        "forced"         — user explicitly requested fresh results
        "safety"         — safety-sensitive query detected (never cached)
        "no_cache"       — no cache entry exists
        "cache_weak"     — cache exists but has too few results
        "cache_expired"  — cache entry expired
        "cache_hit"      — strong cached evidence available (skip search)
        "new_trigger"    — trigger not seen before (no cached data)
    """
    if force:
        return True, "forced", None

    # Safety queries always search — never trust cache for safety
    if intent is not None and intent.safety_level in ("HIGH", "MEDIUM"):
        log.info("Trigger policy: safety override (level=%s rule=%s)",
                 intent.safety_level, intent.matched_rule)
        return True, "safety", None

    # Legacy fallback if no intent provided
    if intent is None:
        trigger = extract_trigger_key(query)
        _SAFETY_TRIGGERS = {"self_harm", "suicide", "violence", "emergency", "abuse"}
        if trigger and trigger in _SAFETY_TRIGGERS:
            return True, "safety", None

    # Check cache
    normalized = normalize_query(query)
    cache_key = compute_cache_key(normalized, language, date.today())

    cached = await check_cache(pool, cache_key, language, date.today())

    if cached is None:
        trigger = extract_trigger_key(query)
        reason = "new_trigger" if trigger else "no_cache"
        return True, reason, None

    # Cache exists — check quality
    results = cached.get("retrieval_results")
    if not results or (isinstance(results, list) and len(results) < _MIN_CACHED_RESULTS):
        return True, "cache_weak", cached

    # Strong cache hit — skip search
    log.info("Trigger policy: cache hit for query=%r", query[:50])
    return False, "cache_hit", cached
