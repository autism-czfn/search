from __future__ import annotations
"""
Search trigger policy (P5 + add_safety.txt): decides whether to run
evidence search for a given trigger/query, or skip because cached
evidence is strong enough.

Uses the intent classifier for safety detection instead of hardcoded
keyword sets. Safety queries always bypass cache.
"""

import logging
from datetime import date

from .cache import check_cache, compute_cache_key, normalize_query, extract_trigger_key
from .intent_classifier import IntentResult

log = logging.getLogger(__name__)

# Minimum cache freshness threshold — if cached evidence has fewer than this
# many results, re-search even if cache is valid
_MIN_CACHED_RESULTS = 2


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
