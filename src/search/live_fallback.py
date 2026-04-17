from __future__ import annotations
"""
Search routing engine (P9 + add_safety.txt [2][3]).

Decides WHEN to trigger live search, selects WHICH sites to search,
and calls site_search.py to execute the searches.

Routing modes:
  LOCAL_ONLY — local results strong, no live search needed
  HYBRID     — local results weak, live search top N sites to supplement
  LIVE_ONLY  — no local results, live search all sites

Safety override (add_safety.txt [3]):
  safety_level HIGH → always HYBRID (live search mandatory)
  safety_level MEDIUM → HYBRID preferred when local is weak
"""

import logging

from ..config import settings as app_settings
from ..sources.registry import get_registry
from .intent_classifier import IntentResult
from .site_search import load_live_search_configs, live_search_all, adapt_live_results

log = logging.getLogger(__name__)

# Kept for backward compatibility — other modules import this set.
# The intent classifier now handles semantic matching; this set is
# only used as a fast-path fallback if classify_intent is bypassed.
_SAFETY_TRIGGERS = {"self_harm", "suicide", "violence", "emergency", "abuse", "aggression", "elopement"}

# In-memory counter for stats
fallback_count = 0


def determine_route(
    local_results: list[dict],
    query: str,
    intent: IntentResult | None = None,
) -> str:
    """Deterministic routing decision.

    Args:
        local_results: merged local search results
        query: original user query (used as fallback if intent is None)
        intent: IntentResult from classify_intent(); if None, falls back
                to legacy keyword matching for backward compatibility

    Returns:
        "LOCAL_ONLY" — local results strong, no live search needed
        "HYBRID"     — local results weak, supplement with live search
        "LIVE_ONLY"  — no local results, live search is primary

    Rules (in order):
        1. If not live_search_enabled → LOCAL_ONLY
        2. If safety_level HIGH → HYBRID (mandatory, add_safety.txt [3])
        3. If safety_level MEDIUM → HYBRID (precautionary)
        4. If no local results → LIVE_ONLY
        5. If too few local results → HYBRID
        6. If best local score too low → HYBRID
        7. Otherwise → LOCAL_ONLY
    """
    if not app_settings.live_search_enabled:
        return "LOCAL_ONLY"

    # Safety check via intent classifier (preferred) or legacy keywords (fallback)
    if intent is not None:
        if intent.safety_level == "HIGH":
            log.info("Routing: HYBRID (safety=HIGH rule=%s)", intent.matched_rule)
            return "HYBRID"
        if intent.safety_level == "MEDIUM":
            log.info("Routing: HYBRID (safety=MEDIUM rule=%s)", intent.matched_rule)
            return "HYBRID"
    else:
        # Legacy fallback — keyword scan (backward compat)
        query_lower = query.lower()
        for safety_term in _SAFETY_TRIGGERS:
            if safety_term in query_lower or safety_term.replace("_", " ") in query_lower:
                log.info("Routing: HYBRID (legacy safety term=%s)", safety_term)
                return "HYBRID"

    if not local_results:
        return "LIVE_ONLY"

    if len(local_results) < app_settings.live_search_min_local_results:
        return "HYBRID"

    best_score = max(
        (r.get("combined_score", 0) for r in local_results), default=0
    )
    if best_score < app_settings.live_search_min_local_score:
        return "HYBRID"

    return "LOCAL_ONLY"


async def run_live_search(
    query: str,
    local_results: list[dict],
) -> list[dict]:
    """Run live search on official sites.

    Selects sites not already represented in local results,
    prioritizes by authority tier, searches in parallel,
    deduplicates by URL, returns SearchResult-compatible dicts.
    """
    global fallback_count
    fallback_count += 1

    # Load live search configs
    ls_configs = load_live_search_configs()
    if not ls_configs:
        log.warning("No live search configs found — skipping live search")
        return []

    # Determine which sites to search (skip sites already in local results)
    local_sources = {item.get("source") for item in local_results if item.get("source")}
    local_urls = {item.get("url") for item in local_results if item.get("url")}

    registry = get_registry()
    candidates = [
        s for s in registry.get_active_sources()
        if s.surface_key not in local_sources
        and s.source_id in ls_configs
    ]

    # Sort by authority tier (tier 1 first)
    candidates.sort(key=lambda s: s.authority_tier or 99)

    # Limit sites to search
    targets = candidates[:app_settings.live_search_max_sources]

    if not targets:
        log.info("LIVE: no candidate sites to search (all represented locally)")
        return []

    site_names = [t.source_id for t in targets]
    log.info("LIVE sites=%s", ",".join(site_names))

    # Search in parallel
    raw_results = await live_search_all(
        query, targets, ls_configs,
        timeout=app_settings.live_search_timeout_sec,
    )

    # Adapt to SearchResult format
    adapted = adapt_live_results(raw_results)

    # Deduplicate by URL against local results
    deduped = [r for r in adapted if r.get("url") not in local_urls]
    removed = len(adapted) - len(deduped)
    if removed:
        log.info("LIVE DEDUP removed=%d (URL overlap with local)", removed)

    return deduped
