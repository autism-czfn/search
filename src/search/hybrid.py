from __future__ import annotations
"""
Hybrid merge: combines semantic + keyword result sets, normalises scores,
applies 5-factor ranking (P4), and returns the top-N results.

Score contract:
  - semantic_score: cosine similarity returned by semantic.py (0–1 range, but normalised anyway)
  - keyword_score:  ts_rank returned by keyword.py (unbounded — MUST be normalised before merge)
  - missing dimension: defaults to 0.0

Modes:
  "hybrid"       — both result sets available
  "keyword_only" — semantic results empty (cold start / API failure)

Ranking uses the 5-factor formula from ranking.py (P4):
  0.40 * source_authority + 0.30 * trigger_match + 0.15 * context_match
  + 0.10 * language_match + 0.05 * recency
"""

import logging
from typing import Literal

from ..sources.registry import get_registry
from .ranking import compute_search_score

log = logging.getLogger(__name__)

SearchMode = Literal["hybrid", "keyword_only"]


def _normalise(values: list[float]) -> list[float]:
    """Min-max normalise a list of floats to [0, 1]."""
    if not values:
        return values
    lo, hi = min(values), max(values)
    span = hi - lo
    if span < 1e-9:
        # All values identical — return 1.0 for non-zero, 0.0 for zero
        return [1.0 if v > 0 else 0.0 for v in values]
    return [(v - lo) / span for v in values]


def _enrich_with_registry(item: dict) -> dict:
    """Add registry metadata (source_id, organization_name, authority_tier, audience_type) to a result dict."""
    registry = get_registry()
    # Check surface_key first (actual source identity), then fall back to source
    # (which may be a platform name like "html_crawl" for crawled official sites)
    source_key = (item.get("surface_key") or item.get("source") or item.get("source_id") or "")
    entry = registry.get_source_by_key(source_key)
    if entry is not None:
        item["source_id"] = entry.source_id
        item["source_domain"] = entry.domain
        item["organization_name"] = entry.organization_name
        item["authority_tier"] = entry.authority_tier
        item["audience_type"] = entry.audience_type
    else:
        item.setdefault("source_id", None)
        item.setdefault("source_domain", None)
        item.setdefault("organization_name", None)
        item.setdefault("authority_tier", None)
        item.setdefault("audience_type", None)
    return item


def merge_and_rerank(
    semantic_results: list[dict],
    keyword_results: list[dict],
    top_n: int,
    query_text: str = "",
    log_context: dict | None = None,
    user_lang: str = "en",
    cross_lingual: bool = False,
    target_lang: str | None = None,
) -> tuple[list[dict], SearchMode]:
    """
    Merge semantic and keyword result dicts, normalise scores, apply
    5-factor ranking (P4), and return top-N results.

    Ranking formula (from ranking.py):
      0.40 * source_authority + 0.30 * trigger_match + 0.15 * context_match
      + 0.10 * language_match + 0.05 * recency

    Returns:
        (ranked_results, search_mode)
    """
    # ── Determine mode ──────────────────────────────────────────────────────
    if not semantic_results:
        mode: SearchMode = "keyword_only"
        log.info("Hybrid merge: no semantic results — keyword_only mode")
    else:
        mode = "hybrid"

    # ── Index both lists by item id ──────────────────────────────────────────
    # Live results (P7) use id=-1 sentinel — they can't be deduped by id.
    # Give each a unique negative id so they don't collide.
    _live_counter = -1
    by_id: dict[int, dict] = {}

    for row in semantic_results:
        item = dict(row)
        item["_sem_raw"] = float(item.get("semantic_score", 0.0))
        item["_kw_raw"] = 0.0          # default: not in keyword results
        by_id[item["id"]] = item

    for row in keyword_results:
        item = dict(row)
        item_id = item["id"]
        # Assign unique negative ids to live results so they don't collide
        if item_id < 0:
            item["id"] = _live_counter
            item_id = _live_counter
            _live_counter -= 1
        if item_id in by_id:
            by_id[item_id]["_kw_raw"] = float(item.get("keyword_score", 0.0))
        else:
            item["_sem_raw"] = 0.0     # default: not in semantic results
            item["_kw_raw"] = float(item.get("keyword_score", 0.0))
            by_id[item_id] = item

    items = list(by_id.values())

    # ── Normalise raw scores ────────────────────────────────────────────────
    sem_raw = [it["_sem_raw"] for it in items]
    kw_raw  = [it["_kw_raw"]  for it in items]

    sem_norm = _normalise(sem_raw)
    kw_norm  = _normalise(kw_raw)

    # ── Compute combined score via 5-factor ranking (P4) ─────────────────────
    for it, sn, kn in zip(items, sem_norm, kw_norm):
        it["semantic_score"] = round(sn, 6)
        it["keyword_score"]  = round(kn, 6)

        # Enrich with registry metadata (needed by ranking factors)
        _enrich_with_registry(it)

        # Apply 5-factor ranking formula
        it["combined_score"] = compute_search_score(
            it,
            query_text=query_text,
            log_context=log_context,
            user_lang=user_lang,
            cross_lingual=cross_lingual,
            target_lang=target_lang,
        )

        # Clean up internal keys
        it.pop("_sem_raw", None)
        it.pop("_kw_raw", None)
        it.pop("_is_live", None)

    # ── Filter non-active sources (safety guard) ─────────────────────────────
    _registry = get_registry()
    active_items = []
    for it in items:
        source_key = it.get("source") or ""
        entry = _registry.get_source_by_key(source_key)
        # Keep if: source not in registry (unknown), or source is active
        if entry is None or entry.is_active:
            active_items.append(it)

    if not active_items and items:
        log.warning(
            "Registry filter removed ALL %d results — returning unfiltered. "
            "Check config/sources.json for accidentally deactivated sources.",
            len(items),
        )
        active_items = items

    # ── Rerank and truncate ──────────────────────────────────────────────────
    active_items.sort(key=lambda x: x["combined_score"], reverse=True)
    return active_items[:top_n], mode
