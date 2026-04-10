from __future__ import annotations
"""
Hybrid merge: combines semantic + keyword result sets, normalises scores,
applies weighted reranking, and returns the top-N results.

Score contract:
  - semantic_score: cosine similarity returned by semantic.py (0–1 range, but normalised anyway)
  - keyword_score:  ts_rank returned by keyword.py (unbounded — MUST be normalised before merge)
  - missing dimension: defaults to 0.0

Modes:
  "hybrid"       — both result sets available; weighted merge 0.7 semantic + 0.3 keyword
  "keyword_only" — semantic results empty (cold start / API failure); combined = keyword_norm
"""

import logging
from typing import Literal

log = logging.getLogger(__name__)

SearchMode = Literal["hybrid", "keyword_only"]

SEMANTIC_WEIGHT = 0.7
KEYWORD_WEIGHT = 0.3

# Sources considered authoritative; receive a score boost during reranking
AUTHORITY_SOURCES = {
    "cdc", "nih", "who", "pubmed", "mayoclinic", "webmd",
    "autism_speaks", "aap", "nimh", "nhs",
}
AUTHORITY_BOOST = 0.15   # additive bonus on top of combined score (0–1 range)


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


def merge_and_rerank(
    semantic_results: list[dict],
    keyword_results: list[dict],
    top_n: int,
) -> tuple[list[dict], SearchMode]:
    """
    Merge semantic and keyword result dicts, normalise scores, rerank.

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
    by_id: dict[int, dict] = {}

    for row in semantic_results:
        item = dict(row)
        item["_sem_raw"] = float(item.get("semantic_score", 0.0))
        item["_kw_raw"] = 0.0          # default: not in keyword results
        by_id[item["id"]] = item

    for row in keyword_results:
        item_id = row["id"]
        if item_id in by_id:
            by_id[item_id]["_kw_raw"] = float(row.get("keyword_score", 0.0))
        else:
            item = dict(row)
            item["_sem_raw"] = 0.0     # default: not in semantic results
            item["_kw_raw"] = float(item.get("keyword_score", 0.0))
            by_id[item_id] = item

    items = list(by_id.values())

    # ── Normalise ────────────────────────────────────────────────────────────
    sem_raw = [it["_sem_raw"] for it in items]
    kw_raw  = [it["_kw_raw"]  for it in items]

    sem_norm = _normalise(sem_raw)
    kw_norm  = _normalise(kw_raw)

    # ── Compute combined score ───────────────────────────────────────────────
    for it, sn, kn in zip(items, sem_norm, kw_norm):
        if mode == "keyword_only":
            combined = kn
        else:
            combined = SEMANTIC_WEIGHT * sn + KEYWORD_WEIGHT * kn

        # Boost authoritative sources so they rank above community posts
        source = (it.get("source") or "").lower().replace("-", "_").replace(" ", "_")
        if source in AUTHORITY_SOURCES:
            combined = min(combined + AUTHORITY_BOOST, 1.0)

        it["semantic_score"] = round(sn, 6)
        it["keyword_score"]  = round(kn, 6)
        it["combined_score"] = round(combined, 6)

        # Clean up internal keys
        it.pop("_sem_raw", None)
        it.pop("_kw_raw", None)

    # ── Rerank and truncate ──────────────────────────────────────────────────
    items.sort(key=lambda x: x["combined_score"], reverse=True)
    return items[:top_n], mode
