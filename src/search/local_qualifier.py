from __future__ import annotations
"""
Local context qualification gate (add_safety.txt [4]).

Decides whether local DB results are good enough to include in the
final answer, especially when mixed with live search results.

When safety_level=HIGH, local results must pass ALL four quality
checks to be included. Otherwise, local results are always included
(quality gate is informational only).
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

log = logging.getLogger(__name__)


# ── Thresholds (from add_safety.txt configuration) ─────────────────────────

RELEVANCE_THRESHOLD = 0.75    # cosine_similarity minimum
QUALITY_THRESHOLD = 0.6       # combined_score minimum
RECENCY_THRESHOLD = 0.5       # recency_score minimum (~ within 7 days)
MIN_RELEVANT_ITEMS = 2        # at least 2 relevant results required


@dataclass(frozen=True)
class QualificationResult:
    include_local: bool
    relevance_ok: bool
    quality_ok: bool
    recency_ok: bool
    coverage_ok: bool
    best_relevance: float
    best_quality: float
    best_recency: float
    relevant_count: int
    reason: str


def _recency_score(item: dict) -> float:
    """Compute recency score with 7-day half-life (0.0–1.0).

    Items within ~7 days score >= 0.5, older items decay exponentially.
    """
    pub = item.get("published_at") or item.get("collected_at")
    if pub is None:
        return 0.3  # unknown date → conservative default

    if isinstance(pub, str):
        try:
            pub = datetime.fromisoformat(pub)
        except (ValueError, TypeError):
            return 0.3
    if pub.tzinfo is None:
        pub = pub.replace(tzinfo=timezone.utc)

    now = datetime.now(timezone.utc)
    days_old = max(0, (now - pub).days)
    # Half-life = 7 days → score = 2^(-days/7)
    return 2.0 ** (-days_old / 7.0)


def qualify_local_results(
    local_results: list[dict],
    safety_level: str = "LOW",
) -> QualificationResult:
    """Evaluate whether local results meet quality bar for inclusion.

    For safety_level HIGH, ALL four conditions must pass:
      1. RELEVANCE:  best semantic_score >= 0.75
      2. QUALITY:    best combined_score >= 0.6
      3. RECENCY:    best recency_score >= 0.5
      4. COVERAGE:   at least 2 results passing relevance threshold

    For non-HIGH safety, local results are always included (gate is
    informational — logged but not enforced).

    Args:
        local_results: merged search results from local DB
        safety_level: "LOW" | "MEDIUM" | "HIGH" from intent classifier

    Returns:
        QualificationResult with include_local decision and diagnostics
    """
    if not local_results:
        return QualificationResult(
            include_local=False,
            relevance_ok=False, quality_ok=False,
            recency_ok=False, coverage_ok=False,
            best_relevance=0.0, best_quality=0.0,
            best_recency=0.0, relevant_count=0,
            reason="no_local_results",
        )

    # Compute scores across all local results
    best_relevance = max(
        (r.get("semantic_score", 0.0) for r in local_results), default=0.0
    )
    best_quality = max(
        (r.get("combined_score", 0.0) for r in local_results), default=0.0
    )
    best_recency = max(
        (_recency_score(r) for r in local_results), default=0.0
    )
    relevant_count = sum(
        1 for r in local_results
        if r.get("semantic_score", 0.0) >= RELEVANCE_THRESHOLD
    )

    # Evaluate each condition
    relevance_ok = best_relevance >= RELEVANCE_THRESHOLD
    quality_ok = best_quality >= QUALITY_THRESHOLD
    recency_ok = best_recency >= RECENCY_THRESHOLD
    coverage_ok = relevant_count >= MIN_RELEVANT_ITEMS

    all_pass = relevance_ok and quality_ok and recency_ok and coverage_ok

    # Decision: for HIGH safety, enforce the gate; otherwise always include
    if safety_level == "HIGH":
        include_local = all_pass
        if not include_local:
            failed = []
            if not relevance_ok:
                failed.append(f"relevance({best_relevance:.2f}<{RELEVANCE_THRESHOLD})")
            if not quality_ok:
                failed.append(f"quality({best_quality:.2f}<{QUALITY_THRESHOLD})")
            if not recency_ok:
                failed.append(f"recency({best_recency:.2f}<{RECENCY_THRESHOLD})")
            if not coverage_ok:
                failed.append(f"coverage({relevant_count}<{MIN_RELEVANT_ITEMS})")
            reason = f"safety_high_gate_failed: {', '.join(failed)}"
            log.warning("Local qualification FAILED (safety=HIGH): %s", reason)
        else:
            reason = "safety_high_gate_passed"
            log.info("Local qualification PASSED (safety=HIGH)")
    else:
        include_local = True
        reason = f"safety_{safety_level.lower()}_gate_bypassed"
        if not all_pass:
            log.info(
                "Local qualification: %s (would fail strict gate: rel=%.2f qual=%.2f rec=%.2f cov=%d)",
                reason, best_relevance, best_quality, best_recency, relevant_count,
            )

    return QualificationResult(
        include_local=include_local,
        relevance_ok=relevance_ok,
        quality_ok=quality_ok,
        recency_ok=recency_ok,
        coverage_ok=coverage_ok,
        best_relevance=best_relevance,
        best_quality=best_quality,
        best_recency=best_recency,
        relevant_count=relevant_count,
        reason=reason,
    )
