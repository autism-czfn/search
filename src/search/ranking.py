from __future__ import annotations
"""
5-factor ranking formula (P4).

Search result ranking:
  0.40 * source_authority + 0.30 * trigger_match + 0.15 * context_match
  + 0.10 * language_match + 0.05 * recency

Evidence ranking (post-extraction):
  0.35 * source_authority + 0.25 * extraction_confidence + 0.20 * trigger_relevance
  + 0.10 * actionability + 0.10 * specificity

All factors deterministic, [0,1], with explicit fallbacks.
Weights loaded from config/ranking.json at module level.
"""

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from ..sources.registry import get_registry

log = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "ranking.json"

# ── Load weights ─────────────────────────────────────────────────────────────

_DEFAULT_SEARCH_WEIGHTS = {
    "source_authority": 0.40,
    "trigger_match": 0.30,
    "context_match": 0.15,
    "language_match": 0.10,
    "recency": 0.05,
}

_DEFAULT_EVIDENCE_WEIGHTS = {
    "source_authority": 0.35,
    "extraction_confidence": 0.25,
    "trigger_relevance": 0.20,
    "actionability": 0.10,
    "specificity": 0.10,
}


def _load_weights() -> tuple[dict[str, float], dict[str, float]]:
    try:
        data = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
        sw = data.get("search_result_weights", _DEFAULT_SEARCH_WEIGHTS)
        ew = data.get("evidence_weights", _DEFAULT_EVIDENCE_WEIGHTS)
        # Validate sums
        for name, w in [("search_result_weights", sw), ("evidence_weights", ew)]:
            total = sum(w.values())
            if abs(total - 1.0) > 0.01:
                log.warning("Ranking weights '%s' sum to %.3f (expected 1.0)", name, total)
        return sw, ew
    except Exception as e:
        log.warning("Failed to load ranking config: %s — using defaults", e)
        return _DEFAULT_SEARCH_WEIGHTS, _DEFAULT_EVIDENCE_WEIGHTS


SEARCH_WEIGHTS, EVIDENCE_WEIGHTS = _load_weights()

# ── Trigger vocabulary (stopgap) ─────────────────────────────────────────────

_TRIGGER_TERMS = {
    "noise", "transition", "sleep", "crowd", "routine", "sensory",
    "school", "food", "social", "screens", "meltdown", "anxiety",
    "aggression", "stimming", "communication", "separation", "overload",
}

# ── Action/specificity patterns ──────────────────────────────────────────────

_ACTION_PATTERNS = re.compile(
    r"\b(try|consider|use|avoid|consult|recommended|suggested|guideline|should|"
    r"encourage|implement|introduce|reduce|limit|provide|offer|establish)\b",
    re.IGNORECASE,
)

_SPECIFICITY_PATTERNS = re.compile(
    r"\b(\d+\s*(?:mg|ml|minutes|hours|weeks|months|years|%|percent)|"
    r"age\s*\d|ages?\s*\d|ABA|CBT|EIBI|PECS|TEACCH|floortime|"
    r"melatonin|risperidone|aripiprazole|occupational therapy|speech therapy)\b",
    re.IGNORECASE,
)

# ── Factor computations ─────────────────────────────────────────────────────


def _source_authority(item: dict) -> float:
    """tier 1 → 1.0, tier 2 → 0.7, tier 3 → 0.3, untiered → 0.0"""
    registry = get_registry()
    source_key = item.get("source") or item.get("source_id") or ""
    entry = registry.get_source_by_key(source_key)
    if entry is None or entry.authority_tier is None:
        return 0.0
    return {1: 1.0, 2: 0.7, 3: 0.3}.get(entry.authority_tier, 0.0)


def _trigger_match(item: dict, query_text: str) -> float:
    """Keyword overlap between query and trigger vocabulary."""
    if not query_text:
        return 0.5
    words = set(query_text.lower().split())
    matches = words & _TRIGGER_TERMS
    if not matches:
        # Fallback to semantic_score if available
        return item.get("semantic_score", 0.5)
    # Score = proportion of query words that are known triggers
    return min(1.0, len(matches) / max(1, len(words)))


def _context_match(item: dict, log_context: dict | None) -> float:
    """Keyword overlap between user's top triggers/outcomes and result content."""
    if not log_context:
        return 0.5
    context_terms = set()
    for t in log_context.get("top_triggers", []):
        context_terms.update(t.lower().split("_"))
    for o in log_context.get("top_outcomes", []):
        context_terms.update(o.lower().split("_"))

    if not context_terms:
        return 0.5

    text = f"{item.get('title', '')} {item.get('description', '')}".lower()
    text_words = set(text.split())
    matched = context_terms & text_words
    return len(matched) / len(context_terms) if context_terms else 0.5


def _language_match(
    item: dict,
    user_lang: str = "en",
    cross_lingual: bool = False,
    target_lang: str | None = None,
) -> float:
    """Binary: 1.0 if lang matches, 0.0 otherwise. Cross-lingual override."""
    result_lang = item.get("lang")
    if result_lang is None:
        return 1.0  # assume match if unknown
    if result_lang == user_lang:
        return 1.0
    if cross_lingual and target_lang and result_lang == target_lang:
        return 1.0
    return 0.0


def _recency(item: dict) -> float:
    """Exponential decay with half-life 365 days."""
    pub = item.get("published_at")
    if pub is None:
        return 0.5
    if isinstance(pub, str):
        try:
            pub = datetime.fromisoformat(pub)
        except (ValueError, TypeError):
            return 0.5
    if pub.tzinfo is None:
        pub = pub.replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    days_old = max(0, (now - pub).days)
    return 2.0 ** (-days_old / 365.0)


def _extraction_confidence(query_text: str, snippet: str) -> float:
    """Jaccard similarity between query tokens and snippet tokens."""
    if not snippet or not query_text:
        return 0.3
    q_tokens = set(query_text.lower().split())
    s_tokens = set(snippet.lower().split())
    if not q_tokens or not s_tokens:
        return 0.3
    intersection = q_tokens & s_tokens
    union = q_tokens | s_tokens
    return len(intersection) / len(union) if union else 0.3


def _actionability(snippet: str) -> float:
    """Count action-indicating patterns in snippet."""
    if not snippet:
        return 0.3
    matches = _ACTION_PATTERNS.findall(snippet)
    return min(1.0, len(matches) / 3.0)


def _specificity(snippet: str) -> float:
    """Count detail markers (named entities, numbers, therapy names)."""
    if not snippet:
        return 0.3
    matches = _SPECIFICITY_PATTERNS.findall(snippet)
    return min(1.0, len(matches) / 4.0)


# ── Public scoring functions ─────────────────────────────────────────────────


def compute_search_score(
    item: dict,
    query_text: str = "",
    log_context: dict | None = None,
    user_lang: str = "en",
    cross_lingual: bool = False,
    target_lang: str | None = None,
) -> float:
    """Compute the 5-factor search result ranking score."""
    w = SEARCH_WEIGHTS
    score = (
        w["source_authority"] * _source_authority(item)
        + w["trigger_match"] * _trigger_match(item, query_text)
        + w["context_match"] * _context_match(item, log_context)
        + w["language_match"] * _language_match(item, user_lang, cross_lingual, target_lang)
        + w["recency"] * _recency(item)
    )
    return round(min(1.0, max(0.0, score)), 6)


def compute_evidence_score(
    query_text: str,
    snippet: str,
    source_key: str = "",
) -> float:
    """Compute the 5-factor evidence ranking score (post-extraction)."""
    w = EVIDENCE_WEIGHTS
    item = {"source": source_key}
    score = (
        w["source_authority"] * _source_authority(item)
        + w["extraction_confidence"] * _extraction_confidence(query_text, snippet)
        + w["trigger_relevance"] * _trigger_match(item, snippet)
        + w["actionability"] * _actionability(snippet)
        + w["specificity"] * _specificity(snippet)
    )
    return round(min(1.0, max(0.0, score)), 6)
