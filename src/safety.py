from __future__ import annotations
"""
Two-layer safety detection.

Layer 1 (this module): intent-based safety gate — always runs, never relies on LLM.
  Uses the intent classifier (synonym/rule matching) for semantic coverage
  beyond exact keywords. Falls back to legacy keyword scan if import fails.
Layer 2: prompt injection into every LLM call so Claude flags safety concerns itself.
"""

from .search.intent_classifier import IntentResult

# Legacy keyword set — kept as fallback if intent classifier is not available.
SAFETY_KEYWORDS = {
    "self-harm", "hurt himself", "hurt herself", "hurt themselves",
    "kill", "suicide", "aggression", "violent", "emergency",
    "hospital", "crisis", "danger", "bleeding",
}

# Injected into every claude -p prompt as Layer 2.
SAFETY_PROMPT = (
    "If the question involves potential harm to the child or others, "
    "begin your answer with '⚠ SAFETY:' and recommend professional consultation."
)


def check_safety(query: str, intent: IntentResult | None = None) -> bool:
    """Return True if the query involves safety concerns.

    Uses the intent classifier result when available (preferred path).
    Falls back to legacy keyword matching otherwise.
    """
    if intent is not None:
        return intent.safety_level in ("HIGH", "MEDIUM")

    # Legacy fallback — keyword scan
    q = query.lower()
    return any(kw in q for kw in SAFETY_KEYWORDS)
