from __future__ import annotations
"""
Two-layer safety detection.

Layer 1 (this module): deterministic keyword gate — always runs, never relies on LLM.
Layer 2: prompt injection into every LLM call so Claude flags safety concerns itself.
"""

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


def check_safety(query: str) -> bool:
    """Return True if any safety keyword is found in the query (case-insensitive)."""
    q = query.lower()
    return any(kw in q for kw in SAFETY_KEYWORDS)
