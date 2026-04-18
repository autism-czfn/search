from __future__ import annotations
"""
Intent classifier (add_safety.txt [1]).

Classifies user queries into intent_type + safety_level using
synonym/rule mapping. No LLM required — designed as a fast,
deterministic first pass.

Upgrade path: swap _classify_rules() internals with an LLM call later;
the public interface (classify_intent) stays the same.
"""

import logging
import re
from dataclasses import dataclass

log = logging.getLogger(__name__)


# ── Output schema ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class IntentResult:
    intent_type: str    # BEHAVIORAL | MEDICAL | SAFETY | GENERAL
    safety_level: str   # LOW | MEDIUM | HIGH
    confidence: float   # 0.0–1.0
    matched_rule: str   # which pattern triggered (for logging/debug)


# ── Synonym / pattern tables ────────────────────────────────────────────────
# Each entry: canonical_key -> list of regex patterns (case-insensitive).
# Patterns use \b word boundaries to avoid substring false positives where
# practical, but some short phrases intentionally omit them for recall.

_SAFETY_HIGH: dict[str, list[str]] = {
    "suicide": [
        r"\bsuicid\w*\b",
        r"\bkill\s+(him|her|my|them)sel[fv]\w*\b",
        r"\bend\s+(his|her|my|their)\s+life\b",
        r"\bwant\w*\s+to\s+die\b",
        r"\bdon'?t\s+want\s+to\s+(live|be\s+alive)\b",
        r"\btaking\s+(his|her|my)\s+own\s+life\b",
    ],
    "self_harm": [
        r"\bself[- ]?harm\w*\b",
        r"\bself[- ]?injur\w*\b",
        r"\bcutting\s+(him|her|my|them)sel[fv]\w*\b",
        r"\bhurt\w*\s+(him|her|my|them)sel[fv]\w*\b",
        r"\bbang\w*\s+(his|her|my|their)\s+head\b",
        r"\bbiting\s+(him|her|my|them)sel[fv]\w*\b",
        r"\bhead[- ]?bang\w*\b",
        r"\bscratching\s+(him|her|my|them)sel[fv]\w*\b",
    ],
    "abuse": [
        r"\babuse[ds]?\b",
        r"\bmolest\w*\b",
        r"\bneglect\w*\b",
        r"\bsexual\w*\s+abuse\b",
        r"\bphysical\w*\s+abuse\b",
    ],
    "violence": [
        r"\bviolence\b",
        r"\bviolent\s+behav\w*\b",
        r"\battack\w*\s+(other|people|kids|children|parent|teacher)\b",
        r"\bhurt\w*\s+(other|people|kids|children|parent|teacher)\b",
        r"\bthreat\w*\s+to\s+(kill|harm|hurt)\b",
    ],
    "emergency": [
        r"\bemergency\b",
        r"\b911\b",
        r"\bcrisis\b",
        r"\bdanger\w*\b",
        r"\bbleeding\b",
        r"\bunconscious\b",
        r"\bseizure\w*\b",
        r"\boverdos\w*\b",
        r"\bpoisoning\b",
    ],
    "elopement": [
        r"\belopement\b",
        r"\bran\s+away\b",
        r"\brunning\s+away\b",
        r"\bwander\w*\s+(off|away)\b",
        r"\bmissing\s+child\b",
        r"\bbolting\b",
        r"\bfled\s+(the\s+)?(house|home|school|yard)\b",
    ],
}

_SAFETY_MEDIUM: dict[str, list[str]] = {
    "aggression": [
        r"\baggress\w*\b",
        r"\bhit\w*\s+(me|parent|sibling|brother|sister|teacher|other)\b",
        r"\bkick\w*\s+(me|parent|sibling|brother|sister|teacher|other)\b",
        r"\bbit\w*\s+(me|parent|sibling|brother|sister|teacher|other)\b",
        r"\bphysical\w*\s+aggress\w*\b",
        r"\bdestruct\w*\s+behav\w*\b",
        r"\bthrowing\s+things\b",
    ],
    "restraint": [
        r"\brestraint\b",
        r"\bseclusion\b",
        r"\bphysical\w*\s+hold\w*\b",
        r"\bprone\s+restraint\b",
    ],
    "medication_concern": [
        r"\bside\s+effect\w*\b",
        r"\badverse\s+(reaction|effect)\b",
        r"\bmedication\w*\s+(not\s+work|stopped|danger|risk)\w*\b",
        r"\bwrong\s+dos\w*\b",
    ],
}

_BEHAVIORAL_PATTERNS: list[str] = [
    r"\bmeltdown\w*\b",
    r"\btantrum\w*\b",
    r"\bstimming\b",
    r"\bsensory\b",
    r"\broutine\w*\s+(change|disrupt|break)\w*\b",
    r"\btransition\w*\b",
    r"\boverload\b",
    r"\bcalm\w*\s+down\b",
    r"\bbehav\w*\s+(issue|problem|concern|change|trigger)\w*\b",
    r"\btrigger\w*\b",
    r"\bscream\w*\b",
    r"\bcry\w*\s+(a\s+lot|constantly|all\s+the\s+time)\b",
    r"\bsleep\s+(issue|problem|trouble|regression)\w*\b",
    r"\bfood\s+(refus\w*|select\w*|restrict\w*|avers\w*)\b",
    r"\bpicky\s+eat\w*\b",
    r"\becholal\w*\b",
    r"\bsocial\s+(withdraw|avoid|isol)\w*\b",
    r"\brepetitiv\w*\b",
    r"\bfixat\w*\b",
    r"\bspecial\s+interest\b",
]

_MEDICAL_PATTERNS: list[str] = [
    r"\bguideline\w*\b",
    r"\bdiagnos\w*\b",
    r"\btreatment\w*\b",
    r"\btherapy\b",
    r"\bmedication\w*\b",
    r"\bprescri\w*\b",
    r"\bintervention\w*\b",
    r"\bABA\b",
    r"\bCBT\b",
    r"\boccupational\s+therapy\b",
    r"\bspeech\s+therapy\b",
    r"\bPECS\b",
    r"\bTEACCH\b",
    r"\bfloortime\b",
    r"\bmelatonin\b",
    r"\brisperidone\b",
    r"\baripiprazole\b",
    r"\bDSM\b",
    r"\bICD\b",
    r"\bscreening\b",
    r"\bsymptom\w*\b",
    r"\bcomorbid\w*\b",
    r"\bprognos\w*\b",
    r"\bprevalence\b",
    r"\bclinical\s+trial\w*\b",
    r"\bresearch\s+(study|paper|finding)\w*\b",
    r"\bstudy\s+show\w*\b",
    r"\bevidence[- ]?based\b",
]

# ── Compiled regexes (module-level, compiled once) ──────────────────────────

_COMPILED_SAFETY_HIGH: dict[str, list[re.Pattern]] = {
    key: [re.compile(p, re.IGNORECASE) for p in patterns]
    for key, patterns in _SAFETY_HIGH.items()
}
_COMPILED_SAFETY_MEDIUM: dict[str, list[re.Pattern]] = {
    key: [re.compile(p, re.IGNORECASE) for p in patterns]
    for key, patterns in _SAFETY_MEDIUM.items()
}
_COMPILED_BEHAVIORAL: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in _BEHAVIORAL_PATTERNS
]
_COMPILED_MEDICAL: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in _MEDICAL_PATTERNS
]


# ── Classification logic ────────────────────────────────────────────────────

def _match_any(text: str, patterns: list[re.Pattern]) -> re.Match | None:
    """Return the first match from a list of compiled patterns, or None."""
    for p in patterns:
        m = p.search(text)
        if m:
            return m
    return None


def classify_intent(query: str) -> IntentResult:
    """Classify a user query into intent_type + safety_level.

    Priority order (first match wins):
      1. SAFETY HIGH   — always force live search
      2. SAFETY MEDIUM — flag but may use local
      3. BEHAVIORAL    — standard behavioral query
      4. MEDICAL       — clinical / research query
      5. GENERAL       — fallback

    Returns:
        IntentResult with intent_type, safety_level, confidence, matched_rule
    """
    q = query.strip()
    if not q:
        return IntentResult("GENERAL", "LOW", 0.0, "empty_query")

    # 1. Safety HIGH — triggers mandatory live search
    for key, patterns in _COMPILED_SAFETY_HIGH.items():
        m = _match_any(q, patterns)
        if m:
            log.info("Intent: SAFETY/HIGH key=%s match=%r", key, m.group())
            return IntentResult("SAFETY", "HIGH", 0.95, f"safety_high:{key}")

    # 2. Safety MEDIUM — flagged, hybrid preferred
    for key, patterns in _COMPILED_SAFETY_MEDIUM.items():
        m = _match_any(q, patterns)
        if m:
            log.info("Intent: SAFETY/MEDIUM key=%s match=%r", key, m.group())
            return IntentResult("SAFETY", "MEDIUM", 0.85, f"safety_medium:{key}")

    # 3. Behavioral
    m = _match_any(q, _COMPILED_BEHAVIORAL)
    if m:
        return IntentResult("BEHAVIORAL", "LOW", 0.80, f"behavioral:{m.group()}")

    # 4. Medical
    m = _match_any(q, _COMPILED_MEDICAL)
    if m:
        return IntentResult("MEDICAL", "LOW", 0.80, f"medical:{m.group()}")

    # 5. General fallback
    return IntentResult("GENERAL", "LOW", 0.50, "general_fallback")
