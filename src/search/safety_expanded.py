from __future__ import annotations
"""
SAFETY_EXPANDED_MODE search implementation (P-SRC-6).

Activates when:
  - intent IN {self_harm, suicide, violence, abuse}
  - OR collect safety_webhook received (get_safety_flag returns a value)

Spec: plan.txt §3 SAFETY SEARCH POLICY v2

Key behaviours:
  - Cache BYPASSED entirely
  - Phase 1: parallel fan-out across ALL 7 Tier-1 sources (CDC, NIMH, NICE,
    NHS, AAP, Mayo Clinic, PubMed) — per-site timeout 3-5s, total budget 5-7s
  - Phase 2 (only if insufficient coverage): adds ALL 4 Tier-2 sources
    (Europe PMC, Semantic Scholar, OpenAlex, ClinicalTrials.gov)
  - Diversity constraints: >= 4 distinct domains, >= 2 Tier-1 sources,
    no single source > 40% of results
  - safety_incomplete = True if constraints unmet after Phase 1+2
  - Ranking: raw_score = 0.45*authority_score + 0.35*combined_score
             final_score = _sigmoid(raw_score)
    (all other factors disabled)
  - Crisis language override: >= 3 Tier-1 sources if crisis language detected
  - Extras (for self_harm intent): cross_source_consensus, key_clinical_guidance,
    urgent_help_section (ALWAYS present — hardcoded fallback if LLM fails)
"""

import asyncio
import logging
import math
import time
from datetime import datetime, timezone

from ..sources.registry import get_registry
from .site_search import load_live_search_configs, live_search_all, adapt_live_results
from .hybrid import _enrich_with_registry

log = logging.getLogger(__name__)

# Per-site timeout (seconds)
_SITE_TIMEOUT = 4.0

# Total budget (seconds)
_TOTAL_BUDGET = 6.0

# Hardcoded fallback for urgent_help_section — NEVER omit for self_harm
URGENT_HELP_FALLBACK = (
    "If in immediate danger, call 911. "
    "Mental health crisis line: 988 (US). "
    "Crisis Text Line: text HOME to 741741."
)

# Tier-1 source IDs as defined by the plan (CDC, NIMH, NICE, NHS, AAP, Mayo Clinic, PubMed)
_TIER1_SOURCE_IDS = {
    "cdc_autism",
    "nimh_autism",
    "nice_autism",
    "nhs_autism",
    "aap_autism",
    "mayo_autism",
    "pubmed",
}

# Tier-2 source IDs (Europe PMC, Semantic Scholar, OpenAlex, ClinicalTrials.gov)
_TIER2_SOURCE_IDS = {
    "europepmc_autism",
    "semanticscholar_autism",
    "openalex_autism",
    "clinicaltrials_autism",
}

# Crisis language patterns that trigger force-include >= 3 Tier-1 sources
_CRISIS_PHRASES = [
    "want to die", "wants to die",
    "kill myself", "kill himself", "kill herself",
    "suicide", "suicidal",
    "end my life", "end his life", "end her life",
    "self harm", "self-harm", "hurt myself", "hurt himself", "hurt herself",
    "不想活", "想死", "自杀",  # Chinese crisis phrases
]


def _has_crisis_language(query: str) -> bool:
    """Return True if the query contains explicit crisis language."""
    q_lower = query.lower()
    return any(phrase in q_lower for phrase in _CRISIS_PHRASES)


def _authority_score(item: dict) -> float:
    """authority_score: tier1=1.0, tier2=0.7, tier3=0.3, unknown=0.0"""
    tier = item.get("authority_tier")
    if tier is None:
        return 0.0
    return {1: 1.0, 2: 0.7, 3: 0.3}.get(int(tier), 0.0)


def _combined_score(item: dict) -> float:
    """combined_score = 0.6 * semantic_score + 0.4 * keyword_score"""
    sem = float(item.get("semantic_score", 0.0))
    kw = float(item.get("keyword_score", 0.0))
    return 0.6 * sem + 0.4 * kw


def _sigmoid(x: float) -> float:
    """Standard sigmoid: 1 / (1 + exp(-x))"""
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def _safety_score(item: dict) -> float:
    """
    SAFETY_EXPANDED_MODE ranking:
      raw_score  = 0.45 * authority_score + 0.35 * combined_score
      final_score = _sigmoid(raw_score)
    All other factors (trigger_match, language_match, context_match, recency,
    personalization) = 0.0 in this mode.
    """
    raw = 0.45 * _authority_score(item) + 0.35 * _combined_score(item)
    return _sigmoid(raw)


def _get_domain(item: dict) -> str:
    """Extract domain from URL or source_id."""
    url = item.get("url", "")
    if url:
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            return parsed.netloc or item.get("source", "")
        except Exception:
            pass
    return item.get("source", "") or item.get("source_id", "")


def _is_tier1_source(item: dict) -> bool:
    """Return True if item is from a Tier-1 source."""
    return item.get("authority_tier") == 1


def _check_diversity(results: list[dict]) -> tuple[bool, int, int]:
    """
    Check diversity constraints:
      - >= 4 distinct domains
      - >= 2 Tier-1 sources represented
      - no single source > 40% of results

    Returns (constraints_met, distinct_domain_count, tier1_source_count)
    """
    if not results:
        return False, 0, 0

    domains = {_get_domain(r) for r in results}
    tier1_sources = {r.get("source") or r.get("source_id", "") for r in results if _is_tier1_source(r)}
    n = len(results)

    # Check source concentration
    source_counts: dict[str, int] = {}
    for r in results:
        src = r.get("source") or r.get("source_id", "")
        source_counts[src] = source_counts.get(src, 0) + 1

    max_pct = max(count / n for count in source_counts.values()) if n > 0 else 0.0

    meets = (
        len(domains) >= 4
        and len(tier1_sources) >= 2
        and max_pct <= 0.40
    )
    return meets, len(domains), len(tier1_sources)


def _enforce_diversity(results: list[dict], force_tier1_count: int = 0) -> list[dict]:
    """
    Enforce diversity constraints on a result set:
      - No single source > 40% of final results
      - At least 2 Tier-1 sources represented
      - At least 4 distinct domains
      - If force_tier1_count > 0: force include that many Tier-1 sources (crisis override)

    Returns a filtered/reordered list preserving ranking order where possible.
    """
    if not results:
        return results

    n = len(results)
    max_per_source = max(1, int(n * 0.40))

    source_counts: dict[str, int] = {}
    selected: list[dict] = []

    # First pass: respect cap per source
    overflow: list[dict] = []
    for r in results:
        src = r.get("source") or r.get("source_id", "")
        if source_counts.get(src, 0) < max_per_source:
            selected.append(r)
            source_counts[src] = source_counts.get(src, 0) + 1
        else:
            overflow.append(r)

    # Second pass: if we lost Tier-1 sources due to capping, try to add them from overflow
    tier1_in_selected = sum(1 for r in selected if _is_tier1_source(r))
    min_tier1 = max(2, force_tier1_count)
    if tier1_in_selected < min_tier1:
        for r in overflow:
            if _is_tier1_source(r) and tier1_in_selected < min_tier1:
                selected.append(r)
                tier1_in_selected += 1

    # Re-sort by safety score
    selected.sort(key=lambda x: x.get("_safety_score", 0.0), reverse=True)
    return selected


async def _fan_out_sources(
    query: str,
    source_ids: set[str],
    timeout: float,
) -> list[dict]:
    """
    Fan-out search across a set of source IDs in parallel.
    Returns adapted live results with registry metadata.
    All queries run to completion or timeout — no short-circuit.
    """
    registry = get_registry()
    ls_configs = load_live_search_configs()

    # Resolve source entries for the requested IDs
    sources = []
    for source_id in source_ids:
        entry = registry.get_source_by_key(source_id)
        if entry is not None and entry.source_id in ls_configs:
            sources.append(entry)
        else:
            # Try by source_id directly
            for s in registry.get_active_sources():
                if s.source_id == source_id and s.source_id in ls_configs:
                    sources.append(s)
                    break

    if not sources:
        log.warning("safety_expanded: no live search configs found for sources=%s", source_ids)
        return []

    log.info("safety_expanded fan-out across %d sources: %s",
             len(sources), [s.source_id for s in sources])

    raw = await live_search_all(query, sources, ls_configs, timeout=timeout)
    adapted = adapt_live_results(raw)

    # Enrich with registry metadata for ranking
    for item in adapted:
        _enrich_with_registry(item)

    return adapted


async def _generate_safety_extras(
    query: str,
    results: list[dict],
    intent_type: str | None,
) -> dict:
    """
    Generate safety extras for self_harm intent.
    Returns dict with: cross_source_consensus, key_clinical_guidance, urgent_help_section.
    LLM failures → field: null, fallback_message: "Insufficient model response"
    urgent_help_section → ALWAYS present via hardcoded fallback.
    """
    extras: dict = {
        "cross_source_consensus": None,
        "cross_source_consensus_fallback": None,
        "key_clinical_guidance": None,
        "key_clinical_guidance_fallback": None,
        "urgent_help_section": URGENT_HELP_FALLBACK,
    }

    if not results:
        extras["cross_source_consensus_fallback"] = "Insufficient model response"
        extras["key_clinical_guidance_fallback"] = "Insufficient model response"
        return extras

    # Build context from top results
    top = results[:7]
    sources_text = "\n".join(
        f"[{i+1}] {r.get('title','(untitled)')} ({r.get('source_name') or r.get('source','')}): "
        f"{(r.get('description') or r.get('snippet',''))[:300]}"
        for i, r in enumerate(top)
    )

    consensus_prompt = (
        "You are a clinical information assistant for a caregiver platform. "
        "Below are search results from multiple trusted medical authorities about "
        f"the query: {query!r}\n\n"
        f"Sources:\n{sources_text}\n\n"
        "Provide a brief (3-5 sentence) consensus summary of what these sources "
        "say, noting any key agreements or important points. Focus on evidence-based "
        "guidance. Do not add new information not in the sources. "
        "Output plain text only."
    )

    guidance_prompt = (
        "You are a clinical information assistant for a caregiver platform. "
        "Based on the following evidence about "
        f"{query!r}:\n\n"
        f"Sources:\n{sources_text}\n\n"
        "List 3-5 specific, actionable clinical guidance points for caregivers. "
        "Format as a numbered list. Each point must be grounded in the sources above. "
        "Do NOT suggest specific medications. Focus on when to seek help, "
        "safe de-escalation, and monitoring. Output plain text only."
    )

    urgent_prompt = (
        "You are a clinical information assistant. "
        f"For a caregiver whose child may be experiencing {query!r}, "
        "write 2-3 sentences describing WHEN TO SEEK URGENT HELP. "
        "Must include: specific warning signs, emergency contact (911), "
        "and mental health crisis line (988 US). Output plain text only."
    )

    async def _call_llm(prompt: str, timeout_s: float = 20) -> str | None:
        """Run claude -p with the given prompt. Returns output or None on failure."""
        import asyncio
        import os
        try:
            env = os.environ.copy()
            env.pop("CLAUDECODE", None)
            proc = await asyncio.create_subprocess_exec(
                "claude", "--disable-slash-commands", "-p", prompt,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            try:
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.communicate()
                log.warning("safety_expanded LLM timeout after %.0fs", timeout_s)
                return None
            if proc.returncode != 0:
                return None
            output = stdout.decode(errors="replace").strip()
            return output if output else None
        except FileNotFoundError:
            log.warning("safety_expanded: claude CLI not found")
            return None
        except Exception as e:
            log.warning("safety_expanded LLM error: %s", e)
            return None

    # Run all three LLM calls concurrently
    consensus_text, guidance_text, urgent_text = await asyncio.gather(
        _call_llm(consensus_prompt, timeout_s=20),
        _call_llm(guidance_prompt, timeout_s=20),
        _call_llm(urgent_prompt, timeout_s=15),
        return_exceptions=False,
    )

    # cross_source_consensus
    if consensus_text:
        extras["cross_source_consensus"] = consensus_text
    else:
        extras["cross_source_consensus_fallback"] = "Insufficient model response"
        log.warning("safety_expanded: cross_source_consensus LLM failed")

    # key_clinical_guidance
    if guidance_text:
        extras["key_clinical_guidance"] = guidance_text
    else:
        extras["key_clinical_guidance_fallback"] = "Insufficient model response"
        log.warning("safety_expanded: key_clinical_guidance LLM failed")

    # urgent_help_section — ALWAYS present; LLM output preferred, hardcoded fallback otherwise
    if urgent_text:
        extras["urgent_help_section"] = urgent_text
    else:
        log.warning("safety_expanded: urgent_help_section LLM failed — using hardcoded fallback")
        extras["urgent_help_section"] = URGENT_HELP_FALLBACK

    return extras


async def run_safety_expanded_search(
    query: str,
    intent_type: str | None = None,
    child_id: str = "default",
) -> dict:
    """
    Run SAFETY_EXPANDED_MODE search.

    Returns a dict compatible with SafetySearchResponse:
      results: list[dict]  (SearchResult-compatible)
      safety_flag: True
      safety_incomplete: bool
      extras: dict | None  (for self_harm intent)
      search_time_ms: int
      search_mode: "safety_expanded"
    """
    t0 = time.monotonic()
    is_crisis = _has_crisis_language(query)
    is_self_harm = (intent_type or "").lower() in ("self_harm", "suicide")

    log.info(
        "safety_expanded START query=%r intent=%s crisis=%s child_id=%s",
        query, intent_type, is_crisis, child_id,
    )

    # ── Phase 1: fan-out across ALL 7 Tier-1 sources ────────────────────────
    log.info("safety_expanded PHASE1: fan-out across all Tier-1 sources")

    # Use asyncio.wait with total budget timeout
    phase1_task = asyncio.create_task(
        _fan_out_sources(query, _TIER1_SOURCE_IDS, timeout=_SITE_TIMEOUT)
    )
    try:
        done, pending = await asyncio.wait(
            {phase1_task}, timeout=_TOTAL_BUDGET
        )
        if pending:
            log.warning("safety_expanded PHASE1: timed out, cancelling %d tasks", len(pending))
            for t in pending:
                t.cancel()
                try:
                    await t
                except (asyncio.CancelledError, Exception):
                    pass
        phase1_results = phase1_task.result() if phase1_task in done else []
    except Exception as e:
        log.error("safety_expanded PHASE1 error: %s", e)
        phase1_results = []

    elapsed_p1 = int((time.monotonic() - t0) * 1000)
    log.info("safety_expanded PHASE1 done: %d results in %dms", len(phase1_results), elapsed_p1)

    # Apply safety ranking to Phase 1 results
    for item in phase1_results:
        item["_safety_score"] = _safety_score(item)

    # ── Check coverage after Phase 1 ────────────────────────────────────────
    constraints_met, domain_count, tier1_count = _check_diversity(phase1_results)
    insufficient = not constraints_met or domain_count < 4 or tier1_count < 2

    phase2_results: list[dict] = []
    if insufficient:
        log.info(
            "safety_expanded PHASE1 insufficient coverage: domains=%d tier1=%d — triggering Phase 2",
            domain_count, tier1_count,
        )
        # ── Phase 2: add ALL 4 Tier-2 sources ───────────────────────────────
        remaining_budget = max(1.0, _TOTAL_BUDGET - (time.monotonic() - t0))
        phase2_task = asyncio.create_task(
            _fan_out_sources(query, _TIER2_SOURCE_IDS, timeout=_SITE_TIMEOUT)
        )
        try:
            done2, pending2 = await asyncio.wait(
                {phase2_task}, timeout=remaining_budget
            )
            if pending2:
                log.warning("safety_expanded PHASE2: timed out")
                for t in pending2:
                    t.cancel()
                    try:
                        await t
                    except (asyncio.CancelledError, Exception):
                        pass
            phase2_results = phase2_task.result() if phase2_task in done2 else []
        except Exception as e:
            log.error("safety_expanded PHASE2 error: %s", e)
            phase2_results = []

        for item in phase2_results:
            item["_safety_score"] = _safety_score(item)

        elapsed_p2 = int((time.monotonic() - t0) * 1000)
        log.info("safety_expanded PHASE2 done: %d results in %dms", len(phase2_results), elapsed_p2)
    else:
        log.info(
            "safety_expanded PHASE1 sufficient: domains=%d tier1=%d — skipping Phase 2",
            domain_count, tier1_count,
        )

    # ── Merge all results ────────────────────────────────────────────────────
    all_results = phase1_results + phase2_results

    # Deduplicate by URL
    seen_urls: set[str] = set()
    deduped: list[dict] = []
    for r in all_results:
        url = r.get("url", "")
        if url and url in seen_urls:
            continue
        seen_urls.add(url)
        deduped.append(r)

    # Sort by safety score
    deduped.sort(key=lambda x: x.get("_safety_score", 0.0), reverse=True)

    # ── Diversity enforcement ────────────────────────────────────────────────
    force_tier1 = 3 if is_crisis else 2
    final_results = _enforce_diversity(deduped, force_tier1_count=force_tier1)

    # ── Check final constraints ──────────────────────────────────────────────
    final_constraints_met, final_domains, final_tier1 = _check_diversity(final_results)

    # If still insufficient after Phase 1+2: return Tier-1 only, set safety_incomplete
    safety_incomplete = False
    if not final_constraints_met:
        log.warning(
            "safety_expanded: constraints not met after Phase1+2 (domains=%d tier1=%d) "
            "— returning Tier-1 only, safety_incomplete=True",
            final_domains, final_tier1,
        )
        safety_incomplete = True
        # Fall back to Tier-1 results only, sorted by authority
        tier1_only = [r for r in deduped if _is_tier1_source(r)]
        final_results = sorted(tier1_only, key=lambda x: x.get("_safety_score", 0.0), reverse=True)

    # Clean internal score key before returning
    for r in final_results:
        r.pop("_safety_score", None)
        # Populate combined_score with safety score for serialization
        r["combined_score"] = round(_safety_score(r), 6)

    # Limit to 5-12 results
    final_results = final_results[:12]

    elapsed_ms = int((time.monotonic() - t0) * 1000)
    log.info(
        "safety_expanded DONE: %d results, incomplete=%s, domains=%d, tier1=%d, elapsed=%dms",
        len(final_results), safety_incomplete, final_domains, final_tier1, elapsed_ms,
    )

    # ── Generate extras for self_harm intent ────────────────────────────────
    extras = None
    if is_self_harm or is_crisis:
        log.info("safety_expanded: generating extras for self_harm/crisis query")
        extras = await _generate_safety_extras(query, final_results, intent_type)

    return {
        "results": final_results,
        "safety_flag": True,
        "safety_incomplete": safety_incomplete,
        "extras": extras,
        "search_time_ms": elapsed_ms,
        "search_mode": "safety_expanded",
    }
