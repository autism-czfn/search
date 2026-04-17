from __future__ import annotations
"""
Curated evidence retrieval for the Insights tab (P8).

Returns clean, parent-safe EvidenceCard objects with NO raw scores,
NO crawl metadata, NO retrieval traces. Only trusted sources
(authority_tier 1/2/3) are included.
"""

import asyncio
import json
import logging
import re

from ..api.models import EvidenceCard, InsightRecommendation
from ..embedder import embed_query
from ..search.keyword import keyword_search
from ..search.semantic import semantic_search
from ..search.hybrid import merge_and_rerank
from ..search.live_fallback import determine_route, run_live_search
from .sources import is_evidence_source, get_source_entry

log = logging.getLogger(__name__)

# Max summary length in EvidenceCard (prevents leaking full content)
MAX_SUMMARY_LEN = 300

# LLM timeout for recommendation generation
RECOMMEND_TIMEOUT = 15


def _confidence_tag(item: dict) -> str | None:
    """Deterministic confidence tag from ranking score + authority tier."""
    entry = get_source_entry(item.get("surface_key") or item.get("source") or item.get("source_id") or "")
    tier = entry.authority_tier if entry else None
    score = item.get("combined_score", 0)
    if tier == 1 and score >= 0.5:
        return "high"
    if score >= 0.4:
        return "medium"
    return None


def _to_evidence_card(item: dict) -> EvidenceCard:
    """Convert a search result dict to a clean EvidenceCard."""
    source_key = item.get("surface_key") or item.get("source") or item.get("source_id") or ""
    entry = get_source_entry(source_key)

    summary_raw = item.get("description") or item.get("snippet") or ""
    summary = summary_raw[:MAX_SUMMARY_LEN]

    return EvidenceCard(
        source_title=item.get("title") or "(untitled)",
        organization_name=entry.organization_name if entry else source_key,
        publication_type=entry.publication_type if entry else None,
        link=item.get("url"),
        summary=summary,
        confidence_tag=_confidence_tag(item),
        is_live_result=bool(item.get("is_live_result")),
    )


async def fetch_curated_evidence(
    pool,
    query: str,
    limit: int = 5,
) -> list[EvidenceCard]:
    """
    Fetch curated evidence cards for a query.

    Pipeline:
      1. Embed query → semantic search → keyword search → merge/rerank
      2. Filter: keep only trusted sources (authority_tier 1/2/3)
      3. Take top `limit` results
      4. Convert to clean EvidenceCard objects (no raw scores)
    """
    fetch_n = limit * 3  # fetch more to allow filtering

    embedding = await embed_query(query)

    sem_results = []
    if embedding is not None:
        sem_results = await semantic_search(pool, embedding, fetch_n, None, None)

    kw_results = await keyword_search(pool, query, fetch_n, None, None)

    if not sem_results and not kw_results:
        return []

    merged, _mode = merge_and_rerank(sem_results, kw_results, top_n=fetch_n)

    # ── Live search: supplement if local results are weak ────────────────
    route = determine_route(merged, query)
    if route in ("HYBRID", "LIVE_ONLY"):
        live_results = await run_live_search(query, merged)
        if live_results:
            log.info("evidence: live search added %d results (route=%s)", len(live_results), route)
            merged = merged + live_results

    # Filter to evidence-quality sources only.
    # Check surface_key first (actual source identity), then fall back to source
    # (which may just be the collector platform name like "html_crawl").
    filtered = [
        item for item in merged
        if is_evidence_source(item.get("surface_key") or item.get("source") or item.get("source_id") or "")
    ]

    # Separate local and live results so live results aren't cut off by limit.
    # Local results: take up to `limit`.  Live results: always included.
    local = [item for item in filtered if not item.get("is_live_result")]
    live = [item for item in filtered if item.get("is_live_result")]
    selected = local[:limit] + live[:limit]
    return [_to_evidence_card(item) for item in selected]


def _extract_json_array(text: str) -> list[dict] | None:
    """Robustly extract a JSON array from LLM output that may contain markdown fences."""
    # Step 1: strip markdown code fences
    cleaned = re.sub(r"```json?\s*", "", text)
    cleaned = re.sub(r"```\s*$", "", cleaned, flags=re.MULTILINE)
    cleaned = cleaned.strip()

    # Step 2: find first [ and last ]
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return None

    # Step 3: extract and parse
    try:
        data = json.loads(cleaned[start : end + 1])
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass
    return None


async def generate_recommendations(
    pattern: dict,
    evidence: list[EvidenceCard],
) -> list[InsightRecommendation]:
    """
    Generate 1-3 concrete parent actions using claude -p.

    Falls back to empty list on any failure.
    """
    evidence_text = "\n".join(
        f"- {card.organization_name}: {card.summary}" for card in evidence
    )

    trigger = pattern.get("trigger", "unknown")
    outcome = pattern.get("outcome", "unknown")
    sample_count = pattern.get("sample_count", 0)
    pct = pattern.get("co_occurrence_pct", 0)

    prompt = (
        "Based on the following behavioral pattern and evidence, "
        "suggest 1-3 concrete, testable actions a parent can try this week.\n\n"
        f"Pattern: {trigger} -> {outcome}\n"
        f"(observed in {sample_count} episodes, {pct}%)\n\n"
        f"Evidence:\n{evidence_text}\n\n"
        "Rules:\n"
        "- Each action must be specific and measurable\n"
        "- Each must be something a parent can do at home\n"
        "- Each must connect to the detected pattern\n"
        "- Do NOT suggest medical treatments or diagnoses\n"
        '- Output as JSON array: [{"text": "..."}]\n'
    )

    try:
        import os

        env = os.environ.copy()
        env.pop("CLAUDECODE", None)

        proc = await asyncio.create_subprocess_exec(
            "claude", "--disable-slash-commands",
            "--dangerously-skip-permissions",
            "-p", prompt,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=RECOMMEND_TIMEOUT
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            log.warning("Recommendation LLM timed out after %ds", RECOMMEND_TIMEOUT)
            return []

        if proc.returncode != 0:
            log.warning("Recommendation LLM failed: exit=%d", proc.returncode)
            return []

        output = stdout.decode(errors="replace").strip()
        if not output:
            return []

        # Robust JSON extraction
        data = _extract_json_array(output)
        if data is None:
            log.warning("Recommendation LLM output not parseable as JSON array")
            return []

        # Validate structure
        results = []
        for item in data:
            if isinstance(item, dict) and "text" in item:
                results.append(InsightRecommendation(text=item["text"]))
        return results[:3]  # cap at 3

    except FileNotFoundError:
        log.warning("claude CLI not found — skipping recommendations")
        return []
    except Exception as e:
        log.warning("Recommendation generation failed: %s", e)
        return []
