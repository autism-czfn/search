from __future__ import annotations
"""
PostgreSQL full-text search over crawled_items.
Searches title + description + content_body via tsvector / plainto_tsquery.
Returns up to *fetch_limit* rows ordered by ts_rank DESC.
"""

import logging
import asyncpg

from ..sources.registry import get_registry

log = logging.getLogger(__name__)

# Columns fetched by every search variant (no embedding, no raw_payload)
_SELECT = """
    id, external_id, source, surface_key,
    title, url, description, content_body,
    author, authors_json, journal, open_access, doi,
    published_at, collected_at, lang, engagement
"""


def _fts_vector(fts_config: str) -> str:
    """Build the tsvector expression for a given FTS config."""
    return f"""
    to_tsvector('{fts_config}',
        coalesce(title, '') || ' ' ||
        coalesce(description, '') || ' ' ||
        coalesce(content_body, '')
    )
"""


async def keyword_search(
    pool: asyncpg.Pool,
    query: str,
    fetch_limit: int = 20,
    source: str | None = None,
    days: int | None = None,
    fts_config: str = "english",
    official_only: bool = False,
) -> list[dict]:
    """
    Full-text search. Returns a list of dicts with a 'keyword_score' key.
    Returns [] if no rows match or on any error.

    Args:
        fts_config: PostgreSQL FTS configuration name.
            'english' (default) — standard English stemming/stopwords.
            'simple' — language-agnostic (splits on whitespace, lowercases).
                       Use for non-English queries (P7 multilingual search).
        official_only: When True, restricts results to authority_tier=1 (official)
            sources only (e.g. CDC, NIH, NHS, NICE). Non-official crawled items
            are excluded at the SQL level for efficiency.
    """
    # Validate fts_config to prevent SQL injection (only allow known configs)
    if fts_config not in ("english", "simple", "french", "german", "spanish"):
        fts_config = "english"

    fts_vec = _fts_vector(fts_config)
    params: list = [query]
    extra_filters: list[str] = []
    p = 2  # next param index

    if source is not None:
        extra_filters.append(f"AND source = ${p}")
        params.append(source)
        p += 1

    if days is not None:
        extra_filters.append(f"AND published_at >= now() - (${p}::int || ' days')::interval")
        params.append(days)
        p += 1

    if official_only:
        official_keys = get_registry().get_official_surface_keys()
        if official_keys:
            extra_filters.append(f"AND surface_key = ANY(${p}::text[])")
            params.append(official_keys)
            p += 1
        else:
            log.warning("keyword_search: official_only=True but registry has no tier-1 sources — filter skipped")

    filters_sql = "\n        ".join(extra_filters)

    sql = f"""
        SELECT
            {_SELECT},
            ts_rank({fts_vec}, plainto_tsquery('{fts_config}', $1)) AS keyword_score
        FROM crawled_items
        WHERE {fts_vec} @@ plainto_tsquery('{fts_config}', $1)
        {filters_sql}
        ORDER BY keyword_score DESC
        LIMIT {fetch_limit}
    """

    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)
        return [dict(r) for r in rows]
    except Exception as e:
        log.error("Keyword search error: %s", e)
        return []
