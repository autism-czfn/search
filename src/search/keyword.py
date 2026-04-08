from __future__ import annotations
"""
PostgreSQL full-text search over crawled_items.
Searches title + description + content_body via tsvector / plainto_tsquery.
Returns up to *fetch_limit* rows ordered by ts_rank DESC.
"""

import logging
import asyncpg

log = logging.getLogger(__name__)

# Columns fetched by every search variant (no embedding, no raw_payload)
_SELECT = """
    id, external_id, source, surface_key,
    title, url, description, content_body,
    author, authors_json, journal, open_access, doi,
    published_at, collected_at, lang, engagement
"""

_FTS_VECTOR = """
    to_tsvector('english',
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
) -> list[dict]:
    """
    Full-text search. Returns a list of dicts with a 'keyword_score' key.
    Returns [] if no rows match or on any error.
    """
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

    filters_sql = "\n        ".join(extra_filters)

    sql = f"""
        SELECT
            {_SELECT},
            ts_rank({_FTS_VECTOR}, plainto_tsquery('english', $1)) AS keyword_score
        FROM crawled_items
        WHERE {_FTS_VECTOR} @@ plainto_tsquery('english', $1)
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
