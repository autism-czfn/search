from __future__ import annotations
"""
pgvector cosine-similarity search over crawled_items.embedding.
Requires the IVFFlat index to exist and embeddings to be populated.
Returns [] (triggers keyword-only fallback) when:
  - no rows have embeddings yet (cold start)
  - any DB error occurs
"""

import logging
import asyncpg

log = logging.getLogger(__name__)

_SELECT = """
    id, external_id, source, surface_key,
    title, url, description, content_body,
    author, authors_json, journal, open_access, doi,
    published_at, collected_at, lang, engagement
"""


async def semantic_search(
    pool: asyncpg.Pool,
    embedding: list[float],
    fetch_limit: int = 20,
    source: str | None = None,
    days: int | None = None,
) -> list[dict]:
    """
    Cosine-similarity search using pgvector.
    Returns a list of dicts with a 'semantic_score' key (cosine similarity 0–1).
    Returns [] on cold start, no embeddings, or any error.
    """
    params: list = [embedding]
    extra_filters: list[str] = []
    p = 2

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
            1 - (embedding <=> $1::vector) AS semantic_score
        FROM crawled_items
        WHERE embedding IS NOT NULL
        {filters_sql}
        ORDER BY embedding <=> $1::vector
        LIMIT {fetch_limit}
    """

    try:
        async with pool.acquire() as conn:
            # Must be set per-connection before the ANN query for recall tuning.
            # probes/lists ≈ 10% → ~95% recall; increase for higher accuracy.
            await conn.execute("SET ivfflat.probes = 10")
            rows = await conn.fetch(sql, *params)

        if not rows:
            log.info("Semantic search: no embedded rows found (cold start or empty index)")
            return []

        return [dict(r) for r in rows]

    except Exception as e:
        log.warning("Semantic search error: %s — results will be empty", e)
        return []
