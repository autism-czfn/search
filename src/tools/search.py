from __future__ import annotations
"""
CLI wrapper: hybrid search against the local DB.
Called by claude -p via Bash during the agent loop.

Usage:
    python -m src.tools.search "autism diagnosis" [--source pubmed] [--days 365] [--limit 10]

Output: JSON array to stdout. Exit 0 on success, exit 1 on error.
"""

import argparse
import asyncio
import json
import logging
import sys

import asyncpg

from ..config import settings
from ..db import init_connection
from ..embedder import embed_query
from ..search.hybrid import merge_and_rerank
from ..search.keyword import keyword_search
from ..search.semantic import semantic_search

log = logging.getLogger(__name__)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid local DB search CLI")
    parser.add_argument("query", help="Search query text")
    parser.add_argument("--source", default=None, help="Filter by source (e.g. pubmed, reddit)")
    parser.add_argument("--days",   type=int, default=None, help="Only items published within N days")
    parser.add_argument("--limit",  type=int, default=10,   help="Max results to return (default 10)")
    args = parser.parse_args()

    pool = None   # initialise before try so finally block is always safe
    try:
        pool = await asyncpg.create_pool(
            settings.database_url,
            init=init_connection,
            min_size=1,
            max_size=3,
        )

        embedding = await embed_query(args.query)
        sem_results: list[dict] = []
        if embedding is not None:
            sem_results = await semantic_search(
                pool, embedding, args.limit * 2, args.source, args.days,
                official_only=True,
            )
        kw_results = await keyword_search(
            pool, args.query, args.limit * 2, args.source, args.days,
            official_only=True,
        )

        if not sem_results and not kw_results:
            print(json.dumps([]))
            return

        merged, _ = merge_and_rerank(sem_results, kw_results, top_n=args.limit,
                                     official_only=True)

        # Serialise — datetime fields are not JSON-serialisable
        output = []
        for r in merged:
            item = dict(r)
            for key in ("published_at", "collected_at", "embedded_at"):
                if item.get(key) is not None:
                    item[key] = item[key].isoformat()
            item.pop("embedding", None)   # drop raw vector — too large for Claude
            output.append(item)

        print(json.dumps(output, indent=2))

    except Exception as e:
        log.error("search CLI error: %s", e)
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        if pool is not None:
            await pool.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    asyncio.run(main())
