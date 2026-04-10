from __future__ import annotations
"""
CLI wrapper: live PubMed E-utilities search.
Called by claude -p via Bash during the agent loop.

Usage:
    python -m src.tools.pubmed "autism biomarkers" [--max 5]

Output: JSON array to stdout. Exit 0 on success, exit 1 on error.
"""

import argparse
import asyncio
import json
import logging
import sys

from ..config import settings
from ..search.pubmed import pubmed_search

log = logging.getLogger(__name__)


async def main() -> None:
    parser = argparse.ArgumentParser(description="PubMed E-utilities search CLI")
    parser.add_argument("query", help="PubMed search query (MeSH terms supported)")
    parser.add_argument("--max", type=int, default=5, help="Max articles to return (1–10, default 5)")
    args = parser.parse_args()

    max_results = max(1, min(args.max, 10))   # clamp to [1, 10]

    try:
        articles = await pubmed_search(
            args.query,
            max_results,
            api_key=settings.ncbi_api_key,
        )
        print(json.dumps(articles, indent=2))
    except Exception as e:
        log.error("pubmed CLI error: %s", e)
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    asyncio.run(main())
