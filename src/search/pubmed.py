from __future__ import annotations
"""
Async PubMed E-utilities HTTP client.
Returns article metadata as a list of dicts.
Returns [] on any network error, bad response, or empty result set.

Each returned dict has keys:
    pmid, title, journal, pubdate, authors (list[str]), doi, url
"""

import logging

import httpx

log = logging.getLogger(__name__)

ESEARCH_URL  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
PUBMED_BASE  = "https://pubmed.ncbi.nlm.nih.gov/"
HTTP_TIMEOUT = 10    # seconds per request


async def pubmed_search(
    query: str,
    max_results: int = 5,
    api_key: str | None = None,
) -> list[dict]:
    """
    Search PubMed via NCBI E-utilities (esearch → esummary).

    Returns a list of article dicts. Returns [] on any network error,
    bad API response, or empty result set. Does NOT raise.
    """
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:

            # ── Step 1: esearch — get PMIDs ──────────────────────────────────
            esearch_params: dict = {
                "db":      "pubmed",
                "term":    query,
                "retmax":  max_results,
                "retmode": "json",
            }
            if api_key:
                esearch_params["api_key"] = api_key

            resp = await client.get(ESEARCH_URL, params=esearch_params)
            resp.raise_for_status()
            idlist: list[str] = resp.json()["esearchresult"]["idlist"]
            if not idlist:
                return []

            # ── Step 2: esummary — get metadata per PMID ─────────────────────
            esummary_params: dict = {
                "db":      "pubmed",
                "id":      ",".join(idlist),
                "retmode": "json",
            }
            if api_key:
                esummary_params["api_key"] = api_key

            resp = await client.get(ESUMMARY_URL, params=esummary_params)
            resp.raise_for_status()
            result = resp.json()["result"]

            articles = []
            for pmid in idlist:
                try:
                    item = result[pmid]
                    # Use .get() for authors — some records omit "name" on
                    # individual author dicts; direct access would KeyError
                    # and discard the whole article via the per-PMID except.
                    authors_raw = [a.get("name", "") for a in item.get("authors", [])]
                    authors = [n for n in authors_raw if n]
                    doi = next(
                        (
                            x["value"]
                            for x in item.get("articleids", [])
                            if x.get("idtype") == "doi"
                        ),
                        None,
                    )
                    articles.append({
                        "pmid":    item["uid"],
                        "title":   item["title"],
                        "journal": item["source"],
                        "pubdate": item["pubdate"],
                        "authors": authors,
                        "doi":     doi,
                        "url":     f"{PUBMED_BASE}{item['uid']}/",
                    })
                except KeyError:
                    # Skip missing/retracted records without dropping the batch
                    continue

            return articles

    except Exception as e:
        log.warning("pubmed_search FAIL query=%r error=%s", query, e)
        return []
