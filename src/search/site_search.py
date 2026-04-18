from __future__ import annotations
"""
Live site search engine (P9).

Searches official sites DIRECTLY using each site's own API or search endpoint.
No Google. No Bing. No third-party search engines.

Methods:
  - "api": JSON API (PubMed, Europe PMC, Semantic Scholar, etc.)
  - "site_search": HTML search page (CDC, NHS, WordPress sites, etc.)
  - "sitemap_index": small sites — fetch all pages, search locally
"""

import asyncio
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

import httpx

from ..sources.registry import get_registry, SourceEntry

log = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "sources.json"

# Module-level cache for live search configs
_ls_configs: dict[str, dict] | None = None


def load_live_search_configs() -> dict[str, dict]:
    """Load live_search config for each source from sources.json.

    Returns: {source_id: live_search_dict} for sources that have it.
    Cached at module level.
    """
    global _ls_configs
    if _ls_configs is not None:
        return _ls_configs

    _ls_configs = {}
    try:
        data = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
        for s in data.get("sources", []):
            ls = s.get("live_search")
            if ls and isinstance(ls, dict) and ls.get("method"):
                _ls_configs[s["source_id"]] = ls
    except Exception as e:
        log.warning("Failed to load live_search configs: %s", e)

    log.info("Live search configs loaded: %d sites", len(_ls_configs))
    return _ls_configs


# ── JSON API result extraction ───────────────────────────────────────────────

def _resolve_dotpath(obj: dict, path: str):
    """Resolve a dot-separated path like 'a.b.c' against a nested dict."""
    for key in path.split("."):
        if isinstance(obj, dict):
            obj = obj.get(key)
        else:
            return None
    return obj


def _extract_json_results(data: dict | list, config: dict) -> list[dict]:
    """Extract results from a JSON API response using config paths."""
    # Navigate to result array via dot-separated path
    result_path = config.get("result_path", "")
    obj = data
    if result_path:
        for key in result_path.split("."):
            if isinstance(obj, dict):
                obj = obj.get(key, [])
            else:
                obj = []
                break

    if not isinstance(obj, list):
        obj = [obj] if obj else []

    results = []
    title_field = config.get("title_field", "title")
    url_field = config.get("url_field")
    url_template = config.get("url_template")
    snippet_field = config.get("snippet_field", "")
    max_results = config.get("max_results", 5)

    for item in obj[:max_results]:
        if not isinstance(item, dict):
            continue

        # Use dot-path resolution for nested fields (e.g. ClinicalTrials)
        title = _resolve_dotpath(item, title_field) if "." in title_field else item.get(title_field, "")
        # Some APIs (Crossref) return title as a list: ["Title text"]
        if isinstance(title, list):
            title = title[0] if title else ""
        if not title:
            continue

        # Build URL
        url = ""
        if url_field:
            url = (_resolve_dotpath(item, url_field) if "." in url_field else item.get(url_field, "")) or ""
        elif url_template:
            try:
                # Flatten nested dicts for template substitution
                url = url_template.format(**item)
            except (KeyError, IndexError):
                url = ""

        snippet = ""
        if snippet_field:
            snippet = (_resolve_dotpath(item, snippet_field) if "." in snippet_field else item.get(snippet_field, "")) or ""
            if isinstance(snippet, list):
                snippet = " ".join(str(s) for s in snippet)

        results.append({
            "title": str(title)[:200],
            "url": str(url),
            "snippet": str(snippet)[:500],
        })

    return results


# ── HTML site search result extraction ───────────────────────────────────────

def _extract_html_results(html: str, config: dict) -> list[dict]:
    """Extract results from an HTML search page using regex patterns."""
    results = []
    max_results = config.get("max_results", 5)

    # Primary: use configured regex patterns
    result_pattern = config.get("result_pattern")
    snippet_pattern = config.get("snippet_pattern")

    if result_pattern:
        # result_pattern should capture (url, title) or (title, url)
        matches = re.findall(result_pattern, html, re.DOTALL | re.IGNORECASE)
        for match in matches[:max_results]:
            if isinstance(match, tuple) and len(match) >= 2:
                url, title = match[0], match[1]
                # Clean HTML entities
                title = re.sub(r"<[^>]+>", "", title).strip()
                title = _unescape_html(title)
                results.append({"title": title[:200], "url": url, "snippet": ""})
    else:
        # Generic fallback: find links with substantial text
        link_pattern = r'<a[^>]+href="(https?://[^"]+)"[^>]*>([^<]{10,})</a>'
        matches = re.findall(link_pattern, html, re.IGNORECASE)
        seen_urls = set()
        for url, title in matches:
            if url in seen_urls:
                continue
            seen_urls.add(url)
            title = _unescape_html(title.strip())
            # Skip navigation/footer links
            if len(title) < 15 or title.lower() in ("read more", "learn more", "click here"):
                continue
            results.append({"title": title[:200], "url": url, "snippet": ""})
            if len(results) >= max_results:
                break

    # Extract snippets if pattern provided
    if snippet_pattern and results:
        snippet_matches = re.findall(snippet_pattern, html, re.DOTALL | re.IGNORECASE)
        for i, snippet in enumerate(snippet_matches):
            if i < len(results):
                cleaned = re.sub(r"<[^>]+>", " ", snippet).strip()
                cleaned = re.sub(r"\s+", " ", cleaned)
                results[i]["snippet"] = _unescape_html(cleaned)[:500]

    return results


def _unescape_html(text: str) -> str:
    """Unescape common HTML entities."""
    replacements = {
        "&amp;": "&", "&lt;": "<", "&gt;": ">",
        "&quot;": '"', "&#39;": "'", "&nbsp;": " ",
        "&#8211;": "–", "&#8212;": "—", "&#8217;": "'",
    }
    for entity, char in replacements.items():
        text = text.replace(entity, char)
    return text


# ── Sitemap index (small sites) ─────────────────────────────────────────────

# Cache fetched sitemaps per day
_sitemap_cache: dict[str, tuple[str, list[str]]] = {}  # domain -> (date, [urls])


async def _search_sitemap_index(
    query: str,
    config: dict,
    source: SourceEntry,
    client: httpx.AsyncClient,
) -> list[dict]:
    """For small sites: fetch all pages, search text for query relevance."""
    base_url = config.get("search_url", "")
    if not base_url:
        return []

    domain = source.domain
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Check sitemap cache
    if domain in _sitemap_cache and _sitemap_cache[domain][0] == today:
        page_urls = _sitemap_cache[domain][1]
    else:
        # Discover pages from the base URL
        try:
            resp = await client.get(base_url, follow_redirects=True)
            if resp.status_code != 200:
                return []
            # Extract all internal links
            links = re.findall(r'href="(/[^"]*)"', resp.text)
            page_urls = list({base_url.rstrip("/") + link for link in links if not link.startswith("#")})
            page_urls = page_urls[:30]  # cap at 30 pages
            _sitemap_cache[domain] = (today, page_urls)
        except Exception as e:
            log.debug("Sitemap discovery failed for %s: %s", domain, e)
            return []

    # Search pages for query relevance
    query_terms = set(query.lower().split())
    results = []
    max_results = config.get("max_results", 5)

    for page_url in page_urls[:20]:
        try:
            resp = await client.get(page_url, follow_redirects=True)
            if resp.status_code != 200:
                continue
            text = re.sub(r"<[^>]+>", " ", resp.text)
            text_lower = text.lower()

            # Check relevance: at least 2 query terms appear
            matched = sum(1 for t in query_terms if t in text_lower)
            if matched < 2:
                continue

            # Extract title from <title> tag
            title_match = re.search(r"<title>([^<]+)</title>", resp.text, re.IGNORECASE)
            title = _unescape_html(title_match.group(1)) if title_match else page_url

            # Snippet: first 300 chars of cleaned text
            cleaned = re.sub(r"\s+", " ", text).strip()
            snippet = cleaned[:300]

            results.append({"title": title[:200], "url": page_url, "snippet": snippet})
            if len(results) >= max_results:
                break
        except Exception:
            continue

    return results


# ── Core search functions ────────────────────────────────────────────────────

async def live_search_site(
    query: str,
    source: SourceEntry,
    live_config: dict,
    timeout: float = 5.0,
) -> list[dict]:
    """Search a single site using its configured method.

    Returns: list of {title, url, snippet, source_id, lang}
    """
    method = live_config.get("method", "")
    search_url = live_config.get("search_url", "")

    # Substitute {q} in URL path if present (e.g., DOAJ path-based API)
    if "{q}" in search_url:
        search_url = search_url.replace("{q}", query)

    if not method or not search_url:
        return []

    try:
        async with httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": "AutismSearchBot/1.0 (research; contact@example.com)"},
        ) as client:

            if method == "api":
                # Build API request
                params = {}
                for k, v in live_config.get("params_template", {}).items():
                    params[k] = v.replace("{q}", query) if isinstance(v, str) else v

                resp = await client.get(search_url, params=params)
                resp.raise_for_status()
                data = resp.json()
                raw_results = _extract_json_results(data, live_config)

            elif method == "site_search":
                # Build site search request
                params = {}
                for k, v in live_config.get("params_template", {}).items():
                    params[k] = v.replace("{q}", query) if isinstance(v, str) else v

                resp = await client.get(search_url, params=params)
                resp.raise_for_status()
                raw_results = _extract_html_results(resp.text, live_config)

            elif method == "sitemap_index":
                raw_results = await _search_sitemap_index(query, live_config, source, client)

            else:
                log.warning("Unknown live search method: %s for %s", method, source.source_id)
                return []

        # Tag each result with source metadata
        tagged = []
        for r in raw_results:
            tagged.append({
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": r.get("snippet", ""),
                "source_id": source.source_id,
                "lang": source.language,
            })

        log.info("LIVE %s: %d results via %s", source.source_id, len(tagged), method)
        return tagged

    except httpx.TimeoutException:
        log.warning("LIVE %s: timeout after %.1fs", source.source_id, timeout)
        return []
    except Exception as e:
        log.warning("LIVE %s: failed — %s", source.source_id, e)
        return []


async def live_search_all(
    query: str,
    sources: list[SourceEntry],
    ls_configs: dict[str, dict],
    timeout: float = 5.0,
) -> list[dict]:
    """Search multiple sites concurrently.

    Runs live_search_site() for each source in parallel.
    Failed sites are logged but don't block others.
    Returns combined results from all sites.
    """
    if not sources:
        return []

    async def _search_one(source: SourceEntry) -> list[dict]:
        config = ls_configs.get(source.source_id)
        if not config:
            return []
        return await live_search_site(query, source, config, timeout)

    tasks = [_search_one(s) for s in sources]
    results_lists = await asyncio.gather(*tasks, return_exceptions=True)

    all_results = []
    succeeded = 0
    for i, r in enumerate(results_lists):
        if isinstance(r, list):
            all_results.extend(r)
            if r:
                succeeded += 1
        elif isinstance(r, Exception):
            log.warning("LIVE %s: task exception — %s", sources[i].source_id, r)

    log.info("LIVE total: %d results from %d/%d sites",
             len(all_results), succeeded, len(sources))
    return all_results


def adapt_live_results(raw_results: list[dict]) -> list[dict]:
    """Convert live search results to SearchResult-compatible dicts."""
    now = datetime.now(timezone.utc)
    adapted = []

    for r in raw_results:
        adapted.append({
            "id": -1,
            "external_id": None,
            "source": r.get("source_id", ""),
            "surface_key": r.get("source_id", ""),
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "description": r.get("snippet", ""),
            "content_body": None,
            "author": None,
            "authors_json": None,
            "journal": None,
            "open_access": None,
            "doi": None,
            "published_at": None,
            "collected_at": now,
            "lang": r.get("lang", "en"),
            "engagement": None,
            "semantic_score": 0.0,
            "keyword_score": 0.0,
            "combined_score": 0.0,
            "is_live_result": True,
        })

    return adapted
