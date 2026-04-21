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
from urllib.parse import urlparse

import httpx

from ..sources.registry import get_registry, SourceEntry
from .ranking import _TRIGGER_TERMS

# ── Non-content URL path segments to discard ────────────────────────────────
# Checked against every segment of the path, so /en/shopaap/cart/ is caught
# just as reliably as /cart/ directly.
_BLOCKED_SEGMENTS = {
    # Auth / account
    "login", "logout", "register", "signup", "sign-up",
    "account", "my-account", "profile", "password",
    # Shop / commerce
    "cart", "checkout", "shop", "shopaap", "store",
    "shipping", "payment", "order", "orders",
    # Membership / org admin
    "membership", "membership-application", "join", "join-aap",
    "donate", "ways-to-give", "giving", "fundraising",
    "career-resources", "careers", "jobs",
    # Nav / utility pages
    "contact", "contact-us", "about", "about-us",
    "sitemap", "site-index", "rss", "feed",
    "newsletter", "subscribe", "unsubscribe",
    "accessibility", "privacy", "disclaimer", "ad-disclaimer",
    "terms", "terms-of-use", "legal", "policies", "policy",
    "press", "media", "news-room", "newsroom",
    "social", "share",
    # Learning / platform portals (not content articles)
    "pedialink", "cme",
}

# Minimum URL path depth (segments) for a result to be considered content.
# Blocks bare root paths like "/", "/en", "/us", etc.
_MIN_PATH_DEPTH = 2

# Stop words excluded when building content-check terms from the query.
# These are generic words unlikely to appear in relevant content uniquely.
_CONTENT_STOP_WORDS = {
    "what", "which", "how", "why", "when", "where", "who",
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "can", "could", "should", "would", "will", "may", "might",
    "to", "in", "of", "and", "or", "for", "with", "that", "this",
    "do", "does", "did", "has", "have", "had",
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "they", "it",
    "as", "by", "at", "on", "up", "if", "so", "no", "not", "new",
    "autism", "autistic", "child", "children", "help", "make",
}

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
            val = (_resolve_dotpath(item, url_field) if "." in url_field else item.get(url_field, "")) or ""
            # Handle list-of-dicts (e.g. DOAJ bibjson.link = [{type, url}, ...])
            if isinstance(val, list):
                for entry in val:
                    if isinstance(entry, dict) and entry.get("url"):
                        val = entry["url"]
                        break
                else:
                    val = ""
            url = str(val) if val else ""
        elif url_template:
            try:
                # Support dot-path keys in template e.g. {protocolSection.identificationModule.nctId}
                resolved = url_template
                for m in re.finditer(r"\{([^}]+)\}", url_template):
                    key = m.group(1)
                    value = _resolve_dotpath(item, key) if "." in key else item.get(key, "")
                    resolved = resolved.replace(m.group(0), str(value or ""))
                url = resolved
            except Exception:
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
        seen_urls: set[str] = set()
        for match in matches:
            if len(results) >= max_results:
                break
            if isinstance(match, tuple) and len(match) >= 2:
                url, title = match[0], match[1]
                if url in seen_urls:
                    continue
                seen_urls.add(url)
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


def _is_content_url(url: str, source_domain: str) -> bool:
    """Return True only if the URL belongs to source_domain and looks like real content.

    Rejects:
      - URLs from a different domain (e.g. reddit.com links embedded on a CDC page)
      - URLs whose path starts with a known nav/utility prefix (/contact, /about…)
      - URLs that are too shallow (fewer than _MIN_PATH_DEPTH path segments)
    """
    if not url:
        return False
    try:
        parsed = urlparse(url)
        # Domain check — strip leading "www." for comparison
        url_host = parsed.netloc.lstrip("www.")
        src_host = source_domain.lstrip("www.")
        if url_host != src_host:
            return False
        path = parsed.path.rstrip("/") or "/"
        # Split into segments and check each against the blocked set.
        # This catches /en/shopaap/cart/ just as reliably as /cart/.
        segments = [s for s in path.lower().split("/") if s]
        if len(segments) < _MIN_PATH_DEPTH:
            return False
        if any(seg in _BLOCKED_SEGMENTS for seg in segments):
            return False
        return True
    except Exception:
        return False


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


# ── Content verification ─────────────────────────────────────────────────────

def _build_check_terms(query: str) -> set[str]:
    """Extract meaningful terms from a query for content relevance checking.

    Priority:
      1. Exact trigger terms present in query ("food", "routine")
      2. Stemmed trigger terms — strip trailing 's' to catch plurals
         ("foods" → "food", "routines" → "routine")
      3. All non-stop-word query tokens as a broad fallback
    """
    clean = re.sub(r"[^a-z\s]", "", query.lower())
    words = set(clean.split())

    # 1. Exact trigger match
    exact = words & _TRIGGER_TERMS
    if exact:
        return exact

    # 2. Stemmed trigger match (plurals / simple inflections)
    stemmed = {
        w[:-1] for w in words
        if w.endswith("s") and len(w) > 4 and w[:-1] in _TRIGGER_TERMS
    }
    if stemmed:
        return stemmed

    # 3. Fallback: all meaningful non-stop tokens
    return words - _CONTENT_STOP_WORDS


async def _verify_results_content(
    results: list[dict],
    check_terms: set[str],
    client: httpx.AsyncClient,
    page_timeout: float = 4.0,
) -> list[dict]:
    """Fetch each result URL and keep only those whose page content is relevant.

    Fetches all URLs in parallel. A result is kept if at least one check_term
    appears in the fetched page text. The snippet is replaced with a relevant
    excerpt centred on the first match — better than the search-page snippet.

    Results that time out or return a non-200 status are dropped so that
    unreachable or paywalled pages never appear in Live Sources.

    Args:
        results:      list of {title, url, snippet} dicts from site search
        check_terms:  terms that must appear somewhere in the page content
        client:       shared httpx async client (connection pool reuse)
        page_timeout: per-page read timeout in seconds

    Returns:
        Filtered list with content-verified results and updated snippets.
    """
    if not results or not check_terms:
        return results

    async def _check_one(result: dict) -> dict | None:
        url = result.get("url", "")
        if not url:
            return None
        try:
            resp = await client.get(url, timeout=page_timeout, follow_redirects=True)
            if resp.status_code != 200:
                log.debug("CONTENT DROP %s — HTTP %d", url, resp.status_code)
                return None

            # Strip HTML to plain text
            text = re.sub(r"<[^>]+>", " ", resp.text)
            text = re.sub(r"\s+", " ", text).strip()
            text_lower = text.lower()

            # Must contain at least one check term
            matched = [t for t in check_terms if t in text_lower]
            if not matched:
                log.debug("CONTENT DROP %s — none of %s found", url, check_terms)
                return None

            # Extract snippet centred on the first matched term
            idx = text_lower.find(matched[0])
            start = max(0, idx - 120)
            end = min(len(text), idx + 280)
            snippet = _unescape_html(re.sub(r"\s+", " ", text[start:end]).strip())[:500]

            return {**result, "snippet": snippet}

        except Exception as exc:
            log.debug("CONTENT FETCH failed for %s: %s", url, exc)
            return None

    outcomes = await asyncio.gather(*[_check_one(r) for r in results],
                                    return_exceptions=True)
    kept = [r for r in outcomes if isinstance(r, dict)]
    dropped = len(results) - len(kept)
    if dropped:
        log.info("CONTENT VERIFY: kept %d/%d (dropped %d off-topic or unreachable)",
                 len(kept), len(results), dropped)
    return kept


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
            # Use only scheme+host for link reconstruction (base_url may be a sub-path)
            parsed = urlparse(base_url)
            origin = f"{parsed.scheme}://{parsed.netloc}"
            url_prefix = config.get("url_prefix", "")
            links = re.findall(r'href="(/[^"]*)"', resp.text)
            page_urls = list({
                origin + link
                for link in links
                if not link.startswith("#")
                and "." not in link.split("/")[-1]   # skip .css/.js/.png etc
                and "?" not in link                   # skip query-string hrefs
                and (not url_prefix or link.startswith(url_prefix))
            })
            page_urls = page_urls[:30]  # cap at 30 pages
            _sitemap_cache[domain] = (today, page_urls)
        except Exception as e:
            log.debug("Sitemap discovery failed for %s: %s", domain, e)
            return []

    # Search pages for query relevance
    query_terms = set(query.lower().split())
    min_match = config.get("min_match_terms", 2)
    results = []
    max_results = config.get("max_results", 5)

    for page_url in page_urls[:20]:
        try:
            resp = await client.get(page_url, follow_redirects=True)
            if resp.status_code != 200:
                continue
            text = re.sub(r"<[^>]+>", " ", resp.text)
            text_lower = text.lower()

            # Check relevance: at least min_match_terms query terms appear
            matched = sum(1 for t in query_terms if t in text_lower)
            if matched < min_match:
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
            headers={"User-Agent": "Mozilla/5.0 (compatible; autism-research/1.0)"},
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

            # ── Content verification ───────────────────────────────────────────
            # For api and site_search results: fetch each result URL and verify
            # the page content actually contains query-relevant terms.
            # sitemap_index already does this during page traversal — skip it.
            if method in ("api", "site_search") and raw_results:
                check_terms = _build_check_terms(query)
                if check_terms:
                    log.debug("CONTENT VERIFY %s: checking %d results for terms %s",
                              source.source_id, len(raw_results), check_terms)
                    raw_results = await _verify_results_content(
                        raw_results, check_terms, client, page_timeout=4.0
                    )

        # Tag each result with source metadata, enforcing domain + content-path rules.
        # This rejects nav/footer links (e.g. "Contact CDC") and any external URLs
        # that leaked in via social-share buttons or API url_field pointing elsewhere.
        tagged = []
        rejected = 0
        for r in raw_results:
            url = r.get("url", "")
            if not _is_content_url(url, source.domain):
                rejected += 1
                log.debug("LIVE %s: rejected non-content URL %s", source.source_id, url)
                continue
            tagged.append({
                "title": r.get("title", ""),
                "url": url,
                "snippet": r.get("snippet", ""),
                "source_id": source.source_id,
                "lang": source.language,
            })

        if rejected:
            log.info("LIVE %s: dropped %d non-content URLs, kept %d via %s",
                     source.source_id, rejected, len(tagged), method)
        else:
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
