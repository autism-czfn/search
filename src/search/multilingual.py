from __future__ import annotations
"""
Multilingual search support (P7).

Translates English queries to target languages and searches LOCAL DB
with the translated query. All non-English sources are already crawled
into crawled_items — no external search API needed.

Pipeline:
  1. translate_query() — English → target language
  2. search_local_translated() — keyword + semantic search on local DB
  3. translate_snippets() — translate result snippets back to English

No Google CSE or external search API keys required.
Translation API key (Google Translate or DeepL) needed for query
translation; without it, falls back to English query (degraded quality).
"""

import asyncio
import logging
import os

import httpx

from ..config import settings
from ..sources.registry import get_registry
from ..embedder import embed_query
from .keyword import keyword_search
from .semantic import semantic_search

log = logging.getLogger(__name__)

# Translation cache (in-memory, process-level)
_translation_cache: dict[str, str] = {}

# FTS config mapping for supported languages
_LANG_FTS_CONFIG = {
    "en": "english",
    "fr": "french",
    "de": "german",
    "es": "spanish",
    # Japanese has no built-in PG FTS config — use 'simple' (whitespace split)
    "ja": "simple",
    # Fallback for any other language
}


def _get_fts_config(lang: str) -> str:
    """Get PostgreSQL FTS config for a language. Defaults to 'simple'."""
    return _LANG_FTS_CONFIG.get(lang, "simple")


async def translate_query(query_en: str, target_lang: str) -> str:
    """Translate an English query to the target language.

    Uses Google Translate or DeepL API if TRANSLATION_API_KEY is set.
    Falls back to returning the original English query.
    Caches results in-memory.
    """
    if target_lang == "en":
        return query_en

    cache_key = f"{query_en}|{target_lang}"
    if cache_key in _translation_cache:
        return _translation_cache[cache_key]

    # Use settings (respects defaults and env-var overrides) rather than
    # reading the raw env var directly, which would miss the default "google".
    api_provider = settings.translation_api or ""
    if not api_provider:
        log.debug("TRANSLATION_API not configured — skipping translation")
        return query_en

    api_key = settings.translation_api_key or os.environ.get("TRANSLATION_API_KEY")
    if not api_key:
        log.debug("No TRANSLATION_API_KEY — returning English query as-is")
        return query_en

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            if api_provider == "deepl":
                resp = await client.post(
                    "https://api-free.deepl.com/v2/translate",
                    data={
                        "auth_key": api_key,
                        "text": query_en,
                        "target_lang": target_lang.upper(),
                    },
                )
                resp.raise_for_status()
                translated = resp.json()["translations"][0]["text"]
            else:
                # Google Translate v2
                resp = await client.post(
                    "https://translation.googleapis.com/language/translate/v2",
                    params={"key": api_key},
                    json={
                        "q": query_en,
                        "target": target_lang,
                        "source": "en",
                        "format": "text",
                    },
                )
                resp.raise_for_status()
                translated = resp.json()["data"]["translations"][0]["translatedText"]

        _translation_cache[cache_key] = translated
        log.info("Translated query to %s: %r → %r", target_lang, query_en[:50], translated[:50])
        return translated

    except Exception as e:
        log.warning("Translation failed (en→%s): %s — using English query", target_lang, e)
        return query_en


async def search_local_translated(
    pool,
    query_translated: str,
    lang: str,
    fetch_limit: int = 20,
) -> list[dict]:
    """Search local DB with a translated query.

    Uses keyword_search() with language-appropriate FTS config
    ('simple' for CJK, language-specific for European languages).
    Also runs semantic_search() with embedded translated query
    (nomic-embed-text has some multilingual capability).

    Returns list of result dicts (same format as keyword/semantic search).
    """
    fts_config = _get_fts_config(lang)

    # Keyword search with language-appropriate FTS config
    kw_results = await keyword_search(
        pool, query_translated, fetch_limit,
        source=None, days=None, fts_config=fts_config,
    )
    log.info("Multilingual keyword search (%s, fts=%s): %d results",
             lang, fts_config, len(kw_results))

    # Semantic search with embedded translated query
    sem_results = []
    embedding = await embed_query(query_translated)
    if embedding is not None:
        sem_results = await semantic_search(
            pool, embedding, fetch_limit, source=None, days=None,
        )
        log.info("Multilingual semantic search (%s): %d results", lang, len(sem_results))

    # Combine (dedup by id will happen in merge_and_rerank)
    return kw_results + sem_results


async def translate_snippets(
    results: list[dict],
    source_lang: str,
    target: str = "en",
    max_results: int = 3,
) -> list[dict]:
    """Translate top N result snippets back to English.

    Preserves originals in snippet_original field.
    """
    for r in results[:max_results]:
        snippet = r.get("description") or ""
        if snippet and source_lang != target:
            r["snippet_original"] = snippet
            r["description"] = await translate_query(snippet, target)

    return results


async def run_multilingual_search(
    query_en: str,
    pool,
    target_langs: list[str] | None = None,
    fetch_limit: int = 10,
) -> list[dict]:
    """Run multilingual search across configured non-English sources.

    Translates query to each target language, searches local DB,
    translates result snippets back to English.

    Args:
        query_en: English query text
        pool: asyncpg pool for crawled_items DB
        target_langs: specific languages to search, or None for all
        fetch_limit: max results per language

    Returns:
        List of result dicts (SearchResult-compatible)
    """
    registry = get_registry()

    # Find non-English sources that are active and have query_lang set
    all_sources = registry.get_active_sources()
    non_en_sources = [
        s for s in all_sources
        if s.query_lang and s.query_lang != "en"
    ]

    if target_langs:
        non_en_sources = [s for s in non_en_sources if s.query_lang in target_langs]

    if not non_en_sources:
        return []

    # Group by language (multiple sources may share a language)
    langs = list({s.query_lang for s in non_en_sources if s.query_lang})

    async def _search_lang(lang: str) -> list[dict]:
        translated = await translate_query(query_en, lang)
        results = await search_local_translated(pool, translated, lang, fetch_limit)
        # Mark results with source language
        for r in results:
            r.setdefault("source_lang", lang)
        # Translate snippets back to English
        await translate_snippets(results, lang, "en")
        return results

    tasks = [_search_lang(lang) for lang in langs]
    results_lists = await asyncio.gather(*tasks, return_exceptions=True)

    all_results = []
    for r in results_lists:
        if isinstance(r, list):
            all_results.extend(r)
        elif isinstance(r, Exception):
            log.warning("Multilingual search task failed: %s", r)

    log.info("Multilingual search: %d total results from %d languages",
             len(all_results), len(langs))
    return all_results
