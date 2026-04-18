from __future__ import annotations
"""
Source registry: loads config/sources.json and provides lookup functions
for source metadata (authority tier, organization name, publication type, etc.).

Singleton pattern — loaded once at import time, cached for the process lifetime.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

# Resolve config path relative to project root:
# src/sources/registry.py -> src/sources -> src -> project_root -> config/sources.json
_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "sources.json"


@dataclass(frozen=True)
class SourceEntry:
    source_id: str
    surface_key: str
    domain: str
    organization_name: str
    language: str
    country: str | None
    source_type: str
    authority_tier: int | None
    audience_type: str
    publication_type: str | None
    access_mode: str  # "crawl" | "api" | "live_search"
    query_lang: str | None
    search_method: str | None
    search_url: str | None
    is_active: bool
    update_frequency: str | None
    notes: str | None


class SourceRegistry:
    """In-memory source registry loaded from config/sources.json."""

    def __init__(self, config_path: Path | None = None) -> None:
        self._by_key: dict[str, SourceEntry] = {}
        self._by_domain: dict[str, SourceEntry] = {}
        self._all: list[SourceEntry] = []
        self._load(config_path or _CONFIG_PATH)

    def _load(self, path: Path) -> None:
        if not path.exists():
            log.warning("Source registry config not found at %s — registry is empty", path)
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            sources = data.get("sources", [])
            for s in sources:
                entry = SourceEntry(
                    source_id=s["source_id"],
                    surface_key=s["surface_key"],
                    domain=s["domain"],
                    organization_name=s["organization_name"],
                    language=s.get("language", "en"),
                    country=s.get("country"),
                    source_type=s.get("source_type", "unknown"),
                    authority_tier=s.get("authority_tier"),
                    audience_type=s.get("audience_type", "mixed"),
                    publication_type=s.get("publication_type"),
                    access_mode=s.get("access_mode", "crawl"),
                    query_lang=s.get("query_lang"),
                    search_method=s.get("search_method"),
                    search_url=s.get("search_url"),
                    is_active=s.get("is_active", True),
                    update_frequency=s.get("update_frequency"),
                    notes=s.get("notes"),
                )
                self._all.append(entry)
                self._by_key[entry.surface_key] = entry
                # Store first entry per domain (prefer higher tier)
                if entry.domain not in self._by_domain:
                    self._by_domain[entry.domain] = entry
            log.info(
                "Source registry loaded: %d sources (%d active)",
                len(self._all),
                sum(1 for s in self._all if s.is_active),
            )
        except Exception as e:
            log.error("Failed to load source registry from %s: %s", path, e)

    def get_source_by_key(self, surface_key: str) -> SourceEntry | None:
        """Look up by surface_key (e.g. 'cdc_autism'). Also tries normalized forms."""
        if surface_key in self._by_key:
            return self._by_key[surface_key]
        # Try normalizing: "cdc" -> check keys starting with "cdc_"
        normalized = surface_key.lower().replace("-", "_").replace(" ", "_")
        if normalized in self._by_key:
            return self._by_key[normalized]
        # Try prefix match for bare source names like "cdc", "nih", "pubmed"
        for key, entry in self._by_key.items():
            if key.startswith(normalized + "_") or key == normalized:
                return entry
        return None

    def get_source_by_domain(self, domain: str) -> SourceEntry | None:
        """Look up by domain (e.g. 'www.cdc.gov')."""
        return self._by_domain.get(domain)

    def get_sources_by_tier(self, tier: int) -> list[SourceEntry]:
        """Get all sources at a given authority tier."""
        return [s for s in self._all if s.authority_tier == tier]

    def get_active_sources(self) -> list[SourceEntry]:
        """Get all active sources."""
        return [s for s in self._all if s.is_active]

    def get_live_search_sources(self) -> list[SourceEntry]:
        """Get active sources with access_mode == 'live_search'."""
        return [s for s in self._all if s.access_mode == "live_search" and s.is_active]

    def get_authority_boost(self, surface_key: str) -> float:
        """Return additive authority boost for a source. Tier 1→+0.20, 2→+0.15, 3→+0.05."""
        entry = self.get_source_by_key(surface_key)
        if entry is None or entry.authority_tier is None:
            return 0.0
        return {1: 0.20, 2: 0.15, 3: 0.05}.get(entry.authority_tier, 0.0)


# ── Module-level singleton ───────────────────────────────────────────────────
_registry: SourceRegistry | None = None


def get_registry() -> SourceRegistry:
    """Return the global source registry (lazy-loaded singleton)."""
    global _registry
    if _registry is None:
        _registry = SourceRegistry()
    return _registry
