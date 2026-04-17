from __future__ import annotations
"""
Evidence source helpers: determine which sources are trusted enough
for curated evidence cards.

All lookups go through the P1 source registry — no hardcoded allow-lists.
Sources with authority_tier 1, 2, or 3 are evidence-eligible.
Sources without a tier (community, social, blog) are blocked.
"""

from ..sources.registry import get_registry, SourceEntry


def is_evidence_source(surface_key: str) -> bool:
    """True if source has authority_tier 1, 2, or 3 in registry."""
    registry = get_registry()
    source = registry.get_source_by_key(surface_key)
    return source is not None and source.authority_tier in (1, 2, 3)


def get_source_entry(surface_key: str) -> SourceEntry | None:
    """Look up a source entry by surface_key (convenience wrapper)."""
    return get_registry().get_source_by_key(surface_key)
