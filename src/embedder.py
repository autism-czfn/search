from __future__ import annotations
"""
Embed a query string using fastembed (local, no API key required).
Returns a list of floats (vector of length 768 for nomic-embed-text-v1.5).
Falls back gracefully — callers handle None to switch to keyword-only mode.
"""

import asyncio
import logging
import time
from fastembed import TextEmbedding

log = logging.getLogger(__name__)

MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"

_model: TextEmbedding | None = None


def _get_model() -> TextEmbedding:
    global _model
    if _model is None:
        log.info("Loading embedding model: %s", MODEL_NAME)
        _model = TextEmbedding(model_name=MODEL_NAME)
    return _model


async def embed_query(text: str) -> list[float] | None:
    """
    Embed *text* using fastembed (runs in-process, no API call).
    Returns the embedding vector, or None on any failure.
    None signals the caller to fall back to keyword-only search.
    """
    if not text.strip():
        return None
    try:
        cold = _model is None
        model = _get_model()
        log.info("embed START cold_load=%s text_len=%d", cold, len(text.strip()))
        t0 = time.monotonic()
        loop = asyncio.get_event_loop()
        vectors = await loop.run_in_executor(
            None, lambda: list(model.embed([text.strip()]))
        )
        log.info("embed DONE dims=%d elapsed=%dms", len(vectors[0]), int((time.monotonic() - t0) * 1000))
        return vectors[0].tolist()
    except Exception as e:
        log.warning("embed FAIL %s — falling back to keyword-only search", e)
        return None
