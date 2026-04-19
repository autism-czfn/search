from __future__ import annotations
"""
Redis-backed safety state for SAFETY_EXPANDED_MODE (P-SRC-6b).

Key:   safety:{child_id}
Value: {"activated": true, "trigger": "<trigger_type>", "timestamp": <unix_time>}
TTL:   1800 seconds (30 minutes — matches clinical "recent escalation window")

Rules:
- set_safety_flag(child_id, trigger_type): SET key with TTL 1800s
- get_safety_flag(child_id): returns dict if key exists and not expired, else None
- Redis unavailable: log error, return None from get / silently skip set — NEVER crash
- Uses redis.asyncio (async Redis client)
"""

import json
import logging
import time

from .config import settings

log = logging.getLogger(__name__)

_SAFETY_TTL = 1800  # 30 minutes


def _get_redis_client():
    """Return a new async Redis client from REDIS_URL, or None if not configured."""
    try:
        import redis.asyncio as aioredis  # type: ignore[import]
        url = settings.redis_url
        if not url:
            log.debug("safety_state: REDIS_URL not configured — safety flag disabled")
            return None
        return aioredis.from_url(url, decode_responses=True, socket_connect_timeout=2)
    except ImportError:
        log.warning("safety_state: redis package not installed — safety flag disabled")
        return None
    except Exception as e:
        log.error("safety_state: failed to create Redis client: %s", e)
        return None


async def set_safety_flag(child_id: str, trigger_type: str) -> None:
    """
    SET Redis key safety:{child_id} with trigger info, TTL 1800s.
    If Redis is unavailable: log the error and silently skip — never crash.
    """
    client = _get_redis_client()
    if client is None:
        return

    key = f"safety:{child_id}"
    value = json.dumps({
        "activated": True,
        "trigger": trigger_type,
        "timestamp": int(time.time()),
    })
    try:
        await client.set(key, value, ex=_SAFETY_TTL)
        log.info("safety_state SET key=%s trigger=%s ttl=%ds", key, trigger_type, _SAFETY_TTL)
    except Exception as e:
        log.error("safety_state: Redis SET failed for key=%s: %s", key, e)
    finally:
        try:
            await client.aclose()
        except Exception:
            pass


async def get_safety_flag(child_id: str) -> dict | None:
    """
    GET Redis key safety:{child_id}.
    Returns the parsed dict if the key exists and has not expired, else None.
    If Redis is unavailable: log the error and return None — never crash.
    """
    client = _get_redis_client()
    if client is None:
        return None

    key = f"safety:{child_id}"
    try:
        raw = await client.get(key)
        if raw is None:
            return None
        data = json.loads(raw)
        log.info("safety_state GET key=%s found trigger=%s", key, data.get("trigger"))
        return data
    except Exception as e:
        log.error("safety_state: Redis GET failed for key=%s: %s", key, e)
        return None
    finally:
        try:
            await client.aclose()
        except Exception:
            pass
