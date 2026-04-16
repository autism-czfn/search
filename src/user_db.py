from __future__ import annotations
"""
Read-only asyncpg connection pool to the user data database (mzhu_test_ tables).
Search never writes here — collect owns all writes.
"""

import json
import logging
import asyncpg

log = logging.getLogger(__name__)

_pool: asyncpg.Pool | None = None


async def _init_connection(conn: asyncpg.Connection) -> None:
    await conn.set_type_codec(
        "jsonb",
        encoder=json.dumps,
        decoder=json.loads,
        schema="pg_catalog",
    )


async def get_user_pool() -> asyncpg.Pool | None:
    """Return the user DB pool, or None if not configured."""
    return _pool


async def connect_user_db(dsn: str) -> asyncpg.Pool:
    global _pool
    log.info("Connecting to user database …")
    _pool = await asyncpg.create_pool(
        dsn,
        init=_init_connection,
        min_size=2,
        max_size=10,
        command_timeout=30,
    )
    async with _pool.acquire() as conn:
        await conn.fetchval("SELECT 1")
    log.info("User database pool ready")
    return _pool


async def disconnect_user_db() -> None:
    global _pool
    if _pool:
        await _pool.close()
        _pool = None
        log.info("User database pool closed")
