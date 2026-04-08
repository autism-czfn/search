from __future__ import annotations
"""
Read-only asyncpg connection pool to the autism-crawler PostgreSQL database.
Registers the pgvector codec so vector columns are returned as lists of floats.
"""

import json
import logging
import asyncpg
from pgvector.asyncpg import register_vector

log = logging.getLogger(__name__)

_pool: asyncpg.Pool | None = None


async def init_connection(conn: asyncpg.Connection) -> None:
    """Called for every new connection in the pool."""
    await register_vector(conn)
    # asyncpg returns JSONB columns as raw strings by default — register
    # a codec so they are automatically decoded to Python dicts/lists.
    await conn.set_type_codec(
        "jsonb",
        encoder=json.dumps,
        decoder=json.loads,
        schema="pg_catalog",
    )


async def get_pool() -> asyncpg.Pool:
    """Return the shared connection pool (must call connect() first)."""
    if _pool is None:
        raise RuntimeError("Database pool not initialised — call connect() first")
    return _pool


async def connect(dsn: str) -> asyncpg.Pool:
    """Create the pool; called once at application startup."""
    global _pool
    log.info("Connecting to database …")
    _pool = await asyncpg.create_pool(
        dsn,
        init=init_connection,
        min_size=2,
        max_size=10,
        command_timeout=30,
    )
    # Smoke-test the connection
    async with _pool.acquire() as conn:
        await conn.fetchval("SELECT 1")
    log.info("Database pool ready")
    return _pool


async def disconnect() -> None:
    """Close the pool; called at application shutdown."""
    global _pool
    if _pool:
        await _pool.close()
        _pool = None
        log.info("Database pool closed")
