from __future__ import annotations
"""
autism-search — FastAPI application entry point.
Run with: uvicorn src.main:app --reload --port 3001
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from .config import settings
from .db import connect, disconnect, get_pool
from .api.routes import router

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
log = logging.getLogger(__name__)


class PrivateNetworkAccessMiddleware(BaseHTTPMiddleware):
    """
    Respond to Chrome's Private Network Access preflight and add the required
    header to every response so public-origin pages can call this private-IP API.
    See: https://developer.chrome.com/blog/private-network-access-preflight/
    """
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["Access-Control-Allow-Private-Network"] = "true"
        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──────────────────────────────────────────────────────────────
    pool = await connect(settings.database_url)
    app.state.pool = pool
    yield
    # ── Shutdown ─────────────────────────────────────────────────────────────
    await disconnect()


app = FastAPI(
    title="Autism Search API",
    description="Semantic + keyword hybrid search over crawled autism content.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "OPTIONS"],
    allow_headers=["*"],
)
# Must be added AFTER CORSMiddleware so it wraps it and can inject the PNA
# header into CORS preflight responses as well as regular responses.
app.add_middleware(PrivateNetworkAccessMiddleware)

app.include_router(router)


# Override the get_pool dependency to return app.state.pool
# This allows Depends(get_pool) in routes to work without global state.
async def _get_pool_from_state():
    return app.state.pool

app.dependency_overrides[get_pool] = _get_pool_from_state
