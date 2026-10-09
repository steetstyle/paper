"""FastAPI application factory.

    search -> metadata -> fetch (HTML? PDF) -> MinerU -> chunk -> embed
    -> relational DB (papers/chunks) + vector store (embeddings)
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware

from paper_app import __version__
from paper_app.api.router import api_router
from paper_app.config import AppSettings, get_settings
from paper_app.container import Container, get_container
from paper_app.logging import bind_request_id, configure_logging, get_logger

logger = get_logger(__name__)

DESCRIPTION = """
ArXiv AI assistant backend.

**Flow**

1. `GET /api/v1/arxiv/search` — query the official ArXiv Atom API.
2. `POST /api/v1/ingest` — fetch HTML (or PDF), run MinerU, chunk and embed.
3. `POST /api/v1/search/semantic` — retrieve from the vector store, hydrated
   with relational metadata.

Embeddings and chunk text are swappable: relational rows live in Postgres/SQLite,
vectors in pgvector, Qdrant or memory (`VECTOR_BACKEND`).
"""


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: AppSettings = getattr(app.state, "settings", None) or get_settings()
    configure_logging(settings.logging.level, settings.logging.json_output)

    container: Container = getattr(app.state, "container", None) or get_container(settings)
    app.state.container = container

    logger.info(
        "startup",
        extra={
            "version": __version__,
            "env": settings.env,
            "embedding_provider": settings.embedding.provider,
            "vector_backend": settings.vector.backend.value,
        },
    )
    try:
        await container.startup()
    except Exception as exc:  # noqa: BLE001 - stay up and report degraded health
        logger.error("startup_degraded", extra={"error": str(exc)})
    try:
        yield
    finally:
        await container.aclose()
        logger.info("shutdown")


def create_app(settings: AppSettings | None = None, container: Container | None = None) -> FastAPI:
    settings = settings or get_settings()

    app = FastAPI(
        title="ArXiv AI Assistant",
        version=__version__,
        description=DESCRIPTION,
        lifespan=lifespan,
        root_path=settings.api.root_path,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )
    app.state.settings = settings
    app.state.container = container or get_container(settings)

    if settings.api.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.api.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    @app.middleware("http")
    async def request_context(request: Request, call_next):  # noqa: ANN001, ANN202
        request_id = bind_request_id(request.headers.get("x-request-id"))
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            logger.exception(
                "request_failed", extra={"method": request.method, "path": request.url.path}
            )
            raise
        duration_ms = (time.perf_counter() - started) * 1000
        response.headers["x-request-id"] = request_id
        response.headers["x-response-time-ms"] = f"{duration_ms:.2f}"
        logger.info(
            "request",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "duration_ms": round(duration_ms, 2),
            },
        )
        return response

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError) -> Any:
        logger.warning("bad_request", extra={"path": request.url.path, "error": str(exc)})
        from fastapi.responses import JSONResponse

        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, content={"detail": str(exc)}
        )

    app.include_router(api_router, prefix="/api/v1")

    @app.get("/", include_in_schema=False)
    async def root() -> dict[str, str]:
        return {
            "service": settings.name,
            "version": __version__,
            "docs": "/docs",
            "health": "/api/v1/health",
        }

    return app


app = create_app()