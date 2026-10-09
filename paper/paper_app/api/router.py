"""API v1 router aggregation."""

from fastapi import APIRouter

from paper_app.api.routes import arxiv, health, ingestion, papers, projects, search, spaces

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(arxiv.router)
api_router.include_router(papers.router)
api_router.include_router(projects.router)
api_router.include_router(ingestion.router)
api_router.include_router(search.router)
api_router.include_router(spaces.router)

__all__ = ["api_router"]