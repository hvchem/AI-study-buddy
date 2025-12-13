"""API package initialization."""
from app.api.documents import router as documents_router
from app.api.study import router as study_router

__all__ = ["documents_router", "study_router"]
