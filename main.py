"""Vercel FastAPI entrypoint (must expose `app` at a supported path)."""

from app.main import app

__all__ = ["app"]
