"""Vercel serverless entrypoint for the FastAPI app."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mangum import Mangum

from app.main import app

handler = Mangum(app, lifespan="off")
