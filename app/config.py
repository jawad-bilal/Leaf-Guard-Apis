import os
from pathlib import Path

# backend/ is the project root for this service
BACKEND_ROOT = Path(__file__).resolve().parents[1]

MODEL_CANDIDATES = [
    BACKEND_ROOT / "models" / "potato_disease.tflite",
]

IMAGE_SIZE = 256
CHANNELS = 3

CLASS_NAMES = [
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
]

DISPLAY_NAMES = {
    "Potato___Early_blight": "Early Blight",
    "Potato___Late_blight": "Late Blight",
    "Potato___healthy": "Healthy",
}

ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".jfif",
    ".png",
    ".bmp",
    ".webp",
    ".gif",
    ".tif",
    ".tiff",
    ".ico",
}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB


def resolve_model_path() -> Path:
    for path in MODEL_CANDIDATES:
        if path.is_file():
            return path
    checked = ", ".join(str(p) for p in MODEL_CANDIDATES)
    raise FileNotFoundError(f"No trained model found. Checked: {checked}")


def allowed_origins() -> list[str]:
    """Local defaults + optional comma-separated FRONTEND_ORIGIN / CORS_ORIGINS."""
    origins = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://aetherleaf.vercel.app",
    ]
    extra = os.getenv("FRONTEND_ORIGIN") or os.getenv("CORS_ORIGINS") or ""
    for item in extra.split(","):
        value = item.strip().rstrip("/")
        if value and value not in origins:
            origins.append(value)
    return origins
