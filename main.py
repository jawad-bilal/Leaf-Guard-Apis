from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from core.config import ALLOWED_EXTENSIONS, MAX_UPLOAD_BYTES, allowed_origins, resolve_model_path
from core.schemas import HealthResponse, PredictionResponse
from core.services.predictor import get_predictor

app = FastAPI(
    title="Potato Disease Classifier API",
    description="CNN inference for Early Blight, Late Blight, and Healthy potato leaves.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    try:
        predictor = get_predictor()
        return HealthResponse(
            status="ok",
            model_loaded=True,
            model_path=str(predictor.model_path),
        )
    except Exception as exc:  # noqa: BLE001
        return HealthResponse(status=f"error: {exc}", model_loaded=False)


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file name provided")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Use: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="File too large (max 10 MB)")

    try:
        return get_predictor().predict_bytes(image_bytes)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc


@app.get("/")
def root(request: Request) -> dict[str, str]:
    return {
        "message": "Potato Disease Classifier API",
        "docs": "/docs",
        "health": "/health",
        "predict": "POST /predict (multipart form field: file)",
        "model": str(resolve_model_path()),
        "path": request.url.path,
    }
