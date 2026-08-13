from pydantic import BaseModel, Field


class ClassProbability(BaseModel):
    class_id: str
    label: str
    confidence: float = Field(..., description="Probability in [0, 1]")


class PredictionResponse(BaseModel):
    class_id: str
    label: str
    confidence: float
    probabilities: list[ClassProbability]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str | None = None
