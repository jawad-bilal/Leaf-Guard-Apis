from __future__ import annotations

import io
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from app.config import (
    CLASS_NAMES,
    DISPLAY_NAMES,
    IMAGE_SIZE,
    resolve_model_path,
)
from app.schemas import ClassProbability, PredictionResponse


def _load_interpreter(model_path: Path):
    """Prefer LiteRT / TFLite runtime (Vercel-friendly); fall back to TensorFlow."""
    try:
        from ai_edge_litert.interpreter import Interpreter  # type: ignore

        return Interpreter(model_path=str(model_path))
    except Exception:
        pass

    try:
        from tflite_runtime.interpreter import Interpreter  # type: ignore

        return Interpreter(model_path=str(model_path))
    except Exception:
        pass

    import tensorflow as tf

    return tf.lite.Interpreter(model_path=str(model_path))


class PotatoDiseasePredictor:
    """Runs potato-leaf inference with the exported TFLite CNN."""

    def __init__(self, model_path: Path | None = None) -> None:
        self.model_path = Path(model_path) if model_path else resolve_model_path()
        self.interpreter = _load_interpreter(self.model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()[0]
        self.class_names = CLASS_NAMES

    def predict_bytes(self, image_bytes: bytes) -> PredictionResponse:
        image = self._load_image(image_bytes)
        batch = np.expand_dims(image, axis=0).astype(self.input_details["dtype"])

        self.interpreter.set_tensor(self.input_details["index"], batch)
        self.interpreter.invoke()
        probs = self.interpreter.get_tensor(self.output_details["index"])[0]

        best_idx = int(np.argmax(probs))
        class_id = self.class_names[best_idx]

        probabilities = [
            ClassProbability(
                class_id=name,
                label=DISPLAY_NAMES[name],
                confidence=round(float(probs[i]), 4),
            )
            for i, name in enumerate(self.class_names)
        ]
        probabilities.sort(key=lambda item: item.confidence, reverse=True)

        return PredictionResponse(
            class_id=class_id,
            label=DISPLAY_NAMES[class_id],
            confidence=round(float(probs[best_idx]), 4),
            probabilities=probabilities,
        )

    @staticmethod
    def _load_image(image_bytes: bytes) -> np.ndarray:
        """RGB float32 256×256 in 0–255; model still applies /255 rescale."""
        with Image.open(io.BytesIO(image_bytes)) as img:
            img = img.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
            arr = np.asarray(img, dtype=np.float32)
        if arr.ndim != 3 or arr.shape[-1] != 3:
            raise ValueError("Expected an RGB image")
        return arr


@lru_cache(maxsize=1)
def get_predictor() -> PotatoDiseasePredictor:
    override = os.getenv("MODEL_PATH")
    return PotatoDiseasePredictor(Path(override) if override else None)
