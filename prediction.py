from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps


BASE_DIR = Path(__file__).resolve().parent
GENDER_MODEL_PATH = BASE_DIR / "gender_model.h5"
AGE_MODEL_PATH = BASE_DIR / "age_model.h5"
IMAGE_SIZE = (64, 64)
GENDER_LABELS = ("Male", "Female")
MIN_DISPLAY_AGE = 0
MAX_DISPLAY_AGE = 100


@dataclass(frozen=True)
class PredictionResult:
    gender: str
    gender_confidence: float
    age: int
    raw_age: float


def load_prediction_models(
    gender_path: Path = GENDER_MODEL_PATH,
    age_path: Path = AGE_MODEL_PATH,
) -> tuple[Any, Any]:
    if not gender_path.exists():
        raise FileNotFoundError(f"Missing gender model: {gender_path.name}")
    if not age_path.exists():
        raise FileNotFoundError(f"Missing age model: {age_path.name}")

    from tensorflow.keras.models import load_model

    return (
        load_model(gender_path, compile=False),
        load_model(age_path, compile=False),
    )


def preprocess_image(
    image: Image.Image,
    image_size: tuple[int, int] = IMAGE_SIZE,
) -> np.ndarray:
    prepared_image = ImageOps.exif_transpose(image).convert("RGB").resize(image_size)
    normalized_image = np.asarray(prepared_image, dtype=np.float32) / 255.0
    return np.expand_dims(normalized_image, axis=0)


def normalize_gender_prediction(raw_prediction: Any) -> tuple[str, float]:
    scores = np.asarray(raw_prediction, dtype=np.float32).reshape(-1)
    if scores.size == 0:
        raise ValueError("Gender model returned an empty prediction.")

    if scores.size == 1:
        female_probability = float(np.clip(scores[0], 0.0, 1.0))
        probabilities = np.array([1.0 - female_probability, female_probability])
    elif np.all((0.0 <= scores) & (scores <= 1.0)) and float(scores.sum()) > 0.0:
        probabilities = scores / scores.sum()
    else:
        shifted_scores = scores - scores.max()
        exp_scores = np.exp(shifted_scores)
        probabilities = exp_scores / exp_scores.sum()

    class_index = int(np.argmax(probabilities))
    label = (
        GENDER_LABELS[class_index]
        if class_index < len(GENDER_LABELS)
        else f"Class {class_index}"
    )
    confidence = float(np.clip(probabilities[class_index], 0.0, 1.0))
    return label, confidence


def normalize_age_prediction(raw_prediction: Any) -> tuple[int, float]:
    scores = np.asarray(raw_prediction, dtype=np.float32).reshape(-1)
    if scores.size == 0:
        raise ValueError("Age model returned an empty prediction.")

    raw_age = float(scores[0])
    display_age = int(round(np.clip(raw_age, MIN_DISPLAY_AGE, MAX_DISPLAY_AGE)))
    return display_age, raw_age


def predict_age_gender(
    image: Image.Image,
    gender_model: Any,
    age_model: Any,
) -> PredictionResult:
    model_input = preprocess_image(image)
    raw_gender_prediction = gender_model.predict(model_input, verbose=0)
    raw_age_prediction = age_model.predict(model_input, verbose=0)

    gender, gender_confidence = normalize_gender_prediction(raw_gender_prediction)
    age, raw_age = normalize_age_prediction(raw_age_prediction)

    return PredictionResult(
        gender=gender,
        gender_confidence=gender_confidence,
        age=age,
        raw_age=raw_age,
    )
