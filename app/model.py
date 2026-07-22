"""Loads the trained pipeline and runs predictions."""

from functools import lru_cache
from pathlib import Path

import joblib

MODEL_PATH = Path(__file__).parent.parent / "model" / "model.joblib"


@lru_cache
def get_model_bundle() -> dict:
    """Loads the model bundle once and caches it for the life of the process."""
    return joblib.load(MODEL_PATH)


def predict(features: list[float]) -> tuple[str, float]:
    """Runs a single prediction and returns (label, malignant_probability)."""
    bundle = get_model_bundle()
    pipeline = bundle["pipeline"]
    target_names = bundle["target_names"]

    proba = pipeline.predict_proba([features])[0]
    predicted_index = int(proba.argmax())
    label = target_names[predicted_index]

    # In the scikit-learn Wisconsin dataset, class index 0 is "malignant" and
    # index 1 is "benign" — target_names preserves whatever order the loaded
    # bundle was trained with, so look up "malignant" by name rather than index.
    malignant_index = list(target_names).index("malignant")
    malignant_probability = float(proba[malignant_index])

    return str(label), malignant_probability
