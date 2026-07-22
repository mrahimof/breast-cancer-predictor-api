"""FastAPI service for the breast cancer predictor."""

from fastapi import FastAPI, HTTPException

from app.model import get_model_bundle, predict
from app.schemas import HealthResponse, PredictionRequest, PredictionResponse

app = FastAPI(
    title="Breast Cancer Predictor API",
    description=(
        "Serves a scikit-learn classifier trained on the Wisconsin Breast "
        "Cancer dataset. Given 30 tumor measurements, predicts whether a "
        "tumor is malignant or benign."
    ),
    version="1.0.0",
)


@app.get("/", tags=["meta"])
def root() -> dict:
    return {
        "message": "Breast Cancer Predictor API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    try:
        get_model_bundle()
        loaded = True
    except Exception:
        loaded = False
    return HealthResponse(status="ok" if loaded else "error", model_loaded=loaded)


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict_endpoint(request: PredictionRequest) -> PredictionResponse:
    try:
        label, malignant_probability = predict(request.features)
    except Exception as exc:  # pragma: no cover - defensive guard
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return PredictionResponse(
        prediction=label, malignant_probability=malignant_probability
    )
