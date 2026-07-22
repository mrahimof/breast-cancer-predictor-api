# breast-cancer-predictor-api

A machine learning model served as a production style REST API. This project takes a classifier trained on the Wisconsin Breast Cancer dataset and wraps it in a FastAPI service, containerized with Docker and tested automatically with GitHub Actions, ready to deploy on Render. It's meant as a small, complete example of the path from a trained model to something a real client could actually call.

## Overview

The model itself is a logistic regression pipeline (with feature scaling) trained on the classic Wisconsin Breast Cancer dataset from scikit-learn, 30 measurements taken from digitized images of breast mass biopsies, used to classify a tumor as malignant or benign. On a held out test set it reaches about 98% accuracy and a ROC-AUC of 0.995. Training happens in `model/train.py`, which also saves the fitted pipeline and a metrics.json summary.

The API lives in `app/`. `app/main.py` defines the FastAPI app with three endpoints: a root endpoint with basic info, `/health` for checking whether the model loaded correctly, and `/predict`, which takes the 30 features as a JSON list and returns a prediction ("malignant" or "benign") along with the predicted probability of malignancy. Input validation is handled by Pydantic in `app/schemas.py`, so a request with the wrong number of features gets rejected with a clear 422 error instead of crashing the model. `app/model.py` loads the trained pipeline once at startup and reuses it across requests.

`tests/test_api.py` covers the main paths: the health check, a real malignant example, a real benign example, and the validation error case. The GitHub Actions workflow in `.github/workflows/ci.yml` retrains the model and runs this test suite on every push and pull request to main, then builds the Docker image as a second job to confirm it builds cleanly.

## Running locally

Install dependencies and train the model:

```
pip install -r requirements-dev.txt
python model/train.py
```

Start the API:

```
uvicorn app.main:app --reload
```

Interactive API docs are then available at `http://localhost:8000/docs`. A prediction request looks like this:

```
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189]}'
```

## Running with Docker

```
docker build -t breast-cancer-predictor-api .
docker run -p 8000:8000 breast-cancer-predictor-api
```

## Deploying on Render

This repo includes a `render.yaml` for Render's Blueprint deployment. Connect the repository in the Render dashboard, choose "New Blueprint Instance," and Render will build the Docker image and expose `/health` as the health check path automatically. No extra configuration is needed since the trained model is committed to the repo under `model/model.joblib`.

## Project structure

`app/` holds the FastAPI service, `model/` holds the training script and the saved model artifact, `tests/` holds the pytest suite, and `.github/workflows/ci.yml` runs everything on every push.

## Disclaimer

This is a portfolio project built on a public research dataset, not a medical device. It shouldn't be used for actual diagnosis.
