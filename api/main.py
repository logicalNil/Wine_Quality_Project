from fastapi import FastAPI, Request, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
import os
from pathlib import Path
import numpy as np
import joblib
import time
import logging
from logging.handlers import RotatingFileHandler
import sqlite3
from datetime import datetime
from prometheus_client import Counter, Histogram, generate_latest, REGISTRY
from prometheus_client.exposition import CONTENT_TYPE_LATEST

from api.schemas import WineFeatures, PredictionResponse
from api.database import log_prediction

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler("logs/wine_api.log", maxBytes=10485760, backupCount=5),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter('request_count', 'Total API request count', ['method', 'endpoint', 'status_code'])
REQUEST_LATENCY = Histogram('request_latency_seconds', 'Request latency in seconds', ['endpoint'])
PREDICTION_COUNT = Counter('prediction_count', 'Prediction count by quality class', ['quality_class'])

app = FastAPI(title="Wine Quality API", version="1.0.0")

# Load model
base_dir = Path(__file__).parent.parent  # Go up to project root from api/
model_path = base_dir / "models" / "best_model.pkl"

# Load model
try:
    model = joblib.load(str(model_path))
    logger.info(f"Model loaded successfully from {model_path}")
except Exception as e:
    logger.error(f"Error loading model from {model_path}: {e}")
    model = None

class_labels = {
    0: "Poor",
    1: "Average",
    2: "Good",
    3: "Excellent"
}


@app.get("/")
async def root():
    return {"message": "Wine Quality Classification API"}


@app.get("/health")
async def health():
    return {"status": "OK"}


@app.get("/metrics")
async def metrics():
    return generate_latest(REGISTRY)


@app.post("/predict", response_model=PredictionResponse)
async def predict(wine: WineFeatures, request: Request):
    start_time = time.time()

    try:
        # Convert input to DataFrame
        input_data = pd.DataFrame([wine.dict()])

        # Make prediction
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0].tolist()

        # Log prediction
        latency = time.time() - start_time
        log_prediction(
            wine.dict(),
            int(prediction),
            probabilities,
            latency,
            request.client.host
        )

        # Update metrics
        PREDICTION_COUNT.labels(quality_class=int(prediction)).inc()
        REQUEST_LATENCY.labels(endpoint='/predict').observe(latency)
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint='/predict',
            status_code=200
        ).inc()

        return PredictionResponse(
            quality_class=int(prediction),
            probabilities=probabilities,
            class_labels=class_labels
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint='/predict',
            status_code=500
        ).inc()
        return JSONResponse(
            status_code=500,
            content={"message": f"Prediction error: {str(e)}"}
        )


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    latency = time.time() - start_time

    logger.info(f"{request.method} {request.url.path} - {response.status_code} - {latency:.4f}s")

    return response


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)