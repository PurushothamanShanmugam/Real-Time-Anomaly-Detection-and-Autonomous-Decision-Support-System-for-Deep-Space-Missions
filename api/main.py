from contextlib import asynccontextmanager
from time import perf_counter

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse

from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from api.schemas import TelemetryRequest, PredictionResponse
from api.inference import load_models, predict_single
from api.metrics import (
    PREDICTION_REQUESTS,
    PREDICTION_FAILURES,
    ANOMALIES_DETECTED,
    PREDICTION_LATENCY,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load ML models once during startup
    load_models()

    # Print useful API links in console
    print("\n" + "=" * 70)
    print(" Space Telemetry Inference API started successfully")
    print("=" * 70)
    print("Root URL       : http://localhost:8000/")
    print("Swagger UI     : http://localhost:8000/docs")
    print("ReDoc          : http://localhost:8000/redoc")
    print("Health Check   : http://localhost:8000/health")
    print("Metrics        : http://localhost:8000/metrics")
    print("OpenAPI JSON   : http://localhost:8000/openapi.json")
    print("Predict POST   : http://localhost:8000/predict")
    print("=" * 70 + "\n")

    yield

    print("\nSpace Telemetry Inference API is shutting down...\n")


app = FastAPI(
    title="Space Telemetry Inference API",
    description="API for anomaly detection and Remaining Useful Life (RUL) prediction",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/")
def root():
    return {
        "message": "Space Telemetry Inference API is running",
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/health",
        "metrics": "/metrics",
        "openapi_json": "/openapi.json",
        "predict_endpoint": "/predict"
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "telemetry-inference-api"
    }


@app.get("/metrics")
def metrics():
    return PlainTextResponse(
        generate_latest().decode("utf-8"),
        media_type=CONTENT_TYPE_LATEST
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: TelemetryRequest):
    PREDICTION_REQUESTS.inc()
    start = perf_counter()

    try:
        result = predict_single(payload.model_dump())

        if result.get("anomaly_detected", False):
            ANOMALIES_DETECTED.inc()

        PREDICTION_LATENCY.observe(perf_counter() - start)
        return result

    except Exception as e:
        PREDICTION_FAILURES.inc()
        raise HTTPException(status_code=500, detail=str(e))