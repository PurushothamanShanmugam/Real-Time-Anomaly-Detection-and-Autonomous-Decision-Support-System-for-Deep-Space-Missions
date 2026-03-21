from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime
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

# ── In-memory live prediction store ───────────────────────────────────
# Stores the last 500 predictions in memory.
# Every call to /predict appends a record here.
# The dashboard polls /live to read from this store in real time.
# Resets automatically when the API container restarts.
LIVE_PREDICTIONS: deque = deque(maxlen=500)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()

    print("\n" + "=" * 70)
    print(" Space Telemetry Inference API started successfully")
    print("=" * 70)
    print("Root URL       : http://localhost:8000/")
    print("Swagger UI     : http://localhost:8000/docs")
    print("Health Check   : http://localhost:8000/health")
    print("Metrics        : http://localhost:8000/metrics")
    print("Live Stream    : http://localhost:8000/live")
    print("Predict POST   : http://localhost:8000/predict")
    print("=" * 70 + "\n")

    yield

    print("\nSpace Telemetry Inference API is shutting down...\n")


app = FastAPI(
    title="Space Telemetry Inference API",
    description="API for anomaly detection and Remaining Useful Life (RUL) prediction",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def root():
    return {
        "message":          "Space Telemetry Inference API is running",
        "docs":             "/docs",
        "health":           "/health",
        "metrics":          "/metrics",
        "live_stream":      "/live",
        "predict_endpoint": "/predict",
    }


@app.get("/health")
def health():
    return {"status": "ok", "service": "telemetry-inference-api"}


@app.get("/metrics")
def metrics():
    return PlainTextResponse(
        generate_latest().decode("utf-8"),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: TelemetryRequest):
    """
    Run inference on a single telemetry row.
    Result is stored in LIVE_PREDICTIONS so /live can serve it
    to the real-time dashboard.
    """
    PREDICTION_REQUESTS.inc()
    start = perf_counter()

    try:
        result = predict_single(payload.model_dump())

        if result.get("anomaly_detected", False):
            ANOMALIES_DETECTED.inc()

        latency = perf_counter() - start
        PREDICTION_LATENCY.observe(latency)

        # ── Append to live store ──────────────────────────────────
        LIVE_PREDICTIONS.append({
            **result,
            "timestamp":  datetime.utcnow().isoformat(),
            "latency_ms": round(latency * 1000, 2),
        })

        return result

    except Exception as e:
        PREDICTION_FAILURES.inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/live")
def live(limit: int = 200):
    """
    Return the last `limit` predictions from the in-memory store.

    The Streamlit dashboard polls this endpoint every 3 seconds
    and re-renders charts, gauges, and the anomaly feed in real time.

    Query param:
        limit — number of recent predictions to return (default 200, max 500)
    """
    limit = min(limit, 500)
    records = list(LIVE_PREDICTIONS)[-limit:]
    return {
        "count":       len(records),
        "predictions": records,
    }


@app.delete("/live/reset")
def live_reset():
    """Clear the live store — useful for starting a fresh streaming session."""
    LIVE_PREDICTIONS.clear()
    return {"message": "Live prediction store cleared."}