from prometheus_client import Counter, Histogram

PREDICTION_REQUESTS = Counter(
    "telemetry_prediction_requests_total",
    "Total number of telemetry prediction requests"
)

PREDICTION_FAILURES = Counter(
    "telemetry_prediction_failures_total",
    "Total number of telemetry prediction failures"
)

ANOMALIES_DETECTED = Counter(
    "telemetry_anomalies_detected_total",
    "Total number of anomaly predictions"
)

PREDICTION_LATENCY = Histogram(
    "telemetry_prediction_latency_seconds",
    "Latency of telemetry prediction requests"
)