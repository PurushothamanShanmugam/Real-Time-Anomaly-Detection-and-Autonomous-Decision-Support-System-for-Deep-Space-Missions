#!/bin/bash
set -e

echo "Starting Space Telemetry Anomaly Detection Dashboard..."

# Optional: run the ML pipeline first if output file is missing
if [ ! -f "/app/outputs/telemetry_results.csv" ]; then
    echo "telemetry_results.csv not found. Running main.py first..."
    python main.py
else
    echo "Found existing telemetry_results.csv. Skipping pipeline run."
fi

echo "Launching Streamlit app..."
exec streamlit run app/streamlit_app.py --server.address=0.0.0.0 --server.port=8501