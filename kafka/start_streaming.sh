#!/bin/bash
# start_streaming.sh — waits for Kafka + API then starts streaming

echo "=============================================="
echo " Kafka Streaming Service Starting..."
echo "=============================================="

KAFKA_HOST=${KAFKA_BOOTSTRAP:-kafka:29092}
API_HOST=${API_URL:-http://telemetry-api:8000/predict}

# Wait for API to be ready
echo "Waiting for API at $API_HOST..."
for i in $(seq 1 30); do
    if curl -sf http://telemetry-api:8000/health > /dev/null 2>&1; then
        echo "API is ready."
        break
    fi
    echo "  Attempt $i/30 — API not ready yet, waiting 5s..."
    sleep 5
done

# Wait a bit more for Kafka to be fully ready
echo "Waiting 15s for Kafka to be fully initialised..."
sleep 15

echo ""
echo "Starting Kafka producer + consumer pipeline..."
echo "  Broker  : $KAFKA_HOST"
echo "  API     : $API_HOST"
echo "  CSV     : ${CSV_PATH}"
echo ""

python kafka/stream_pipeline.py