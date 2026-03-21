#!/bin/bash
set -e

# ═══════════════════════════════════════════════════════════════════════
#  ISTAS — India Space Telemetry Anomaly System
#  Dashboard entrypoint
#  Waits for the pipeline service to finish, then starts Streamlit and
#  prints all service URLs to the console (same as the previous version).
# ═══════════════════════════════════════════════════════════════════════

print_banner() {
  echo ""
  echo "  ╔══════════════════════════════════════════════════════════════╗"
  echo "  ║        INDIA SPACE TELEMETRY ANOMALY SYSTEM (ISTAS)         ║"
  echo "  ║        AI-Based Real-Time Anomaly Detection System          ║"
  echo "  ╚══════════════════════════════════════════════════════════════╝"
  echo ""
}

print_urls() {
  echo ""
  echo "  ┌──────────────────────────────────────────────────────────────┐"
  echo "  │                   SERVICE ACCESS URLS                        │"
  echo "  ├──────────────────────────────────────────────────────────────┤"
  echo "  │  🛸  Streamlit Dashboard  →  http://localhost:8501           │"
  echo "  │  ⚡  FastAPI  (docs)      →  http://localhost:8000/docs      │"
  echo "  │  📡  FastAPI  (predict)   →  http://localhost:8000/predict   │"
  echo "  │  📊  Grafana              →  http://localhost:3000           │"
  echo "  │      (login: admin / admin)                                  │"
  echo "  │  🔭  Prometheus           →  http://localhost:9090           │"
  echo "  ├──────────────────────────────────────────────────────────────┤"
  echo "  │  Kafka broker             →  localhost:9092                  │"
  echo "  │  Zookeeper                →  localhost:2181                  │"
  echo "  └──────────────────────────────────────────────────────────────┘"
  echo ""
  echo "  Tip: run  docker logs telemetry-pipeline  to see ML training logs."
  echo "  Tip: run  docker logs telemetry-api       to see API logs."
  echo ""
}

print_banner

# ── Wait for the pipeline service to produce telemetry_results.csv ──
MAX_WAIT=300   # 5 minutes
WAITED=0

echo "  [dashboard] Waiting for telemetry_results.csv from pipeline service ..."
while [ ! -f "/app/outputs/telemetry_results.csv" ]; do
    if [ "$WAITED" -ge "$MAX_WAIT" ]; then
        echo ""
        echo "  [ERROR] telemetry_results.csv not found after ${MAX_WAIT}s."
        echo "  Check pipeline logs:  docker logs telemetry-pipeline"
        exit 1
    fi
    echo "  [dashboard] Not ready yet — waiting 5s ... (${WAITED}s elapsed)"
    sleep 5
    WAITED=$((WAITED + 5))
done

echo "  [dashboard] telemetry_results.csv found. Starting Streamlit ..."
echo ""

# ── Print all access URLs before handing off to Streamlit ───────────
print_urls

# ── Launch Streamlit (replaces this shell process) ───────────────────
exec streamlit run app/streamlit_app.py \
    --server.address=0.0.0.0 \
    --server.port=8501 \
    --server.headless=true