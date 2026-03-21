# Space Telemetry Anomaly Detection & Predictive Maintenance System

**India Space Academy — AI & Machine Learning in Space Exploration**
*Development of an AI-Based Real-Time Anomaly Detection and Autonomous Decision Support System for Deep Space Missions*

![CI](https://github.com/PurushothamanShanmugam/Real-Time-Anomaly-Detection-and-Autonomous-Decision-Support-System-for-Deep-Space-Missions/actions/workflows/ci.yml/badge.svg)

---

## Project Overview

Deep-space missions like Chandrayaan-3, Aditya-L1, and Mars Perseverance face communication delays that prevent Earth from responding to every onboard alert. This project simulates that problem by building a fully autonomous AI system that can:

- Detect anomalies in spacecraft telemetry using Isolation Forest + deep learning Autoencoder
- Predict Remaining Useful Life (RUL) with Random Forest (baseline) and LSTM (deep learning)
- Recommend corrective actions autonomously via a rule-based decision engine
- Score risk (0–1) and estimate decision confidence
- Visualise mission health in a real-time Streamlit mission-control dashboard
- Stream telemetry through Kafka and reflect every prediction live on the dashboard
- Serve predictions via a FastAPI inference API with Prometheus metrics
- Expose dashboards in Grafana for operational monitoring

Dataset: **NASA C-MAPSS Turbofan Engine Degradation Dataset** (FD001).

---

## Quick Start — Docker (Recommended)

```bash
# 1. Clone the repository
git clone https://github.com/PurushothamanShanmugam/Real-Time-Anomaly-Detection-and-Autonomous-Decision-Support-System-for-Deep-Space-Missions.git
cd Real-Time-Anomaly-Detection-and-Autonomous-Decision-Support-System-for-Deep-Space-Missions

# 2. Start everything with one command
docker compose up --build
```

Once the ML pipeline finishes training (3–5 min), the dashboard container
**prints all service URLs** directly to your terminal:

```
  ┌──────────────────────────────────────────────────────────────┐
  │                   SERVICE ACCESS URLS                        │
  ├──────────────────────────────────────────────────────────────┤
  │    Streamlit Dashboard  →  http://localhost:8501             │
  │    FastAPI  (docs)      →  http://localhost:8000/docs        │
  │    FastAPI  (predict)   →  http://localhost:8000/predict     │
  │    Grafana              →  http://localhost:3000             │
  │    (login: admin / admin)                                    │
  │    Prometheus           →  http://localhost:9090             │
  ├──────────────────────────────────────────────────────────────┤
  │  Kafka broker             →  localhost:9092                  │
  │  Zookeeper                →  localhost:2181                  │
  └──────────────────────────────────────────────────────────────┘
```

> **Tip:** if a container exits, run `docker logs telemetry-pipeline` to see ML training output and `docker logs telemetry-api` for API logs.

---

## Manual / Local Run

```bash
# Install dependencies
pip install -r requirements.txt
# Optional (LSTM + Autoencoder):
pip install tensorflow

# Train models + generate outputs
python main.py

# Launch Streamlit dashboard
streamlit run app/streamlit_app.py

# Start FastAPI server (optional)
uvicorn api.main:app --port 8000

# Stream via Kafka (optional, needs a running Kafka broker)
python kafka/producer.py
python kafka/consumer.py
```

---

## System Architecture

```
space-telemetry-anomaly-system/
│
├── main.py                        ← End-to-end ML pipeline (single entry point)
│
├── src/
│   ├── config.py                  ← Central configuration & paths
│   ├── data_loader.py             ← NASA C-MAPSS dataset loader
│   ├── preprocess.py              ← RUL calculation + StandardScaler
│   ├── feature_engineering.py     ← Rolling mean + cycle-diff features
│   ├── train_anomaly.py           ← Isolation Forest anomaly detection
│   ├── train_autoencoder.py       ← Deep learning Autoencoder (Day 10)
│   ├── train_rul.py               ← Random Forest RUL prediction
│   ├── train_rul_lstm.py          ← LSTM RUL prediction (Day 4)
│   ├── decision_engine.py         ← Rule-based decision engine (Day 13)
│   ├── decision_confidence.py     ← Risk score + confidence + RL demo (Days 14, 16–17)
│   ├── evaluate.py                ← Confusion matrix + classification report
│   ├── simulator.py               ← Telemetry stream simulator
│   └── utils.py                   ← Shared utilities
│
├── app/
│   └── streamlit_app.py           ← 5-page mission-control dashboard (Days 18–20)
│
├── api/
│   ├── main.py                    ← FastAPI inference server
│   ├── inference.py               ← Model loading + prediction logic
│   ├── schemas.py                 ← Pydantic request/response models
│   └── metrics.py                 ← Prometheus counters + histograms
│
├── kafka/
│   ├── producer.py                ← Streams telemetry CSV → Kafka topic
│   └── consumer.py                ← Reads Kafka → calls /predict API
│
├── monitoring/
│   ├── prometheus.yml             ← Prometheus scrape config
│   └── grafana/                   ← Grafana provisioning (datasource + dashboard)
│
├── tests/
│   ├── test_decision_engine.py
│   ├── test_decision_confidence.py
│   ├── test_train_anomaly.py
│   ├── test_data_loader.py
│   ├── test_preprocess.py
│   └── test_dashboard_and_entrypoint.py  ← Dashboard syntax + URL banner checks
│
├── data/raw/train_FD001.txt       ← NASA C-MAPSS training data
├── models/                        ← Saved .pkl and .keras model files
├── outputs/                       ← Logs, figures, metrics, results CSV
│
├── Dockerfile                     ← Dashboard container
├── Dockerfile.api                 ← FastAPI container
├── Dockerfile.pipeline            ← ML pipeline container (TensorFlow included)
├── Dockerfile.consumer            ← Kafka consumer container
├── docker-compose.yml             ← Full stack orchestration
├── entrypoint.sh                  ← Dashboard startup script + URL banner
└── .github/workflows/ci.yml       ← GitHub Actions CI pipeline
```

---

## Dashboard Pages

The dashboard uses a NASA mission-control aesthetic with animated status badges,
`Share Tech Mono` telemetry font, and a deep-space dark theme.

| # | Page | Contents |
|---|------|----------|
| 1 | **Overview & Anomaly Detection** | Fleet KPI cards, unit status row, health gauges (alert / health / failure risk / risk score), anomaly score timeline, flag distribution, action summary, data table |
| 2 | **RUL Prediction & Model Comparison** | Actual vs predicted RUL line chart, prediction error area chart, RF vs LSTM metrics, scatter across all units |
| 3 | **Risk Score & Decision Confidence** | Risk gauge, confidence KPI, decision justification card, risk timeline, confidence pie, anomaly event log |
| 4 | **Autoencoder Analysis** | Reconstruction error chart, Isolation Forest vs Autoencoder comparison, model agreement analysis, evaluation report |
| 5 | **Reinforcement Learning Demo** | Rule-based vs RL comparison table, interactive Q-table trainer with sliders, learned policy table, Q-value heatmap |
| 6 | **🔴 Live Stream** | Real-time dashboard that polls `GET /live` every 3 seconds — reflects every Kafka → API prediction live with KPI cards, anomaly timeline, RUL chart, risk area chart, action distribution, and live feed table |

---

## FastAPI Endpoints

```
GET  /              — Health check
GET  /metrics       — Prometheus metrics
GET  /live          — Last N predictions from live stream (polls every 3s by dashboard)
DELETE /live/reset  — Clear the live prediction store
POST /predict       — Run inference on a single telemetry row
```

Example `/predict` request:
```json
{
  "unit_id": 1, "time_cycle": 150,
  "op_setting_1": 0.0, "op_setting_2": 0.0, "op_setting_3": 100.0,
  "sensor_1": 518.67, "sensor_2": 642.43
}
```

Example response:
```json
{
  "unit_id": 1, "time_cycle": 150,
  "anomaly_flag": -1, "anomaly_score": -0.182, "anomaly_detected": true,
  "predicted_rul": 23.4,
  "recommended_action": "HIGH RISK: Schedule urgent inspection",
  "risk_score": 0.74, "confidence": "HIGH",
  "justification": "anomaly detected (score=-0.182); RUL below high-risk threshold (23 cycles)."
}
```

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run only dashboard/entrypoint tests
pytest tests/test_dashboard_and_entrypoint.py -v
```

The CI pipeline (`.github/workflows/ci.yml`) runs on every push and:

1. Installs all dependencies (excluding TensorFlow for speed)
2. Runs all 6 test modules including `test_dashboard_and_entrypoint.py`
3. Syntax-checks every `.py` file in `src/`, `api/`, `kafka/`, `app/`
4. Validates `entrypoint.sh` bash syntax
5. Checks `api/main.py` has `LIVE_PREDICTIONS` store and `/live` endpoint
6. Checks `app/streamlit_app.py` has the Live Stream page
7. Runs a full import check including `collections.deque` and API schemas

---

## Docker Services

| Container | Port | Purpose |
|-----------|------|---------|
| `telemetry-pipeline` | — | Trains all models; writes `outputs/` and `models/` |
| `telemetry-api` | 8000 | FastAPI inference + Prometheus `/metrics` |
| `telemetry-dashboard` | 8501 | Streamlit mission-control dashboard |
| `telemetry-kafka-consumer` | — | Reads `telemetry_results.csv` → Kafka → API |
| `telemetry-zookeeper` | 2181 | Kafka coordination |
| `telemetry-kafka` | 9092 | Kafka message broker |
| `telemetry-prometheus` | 9090 | Scrapes API metrics every 5 s |
| `telemetry-grafana` | 3000 | Grafana dashboards (admin / admin) |

---

## Output Files

| File | Description |
|------|-------------|
| `outputs/telemetry_results.csv` | Full results with all model outputs |
| `outputs/metrics/rul_metrics.txt` | Random Forest evaluation |
| `outputs/metrics/rul_lstm_metrics.txt` | LSTM evaluation |
| `outputs/metrics/rul_model_comparison.txt` | RF vs LSTM side-by-side |
| `outputs/metrics/anomaly_classification_report.txt` | Precision / recall / F1 |
| `outputs/metrics/autoencoder_classification_report.txt` | Autoencoder evaluation |
| `outputs/metrics/rl_q_table.txt` | Learned Q-table |
| `outputs/figures/anomaly_confusion_matrix.png` | Confusion matrix plot |
| `outputs/logs/pipeline.log` | Timestamped execution log |

---

## Author

**Purushothaman Shanmugam**
M25DE1033 · m25de1033@iitj.ac.in
M.Tech — Data Engineering, 
Indian Institue of Technology Jodhpur

---

## License

Educational and research use only, developed as part of the India Space Academy
AI & Machine Learning in Space Exploration programme.
Dataset: NASA C-MAPSS, publicly available via NASA PCoE.