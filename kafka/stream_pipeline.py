"""
kafka/stream_pipeline.py
─────────────────────────
Single script that runs producer + consumer in parallel threads.
Used by Dockerfile.consumer inside Docker.

Environment variables (set by docker-compose.yml):
  KAFKA_BOOTSTRAP  — broker address (default: kafka:29092)
  API_URL          — FastAPI predict endpoint
  TOPIC_NAME       — Kafka topic name
  CSV_PATH         — path to telemetry_results.csv
  DELAY_SECONDS    — pause between messages
"""

import json
import os
import time
import threading
import pandas as pd
import requests
from kafka import KafkaProducer, KafkaConsumer

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "kafka:29092")
API_URL         = os.getenv("API_URL", "http://telemetry-api:8000/predict")
TOPIC_NAME      = os.getenv("TOPIC_NAME", "telemetry-input")
CSV_PATH        = os.getenv("CSV_PATH", "/app/outputs/telemetry_results.csv")
DELAY_SECONDS   = float(os.getenv("DELAY_SECONDS", "0.3"))

REQUIRED_COLUMNS = [
    "unit_id", "time_cycle",
    "op_setting_1", "op_setting_2", "op_setting_3",
    "sensor_1",  "sensor_2",  "sensor_3",  "sensor_4",  "sensor_5",
    "sensor_6",  "sensor_7",  "sensor_8",  "sensor_9",  "sensor_10",
    "sensor_11", "sensor_12", "sensor_13", "sensor_14", "sensor_15",
    "sensor_16", "sensor_17", "sensor_18", "sensor_19", "sensor_20",
    "sensor_21",
]


def run_producer():
    print(f"[Producer] Connecting to Kafka at {KAFKA_BOOTSTRAP}...")
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP,
        value_serializer=lambda m: json.dumps(m).encode("utf-8"),
    )

    df = pd.read_csv(CSV_PATH)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        print(f"[Producer] ERROR — missing columns: {missing}")
        return

    print(f"[Producer] Loaded {len(df)} rows from {CSV_PATH}")
    print(f"[Producer] Publishing to topic '{TOPIC_NAME}' every {DELAY_SECONDS}s...\n")

    for index, row in df[REQUIRED_COLUMNS].iterrows():
        payload = row.to_dict()
        for k, v in payload.items():
            if pd.isna(v):
                payload[k] = 0.0
        payload["unit_id"]    = int(payload["unit_id"])
        payload["time_cycle"] = int(payload["time_cycle"])

        producer.send(TOPIC_NAME, value=payload)
        print(f"[Producer] Sent → unit={payload['unit_id']}  cycle={payload['time_cycle']}")
        time.sleep(DELAY_SECONDS)

    producer.flush()
    producer.close()
    print("[Producer] All records sent.")


def run_consumer():
    # Small delay so producer starts first
    time.sleep(3)
    print(f"[Consumer] Connecting to Kafka at {KAFKA_BOOTSTRAP}...")

    consumer = KafkaConsumer(
        TOPIC_NAME,
        bootstrap_servers=KAFKA_BOOTSTRAP,
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        group_id="telemetry-docker-consumer",
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        consumer_timeout_ms=60000,   # stop after 60s of no messages
    )

    print(f"[Consumer] Listening on '{TOPIC_NAME}', forwarding to {API_URL}...\n")

    for message in consumer:
        payload = message.value
        try:
            response = requests.post(API_URL, json=payload, timeout=10)
            if response.status_code == 200:
                r = response.json()
                flag   = r.get("anomaly_flag", "?")
                rul    = r.get("predicted_rul", 0)
                risk   = r.get("risk_score", 0)
                conf   = r.get("confidence", "?")
                action = r.get("recommended_action", "?")
                alert  = " *** ANOMALY ***" if r.get("anomaly_detected") else ""
                print(f"[Consumer] unit={payload['unit_id']} cycle={payload['time_cycle']} "
                      f"flag={flag} RUL={rul:.1f} risk={risk:.2f} [{conf}] → {action}{alert}")
            else:
                print(f"[Consumer] API error {response.status_code}: {response.text}")
        except Exception as e:
            print(f"[Consumer] Request failed: {e}")


if __name__ == "__main__":
    producer_thread = threading.Thread(target=run_producer, daemon=True)
    consumer_thread = threading.Thread(target=run_consumer, daemon=True)

    producer_thread.start()
    consumer_thread.start()

    producer_thread.join()
    consumer_thread.join()
    print("Streaming pipeline complete.")