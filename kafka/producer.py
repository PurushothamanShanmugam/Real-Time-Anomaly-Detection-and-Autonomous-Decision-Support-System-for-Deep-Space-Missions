"""
kafka/producer.py
-----------------
Reads telemetry_results.csv row by row and publishes each record
to the Kafka topic 'telemetry-input' for the consumer to pick up
and forward to the FastAPI inference endpoint.

Usage:
    # Make sure Kafka is running first:
    docker-compose up -d kafka zookeeper

    # Then run:
    python kafka/producer.py
"""

import json
import time
import pandas as pd
from kafka import KafkaProducer

TOPIC_NAME        = "telemetry-input"
BOOTSTRAP_SERVERS = "localhost:9092"
CSV_PATH          = "outputs/telemetry_results.csv"
DELAY_SECONDS     = 0.5   # pause between messages (lower = faster simulation)

# Columns the FastAPI /predict endpoint expects
REQUIRED_COLUMNS = [
    "unit_id", "time_cycle",
    "op_setting_1", "op_setting_2", "op_setting_3",
    "sensor_1",  "sensor_2",  "sensor_3",  "sensor_4",  "sensor_5",
    "sensor_6",  "sensor_7",  "sensor_8",  "sensor_9",  "sensor_10",
    "sensor_11", "sensor_12", "sensor_13", "sensor_14", "sensor_15",
    "sensor_16", "sensor_17", "sensor_18", "sensor_19", "sensor_20",
    "sensor_21",
]


def serializer(message: dict) -> bytes:
    return json.dumps(message).encode("utf-8")


def main():
    print(f"Connecting to Kafka at {BOOTSTRAP_SERVERS}...")
    producer = KafkaProducer(
        bootstrap_servers=BOOTSTRAP_SERVERS,
        value_serializer=serializer,
    )

    df = pd.read_csv(CSV_PATH)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    print(f"Loaded {len(df)} rows from {CSV_PATH}")
    print(f"Publishing to topic '{TOPIC_NAME}' with {DELAY_SECONDS}s delay...\n")

    for index, row in df[REQUIRED_COLUMNS].iterrows():
        payload = row.to_dict()

        # Clean NaN values
        for key, value in payload.items():
            if pd.isna(value):
                payload[key] = 0.0

        # Ensure correct types
        payload["unit_id"]    = int(payload["unit_id"])
        payload["time_cycle"] = int(payload["time_cycle"])

        producer.send(TOPIC_NAME, value=payload)
        print(f"[Row {index + 1:>5}] unit={payload['unit_id']}  "
              f"cycle={payload['time_cycle']}")

        time.sleep(DELAY_SECONDS)

    producer.flush()
    producer.close()
    print("\nAll telemetry records sent successfully.")


if __name__ == "__main__":
    main()