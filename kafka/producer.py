import json
import time
import pandas as pd
from kafka import KafkaProducer

TOPIC_NAME = "telemetry-input"
BOOTSTRAP_SERVERS = "localhost:9092"
CSV_PATH = "outputs/telemetry_results.csv"  # change later if needed


def serializer(message: dict) -> bytes:
    return json.dumps(message).encode("utf-8")


def main():
    producer = KafkaProducer(
        bootstrap_servers=BOOTSTRAP_SERVERS,
        value_serializer=serializer
    )

    df = pd.read_csv(CSV_PATH)

    required_columns = [
        "unit_id", "time_cycle",
        "op_setting_1", "op_setting_2", "op_setting_3",
        "sensor_1", "sensor_2", "sensor_3", "sensor_4", "sensor_5",
        "sensor_6", "sensor_7", "sensor_8", "sensor_9", "sensor_10",
        "sensor_11", "sensor_12", "sensor_13", "sensor_14", "sensor_15",
        "sensor_16", "sensor_17", "sensor_18", "sensor_19", "sensor_20",
        "sensor_21"
    ]

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    for index, row in df[required_columns].iterrows():
        payload = row.to_dict()

        for key, value in payload.items():
            if pd.isna(value):
                payload[key] = 0.0

        payload["unit_id"] = int(payload["unit_id"])
        payload["time_cycle"] = int(payload["time_cycle"])

        producer.send(TOPIC_NAME, value=payload)
        print(f"Sent row {index + 1}: {payload}")

        time.sleep(1)

    producer.flush()
    producer.close()
    print("Finished sending telemetry messages.")


if __name__ == "__main__":
    main()