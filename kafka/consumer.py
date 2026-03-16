import json
import requests
from kafka import KafkaConsumer

TOPIC_NAME = "telemetry-input"
BOOTSTRAP_SERVERS = "localhost:9092"
API_URL = "http://localhost:8000/predict"


def main():
    consumer = KafkaConsumer(
        TOPIC_NAME,
        bootstrap_servers=BOOTSTRAP_SERVERS,
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        group_id="telemetry-consumer-group",
        value_deserializer=lambda m: json.loads(m.decode("utf-8"))
    )

    print("Kafka consumer started. Waiting for telemetry messages...")

    for message in consumer:
        payload = message.value
        print(f"Received: {payload}")

        try:
            response = requests.post(API_URL, json=payload, timeout=10)
            print(f"API status: {response.status_code}")

            if response.status_code == 200:
                print("Prediction:", response.json())
            else:
                print("API error:", response.text)

        except Exception as e:
            print("Request failed:", str(e))


if __name__ == "__main__":
    main()