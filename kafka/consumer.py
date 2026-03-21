"""
kafka/consumer.py
-----------------
Consumes telemetry messages from the Kafka topic 'telemetry-input',
forwards each one to the FastAPI /predict endpoint, and prints
the full prediction result including risk score and confidence.

Usage:
    # Make sure both Kafka and the API are running:
    docker-compose up -d kafka zookeeper
    uvicorn api.main:app --host 0.0.0.0 --port 8000

    # Then run in a separate terminal:
    python kafka/consumer.py
"""

import json
import requests
from kafka import KafkaConsumer

TOPIC_NAME        = "telemetry-input"
BOOTSTRAP_SERVERS = "localhost:9092"
API_URL           = "http://localhost:8000/predict"
GROUP_ID          = "telemetry-consumer-group"


def main():
    print(f"Connecting to Kafka at {BOOTSTRAP_SERVERS}...")
    consumer = KafkaConsumer(
        TOPIC_NAME,
        bootstrap_servers=BOOTSTRAP_SERVERS,
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        group_id=GROUP_ID,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
    )

    print(f"Listening on topic '{TOPIC_NAME}'. Forwarding to {API_URL}...\n")

    for message in consumer:
        payload = message.value

        print(f"[Received] unit={payload.get('unit_id')}  "
              f"cycle={payload.get('time_cycle')}")

        try:
            response = requests.post(API_URL, json=payload, timeout=10)

            if response.status_code == 200:
                result = response.json()
                print(f"  Anomaly flag   : {result['anomaly_flag']}")
                print(f"  Predicted RUL  : {result['predicted_rul']:.1f}")
                print(f"  Risk score     : {result['risk_score']:.3f}")
                print(f"  Confidence     : {result['confidence']}")
                print(f"  Action         : {result['recommended_action']}")
                print(f"  Justification  : {result['justification']}")

                # Alert on critical cases
                if result.get("anomaly_detected"):
                    print(f"  *** ANOMALY DETECTED — {result['recommended_action']} ***")
            else:
                print(f"  API error {response.status_code}: {response.text}")

        except requests.exceptions.ConnectionError:
            print("  ERROR: Cannot reach API. Is uvicorn running on port 8000?")
        except Exception as e:
            print(f"  ERROR: {e}")

        print()


if __name__ == "__main__":
    main()