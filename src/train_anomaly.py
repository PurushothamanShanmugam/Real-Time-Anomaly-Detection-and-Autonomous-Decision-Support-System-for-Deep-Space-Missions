import joblib
from pathlib import Path
from sklearn.ensemble import IsolationForest
from src.config import ANOMALY_MODEL_PATH, ANOMALY_N_ESTIMATORS, ANOMALY_CONTAMINATION, RANDOM_STATE


def train_anomaly_model(df, feature_cols):
    """
    Train Isolation Forest for anomaly detection.
    """
    model = IsolationForest(
        n_estimators=ANOMALY_N_ESTIMATORS,
        contamination=ANOMALY_CONTAMINATION,
        random_state=RANDOM_STATE
    )

    model.fit(df[feature_cols])

    Path(ANOMALY_MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, ANOMALY_MODEL_PATH)

    print(f"Anomaly detection model trained and saved to {ANOMALY_MODEL_PATH}")

    return model


def detect_anomalies(model, df, feature_cols):
    """
    Detect anomalies in telemetry data.
    """
    df = df.copy()

    scores = model.decision_function(df[feature_cols])
    predictions = model.predict(df[feature_cols])

    df["anomaly_score"] = scores
    df["anomaly_flag"] = predictions

    return df