from pathlib import Path
import joblib
import pandas as pd

from src.config import ANOMALY_MODEL_PATH, RUL_MODEL_PATH, FEATURE_COLUMNS_PATH
from src.decision_engine import recommend_action
from src.feature_engineering import create_telemetry_features


MODEL_CACHE = {
    "anomaly_model": None,
    "rul_model": None,
    "feature_columns": None,
}


def load_models():
    if MODEL_CACHE["anomaly_model"] is None:
        anomaly_path = Path(ANOMALY_MODEL_PATH)
        if not anomaly_path.exists():
            raise FileNotFoundError(f"Anomaly model not found: {ANOMALY_MODEL_PATH}")
        MODEL_CACHE["anomaly_model"] = joblib.load(anomaly_path)

    if MODEL_CACHE["rul_model"] is None:
        rul_path = Path(RUL_MODEL_PATH)
        if not rul_path.exists():
            raise FileNotFoundError(f"RUL model not found: {RUL_MODEL_PATH}")
        MODEL_CACHE["rul_model"] = joblib.load(rul_path)

    if MODEL_CACHE["feature_columns"] is None:
        feature_cols_path = Path(FEATURE_COLUMNS_PATH)
        if not feature_cols_path.exists():
            raise FileNotFoundError(f"Feature columns file not found: {FEATURE_COLUMNS_PATH}")
        MODEL_CACHE["feature_columns"] = joblib.load(feature_cols_path)

    return (
        MODEL_CACHE["anomaly_model"],
        MODEL_CACHE["rul_model"],
        MODEL_CACHE["feature_columns"],
    )


def prepare_input(payload: dict) -> pd.DataFrame:
    df = pd.DataFrame([payload])

    # Recreate engineered features used during training
    df = create_telemetry_features(df)

    _, _, feature_columns = load_models()

    # Add any missing columns so model input exactly matches training
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    # Keep exact order used during training
    df = df[feature_columns]

    return df


def predict_single(payload: dict) -> dict:
    anomaly_model, rul_model, _ = load_models()
    X = prepare_input(payload)

    anomaly_score = float(anomaly_model.decision_function(X)[0])
    anomaly_flag = int(anomaly_model.predict(X)[0])
    predicted_rul = float(rul_model.predict(X)[0])

    action = recommend_action(
        anomaly_flag=anomaly_flag,
        anomaly_score=anomaly_score,
        rul=predicted_rul,
    )

    return {
        "unit_id": int(payload["unit_id"]),
        "time_cycle": int(payload["time_cycle"]),
        "anomaly_score": anomaly_score,
        "anomaly_flag": anomaly_flag,
        "predicted_rul": predicted_rul,
        "recommended_action": action,
    }