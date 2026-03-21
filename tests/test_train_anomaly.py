import pandas as pd
import numpy as np
from src.train_anomaly import train_anomaly_model, detect_anomalies


def _make_df():
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "unit_id": np.repeat([1, 2], n // 2),
        "time_cycle": list(range(n // 2)) * 2,
        "RUL": np.random.randint(0, 200, n),
        "feature_a": np.random.randn(n),
        "feature_b": np.random.randn(n),
    })
    return df


def test_train_anomaly_model():
    df = _make_df()
    feature_cols = ["feature_a", "feature_b"]
    model = train_anomaly_model(df, feature_cols)
    assert model is not None


def test_detect_anomalies_columns():
    df = _make_df()
    feature_cols = ["feature_a", "feature_b"]
    model = train_anomaly_model(df, feature_cols)
    result = detect_anomalies(model, df, feature_cols)
    assert "anomaly_score" in result.columns
    assert "anomaly_flag" in result.columns
    assert set(result["anomaly_flag"].unique()).issubset({1, -1})