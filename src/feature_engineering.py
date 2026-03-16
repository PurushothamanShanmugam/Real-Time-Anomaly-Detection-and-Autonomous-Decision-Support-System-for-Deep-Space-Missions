import pandas as pd
from src.config import SELECTED_SENSORS, ROLLING_WINDOW


def create_telemetry_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create telemetry features for selected sensors:
    1. rolling mean
    2. cycle-to-cycle difference
    """
    df = df.copy()

    for sensor in SELECTED_SENSORS:
        if sensor in df.columns:
            df[f"{sensor}_rolling_mean"] = (
                df.groupby("unit_id")[sensor]
                .transform(lambda x: x.rolling(window=ROLLING_WINDOW, min_periods=1).mean())
            )

            df[f"{sensor}_diff"] = (
                df.groupby("unit_id")[sensor]
                .transform(lambda x: x.diff().fillna(0))
            )

    return df