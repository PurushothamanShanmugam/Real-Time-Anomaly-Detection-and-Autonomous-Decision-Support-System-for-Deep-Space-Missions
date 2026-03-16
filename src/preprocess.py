import pandas as pd
from sklearn.preprocessing import StandardScaler


def add_rul(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add Remaining Useful Life (RUL) column.
    RUL = max cycle of each unit - current cycle
    """
    df = df.copy()

    max_cycle_df = df.groupby("unit_id")["time_cycle"].max().reset_index()
    max_cycle_df.columns = ["unit_id", "max_cycle"]

    df = df.merge(max_cycle_df, on="unit_id", how="left")
    df["RUL"] = df["max_cycle"] - df["time_cycle"]

    df.drop(columns=["max_cycle"], inplace=True)

    return df


def scale_features(df: pd.DataFrame):
    """
    Scale numeric telemetry features for ML models.
    """
    df = df.copy()

    feature_cols = [col for col in df.columns if col not in ["unit_id", "time_cycle", "RUL"]]

    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    return df, scaler, feature_cols