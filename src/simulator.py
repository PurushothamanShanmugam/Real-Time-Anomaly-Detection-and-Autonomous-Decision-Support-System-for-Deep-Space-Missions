import time
import pandas as pd


def simulate_telemetry_stream(df: pd.DataFrame, unit_id: int, delay: float = 0.1):
    """
    Simulate telemetry streaming for a selected engine unit.

    Parameters:
        df (pd.DataFrame): Input dataframe containing telemetry data.
        unit_id (int): Engine unit to simulate.
        delay (float): Delay in seconds between records.

    Yields:
        dict: One telemetry row at a time as a dictionary.
    """
    unit_df = df[df["unit_id"] == unit_id].sort_values("time_cycle").copy()

    for _, row in unit_df.iterrows():
        time.sleep(delay)
        yield row.to_dict()