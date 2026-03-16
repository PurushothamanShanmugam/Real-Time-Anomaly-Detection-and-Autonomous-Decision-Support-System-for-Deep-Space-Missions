import pandas as pd
from pathlib import Path

# Column structure for NASA C-MAPSS telemetry dataset
COLUMNS = (
    ["unit_id", "time_cycle", "op_setting_1", "op_setting_2", "op_setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)

def load_cmapss_data(file_path: str) -> pd.DataFrame:
    """
    Load NASA C-MAPSS dataset from a whitespace-separated file.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None
    )

    # Some versions include extra empty columns
    df = df.dropna(axis=1, how="all")

    if df.shape[1] != len(COLUMNS):
        raise ValueError(
            f"Unexpected number of columns. Got {df.shape[1]}, expected {len(COLUMNS)}"
        )

    df.columns = COLUMNS

    return df