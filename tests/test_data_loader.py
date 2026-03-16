import pandas as pd
from pathlib import Path
from src.data_loader import load_cmapss_data


def test_load_cmapss_data():
    file_path = Path("data/raw/train_FD001.txt")
    df = load_cmapss_data(file_path)

    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] > 0
    assert df.shape[1] == 26
    assert "unit_id" in df.columns
    assert "time_cycle" in df.columns