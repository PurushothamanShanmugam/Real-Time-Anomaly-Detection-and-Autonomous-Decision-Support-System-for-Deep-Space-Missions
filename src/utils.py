import pandas as pd
from pathlib import Path
from datetime import datetime


def ensure_directory(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log_message(message: str):
    print(f"[{get_timestamp()}] {message}")


def load_results_csv(path: str):
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")
    return pd.read_csv(csv_path)