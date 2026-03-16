from pathlib import Path

# Base project directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = BASE_DIR / "models"
FEATURE_COLUMNS_PATH = str(BASE_DIR / "models" / "feature_columns.pkl")

# Main dataset path
TRAIN_DATA_PATH = RAW_DATA_DIR / "train_FD001.txt"

# Output files
RESULT_CSV_PATH = OUTPUT_DIR / "telemetry_results.csv"
RUL_METRICS_PATH = OUTPUT_DIR / "rul_metrics.txt"

# Model files
ANOMALY_MODEL_PATH = MODEL_DIR / "anomaly_model.pkl"
RUL_MODEL_PATH = MODEL_DIR / "rul_model.pkl"

# Random states
RANDOM_STATE = 42

# Anomaly model settings
ANOMALY_N_ESTIMATORS = 100
ANOMALY_CONTAMINATION = 0.05

# RUL model settings
RUL_N_ESTIMATORS = 100

# Feature engineering settings
ROLLING_WINDOW = 5
SELECTED_SENSORS = ["sensor_2", "sensor_3", "sensor_4", "sensor_11", "sensor_15", "sensor_21"]