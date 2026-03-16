from pathlib import Path
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import joblib

from src.data_loader import load_cmapss_data
from src.preprocess import add_rul, scale_features
from src.feature_engineering import create_telemetry_features
from src.train_anomaly import train_anomaly_model, detect_anomalies
from src.train_rul import (
    split_rul_data,
    train_rul_model,
    predict_rul,
    evaluate_rul_model
)
from src.decision_engine import recommend_action
from src.evaluate import create_anomaly_ground_truth, evaluate_anomaly_detection
from src.utils import ensure_directory
from src.config import (
    TRAIN_DATA_PATH,
    RESULT_CSV_PATH,
    RUL_METRICS_PATH,
    OUTPUT_DIR,
    PROCESSED_DATA_DIR,
    FEATURE_COLUMNS_PATH
)


# -----------------------------
# Folder paths
# -----------------------------
LOG_DIR = OUTPUT_DIR / "logs"
FIGURE_DIR = OUTPUT_DIR / "figures"
METRICS_DIR = OUTPUT_DIR / "metrics"
SAMPLE_STREAM_DIR = Path("data") / "sample_stream"
NOTEPAD_DIR = Path("notepad")


# -----------------------------
# Logging helper
# -----------------------------
def write_log(log_file: Path, message: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_message = f"[{timestamp}] {message}"
    print(full_message)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(full_message + "\n")


# -----------------------------
# Save anomaly evaluation text
# -----------------------------
def save_anomaly_evaluation(conf_matrix, class_report, log_file: Path):
    ensure_directory(METRICS_DIR)

    cm_txt_path = METRICS_DIR / "anomaly_confusion_matrix.txt"
    report_path = METRICS_DIR / "anomaly_classification_report.txt"

    with open(cm_txt_path, "w", encoding="utf-8") as f:
        f.write("Confusion Matrix\n")
        f.write("================\n")
        f.write(str(conf_matrix))

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Classification Report\n")
        f.write("=====================\n")
        f.write(class_report)

    write_log(log_file, f"Anomaly confusion matrix text saved to: {cm_txt_path}")
    write_log(log_file, f"Anomaly classification report saved to: {report_path}")


# -----------------------------
# Save confusion matrix figure
# -----------------------------
def save_confusion_matrix_figure(conf_matrix, log_file: Path):
    ensure_directory(FIGURE_DIR)

    fig_path = FIGURE_DIR / "anomaly_confusion_matrix.png"

    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False)
    plt.title("Anomaly Detection Confusion Matrix")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close(fig)

    write_log(log_file, f"Confusion matrix figure saved to: {fig_path}")


# -----------------------------
# Save RUL metrics
# -----------------------------
def save_rul_metrics(metrics, log_file: Path):
    ensure_directory(METRICS_DIR)

    lines = [
        "RUL Model Evaluation on Test Data",
        "================================="
    ]
    for key, value in metrics.items():
        lines.append(f"{key}: {value:.4f}")

    content = "\n".join(lines)

    with open(RUL_METRICS_PATH, "w", encoding="utf-8") as f:
        f.write(content)

    metrics_copy_path = METRICS_DIR / "rul_metrics.txt"
    with open(metrics_copy_path, "w", encoding="utf-8") as f:
        f.write(content)

    write_log(log_file, f"RUL metrics saved to: {RUL_METRICS_PATH}")
    write_log(log_file, f"RUL metrics copy saved to: {metrics_copy_path}")


# -----------------------------
# Save processed dataset
# -----------------------------
def save_processed_dataset(df: pd.DataFrame, log_file: Path):
    ensure_directory(PROCESSED_DATA_DIR)
    processed_path = PROCESSED_DATA_DIR / "telemetry_processed.csv"
    df.to_csv(processed_path, index=False)
    write_log(log_file, f"Processed dataset saved to: {processed_path}")


# -----------------------------
# Save sample stream dataset
# -----------------------------
def save_sample_stream(df: pd.DataFrame, log_file: Path, n_rows: int = 200):
    ensure_directory(SAMPLE_STREAM_DIR)

    sample_stream_path = SAMPLE_STREAM_DIR / "sample_telemetry_stream.csv"

    stream_df = df.head(n_rows).copy()
    stream_df.to_csv(sample_stream_path, index=False)

    write_log(log_file, f"Sample stream dataset saved to: {sample_stream_path}")


# -----------------------------
# Save notepad summary
# -----------------------------
def save_notepad_summary(metrics, conf_matrix, log_file: Path):
    ensure_directory(NOTEPAD_DIR)

    summary_path = NOTEPAD_DIR / "run_summary.txt"

    lines = [
        "Space Telemetry Anomaly Detection System - Run Summary",
        "=====================================================",
        "",
        f"Run timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "Generated folders handled:",
        "- outputs/logs",
        "- outputs/figures",
        "- outputs/metrics",
        "- data/processed",
        "- data/sample_stream",
        "- notepad",
        "",
        "RUL Metrics:"
    ]

    for key, value in metrics.items():
        lines.append(f"- {key}: {value:.4f}")

    lines.extend([
        "",
        "Anomaly Confusion Matrix:",
        str(conf_matrix),
        "",
        "Main output files:",
        f"- {RESULT_CSV_PATH}",
        f"- {RUL_METRICS_PATH}",
        f"- {METRICS_DIR / 'anomaly_confusion_matrix.txt'}",
        f"- {METRICS_DIR / 'anomaly_classification_report.txt'}",
        f"- {FIGURE_DIR / 'anomaly_confusion_matrix.png'}",
        f"- {PROCESSED_DATA_DIR / 'telemetry_processed.csv'}",
        f"- {SAMPLE_STREAM_DIR / 'sample_telemetry_stream.csv'}",
        f"- {FEATURE_COLUMNS_PATH}"
    ])

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    write_log(log_file, f"Run summary saved to: {summary_path}")


# -----------------------------
# Save feature columns for API inference
# -----------------------------
def save_feature_columns(feature_cols, log_file: Path):
    feature_columns_path = Path(FEATURE_COLUMNS_PATH)
    ensure_directory(feature_columns_path.parent)
    joblib.dump(feature_cols, feature_columns_path)
    write_log(log_file, f"Feature columns saved to: {feature_columns_path}")


def main():
    ensure_directory(OUTPUT_DIR)
    ensure_directory(LOG_DIR)
    ensure_directory(FIGURE_DIR)
    ensure_directory(METRICS_DIR)
    ensure_directory(PROCESSED_DATA_DIR)
    ensure_directory(SAMPLE_STREAM_DIR)
    ensure_directory(NOTEPAD_DIR)

    log_file = LOG_DIR / "pipeline.log"

    write_log(log_file, "NEW MAIN FILE RUNNING")

    # Step 1: Load dataset
    df = load_cmapss_data(str(TRAIN_DATA_PATH))
    write_log(log_file, "Dataset loaded successfully.")
    write_log(log_file, f"Original shape: {df.shape}")

    # Step 2: Add RUL
    df = add_rul(df)
    write_log(log_file, "RUL column added successfully.")
    write_log(log_file, f"Shape after RUL: {df.shape}")

    # Step 3: Feature engineering
    df = create_telemetry_features(df)
    write_log(log_file, "Feature engineering completed.")
    write_log(log_file, f"Shape after feature engineering: {df.shape}")

    # Step 4: Scale features
    scaled_df, scaler, feature_cols = scale_features(df)
    write_log(log_file, "Feature scaling completed.")
    write_log(log_file, f"Number of model features: {len(feature_cols)}")

    # Step 4A: Save exact feature column order for API inference
    save_feature_columns(feature_cols, log_file)

    # Save processed data
    save_processed_dataset(scaled_df, log_file)

    # Step 5: Train anomaly model
    anomaly_model = train_anomaly_model(scaled_df, feature_cols)

    # Step 6: Detect anomalies
    result_df = detect_anomalies(anomaly_model, scaled_df, feature_cols)
    write_log(log_file, "Anomaly detection completed.")

    # Step 7: Train/test split for RUL model
    X_train, X_test, y_train, y_test = split_rul_data(result_df, feature_cols)
    write_log(log_file, "Train/test split completed for RUL model.")
    write_log(log_file, f"Training rows: {len(X_train)}")
    write_log(log_file, f"Testing rows : {len(X_test)}")

    # Step 8: Train RUL model
    rul_model = train_rul_model(X_train, y_train)

    # Step 9: Predict on test set
    rul_test_df = predict_rul(rul_model, X_test, y_test)

    # Step 10: Evaluate RUL model
    metrics = evaluate_rul_model(rul_test_df)
    write_log(log_file, "RUL model evaluated successfully.")

    # Step 11: Add predicted RUL to full dataset
    result_df["predicted_RUL"] = rul_model.predict(result_df[feature_cols])

    # Step 12: Decision support
    result_df["recommended_action"] = result_df.apply(
        lambda row: recommend_action(
            row["anomaly_flag"],
            row["anomaly_score"],
            row["predicted_RUL"]
        ),
        axis=1
    )
    write_log(log_file, "Decision support recommendations generated.")

    # Step 13: Anomaly evaluation
    result_df = create_anomaly_ground_truth(result_df, threshold=15)
    conf_matrix, class_report = evaluate_anomaly_detection(result_df)
    write_log(log_file, "Anomaly detection evaluation completed.")

    # Step 14: Save anomaly evaluation text + figure
    save_anomaly_evaluation(conf_matrix, class_report, log_file)
    save_confusion_matrix_figure(conf_matrix, log_file)

    # Step 15: Save telemetry results for Streamlit
    result_df.to_csv(RESULT_CSV_PATH, index=False)
    write_log(log_file, f"Results saved successfully to: {RESULT_CSV_PATH}")

    # Step 16: Save RUL metrics
    save_rul_metrics(metrics, log_file)

    # Step 17: Save sample stream data
    save_sample_stream(result_df, log_file, n_rows=200)

    # Step 18: Save notepad summary
    save_notepad_summary(metrics, conf_matrix, log_file)

    write_log(log_file, "Pipeline completed successfully.")


if __name__ == "__main__":
    main()