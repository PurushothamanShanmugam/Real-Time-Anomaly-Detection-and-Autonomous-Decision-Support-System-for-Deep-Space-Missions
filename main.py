"""
main.py — Complete AI Pipeline for Space Telemetry Anomaly Detection System
============================================================================
Covers all India Space Academy project objectives in one pipeline:

  Phase 1  — Data loading, feature engineering, preprocessing
  Phase 2  — Isolation Forest anomaly detection          (Day 3)
           — Autoencoder anomaly detection               (Day 10)
           — Random Forest RUL prediction                (Day 11)
           — LSTM RUL prediction                         (Day 4)
           — Model evaluation (ROC-AUC, confusion matrix, MAE/RMSE/R2)
  Phase 3  — Rule-based decision engine                  (Day 13)
           — Risk scoring + decision confidence          (Days 16-17)
           — Reinforcement Learning Q-table demo         (Day 14)
  Phase 4  — Dashboard output (telemetry_results.csv)   (Days 18-20)

Usage:
    python main.py

TensorFlow is required for LSTM and Autoencoder:
    pip install tensorflow
If TensorFlow is not installed, those two steps are skipped automatically
and the rest of the pipeline runs fine.
"""

from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import ConfusionMatrixDisplay

# ── Core src imports ────────────────────────────────────────────────
from src.data_loader        import load_cmapss_data
from src.preprocess         import add_rul, scale_features
from src.feature_engineering import create_telemetry_features
from src.train_anomaly      import train_anomaly_model, detect_anomalies
from src.train_rul          import split_rul_data, train_rul_model, predict_rul, evaluate_rul_model
from src.decision_engine    import recommend_action
from src.evaluate           import create_anomaly_ground_truth, evaluate_anomaly_detection
from src.utils              import ensure_directory

# ── New module imports ───────────────────────────────────────────────
from src.train_autoencoder  import (
    train_autoencoder, detect_anomalies_autoencoder,
    evaluate_autoencoder, save_autoencoder_evaluation
)
from src.train_rul_lstm     import (
    split_lstm_data, train_lstm_model,
    evaluate_lstm_model, save_lstm_metrics
)
from src.decision_confidence import (
    recommend_action_with_confidence,
    run_q_learning_demo, print_rl_policy
)

# ── Config imports ───────────────────────────────────────────────────
from src.config import (
    TRAIN_DATA_PATH,
    RESULT_CSV_PATH,
    RUL_METRICS_PATH,
    OUTPUT_DIR,
    PROCESSED_DATA_DIR,
    FEATURE_COLUMNS_PATH,
)

# ── Directory layout ─────────────────────────────────────────────────
LOG_DIR          = OUTPUT_DIR / "logs"
FIGURE_DIR       = OUTPUT_DIR / "figures"
METRICS_DIR      = OUTPUT_DIR / "metrics"
SAMPLE_STREAM_DIR = Path("data") / "sample_stream"
NOTEPAD_DIR      = Path("notepad")


# ════════════════════════════════════════════════════════════════════
#  Helpers
# ════════════════════════════════════════════════════════════════════

def write_log(log_file: Path, message: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def save_anomaly_evaluation(conf_matrix, class_report, log_file: Path):
    ensure_directory(METRICS_DIR)
    cm_path     = METRICS_DIR / "anomaly_confusion_matrix.txt"
    report_path = METRICS_DIR / "anomaly_classification_report.txt"

    cm_path.write_text(
        f"Confusion Matrix\n================\n{conf_matrix}", encoding="utf-8")
    report_path.write_text(
        f"Classification Report\n=====================\n{class_report}", encoding="utf-8")

    write_log(log_file, f"Confusion matrix saved  → {cm_path}")
    write_log(log_file, f"Classification report   → {report_path}")


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
    write_log(log_file, f"Confusion matrix figure → {fig_path}")


def save_rul_metrics(metrics: dict, label: str, log_file: Path):
    """Save RUL metrics for either the RF or LSTM model."""
    ensure_directory(METRICS_DIR)

    filename = "rul_metrics.txt" if label == "RF" else "rul_lstm_metrics.txt"
    title    = "RUL Model Evaluation on Test Data" if label == "RF" \
               else "LSTM RUL Model Evaluation on Test Data"
    sep      = "=================================" if label == "RF" \
               else "======================================="

    lines = [title, sep] + [f"{k}: {v:.4f}" for k, v in metrics.items()]
    content = "\n".join(lines)

    (METRICS_DIR / filename).write_text(content, encoding="utf-8")

    # Keep top-level rul_metrics.txt pointing at RF (for backward compat)
    if label == "RF":
        RUL_METRICS_PATH.write_text(content, encoding="utf-8")

    write_log(log_file, f"{label} metrics saved → {METRICS_DIR / filename}")


def save_model_comparison(rf_metrics: dict, lstm_metrics: dict, log_file: Path):
    path = METRICS_DIR / "rul_model_comparison.txt"
    lines = [
        "RUL Model Comparison: Random Forest vs LSTM",
        "=" * 46,
        f"{'Metric':<10} {'Random Forest':>16} {'LSTM':>12}",
        "-" * 40,
    ]
    for key in ["MAE", "RMSE", "R2"]:
        rf_val   = rf_metrics.get(key, float("nan"))
        lstm_val = lstm_metrics.get(key, float("nan"))
        lines.append(f"{key:<10} {rf_val:>16.4f} {lstm_val:>12.4f}")
    path.write_text("\n".join(lines), encoding="utf-8")
    write_log(log_file, f"Model comparison saved  → {path}")


def save_processed_dataset(df: pd.DataFrame, log_file: Path):
    ensure_directory(PROCESSED_DATA_DIR)
    path = PROCESSED_DATA_DIR / "telemetry_processed.csv"
    df.to_csv(path, index=False)
    write_log(log_file, f"Processed dataset saved → {path}")


def save_sample_stream(df: pd.DataFrame, log_file: Path, n_rows: int = 200):
    ensure_directory(SAMPLE_STREAM_DIR)
    path = SAMPLE_STREAM_DIR / "sample_telemetry_stream.csv"
    df.head(n_rows).to_csv(path, index=False)
    write_log(log_file, f"Sample stream saved     → {path}")


def save_notepad_summary(rf_metrics: dict, lstm_metrics: dict,
                         conf_matrix, log_file: Path):
    ensure_directory(NOTEPAD_DIR)
    path = NOTEPAD_DIR / "run_summary.txt"
    lines = [
        "Space Telemetry Anomaly Detection System — Run Summary",
        "=======================================================",
        "",
        f"Run timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "RF RUL Metrics:",
    ] + [f"  {k}: {v:.4f}" for k, v in rf_metrics.items()]

    if lstm_metrics:
        lines += ["", "LSTM RUL Metrics:"] + [f"  {k}: {v:.4f}" for k, v in lstm_metrics.items()]

    lines += [
        "",
        "Anomaly Confusion Matrix:",
        str(conf_matrix),
        "",
        "Main output files:",
        f"  {RESULT_CSV_PATH}",
        f"  {RUL_METRICS_PATH}",
        f"  {METRICS_DIR / 'anomaly_confusion_matrix.txt'}",
        f"  {METRICS_DIR / 'anomaly_classification_report.txt'}",
        f"  {FIGURE_DIR / 'anomaly_confusion_matrix.png'}",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    write_log(log_file, f"Run summary saved       → {path}")


# ════════════════════════════════════════════════════════════════════
#  Main pipeline
# ════════════════════════════════════════════════════════════════════

def main():

    # ── Create all directories ───────────────────────────────────────
    for d in [OUTPUT_DIR, LOG_DIR, FIGURE_DIR, METRICS_DIR,
               PROCESSED_DATA_DIR, SAMPLE_STREAM_DIR, NOTEPAD_DIR]:
        ensure_directory(d)

    log_file = LOG_DIR / "pipeline.log"
    write_log(log_file, "=" * 60)
    write_log(log_file, "  PIPELINE STARTED")
    write_log(log_file, "=" * 60)

    # ────────────────────────────────────────────────────────────────
    # PHASE 1 — Data loading & preprocessing
    # ────────────────────────────────────────────────────────────────

    # Step 1: Load dataset
    write_log(log_file, "Step 1 | Loading dataset...")
    df = load_cmapss_data(str(TRAIN_DATA_PATH))
    write_log(log_file, f"        Dataset loaded — shape: {df.shape}")

    # Step 2: Add RUL column
    write_log(log_file, "Step 2 | Adding RUL column...")
    df = add_rul(df)
    write_log(log_file, f"        RUL added — shape: {df.shape}")

    # Step 3: Feature engineering
    write_log(log_file, "Step 3 | Feature engineering (rolling mean + diff)...")
    df = create_telemetry_features(df)
    write_log(log_file, f"        Features created — shape: {df.shape}")

    # Step 4: Scale features
    write_log(log_file, "Step 4 | Scaling features (StandardScaler)...")
    scaled_df, scaler, feature_cols = scale_features(df)
    write_log(log_file, f"        Scaled — {len(feature_cols)} feature columns")

    # Save feature columns for API inference
    feature_columns_path = Path(FEATURE_COLUMNS_PATH)
    ensure_directory(feature_columns_path.parent)
    joblib.dump(feature_cols, feature_columns_path)
    write_log(log_file, f"        Feature columns saved → {feature_columns_path}")

    # Save processed dataset
    save_processed_dataset(scaled_df, log_file)

    # ────────────────────────────────────────────────────────────────
    # PHASE 2A — Anomaly Detection: Isolation Forest (Day 3)
    # ────────────────────────────────────────────────────────────────

    write_log(log_file, "Step 5 | Training Isolation Forest anomaly model...")
    anomaly_model = train_anomaly_model(scaled_df, feature_cols)
    result_df     = detect_anomalies(anomaly_model, scaled_df, feature_cols)
    write_log(log_file, "        Isolation Forest anomaly detection complete.")

    # ────────────────────────────────────────────────────────────────
    # PHASE 2B — Anomaly Detection: Autoencoder (Day 10)
    # ────────────────────────────────────────────────────────────────

    write_log(log_file, "Step 6 | Training Autoencoder for anomaly detection...")
    lstm_metrics = {}   # initialise in case LSTM is skipped
    try:
        ae_model, ae_threshold = train_autoencoder(
            result_df, feature_cols,
            rul_threshold=50, epochs=50, batch_size=256
        )
        result_df = detect_anomalies_autoencoder(
            ae_model, result_df, feature_cols, ae_threshold
        )
        ae_cm, ae_report, ae_roc, ae_pr = evaluate_autoencoder(result_df, rul_threshold=15)
        save_autoencoder_evaluation(ae_cm, ae_report, ae_roc, ae_pr, METRICS_DIR)
        write_log(log_file, f"        Autoencoder — ROC-AUC: {ae_roc:.4f}  PR-AUC: {ae_pr:.4f}")

    except RuntimeError as e:
        write_log(log_file, f"        Autoencoder skipped: {e}")
        write_log(log_file, "        Install TensorFlow to enable: pip install tensorflow")
        result_df["reconstruction_error"]     = float("nan")
        result_df["autoencoder_anomaly_flag"] = 0

    # ────────────────────────────────────────────────────────────────
    # PHASE 2C — RUL Prediction: Random Forest baseline (Day 11)
    # ────────────────────────────────────────────────────────────────

    write_log(log_file, "Step 7 | Training Random Forest RUL model...")
    X_train, X_test, y_train, y_test = split_rul_data(result_df, feature_cols)
    write_log(log_file, f"        Train rows: {len(X_train)}  Test rows: {len(X_test)}")

    rul_model   = train_rul_model(X_train, y_train)
    rul_test_df = predict_rul(rul_model, X_test, y_test)
    rf_metrics  = evaluate_rul_model(rul_test_df)
    write_log(log_file, f"        RF — MAE: {rf_metrics['MAE']:.4f}  R2: {rf_metrics['R2']:.4f}")

    # Predict RUL for entire dataset
    result_df["predicted_RUL"] = rul_model.predict(result_df[feature_cols])
    save_rul_metrics(rf_metrics, "RF", log_file)

    # ────────────────────────────────────────────────────────────────
    # PHASE 2D — RUL Prediction: LSTM deep learning (Day 4)
    # ────────────────────────────────────────────────────────────────

    write_log(log_file, "Step 8 | Training LSTM RUL model...")
    try:
        X_tr_seq, X_te_seq, y_tr_seq, y_te_seq = split_lstm_data(
            result_df, feature_cols
        )
        lstm_model   = train_lstm_model(X_tr_seq, y_tr_seq, epochs=30, batch_size=64)
        lstm_metrics = evaluate_lstm_model(lstm_model, X_te_seq, y_te_seq)
        save_lstm_metrics(lstm_metrics, METRICS_DIR)
        save_model_comparison(rf_metrics, lstm_metrics, log_file)
        write_log(log_file, f"        LSTM — MAE: {lstm_metrics['MAE']:.4f}  R2: {lstm_metrics['R2']:.4f}")

    except RuntimeError as e:
        write_log(log_file, f"        LSTM skipped: {e}")
        write_log(log_file, "        Install TensorFlow to enable: pip install tensorflow")

    # ────────────────────────────────────────────────────────────────
    # PHASE 3A — Decision engine: rule-based actions (Day 13)
    # ────────────────────────────────────────────────────────────────

    write_log(log_file, "Step 9 | Generating maintenance recommendations...")
    result_df["recommended_action"] = result_df.apply(
        lambda row: recommend_action(
            row["anomaly_flag"],
            row["anomaly_score"],
            row["predicted_RUL"]
        ), axis=1
    )
    write_log(log_file, "        Recommendations generated.")

    # ────────────────────────────────────────────────────────────────
    # PHASE 3B — Risk scoring + decision confidence (Days 16-17)
    # ────────────────────────────────────────────────────────────────

    write_log(log_file, "Step 10 | Computing risk scores and decision confidence...")
    confidence_cols = result_df.apply(
        lambda row: recommend_action_with_confidence(
            row["anomaly_flag"],
            row["anomaly_score"],
            row["predicted_RUL"]
        ), axis=1, result_type="expand"
    )
    # Add only the new columns (risk_score, confidence, justification)
    for col in ["risk_score", "confidence", "justification"]:
        result_df[col] = confidence_cols[col]
    write_log(log_file, "        Risk scoring and confidence estimation complete.")

    # ────────────────────────────────────────────────────────────────
    # PHASE 3C — Reinforcement Learning Q-table demo (Day 14)
    # ────────────────────────────────────────────────────────────────

    write_log(log_file, "Step 11 | Running Q-learning RL demo...")
    Q = run_q_learning_demo(episodes=500)
    print_rl_policy(Q)
    rl_path = METRICS_DIR / "rl_q_table.txt"
    np.savetxt(str(rl_path), Q, fmt="%.4f",
               header="Q-table (8 states x 4 actions)")
    write_log(log_file, f"        Q-table saved → {rl_path}")

    # ────────────────────────────────────────────────────────────────
    # PHASE 2E — Anomaly evaluation (confusion matrix, report)
    # ────────────────────────────────────────────────────────────────

    write_log(log_file, "Step 12 | Evaluating anomaly detection...")
    result_df   = create_anomaly_ground_truth(result_df, threshold=15)
    conf_matrix, class_report = evaluate_anomaly_detection(result_df)
    save_anomaly_evaluation(conf_matrix, class_report, log_file)
    save_confusion_matrix_figure(conf_matrix, log_file)
    write_log(log_file, "        Anomaly evaluation complete.")

    # ────────────────────────────────────────────────────────────────
    # PHASE 4 — Save all outputs
    # ────────────────────────────────────────────────────────────────

    write_log(log_file, "Step 13 | Saving final outputs...")
    result_df.to_csv(RESULT_CSV_PATH, index=False)
    write_log(log_file, f"        Results CSV → {RESULT_CSV_PATH}")

    save_sample_stream(result_df, log_file, n_rows=200)
    save_notepad_summary(rf_metrics, lstm_metrics, conf_matrix, log_file)

    # ── Final summary ────────────────────────────────────────────────
    write_log(log_file, "=" * 60)
    write_log(log_file, "  PIPELINE COMPLETED SUCCESSFULLY")
    write_log(log_file, "=" * 60)

    print("\n" + "=" * 60)
    print("  Pipeline Summary")
    print("=" * 60)
    print(f"  RF   RUL — MAE: {rf_metrics['MAE']:.2f}  RMSE: {rf_metrics['RMSE']:.2f}  R2: {rf_metrics['R2']:.4f}")
    if lstm_metrics:
        print(f"  LSTM RUL — MAE: {lstm_metrics['MAE']:.2f}  RMSE: {lstm_metrics['RMSE']:.2f}  R2: {lstm_metrics['R2']:.4f}")
    else:
        print("  LSTM RUL — skipped (install TensorFlow to enable)")
    print(f"  Output columns: {list(result_df.columns)}")
    print(f"  Dashboard file: {RESULT_CSV_PATH}")
    print("=" * 60)
    print("\n  Run dashboard: streamlit run app/streamlit_app.py\n")


if __name__ == "__main__":
    main()