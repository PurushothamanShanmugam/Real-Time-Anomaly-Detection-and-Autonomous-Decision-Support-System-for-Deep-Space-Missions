"""
Autoencoder-based anomaly detection model.
Covers Day 10 objective: Deep Learning Autoencoder for anomaly detection.

A dense autoencoder is trained on normal telemetry data only.
At inference, high reconstruction error = anomaly.
Results are compared against Isolation Forest in main_extended.py.
"""

import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)

try:
    import tensorflow as tf
    from tensorflow.keras.models import Model, load_model
    from tensorflow.keras.layers import Input, Dense
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("[Autoencoder] TensorFlow not installed. Run: pip install tensorflow")

from src.config import MODEL_DIR, RANDOM_STATE

AUTOENCODER_MODEL_PATH   = MODEL_DIR / "autoencoder_model.keras"
AUTOENCODER_THRESHOLD_PATH = MODEL_DIR / "autoencoder_threshold.pkl"


# ------------------------------------------------------------------ #
#  Model definition
# ------------------------------------------------------------------ #

def build_autoencoder(n_features):
    """
    Symmetric dense autoencoder.

    Encoder: n_features → 32 → 16 → 8  (bottleneck)
    Decoder: 8 → 16 → 32 → n_features

    The bottleneck forces the model to learn a compressed representation
    of normal telemetry. Anomalous signals produce higher reconstruction error.
    """
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow is required. Install with: pip install tensorflow")

    tf.random.set_seed(RANDOM_STATE)

    inputs  = Input(shape=(n_features,), name="input")
    encoded = Dense(32, activation="relu",    name="enc_32")(inputs)
    encoded = Dense(16, activation="relu",    name="enc_16")(encoded)
    bottleneck = Dense(8, activation="relu",  name="bottleneck")(encoded)
    decoded = Dense(16, activation="relu",    name="dec_16")(bottleneck)
    decoded = Dense(32, activation="relu",    name="dec_32")(decoded)
    outputs = Dense(n_features, activation="linear", name="output")(decoded)

    autoencoder = Model(inputs, outputs, name="telemetry_autoencoder")
    autoencoder.compile(optimizer="adam", loss="mse")
    return autoencoder


# ------------------------------------------------------------------ #
#  Train
# ------------------------------------------------------------------ #

def train_autoencoder(df, feature_cols, rul_threshold=50, epochs=50, batch_size=256):
    """
    Train the autoencoder exclusively on 'normal' samples
    (engines with RUL > rul_threshold), so it learns the pattern
    of healthy operation and flags degraded signals as outliers.

    Parameters
    ----------
    df            : preprocessed and scaled DataFrame
    feature_cols  : list of feature column names
    rul_threshold : cycles above which an engine is considered 'normal'
    epochs        : max training epochs (early stopping applies)
    batch_size    : mini-batch size

    Returns
    -------
    model         : trained Keras autoencoder
    threshold     : float — reconstruction MSE threshold for anomaly detection
    """
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow is required. Install with: pip install tensorflow")

    # Train only on normal (healthy) data
    normal_df  = df[df["RUL"] > rul_threshold]
    X_normal   = normal_df[feature_cols].values.astype(np.float32)

    n_features = X_normal.shape[1]
    model      = build_autoencoder(n_features)

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    print(f"\nTraining Autoencoder on {len(X_normal)} normal samples "
          f"(RUL > {rul_threshold}) out of {len(df)} total rows")

    model.fit(
        X_normal, X_normal,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.15,
        callbacks=[early_stop],
        shuffle=True,
        verbose=1
    )

    # Set threshold at 95th percentile of reconstruction error on normal data
    recon        = model.predict(X_normal, verbose=0)
    mse_normal   = np.mean((X_normal - recon) ** 2, axis=1)
    threshold    = float(np.percentile(mse_normal, 95))

    print(f"\nAnomaly threshold (95th pct of normal MSE): {threshold:.6f}")

    # Save model and threshold
    Path(AUTOENCODER_MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
    model.save(AUTOENCODER_MODEL_PATH)
    joblib.dump(threshold, AUTOENCODER_THRESHOLD_PATH)

    print(f"Autoencoder model saved to {AUTOENCODER_MODEL_PATH}")
    print(f"Threshold saved to {AUTOENCODER_THRESHOLD_PATH}")

    return model, threshold


# ------------------------------------------------------------------ #
#  Inference
# ------------------------------------------------------------------ #

def detect_anomalies_autoencoder(model, df, feature_cols, threshold):
    """
    Add reconstruction_error and autoencoder_anomaly_flag columns to df.

    autoencoder_anomaly_flag:
        1  = anomaly  (reconstruction error > threshold)
        0  = normal
    """
    df = df.copy()
    X  = df[feature_cols].values.astype(np.float32)

    recon = model.predict(X, verbose=0)
    df["reconstruction_error"]       = np.mean((X - recon) ** 2, axis=1)
    df["autoencoder_anomaly_flag"]   = (df["reconstruction_error"] > threshold).astype(int)

    return df


# ------------------------------------------------------------------ #
#  Evaluate
# ------------------------------------------------------------------ #

def evaluate_autoencoder(df, rul_threshold=15):
    """
    Compare autoencoder anomaly flags against the RUL-based ground truth
    (same method used by evaluate_anomaly_detection in evaluate.py).

    Returns confusion matrix and classification report strings.
    """
    df = df.copy()
    y_true = (df["RUL"] <= rul_threshold).astype(int)
    y_pred = df["autoencoder_anomaly_flag"]

    cm     = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=4)

    # ROC-AUC using raw reconstruction error as the score
    try:
        roc_auc = roc_auc_score(y_true, df["reconstruction_error"])
        pr_auc  = average_precision_score(y_true, df["reconstruction_error"])
    except Exception:
        roc_auc = pr_auc = float("nan")

    print("\nAutoencoder Anomaly Detection Evaluation:")
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)
    print(f"ROC-AUC  : {roc_auc:.4f}")
    print(f"PR-AUC   : {pr_auc:.4f}")

    return cm, report, roc_auc, pr_auc


# ------------------------------------------------------------------ #
#  Save evaluation results
# ------------------------------------------------------------------ #

def save_autoencoder_evaluation(cm, report, roc_auc, pr_auc, output_dir):
    """Save autoencoder evaluation outputs alongside existing metric files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cm_path     = output_dir / "autoencoder_confusion_matrix.txt"
    report_path = output_dir / "autoencoder_classification_report.txt"

    cm_path.write_text(
        f"Autoencoder Confusion Matrix\n{'=' * 30}\n{str(cm)}\n"
        f"\nROC-AUC : {roc_auc:.4f}\nPR-AUC  : {pr_auc:.4f}",
        encoding="utf-8"
    )

    report_path.write_text(
        f"Autoencoder Classification Report\n{'=' * 35}\n{report}\n"
        f"ROC-AUC : {roc_auc:.4f}\nPR-AUC  : {pr_auc:.4f}",
        encoding="utf-8"
    )

    print(f"Autoencoder evaluation saved to {output_dir}")