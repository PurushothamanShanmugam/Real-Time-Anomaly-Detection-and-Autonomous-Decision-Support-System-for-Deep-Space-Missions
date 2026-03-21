"""
LSTM-based Remaining Useful Life (RUL) prediction model.
Covers Day 4 objective: LSTM for predictive failure detection.

Uses sliding window sequences of telemetry features to predict RUL.
Results are compared against the Random Forest baseline in main_extended.py.
"""

import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("[LSTM] TensorFlow not installed. Run: pip install tensorflow")

from src.config import MODEL_DIR, RANDOM_STATE

LSTM_MODEL_PATH = MODEL_DIR / "rul_lstm_model.keras"
SEQUENCE_LENGTH = 30  # number of time steps per sequence window


# ------------------------------------------------------------------ #
#  Sequence builder
# ------------------------------------------------------------------ #

def build_sequences(df, feature_cols, sequence_length=SEQUENCE_LENGTH):
    """
    Convert flat telemetry rows into (samples, timesteps, features) arrays.

    For each engine unit, a sliding window of `sequence_length` cycles is
    created. The label for each window is the RUL at the last cycle of
    that window.

    Returns
    -------
    X : np.ndarray  shape (n_samples, sequence_length, n_features)
    y : np.ndarray  shape (n_samples,)
    """
    X_list, y_list = [], []

    for unit_id, group in df.groupby("unit_id"):
        group = group.sort_values("time_cycle")
        features = group[feature_cols].values
        labels   = group["RUL"].values

        # Only create sequences if the unit has enough cycles
        if len(features) < sequence_length:
            continue

        for i in range(len(features) - sequence_length + 1):
            X_list.append(features[i : i + sequence_length])
            y_list.append(labels[i + sequence_length - 1])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    return X, y


# ------------------------------------------------------------------ #
#  Train / test split for sequences
# ------------------------------------------------------------------ #

def split_lstm_data(df, feature_cols, test_size=0.2, sequence_length=SEQUENCE_LENGTH):
    """
    Split units into train/test groups BEFORE building sequences.
    This prevents data leakage (same engine appearing in both splits).
    """
    unit_ids = df["unit_id"].unique()
    np.random.seed(RANDOM_STATE)
    np.random.shuffle(unit_ids)

    split_idx   = int(len(unit_ids) * (1 - test_size))
    train_units = unit_ids[:split_idx]
    test_units  = unit_ids[split_idx:]

    train_df = df[df["unit_id"].isin(train_units)]
    test_df  = df[df["unit_id"].isin(test_units)]

    X_train, y_train = build_sequences(train_df, feature_cols, sequence_length)
    X_test,  y_test  = build_sequences(test_df,  feature_cols, sequence_length)

    return X_train, X_test, y_train, y_test


# ------------------------------------------------------------------ #
#  Model definition
# ------------------------------------------------------------------ #

def build_lstm_model(n_features, sequence_length=SEQUENCE_LENGTH):
    """
    Two-layer stacked LSTM with dropout for regularisation.

    Architecture
    ------------
    LSTM(64, return_sequences=True) → Dropout(0.2)
    LSTM(32)                        → Dropout(0.2)
    Dense(16, relu)
    Dense(1)   ← RUL prediction (regression)
    """
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow is required. Install with: pip install tensorflow")

    tf.random.set_seed(RANDOM_STATE)

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(sequence_length, n_features)),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


# ------------------------------------------------------------------ #
#  Train
# ------------------------------------------------------------------ #

def train_lstm_model(X_train, y_train, epochs=30, batch_size=64):
    """
    Train the LSTM model with early stopping.
    Returns the trained Keras model.
    """
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow is required. Install with: pip install tensorflow")

    n_features = X_train.shape[2]
    model = build_lstm_model(n_features, sequence_length=X_train.shape[1])

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    print(f"\nTraining LSTM model — sequences: {X_train.shape[0]}, "
          f"timesteps: {X_train.shape[1]}, features: {n_features}")

    model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.15,
        callbacks=[early_stop],
        verbose=1
    )

    Path(LSTM_MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
    model.save(LSTM_MODEL_PATH)
    print(f"\nLSTM model saved to {LSTM_MODEL_PATH}")

    return model


# ------------------------------------------------------------------ #
#  Evaluate
# ------------------------------------------------------------------ #

def evaluate_lstm_model(model, X_test, y_test):
    """
    Predict on test sequences and compute regression metrics.
    Returns a dict matching the format used by evaluate_rul_model()
    in train_rul.py so results are directly comparable.
    """
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow is required. Install with: pip install tensorflow")

    y_pred = model.predict(X_test, verbose=0).flatten()

    mae  = mean_absolute_error(y_test, y_pred)
    mse  = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2   = r2_score(y_test, y_pred)

    print("\nLSTM RUL Model — Test Set Evaluation:")
    print(f"  MAE  : {mae:.4f}")
    print(f"  MSE  : {mse:.4f}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  R2   : {r2:.4f}")

    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}


# ------------------------------------------------------------------ #
#  Save metrics
# ------------------------------------------------------------------ #

def save_lstm_metrics(metrics, output_dir):
    """Save LSTM metrics to a text file in the same format as rul_metrics.txt."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    path = output_dir / "rul_lstm_metrics.txt"
    lines = [
        "LSTM RUL Model Evaluation on Test Data",
        "======================================="
    ]
    for key, value in metrics.items():
        lines.append(f"{key}: {value:.4f}")

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"LSTM metrics saved to {path}")