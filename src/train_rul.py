import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from src.config import RUL_MODEL_PATH, RUL_N_ESTIMATORS, RANDOM_STATE


def split_rul_data(df, feature_cols, test_size=0.2, random_state=RANDOM_STATE):
    """
    Split telemetry data into train and test sets for RUL prediction.
    """
    X = df[feature_cols]
    y = df["RUL"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )

    return X_train, X_test, y_train, y_test


def train_rul_model(X_train, y_train):
    """
    Train a Random Forest model to predict Remaining Useful Life (RUL).
    """
    model = RandomForestRegressor(
        n_estimators=RUL_N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    Path(RUL_MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, RUL_MODEL_PATH)

    print(f"RUL prediction model trained and saved to {RUL_MODEL_PATH}")

    return model


def predict_rul(model, X_test, y_test):
    """
    Predict RUL values for the test set.
    """
    predictions = model.predict(X_test)

    result_df = X_test.copy()
    result_df["RUL"] = y_test.values
    result_df["predicted_RUL"] = predictions

    return result_df


def evaluate_rul_model(result_df):
    """
    Evaluate predicted RUL against actual RUL.
    """
    y_true = result_df["RUL"]
    y_pred = result_df["predicted_RUL"]

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_true, y_pred)

    print("\nRUL Model Evaluation on Test Data:")
    print(f"MAE  : {mae:.4f}")
    print(f"MSE  : {mse:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R2   : {r2:.4f}")

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    }