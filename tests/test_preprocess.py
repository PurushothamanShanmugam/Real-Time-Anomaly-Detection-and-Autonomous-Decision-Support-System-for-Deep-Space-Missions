from src.data_loader import load_cmapss_data
from src.preprocess import add_rul, scale_features


def test_add_rul():
    df = load_cmapss_data("data/raw/train_FD001.txt")
    df = add_rul(df)

    assert "RUL" in df.columns
    assert df["RUL"].min() >= 0


def test_scale_features():
    df = load_cmapss_data("data/raw/train_FD001.txt")
    df = add_rul(df)
    scaled_df, scaler, feature_cols = scale_features(df)

    assert len(feature_cols) > 0
    assert "RUL" in scaled_df.columns