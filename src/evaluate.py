from sklearn.metrics import classification_report, confusion_matrix


def create_anomaly_ground_truth(df, threshold=15):
    """
    Create simple anomaly ground truth using low RUL threshold.
    If RUL <= threshold, mark as anomaly (1), else normal (0).
    """
    df = df.copy()
    df["true_anomaly"] = (df["RUL"] <= threshold).astype(int)
    return df


def evaluate_anomaly_detection(df):
    """
    Evaluate anomaly predictions using derived anomaly labels.
    Isolation Forest output:
    -1 = anomaly
     1 = normal

    Converted prediction:
    1 = anomaly
    0 = normal
    """
    df = df.copy()

    y_true = df["true_anomaly"]
    y_pred = df["anomaly_flag"].apply(lambda x: 1 if x == -1 else 0)

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=4)

    print("\nAnomaly Detection Evaluation:")
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)

    return cm, report