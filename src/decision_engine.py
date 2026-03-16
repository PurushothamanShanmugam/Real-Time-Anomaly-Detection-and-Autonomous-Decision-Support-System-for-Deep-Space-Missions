def recommend_action(anomaly_flag, anomaly_score, rul):
    """
    Recommend maintenance action using:
    - anomaly flag
    - anomaly score
    - predicted RUL
    Lower anomaly_score = more abnormal in Isolation Forest.
    """

    if anomaly_flag == -1 and rul <= 5:
        return "CRITICAL: Immediate maintenance required"

    if anomaly_flag == -1 and rul <= 15:
        return "HIGH RISK: Schedule urgent inspection"

    if anomaly_flag == -1 and anomaly_score < -0.10:
        return "WARNING: Strong anomaly detected"

    if anomaly_flag == -1:
        return "WARNING: Monitor this unit closely"

    if rul <= 10:
        return "CAUTION: Unit nearing failure threshold"

    return "NORMAL: System operating normally"