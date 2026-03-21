from src.decision_confidence import (
    compute_risk_score,
    estimate_confidence,
    build_justification,
    recommend_action_with_confidence,
)


def test_risk_score_critical():
    score = compute_risk_score(anomaly_flag=-1, anomaly_score=-0.4, rul=3)
    assert score > 0.7, "Critical state should yield high risk score"


def test_risk_score_normal():
    score = compute_risk_score(anomaly_flag=1, anomaly_score=0.1, rul=120)
    assert score < 0.3, "Safe state should yield low risk score"


def test_risk_score_range():
    for flag in [1, -1]:
        for rul in [0, 5, 15, 50, 100]:
            score = compute_risk_score(flag, -0.1, rul)
            assert 0.0 <= score <= 1.0, f"Risk score out of range: {score}"


def test_confidence_high_both_agree():
    conf = estimate_confidence(anomaly_flag=-1, anomaly_score=-0.2, rul=4)
    assert conf == "HIGH"


def test_confidence_high_all_clear():
    conf = estimate_confidence(anomaly_flag=1, anomaly_score=0.15, rul=90)
    assert conf == "HIGH"


def test_justification_contains_rul():
    just = build_justification(anomaly_flag=-1, anomaly_score=-0.1, rul=3)
    assert "critically low" in just


def test_full_recommendation_keys():
    result = recommend_action_with_confidence(anomaly_flag=-1, anomaly_score=-0.3, rul=5)
    for key in ["recommended_action", "risk_score", "confidence", "justification"]:
        assert key in result, f"Missing key: {key}"