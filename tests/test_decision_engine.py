from src.decision_engine import recommend_action


def test_recommend_action_normal():
    action = recommend_action(anomaly_flag=1, anomaly_score=0.2, rul=100)
    assert "NORMAL" in action


def test_recommend_action_critical():
    action = recommend_action(anomaly_flag=-1, anomaly_score=-0.5, rul=3)
    assert "CRITICAL" in action


def test_recommend_action_high_risk():
    action = recommend_action(anomaly_flag=-1, anomaly_score=-0.2, rul=10)
    assert "HIGH RISK" in action