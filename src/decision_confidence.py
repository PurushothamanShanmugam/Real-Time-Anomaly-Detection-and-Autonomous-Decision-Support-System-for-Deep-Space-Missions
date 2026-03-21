"""
Decision confidence estimation and risk scoring system.
Covers Days 16–17 objectives: risk scoring + decision confidence estimation.

Extends the rule-based decision engine in decision_engine.py with:
  - A normalised risk score (0.0 – 1.0)
  - A confidence level label (HIGH / MEDIUM / LOW)
  - A brief justification explaining what drove the recommendation

Also includes a short primer on how Reinforcement Learning (RL) would
extend this rule-based system — covering the Day 14 RL objective at a
conceptual level, with a minimal Q-table demo on a toy environment.
"""

import numpy as np


# ------------------------------------------------------------------ #
#  Risk score
# ------------------------------------------------------------------ #

# RUL thresholds that define risk bands (cycles remaining)
RUL_CRITICAL  = 5
RUL_HIGH_RISK = 15
RUL_CAUTION   = 30
RUL_SAFE      = 80   # above this, RUL contributes 0 risk

# Anomaly score thresholds (Isolation Forest: more negative = more abnormal)
SCORE_STRONG_ANOMALY  = -0.15
SCORE_WEAK_ANOMALY    = -0.05


def compute_risk_score(anomaly_flag: int, anomaly_score: float, rul: float) -> float:
    """
    Compute a normalised risk score in [0.0, 1.0].

    Two independent signals are combined:
      - RUL component   : higher risk the closer to failure
      - Anomaly component : higher risk for stronger anomalies

    Parameters
    ----------
    anomaly_flag  : int   Isolation Forest output (-1 = anomaly, 1 = normal)
    anomaly_score : float Decision function score (more negative = more anomalous)
    rul           : float Predicted remaining useful life in cycles

    Returns
    -------
    float in [0.0, 1.0]
    """
    # RUL risk: linear ramp from 0 (safe) to 1 (critical)
    rul_clamped  = float(np.clip(rul, 0, RUL_SAFE))
    rul_risk     = 1.0 - (rul_clamped / RUL_SAFE)  # 0 = safe, 1 = imminent failure

    # Anomaly risk: map anomaly score to [0, 1]
    #   anomaly_flag == 1 (normal) contributes 0 base anomaly risk
    #   anomaly_flag == -1 (anomaly) contributes risk based on score magnitude
    if anomaly_flag == 1:
        anomaly_risk = 0.0
    else:
        # score ranges roughly from -0.5 (extreme anomaly) to 0 (boundary)
        score_clamped = float(np.clip(anomaly_score, -0.5, 0.0))
        anomaly_risk  = abs(score_clamped) / 0.5   # normalise to [0, 1]

    # Weighted combination: RUL contributes 60%, anomaly signal 40%
    risk_score = 0.60 * rul_risk + 0.40 * anomaly_risk
    return float(np.clip(risk_score, 0.0, 1.0))


# ------------------------------------------------------------------ #
#  Confidence level
# ------------------------------------------------------------------ #

def estimate_confidence(anomaly_flag: int, anomaly_score: float, rul: float) -> str:
    """
    Estimate decision confidence based on signal clarity.

    HIGH   – both signals agree strongly
    MEDIUM – signals are consistent but not extreme
    LOW    – signals are contradictory or borderline

    Returns
    -------
    str : "HIGH" | "MEDIUM" | "LOW"
    """
    anomaly_detected  = (anomaly_flag == -1)
    strong_anomaly    = anomaly_detected and (anomaly_score < SCORE_STRONG_ANOMALY)
    near_failure      = rul <= RUL_HIGH_RISK

    # Both signals agree: high confidence
    if strong_anomaly and near_failure:
        return "HIGH"
    if (not anomaly_detected) and (rul > RUL_SAFE):
        return "HIGH"

    # One signal is clear, other is borderline
    if strong_anomaly or near_failure:
        return "MEDIUM"

    # Contradictory or marginal: anomaly detected but RUL is still high,
    # or RUL is low but anomaly score is only slightly below threshold
    return "LOW"


# ------------------------------------------------------------------ #
#  Justification
# ------------------------------------------------------------------ #

def build_justification(anomaly_flag: int, anomaly_score: float, rul: float) -> str:
    """
    Return a one-sentence human-readable explanation of the decision.
    """
    parts = []

    if anomaly_flag == -1:
        parts.append(f"anomaly detected (score={anomaly_score:.3f})")
    else:
        parts.append("no anomaly detected")

    if rul <= RUL_CRITICAL:
        parts.append(f"RUL critically low ({rul:.0f} cycles)")
    elif rul <= RUL_HIGH_RISK:
        parts.append(f"RUL below high-risk threshold ({rul:.0f} cycles)")
    elif rul <= RUL_CAUTION:
        parts.append(f"RUL in caution zone ({rul:.0f} cycles)")
    else:
        parts.append(f"RUL acceptable ({rul:.0f} cycles)")

    return "; ".join(parts) + "."


# ------------------------------------------------------------------ #
#  Full enriched decision (drop-in replacement for recommend_action)
# ------------------------------------------------------------------ #

def recommend_action_with_confidence(
    anomaly_flag: int,
    anomaly_score: float,
    rul: float
) -> dict:
    """
    Return a dict with all decision fields, suitable for adding as
    new columns alongside the existing 'recommended_action' column.

    Example output
    --------------
    {
        "risk_score"      : 0.74,
        "confidence"      : "HIGH",
        "justification"   : "anomaly detected (score=-0.18); RUL critically low (4 cycles)."
    }
    """
    from src.decision_engine import recommend_action   # reuse existing logic

    action        = recommend_action(anomaly_flag, anomaly_score, rul)
    risk_score    = compute_risk_score(anomaly_flag, anomaly_score, rul)
    confidence    = estimate_confidence(anomaly_flag, anomaly_score, rul)
    justification = build_justification(anomaly_flag, anomaly_score, rul)

    return {
        "recommended_action": action,
        "risk_score"        : round(risk_score, 4),
        "confidence"        : confidence,
        "justification"     : justification,
    }


# ------------------------------------------------------------------ #
#  Reinforcement Learning primer (Day 14)
# ------------------------------------------------------------------ #

class SpacecraftRLEnvironment:
    """
    Minimal toy RL environment for a spacecraft subsystem.
    Demonstrates the Day 14 concept: rule-based vs AI-based decisions.

    State  : (anomaly_flag, rul_band)
               rul_band: 0=critical(<5), 1=high-risk(5-15), 2=caution(15-30), 3=safe(>30)
    Actions: 0=do_nothing, 1=monitor, 2=inspect, 3=emergency_shutdown
    Reward : shaped to encourage early, appropriate intervention
    """

    ACTIONS = {
        0: "do_nothing",
        1: "monitor_closely",
        2: "schedule_inspection",
        3: "emergency_shutdown",
    }

    def __init__(self):
        self.state = (0, 3)   # start: no anomaly, safe RUL band
        self.step_count = 0

    def reset(self):
        self.state = (0, 3)
        self.step_count = 0
        return self.state

    def step(self, action):
        """
        Simulate environment transition.
        Returns (next_state, reward, done).
        """
        anomaly_flag, rul_band = self.state
        self.step_count += 1
        done = False

        # Simulate degradation: RUL band decreases over time
        new_rul_band = max(0, rul_band - (1 if self.step_count % 5 == 0 else 0))
        # Random anomaly emergence
        new_anomaly  = 1 if (new_rul_band <= 1 and np.random.rand() > 0.5) else anomaly_flag

        # Reward logic
        if new_rul_band == 0 and action != 3:
            reward = -10   # failed to shut down at critical state
        elif new_rul_band == 0 and action == 3:
            reward = +10   # correct emergency action
        elif new_rul_band == 1 and action in (2, 3):
            reward = +5    # good preventive action
        elif new_rul_band >= 2 and action == 3:
            reward = -5    # unnecessary shutdown
        elif new_rul_band >= 2 and action == 0 and new_anomaly:
            reward = -3    # ignoring an anomaly
        else:
            reward = +1    # reasonable action

        done = (new_rul_band == 0 and action == 3) or self.step_count >= 50
        self.state = (new_anomaly, new_rul_band)
        return self.state, reward, done


def run_q_learning_demo(episodes=200, alpha=0.1, gamma=0.9, epsilon=0.3):
    """
    Train a simple Q-table agent on the toy spacecraft RL environment.

    This is intentionally simple — a demonstration of the RL loop:
      observe state → choose action → get reward → update Q-table

    In a real deep-space mission context, a Deep Q-Network (DQN) or
    Proximal Policy Optimisation (PPO) agent would replace this Q-table,
    with the full sensor feature vector as state.

    Returns
    -------
    Q : np.ndarray  shape (n_states, n_actions)  — learned Q-table
    """
    n_states  = 2 * 4   # 2 anomaly flags × 4 RUL bands
    n_actions = 4
    Q = np.zeros((n_states, n_actions))

    env = SpacecraftRLEnvironment()

    def state_idx(s):
        return s[0] * 4 + s[1]

    for episode in range(episodes):
        state = env.reset()
        done  = False

        while not done:
            s_idx = state_idx(state)

            # Epsilon-greedy action selection
            if np.random.rand() < epsilon:
                action = np.random.randint(n_actions)
            else:
                action = int(np.argmax(Q[s_idx]))

            next_state, reward, done = env.step(action)
            ns_idx = state_idx(next_state)

            # Q-learning update
            Q[s_idx, action] += alpha * (
                reward + gamma * np.max(Q[ns_idx]) - Q[s_idx, action]
            )

            state = next_state

    return Q


def print_rl_policy(Q):
    """Print the learned policy from the Q-table in a readable format."""
    env = SpacecraftRLEnvironment()
    print("\nLearned RL Policy (Q-table argmax):")
    print(f"{'State':<35} {'Best Action'}")
    print("-" * 55)
    for anomaly in range(2):
        for rul_band in range(4):
            s_idx  = anomaly * 4 + rul_band
            action = int(np.argmax(Q[s_idx]))
            band_label = ["Critical(<5)", "High-risk(5-15)", "Caution(15-30)", "Safe(>30)"][rul_band]
            anom_label = "Anomaly" if anomaly else "Normal"
            print(f"  {anom_label:<10} | RUL band: {band_label:<16} → {env.ACTIONS[action]}")