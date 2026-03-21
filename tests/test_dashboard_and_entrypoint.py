"""
tests/test_dashboard_and_entrypoint.py

Validates that:
1. The Streamlit dashboard module can be imported (syntax + top-level imports)
2. The helper utilities used by the dashboard work correctly
3. entrypoint.sh is present and syntactically valid
4. api/main.py has the /live endpoint and LIVE_PREDICTIONS store
"""

import subprocess
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import pytest

# ── make sure project root is on sys.path ───────────────────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ════════════════════════════════════════════════════════════════════
# 1. Dashboard module — syntax & key imports
# ════════════════════════════════════════════════════════════════════

def test_dashboard_syntax():
    """app/streamlit_app.py must compile without errors."""
    dashboard = ROOT / "app" / "streamlit_app.py"
    assert dashboard.exists(), "app/streamlit_app.py not found"
    result = subprocess.run(
        [sys.executable, "-m", "py_compile", str(dashboard)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, f"Syntax error in dashboard:\n{result.stderr}"


def test_dashboard_has_live_stream_page():
    """app/streamlit_app.py must contain the Live Stream page."""
    dashboard = ROOT / "app" / "streamlit_app.py"
    content = dashboard.read_text()
    assert "Live Stream" in content, "Dashboard is missing the Live Stream page"
    assert "/live" in content, "Dashboard must poll the /live API endpoint"
    assert "LIVE_PREDICTIONS" in content or "API_BASE" in content, \
        "Dashboard must reference the live API"


def test_simulator_import():
    """src.simulator must be importable (used by dashboard)."""
    from src.simulator import simulate_telemetry_stream  # noqa: F401


def test_decision_confidence_import():
    """src.decision_confidence must be importable (used by RL page)."""
    from src.decision_confidence import (  # noqa: F401
        compute_risk_score,
        estimate_confidence,
        build_justification,
        recommend_action_with_confidence,
        run_q_learning_demo,
        SpacecraftRLEnvironment,
    )


# ════════════════════════════════════════════════════════════════════
# 2. api/main.py — live endpoint validation
# ════════════════════════════════════════════════════════════════════

def test_api_main_syntax():
    """api/main.py must compile without errors."""
    api_main = ROOT / "api" / "main.py"
    assert api_main.exists(), "api/main.py not found"
    result = subprocess.run(
        [sys.executable, "-m", "py_compile", str(api_main)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, f"Syntax error in api/main.py:\n{result.stderr}"


def test_api_main_has_live_store():
    """api/main.py must define the LIVE_PREDICTIONS deque store."""
    api_main = ROOT / "api" / "main.py"
    content = api_main.read_text()
    assert "LIVE_PREDICTIONS" in content, \
        "api/main.py must have LIVE_PREDICTIONS in-memory store"
    assert "deque" in content, \
        "api/main.py must use collections.deque for the live store"


def test_api_main_has_live_endpoint():
    """api/main.py must define the /live GET endpoint."""
    api_main = ROOT / "api" / "main.py"
    content = api_main.read_text()
    assert '"/live"' in content or "'/live'" in content, \
        "api/main.py must define GET /live endpoint"
    assert "live_reset" in content or "/live/reset" in content, \
        "api/main.py must define DELETE /live/reset endpoint"


def test_api_main_predict_appends_to_live_store():
    """api/main.py /predict endpoint must append results to LIVE_PREDICTIONS."""
    api_main = ROOT / "api" / "main.py"
    content = api_main.read_text()
    assert "LIVE_PREDICTIONS.append" in content, \
        "/predict must append each result to LIVE_PREDICTIONS"
    assert "timestamp" in content, \
        "Each live record must include a timestamp"
    assert "latency_ms" in content, \
        "Each live record must include latency_ms"


# ════════════════════════════════════════════════════════════════════
# 3. Simulator helper (powers live telemetry stream in dashboard)
# ════════════════════════════════════════════════════════════════════

def _make_unit_df(n=50):
    np.random.seed(0)
    return pd.DataFrame({
        "unit_id":            [1] * n,
        "time_cycle":         range(1, n + 1),
        "RUL":                range(n - 1, -1, -1),
        "predicted_RUL":      np.random.randint(0, n, n).astype(float),
        "anomaly_score":      np.random.randn(n),
        "anomaly_flag":       np.random.choice([1, -1], n),
        "recommended_action": ["NORMAL OPERATIONS"] * n,
    })


def test_simulator_yields_records():
    from src.simulator import simulate_telemetry_stream
    df = _make_unit_df(20)
    records = list(simulate_telemetry_stream(df, unit_id=1, delay=0))
    assert len(records) == len(df), "Simulator should yield one record per row"


def test_simulator_record_has_required_keys():
    from src.simulator import simulate_telemetry_stream
    df = _make_unit_df(5)
    required = ["unit_id", "time_cycle", "RUL", "predicted_RUL",
                "anomaly_score", "anomaly_flag", "recommended_action"]
    for record in simulate_telemetry_stream(df, unit_id=1, delay=0):
        for key in required:
            assert key in record, f"Missing key in simulator output: {key}"


# ════════════════════════════════════════════════════════════════════
# 4. entrypoint.sh validation
# ════════════════════════════════════════════════════════════════════

def test_entrypoint_exists():
    ep = ROOT / "entrypoint.sh"
    assert ep.exists(), "entrypoint.sh is missing from project root"


def test_entrypoint_bash_syntax():
    ep = ROOT / "entrypoint.sh"
    result = subprocess.run(["bash", "-n", str(ep)], capture_output=True, text=True)
    assert result.returncode == 0, f"Bash syntax error in entrypoint.sh:\n{result.stderr}"


def test_entrypoint_contains_url_banner():
    ep = ROOT / "entrypoint.sh"
    content = ep.read_text()
    assert "localhost:8501" in content, "entrypoint.sh should print Streamlit URL"
    assert "localhost:8000" in content, "entrypoint.sh should print FastAPI URL"
    assert "localhost:3000" in content, "entrypoint.sh should print Grafana URL"
    assert "localhost:9090" in content, "entrypoint.sh should print Prometheus URL"


def test_entrypoint_has_wait_loop():
    ep = ROOT / "entrypoint.sh"
    content = ep.read_text()
    assert "telemetry_results.csv" in content, \
        "entrypoint.sh should wait for telemetry_results.csv"
    assert "MAX_WAIT" in content, "entrypoint.sh should have a timeout guard"