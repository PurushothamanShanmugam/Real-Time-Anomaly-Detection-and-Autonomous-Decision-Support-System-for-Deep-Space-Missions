import sys
from pathlib import Path

# -------------------------------------------------
# Make project root importable so "src" works
# -------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from src.simulator import simulate_telemetry_stream

# -------------------------------------------------
# Streamlit page config must come before sidebar UI
# -------------------------------------------------
st.set_page_config(
    page_title="Telemetry Control Panel",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------
# Custom styling
# -------------------------------------------------
st.markdown("""
<style>
.main {
    background-color: #0e1117;
}
.block-container {
    padding-top: 1.2rem;
    padding-bottom: 1rem;
}
h1, h2, h3 {
    color: #ffffff;
}
.metric-card {
    background: linear-gradient(135deg, #0f172a, #020617);
    padding: 20px;
    border-radius: 14px;
    border: 1px solid #30363d;
    text-align: center;
    box-shadow: 0px 6px 15px rgba(0,0,0,0.35);
    margin-bottom: 10px;
}
.metric-card h3 {
    color: #cbd5e1;
    font-size: 18px;
    margin-bottom: 10px;
}
.metric-card h2 {
    color: #22c55e;
    font-size: 38px;
    font-weight: bold;
    margin: 0;
}
.small-text {
    font-size: 15px;
    color: #cbd5e1;
}
.status-normal {
    color: #22c55e;
    font-weight: bold;
}
.status-warning {
    color: #f59e0b;
    font-weight: bold;
}
.status-critical {
    color: #ef4444;
    font-weight: bold;
}
.health-good {
    color: #22c55e;
    font-weight: bold;
}
.health-medium {
    color: #f59e0b;
    font-weight: bold;
}
.health-bad {
    color: #ef4444;
    font-weight: bold;
}
div[data-testid="stMetric"] {
    background-color: #111827;
    border: 1px solid #30363d;
    padding: 12px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# Sidebar controls
# -------------------------------------------------
st.sidebar.header("Control Panel")
run_simulation = st.sidebar.checkbox("Run telemetry simulation", value=False)
simulation_delay = st.sidebar.slider("Simulation delay (seconds)", 0.0, 1.0, 0.1, 0.05)

# -------------------------------------------------
# Load results file using absolute path
# -------------------------------------------------
file_path = BASE_DIR / "outputs" / "telemetry_results.csv"

if not file_path.exists():
    st.error(f"Result file not found: {file_path}")
    st.info("Please run main.py first to generate telemetry_results.csv.")
    st.stop()

df = pd.read_csv(file_path)

required_columns = [
    "unit_id",
    "time_cycle",
    "RUL",
    "predicted_RUL",
    "anomaly_score",
    "anomaly_flag",
    "recommended_action"
]

missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    st.error(f"Missing required columns in telemetry_results.csv: {missing_columns}")
    st.stop()

st.title("AI-Based Telemetry Anomaly Detection Control Panel")
st.markdown(
    "<p class='small-text'>Telemetry monitoring dashboard based on pipeline output, anomaly detection, RUL prediction, and maintenance decision support</p>",
    unsafe_allow_html=True
)

# -------------------------------------------------
# Unit selection
# -------------------------------------------------
unit_ids = sorted(df["unit_id"].unique().tolist())
selected_unit = st.sidebar.selectbox("Select Engine / Unit", unit_ids)
show_only_anomalies = st.sidebar.checkbox("Show anomalies only", value=False)

unit_df_full = df[df["unit_id"] == selected_unit].copy().sort_values("time_cycle")

if unit_df_full.empty:
    st.warning(f"No records found for Unit {selected_unit}")
    st.stop()

# -------------------------------------------------
# Optional simulation mode
# -------------------------------------------------
if run_simulation:
    simulated_rows = []
    sim_placeholder = st.empty()

    for record in simulate_telemetry_stream(unit_df_full, selected_unit, delay=simulation_delay):
        simulated_rows.append(record)
        live_df = pd.DataFrame(simulated_rows)

        display_cols = [
            col for col in [
                "unit_id",
                "time_cycle",
                "RUL",
                "predicted_RUL",
                "anomaly_score",
                "anomaly_flag",
                "recommended_action"
            ] if col in live_df.columns
        ]

        sim_placeholder.dataframe(
            live_df[display_cols],
            use_container_width=True
        )

    unit_df = pd.DataFrame(simulated_rows)
else:
    unit_df = unit_df_full.copy()

if show_only_anomalies:
    unit_df = unit_df[unit_df["anomaly_flag"] == -1].copy()

if unit_df.empty:
    st.warning("No rows to display for the selected filter.")
    st.stop()

# -------------------------------------------------
# Global summary cards
# -------------------------------------------------
total_records = len(df)
total_anomalies = int((df["anomaly_flag"] == -1).sum())
total_normal = int((df["anomaly_flag"] == 1).sum())
critical_count = int((df["recommended_action"] == "CRITICAL: Immediate maintenance required").sum())

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <h3>Total Records</h3>
        <h2>{total_records}</h2>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <h3>Detected Anomalies</h3>
        <h2>{total_anomalies}</h2>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <h3>Normal Records</h3>
        <h2>{total_normal}</h2>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <h3>Critical Cases</h3>
        <h2>{critical_count}</h2>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# -------------------------------------------------
# Unit summary
# -------------------------------------------------
latest_row = unit_df_full.iloc[-1]
latest_action = str(latest_row["recommended_action"])

unit_total = len(unit_df_full)
unit_anomalies = int((unit_df_full["anomaly_flag"] == -1).sum())
unit_rul = float(latest_row["RUL"])
unit_predicted_rul = float(latest_row["predicted_RUL"])
max_rul = float(unit_df_full["RUL"].max()) if float(unit_df_full["RUL"].max()) != 0 else 1.0
health_score = max(0, min(100, round((unit_predicted_rul / max_rul) * 100, 2)))
latest_error = abs(unit_rul - unit_predicted_rul)

if "CRITICAL" in latest_action:
    status_class = "status-critical"
    alert_text = "CRITICAL ALERT"
    alert_color = "#ef4444"
elif "HIGH RISK" in latest_action or "WARNING" in latest_action or "CAUTION" in latest_action:
    status_class = "status-warning"
    alert_text = "WARNING ALERT"
    alert_color = "#f59e0b"
else:
    status_class = "status-normal"
    alert_text = "NORMAL STATUS"
    alert_color = "#22c55e"

if health_score >= 70:
    health_class = "health-good"
elif health_score >= 40:
    health_class = "health-medium"
else:
    health_class = "health-bad"

st.subheader(f"Telemetry Analysis for Unit {selected_unit}")

summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)

with summary_col1:
    st.markdown(f"**Latest Cycle:** {int(latest_row['time_cycle'])}")
    st.markdown(f"**Actual RUL:** {int(latest_row['RUL'])}")

with summary_col2:
    st.markdown(f"**Predicted RUL:** {latest_row['predicted_RUL']:.2f}")
    st.markdown(f"**Prediction Error:** {latest_error:.2f}")

with summary_col3:
    st.markdown(f"**Latest Anomaly Score:** {latest_row['anomaly_score']:.4f}")
    st.markdown(f"**Latest Anomaly Flag:** {int(latest_row['anomaly_flag'])}")

with summary_col4:
    st.markdown(
        f"**Latest Status:** <span class='{status_class}'>{latest_action}</span>",
        unsafe_allow_html=True
    )

st.markdown("---")

# -------------------------------------------------
# Gauges
# -------------------------------------------------
g1, g2, g3, g4 = st.columns(4)

with g1:
    fig_gauge_alert = go.Figure(go.Indicator(
        mode="gauge+number",
        value=100 if "CRITICAL" in latest_action else 60 if ("HIGH RISK" in latest_action or "WARNING" in latest_action or "CAUTION" in latest_action) else 20,
        title={"text": "Alert Level"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": alert_color},
            "steps": [
                {"range": [0, 35], "color": "#14532d"},
                {"range": [35, 70], "color": "#78350f"},
                {"range": [70, 100], "color": "#7f1d1d"},
            ]
        }
    ))
    fig_gauge_alert.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig_gauge_alert, use_container_width=True)

with g2:
    fig_gauge_health = go.Figure(go.Indicator(
        mode="gauge+number",
        value=health_score,
        title={"text": "Predicted Health Score"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#22c55e" if health_score >= 70 else "#f59e0b" if health_score >= 40 else "#ef4444"},
            "steps": [
                {"range": [0, 40], "color": "#7f1d1d"},
                {"range": [40, 70], "color": "#78350f"},
                {"range": [70, 100], "color": "#14532d"},
            ]
        }
    ))
    fig_gauge_health.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig_gauge_health, use_container_width=True)

with g3:
    failure_risk = min(100, round((unit_anomalies / unit_total) * 100 * 5, 2)) if unit_total else 0
    fig_gauge_failure = go.Figure(go.Indicator(
        mode="gauge+number",
        value=failure_risk,
        title={"text": "Failure Risk Meter"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#ef4444" if failure_risk >= 70 else "#f59e0b" if failure_risk >= 40 else "#22c55e"},
            "steps": [
                {"range": [0, 40], "color": "#14532d"},
                {"range": [40, 70], "color": "#78350f"},
                {"range": [70, 100], "color": "#7f1d1d"},
            ]
        }
    ))
    fig_gauge_failure.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig_gauge_failure, use_container_width=True)

with g4:
    st.markdown("### Live Status")
    st.markdown(f"**Alert Indicator:** <span style='color:{alert_color}; font-weight:bold'>{alert_text}</span>", unsafe_allow_html=True)
    st.markdown(f"**Predicted Unit Health:** <span class='{health_class}'>{health_score}%</span>", unsafe_allow_html=True)
    st.markdown(f"**Detected Unit Anomalies:** {unit_anomalies}")
    st.markdown("**Current Recommended Action:**")
    st.markdown(f"<span class='{status_class}'>{latest_action}</span>", unsafe_allow_html=True)

st.markdown("---")

# -------------------------------------------------
# Charts
# -------------------------------------------------
left_col, right_col = st.columns(2)

with left_col:
    st.subheader("Actual vs Predicted RUL")
    rul_compare_fig = go.Figure()
    rul_compare_fig.add_trace(go.Scatter(
        x=unit_df_full["time_cycle"],
        y=unit_df_full["RUL"],
        mode="lines",
        name="Actual RUL"
    ))
    rul_compare_fig.add_trace(go.Scatter(
        x=unit_df_full["time_cycle"],
        y=unit_df_full["predicted_RUL"],
        mode="lines",
        name="Predicted RUL"
    ))
    rul_compare_fig.update_layout(
        template="plotly_dark",
        height=350,
        xaxis_title="Cycle",
        yaxis_title="RUL"
    )
    st.plotly_chart(rul_compare_fig, use_container_width=True)

with right_col:
    st.subheader("Anomaly Score Over Time")
    score_fig = px.line(
        unit_df,
        x="time_cycle",
        y="anomaly_score",
        markers=False,
        template="plotly_dark"
    )
    score_fig.add_hline(y=0, line_dash="dash", line_color="red")
    score_fig.update_layout(height=350, xaxis_title="Cycle", yaxis_title="Anomaly Score")
    st.plotly_chart(score_fig, use_container_width=True)

st.markdown("---")

st.subheader("RUL Prediction Error Over Time")
error_df = unit_df_full.copy()
error_df["RUL_error"] = (error_df["RUL"] - error_df["predicted_RUL"]).abs()

error_fig = px.line(
    error_df,
    x="time_cycle",
    y="RUL_error",
    template="plotly_dark"
)
error_fig.update_layout(height=350, xaxis_title="Cycle", yaxis_title="Absolute Error")
st.plotly_chart(error_fig, use_container_width=True)

st.markdown("---")

timeline_df = unit_df_full.copy()
timeline_df["anomaly_label"] = timeline_df["anomaly_flag"].map({1: "Normal", -1: "Anomaly"})

st.subheader("Live Anomaly Timeline")
timeline_fig = px.scatter(
    timeline_df,
    x="time_cycle",
    y="anomaly_score",
    color="anomaly_label",
    color_discrete_map={"Normal": "#22c55e", "Anomaly": "#ef4444"},
    hover_data=["RUL", "predicted_RUL", "recommended_action"],
    template="plotly_dark"
)
timeline_fig.add_hline(y=0, line_dash="dash", line_color="white")
timeline_fig.update_layout(height=420, xaxis_title="Cycle", yaxis_title="Anomaly Score")
st.plotly_chart(timeline_fig, use_container_width=True)

st.markdown("---")

dist_col1, dist_col2 = st.columns(2)

with dist_col1:
    st.subheader("Anomaly Flag Distribution")
    anomaly_counts = unit_df_full["anomaly_flag"].value_counts().sort_index()
    anomaly_counts.index = anomaly_counts.index.astype(str)
    anomaly_bar = px.bar(
        x=anomaly_counts.index,
        y=anomaly_counts.values,
        labels={"x": "Anomaly Flag", "y": "Count"},
        template="plotly_dark"
    )
    anomaly_bar.update_layout(height=350)
    st.plotly_chart(anomaly_bar, use_container_width=True)

with dist_col2:
    st.subheader("Recommended Action Summary")
    action_counts = unit_df_full["recommended_action"].value_counts()
    action_bar = px.bar(
        x=action_counts.index,
        y=action_counts.values,
        labels={"x": "Recommended Action", "y": "Count"},
        template="plotly_dark"
    )
    action_bar.update_layout(height=350)
    st.plotly_chart(action_bar, use_container_width=True)

st.markdown("---")

st.subheader("Telemetry Data Table")
display_columns = [
    col for col in [
        "unit_id",
        "time_cycle",
        "RUL",
        "predicted_RUL",
        "anomaly_score",
        "anomaly_flag",
        "recommended_action"
    ] if col in unit_df.columns
]
st.dataframe(unit_df[display_columns], use_container_width=True)