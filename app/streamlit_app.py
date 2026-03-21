import os
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import requests

from src.simulator import simulate_telemetry_stream

# ── API base URL — reads from docker-compose env var, falls back to localhost
API_BASE = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Space Telemetry Control Panel", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>

/* ── 1. Remove white strip at top completely ── */
[data-testid="stDecoration"]   { display: none !important; }
[data-testid="stStatusWidget"] { display: none !important; }
#MainMenu { visibility: hidden !important; }
footer    { visibility: hidden !important; }

/* Make header bar background transparent (keeps Deploy button visible) */
header[data-testid="stHeader"] {
    background:       rgba(184,224,247,0.95) !important;
    background-color: rgba(184,224,247,0.95) !important;
    border-bottom: 1px solid #7bbfe0 !important;
}

/* ── 2. Force Deploy button and toolbar to show ── */
header[data-testid="stHeader"] * { visibility: visible !important; opacity: 1 !important; }
[data-testid="stToolbar"]         { display: flex    !important; visibility: visible !important; opacity: 1 !important; }
[data-testid="stToolbar"] *       { visibility: visible !important; opacity: 1 !important; }
.stDeployButton                   { display: inline-flex !important; visibility: visible !important; opacity: 1 !important; }
.stDeployButton *                 { visibility: visible !important; opacity: 1 !important; }
button[data-testid="baseButton-header"] { display: inline-flex !important; visibility: visible !important; }

/* ── 3. Remove anchor/link icon that appears on heading hover ── */
a.anchor-link                                    { display: none !important; }
.anchor-link                                     { display: none !important; }
h1 .anchor-link, h2 .anchor-link,
h3 .anchor-link, h4 .anchor-link                 { display: none !important; }
h1 a, h2 a, h3 a, h4 a, h5 a, h6 a             { display: none !important; }
h1 a svg, h2 a svg, h3 a svg,
h4 a svg, h5 a svg, h6 a svg                    { display: none !important; }
[data-testid="stHeadingActionElements"]          { display: none !important; }
[data-testid="stMarkdownContainer"] h1 a,
[data-testid="stMarkdownContainer"] h2 a,
[data-testid="stMarkdownContainer"] h3 a,
[data-testid="stMarkdownContainer"] h4 a         { display: none !important; }
.stHeadingWithActionElements a                   { display: none !important; }
.stHeadingWithActionElements svg                 { display: none !important; }
/* Also covers bold text (**text**) hover icons */
[data-testid="stMarkdownContainer"] p a svg      { display: none !important; }
[data-testid="stMarkdownContainer"] strong a     { display: none !important; }

/* ── Background: light sky blue with twinkling stars ── */
.stApp, .main, section[data-testid="stSidebar"] ~ div {
    background: linear-gradient(180deg, #b8e0f7 0%, #ceeaf9 40%, #e0f2fe 100%) !important;
}
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0; width: 100%; height: 100%;
    pointer-events: none;
    z-index: 0;
    background:
        radial-gradient(1.5px 1.5px at  5%  8%, #fff 0%, transparent 100%),
        radial-gradient(1px   1px   at 12% 14%, #fff 0%, transparent 100%),
        radial-gradient(2px   2px   at 20%  5%, #ffffffcc 0%, transparent 100%),
        radial-gradient(1px   1px   at 28% 18%, #fff 0%, transparent 100%),
        radial-gradient(1.5px 1.5px at 35%  3%, #fff 0%, transparent 100%),
        radial-gradient(1px   1px   at 43% 11%, #ffffffcc 0%, transparent 100%),
        radial-gradient(2px   2px   at 50%  7%, #fff 0%, transparent 100%),
        radial-gradient(1px   1px   at 58% 16%, #fff 0%, transparent 100%),
        radial-gradient(1.5px 1.5px at 65%  2%, #ffffffcc 0%, transparent 100%),
        radial-gradient(1px   1px   at 73% 13%, #fff 0%, transparent 100%),
        radial-gradient(2px   2px   at 80%  6%, #fff 0%, transparent 100%),
        radial-gradient(1px   1px   at 88% 19%, #ffffffcc 0%, transparent 100%),
        radial-gradient(1.5px 1.5px at 95%  9%, #fff 0%, transparent 100%),
        radial-gradient(1px   1px   at  8% 30%, #ffffffaa 0%, transparent 100%),
        radial-gradient(1.5px 1.5px at 18% 35%, #ffffffaa 0%, transparent 100%),
        radial-gradient(1px   1px   at 30% 25%, #ffffffaa 0%, transparent 100%),
        radial-gradient(2px   2px   at 42% 38%, #ffffffaa 0%, transparent 100%),
        radial-gradient(1px   1px   at 55% 28%, #ffffffaa 0%, transparent 100%),
        radial-gradient(1.5px 1.5px at 68% 32%, #ffffffaa 0%, transparent 100%),
        radial-gradient(1px   1px   at 78% 22%, #ffffffaa 0%, transparent 100%),
        radial-gradient(2px   2px   at 90% 36%, #ffffffaa 0%, transparent 100%),
        radial-gradient(1px   1px   at 15% 50%, #ffffff88 0%, transparent 100%),
        radial-gradient(1.5px 1.5px at 32% 55%, #ffffff88 0%, transparent 100%),
        radial-gradient(1px   1px   at 48% 45%, #ffffff88 0%, transparent 100%),
        radial-gradient(2px   2px   at 62% 52%, #ffffff88 0%, transparent 100%),
        radial-gradient(1px   1px   at 75% 48%, #ffffff88 0%, transparent 100%),
        radial-gradient(1.5px 1.5px at 85% 58%, #ffffff88 0%, transparent 100%);
    animation: twinkle 3s ease-in-out infinite alternate;
}
@keyframes twinkle {
    0%   { opacity: 0.4; }
    30%  { opacity: 1.0; }
    60%  { opacity: 0.5; }
    100% { opacity: 0.9; }
}

/* ── All text: dark black bold ── */
.block-container { padding-top: 1.2rem; padding-bottom: 1rem; position: relative; z-index: 1; }
h1, h2, h3, h4, h5, h6 { color: #0a0a0a !important; font-weight: 900 !important; }
p, span, label, div { color: #0a0a0a; }
.small-text { font-size: 15px; color: #111111; font-weight: 600; }

/* ── Metric cards ── */
.metric-card {
    background: rgba(255,255,255,0.85);
    padding: 20px; border-radius: 14px;
    border: 2px solid #7bbfe0;
    text-align: center;
    box-shadow: 0px 4px 12px rgba(0,102,204,0.15);
    margin-bottom: 10px;
}
.metric-card h3 { color: #111111 !important; font-size: 18px; margin-bottom: 10px; font-weight: 700 !important; }
.metric-card h2 { color: #004499 !important; font-size: 38px; font-weight: 900 !important; margin: 0; }

/* ── Status colors ── */
.status-normal   { color: #006622; font-weight: 900; }
.status-warning  { color: #7a4a00; font-weight: 900; }
.status-critical { color: #880011; font-weight: 900; }
.health-good     { color: #006622; font-weight: 900; }
.health-medium   { color: #7a4a00; font-weight: 900; }
.health-bad      { color: #880011; font-weight: 900; }
.conf-high   { color: #006622; font-weight: 900; }
.conf-medium { color: #7a4a00; font-weight: 900; }
.conf-low    { color: #880011; font-weight: 900; }

/* ── Streamlit metric boxes ── */
div[data-testid="stMetric"] {
    background-color: rgba(255,255,255,0.85) !important;
    border: 2px solid #7bbfe0 !important;
    padding: 12px; border-radius: 12px;
}
div[data-testid="stMetric"] label { color: #111111 !important; font-weight: 700 !important; }
div[data-testid="stMetric"] div   { color: #0a0a0a !important; font-weight: 900 !important; }

/* ── Info box ── */
.info-box {
    background-color: rgba(255,255,255,0.8);
    border-left: 4px solid #0066cc;
    padding: 12px 16px; border-radius: 8px;
    margin-bottom: 12px; font-size: 14px;
    color: #111111 !important; font-weight: 600;
    border: 1px solid #7bbfe0;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #a8d8f0 0%, #c5e8f8 100%) !important;
    border-right: 2px solid #7bbfe0 !important;
}
section[data-testid="stSidebar"] * { color: #0a0a0a !important; font-weight: 700 !important; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { color: #0a0a0a !important; font-weight: 900 !important; }

/* ── Buttons ── */
.stButton > button {
    background: rgba(255,255,255,0.9) !important;
    border: 2px solid #0066cc !important;
    color: #0a0a0a !important;
    font-weight: 900 !important;
    border-radius: 8px !important;
}
.stButton > button:hover {
    background: #b8dff5 !important;
    box-shadow: 0 4px 12px rgba(0,102,204,0.25) !important;
}

/* ── Dataframe / table text ── */
.dataframe, .dataframe td, .dataframe th { color: #0a0a0a !important; font-weight: 600 !important; }

/* ── General Streamlit text overrides ── */
[data-testid="stMarkdownContainer"] p   { color: #0a0a0a !important; font-weight: 600; }
[data-testid="stMarkdownContainer"] li  { color: #0a0a0a !important; font-weight: 600; }
.stSelectbox label, .stSlider label, .stCheckbox label, .stRadio label {
    color: #0a0a0a !important; font-weight: 700 !important;
}

</style>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────
st.sidebar.header("Control Panel")
page = st.sidebar.radio("Dashboard Section", [
    "Overview & Anomaly Detection",
    "RUL Prediction & Model Comparison",
    "Risk Score & Decision Confidence",
    "Autoencoder Analysis",
    "Reinforcement Learning Demo",
    "🔴 Live Stream",
])
run_simulation   = st.sidebar.checkbox("Run telemetry simulation", value=False)
simulation_delay = st.sidebar.slider("Simulation delay (s)", 0.0, 1.0, 0.1, 0.05)

# ── Load data ─────────────────────────────────────────────────────────
EXTENDED_PATH = BASE_DIR / "outputs" / "telemetry_results_extended.csv"
BASE_PATH     = BASE_DIR / "outputs" / "telemetry_results.csv"

if EXTENDED_PATH.exists():
    df = pd.read_csv(EXTENDED_PATH)
    using_extended = True
elif BASE_PATH.exists():
    df = pd.read_csv(BASE_PATH)
    using_extended = False
else:
    st.error("No result file found. Run main.py first.")
    st.info("Command: python main.py")
    st.stop()

required = ["unit_id","time_cycle","RUL","predicted_RUL","anomaly_score","anomaly_flag","recommended_action"]
missing  = [c for c in required if c not in df.columns]
if missing:
    st.error(f"Missing columns: {missing}")
    st.stop()

has_confidence  = "risk_score" in df.columns and "confidence" in df.columns
has_autoencoder = "reconstruction_error" in df.columns and "autoencoder_anomaly_flag" in df.columns

unit_ids      = sorted(df["unit_id"].unique().tolist())
selected_unit = st.sidebar.selectbox("Select Engine / Unit", unit_ids)
show_anom_only= st.sidebar.checkbox("Show anomalies only", value=False)

unit_df_full = df[df["unit_id"] == selected_unit].copy().sort_values("time_cycle")
if unit_df_full.empty:
    st.warning(f"No records for Unit {selected_unit}")
    st.stop()

if run_simulation:
    rows = []
    placeholder = st.empty()
    for record in simulate_telemetry_stream(unit_df_full, selected_unit, delay=simulation_delay):
        rows.append(record)
        live = pd.DataFrame(rows)
        placeholder.dataframe(live[[c for c in required if c in live.columns]], use_container_width=True)
    unit_df = pd.DataFrame(rows)
else:
    unit_df = unit_df_full.copy()

if show_anom_only:
    unit_df = unit_df[unit_df["anomaly_flag"] == -1].copy()
if unit_df.empty:
    st.warning("No rows to display.")
    st.stop()

# ── Shared values ─────────────────────────────────────────────────────
latest      = unit_df_full.iloc[-1]
latest_act  = str(latest["recommended_action"])
unit_total  = len(unit_df_full)
unit_anom   = int((unit_df_full["anomaly_flag"] == -1).sum())
pred_rul    = float(latest["predicted_RUL"])
actual_rul  = float(latest["RUL"])
max_rul     = max(float(unit_df_full["RUL"].max()), 1.0)
health      = max(0, min(100, round((pred_rul / max_rul) * 100, 2)))
pred_err    = abs(actual_rul - pred_rul)

if   "CRITICAL"  in latest_act: sc, at, ac = "status-critical", "CRITICAL ALERT",  "#ef4444"
elif any(w in latest_act for w in ["HIGH RISK","WARNING","CAUTION"]):
                                  sc, at, ac = "status-warning",  "WARNING ALERT",   "#f59e0b"
else:                             sc, at, ac = "status-normal",   "NORMAL STATUS",   "#22c55e"
hc = "health-good" if health >= 70 else "health-medium" if health >= 40 else "health-bad"


# ══════════════════════════════════════════════════════════════════════
# PAGE 1 — Overview & Anomaly Detection
# ══════════════════════════════════════════════════════════════════════
if page == "Overview & Anomaly Detection":

    st.title("AI-Based Telemetry Anomaly Detection Control Panel")
    st.markdown(
        f"<p class='small-text'>NASA C-MAPSS Dataset · Unit {selected_unit} · "
        f"{'Extended pipeline active (LSTM + Autoencoder + Risk Scoring)' if using_extended else 'Base pipeline active'}</p>",
        unsafe_allow_html=True)

    total_rec   = len(df)
    total_anom  = int((df["anomaly_flag"] == -1).sum())
    total_norm  = int((df["anomaly_flag"] ==  1).sum())
    crit_count  = int(df["recommended_action"].str.contains("CRITICAL", na=False).sum())

    c1,c2,c3,c4 = st.columns(4)
    for col, label, val in zip([c1,c2,c3,c4],
            ["Total Records","Detected Anomalies","Normal Records","Critical Cases"],
            [total_rec, total_anom, total_norm, crit_count]):
        with col:
            st.markdown(f'<div class="metric-card"><h3>{label}</h3><h2>{val:,}</h2></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.subheader(f"Unit {selected_unit} — Latest Status")

    sc1,sc2,sc3,sc4 = st.columns(4)
    with sc1:
        st.markdown(f"**Latest Cycle:** {int(latest['time_cycle'])}")
        st.markdown(f"**Actual RUL:** {int(actual_rul)}")
    with sc2:
        st.markdown(f"**Predicted RUL:** {pred_rul:.2f}")
        st.markdown(f"**Prediction Error:** {pred_err:.2f}")
    with sc3:
        st.markdown(f"**Anomaly Score:** {latest['anomaly_score']:.4f}")
        st.markdown(f"**Anomaly Flag:** {int(latest['anomaly_flag'])}")
    with sc4:
        st.markdown(f"**Status:** <span class='{sc}'>{latest_act}</span>", unsafe_allow_html=True)
        if has_confidence:
            risk = float(latest["risk_score"])
            conf = str(latest["confidence"])
            cfc  = f"conf-{conf.lower()}"
            st.markdown(f"**Risk:** {risk:.2f} · **Confidence:** <span class='{cfc}'>{conf}</span>", unsafe_allow_html=True)

    st.markdown("---")
    g1,g2,g3,g4 = st.columns(4)
    av = 100 if "CRITICAL" in latest_act else 60 if any(w in latest_act for w in ["HIGH RISK","WARNING","CAUTION"]) else 20

    def gauge(val, title, bar_color, steps):
        fig = go.Figure(go.Indicator(mode="gauge+number", value=val, title={"text": title},
            gauge={"axis": {"range": [0,100]}, "bar": {"color": bar_color}, "steps": steps}))
        fig.update_layout(height=270, margin=dict(l=20,r=20,t=50,b=10))
        return fig

    steps_risk    = [{"range":[0,40],"color":"#14532d"},{"range":[40,70],"color":"#78350f"},{"range":[70,100],"color":"#7f1d1d"}]
    steps_health  = [{"range":[0,40],"color":"#7f1d1d"},{"range":[40,70],"color":"#78350f"},{"range":[70,100],"color":"#14532d"}]

    with g1: st.plotly_chart(gauge(av, "Alert Level", ac, steps_risk), use_container_width=True)
    with g2:
        hbar = "#22c55e" if health>=70 else "#f59e0b" if health>=40 else "#ef4444"
        st.plotly_chart(gauge(health, "Health Score (%)", hbar, steps_health), use_container_width=True)
    with g3:
        fr = min(100, round((unit_anom/unit_total)*100*5,2)) if unit_total else 0
        fb = "#ef4444" if fr>=70 else "#f59e0b" if fr>=40 else "#22c55e"
        st.plotly_chart(gauge(fr, "Failure Risk (%)", fb, steps_risk), use_container_width=True)
    with g4:
        if has_confidence:
            rv = float(latest["risk_score"])*100
            rb = "#ef4444" if rv>=70 else "#f59e0b" if rv>=40 else "#22c55e"
            st.plotly_chart(gauge(round(rv,1), "Risk Score (×100)", rb, steps_risk), use_container_width=True)
        else:
            st.markdown("### Live Status")
            st.markdown(f"**Alert:** <span style='color:{ac};font-weight:bold'>{at}</span>", unsafe_allow_html=True)
            st.markdown(f"**Health:** <span class='{hc}'>{health}%</span>", unsafe_allow_html=True)
            st.markdown(f"**Anomalies:** {unit_anom}")
            st.markdown(f"<span class='{sc}'>{latest_act}</span>", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Anomaly Score Timeline")
    tdf = unit_df_full.copy()
    tdf["Label"] = tdf["anomaly_flag"].map({1:"Normal",-1:"Anomaly"})
    hover = ["RUL","predicted_RUL","recommended_action"] + (["risk_score","confidence"] if has_confidence else [])
    fig_t = px.scatter(tdf, x="time_cycle", y="anomaly_score", color="Label",
                       color_discrete_map={"Normal":"#22c55e","Anomaly":"#ef4444"},
                       hover_data=hover, template="plotly_dark")
    fig_t.add_hline(y=0, line_dash="dash", line_color="white")
    fig_t.update_layout(height=400, xaxis_title="Cycle", yaxis_title="Anomaly Score (Isolation Forest)")
    st.plotly_chart(fig_t, use_container_width=True)

    d1,d2 = st.columns(2)
    with d1:
        st.subheader("Anomaly Flag Distribution")
        ac2 = unit_df_full["anomaly_flag"].value_counts().sort_index()
        ac2.index = ac2.index.map({1:"Normal (1)",-1:"Anomaly (-1)"})
        fig = px.bar(x=ac2.index, y=ac2.values, template="plotly_dark",
                     color=ac2.index, color_discrete_map={"Normal (1)":"#22c55e","Anomaly (-1)":"#ef4444"})
        fig.update_layout(height=340, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with d2:
        st.subheader("Recommended Action Summary")
        rc = unit_df_full["recommended_action"].value_counts()
        fig = px.bar(x=rc.index, y=rc.values, template="plotly_dark")
        fig.update_layout(height=340, xaxis_tickangle=-20)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Telemetry Data Table")
    dcols = [c for c in required if c in unit_df.columns]
    if has_confidence:
        dcols += [c for c in ["risk_score","confidence","justification"] if c in unit_df.columns]
    st.dataframe(unit_df[dcols], use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
# PAGE 2 — RUL Prediction & Model Comparison
# ══════════════════════════════════════════════════════════════════════
elif page == "RUL Prediction & Model Comparison":

    st.title("RUL Prediction — Random Forest vs LSTM")
    st.markdown("<p class='small-text'>Day 4 & Day 11 objectives — baseline model vs deep learning sequence model</p>", unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    <b>Random Forest</b> treats each telemetry row independently.
    <b>LSTM</b> uses a 30-cycle sliding window to capture how sensor patterns evolve over time —
    which is closer to how real degradation manifests.
    </div>""", unsafe_allow_html=True)

    fig_rul = go.Figure()
    fig_rul.add_trace(go.Scatter(x=unit_df_full["time_cycle"], y=unit_df_full["RUL"],
        mode="lines", name="Actual RUL", line=dict(color="#22c55e")))
    fig_rul.add_trace(go.Scatter(x=unit_df_full["time_cycle"], y=unit_df_full["predicted_RUL"],
        mode="lines", name="RF Predicted RUL", line=dict(color="#3b82f6", dash="dash")))
    fig_rul.update_layout(template="plotly_dark", height=370,
                           xaxis_title="Cycle", yaxis_title="RUL (cycles)",
                           title=f"Actual vs Predicted RUL — Unit {selected_unit}")
    st.plotly_chart(fig_rul, use_container_width=True)

    err_df = unit_df_full.copy()
    err_df["abs_error"] = (err_df["RUL"] - err_df["predicted_RUL"]).abs()
    fig_err = px.line(err_df, x="time_cycle", y="abs_error", template="plotly_dark")
    fig_err.update_layout(height=280, title="Absolute Prediction Error Over Time",
                           xaxis_title="Cycle", yaxis_title="|Actual − Predicted|")
    st.plotly_chart(fig_err, use_container_width=True)

    st.markdown("---")
    st.subheader("Model Evaluation Metrics")

    rf_path   = BASE_DIR / "outputs" / "metrics" / "rul_metrics.txt"
    lstm_path = BASE_DIR / "outputs" / "metrics" / "rul_lstm_metrics.txt"
    cmp_path  = BASE_DIR / "outputs" / "metrics" / "rul_model_comparison.txt"

    m1, m2 = st.columns(2)
    with m1:
        st.markdown("#### Random Forest (Baseline)")
        if rf_path.exists():
            for line in rf_path.read_text().splitlines():
                if ":" in line and not line.startswith("RUL") and not line.startswith("="):
                    k, v = line.split(":", 1)
                    try: st.metric(k.strip(), f"{float(v.strip()):.4f}")
                    except ValueError: pass
        else:
            st.info("Run main.py to generate RF metrics.")

    with m2:
        st.markdown("#### LSTM (Deep Learning)")
        if lstm_path.exists():
            for line in lstm_path.read_text().splitlines():
                if ":" in line and not line.startswith("LSTM") and not line.startswith("="):
                    k, v = line.split(":", 1)
                    try: st.metric(k.strip(), f"{float(v.strip()):.4f}")
                    except ValueError: pass
        else:
            st.warning("LSTM metrics not found. Re-run the pipeline to generate LSTM metrics.")

    if cmp_path.exists():
        st.markdown("---")
        st.subheader("Side-by-Side Comparison")
        st.code(cmp_path.read_text(), language="text")

    st.markdown("---")
    st.subheader("Actual vs Predicted RUL — Scatter (all units, sampled)")
    sample_df = df.sample(min(5000, len(df)), random_state=42)
    fig_sc = px.scatter(sample_df, x="RUL", y="predicted_RUL", opacity=0.35, template="plotly_dark",
                         labels={"RUL":"Actual RUL","predicted_RUL":"RF Predicted RUL"})
    mv = max(sample_df["RUL"].max(), sample_df["predicted_RUL"].max())
    fig_sc.add_shape(type="line", x0=0, y0=0, x1=mv, y1=mv, line=dict(color="red", dash="dash"))
    fig_sc.update_layout(height=400)
    st.caption("Red line = perfect prediction. Closer to the diagonal = better accuracy.")
    st.plotly_chart(fig_sc, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
# PAGE 3 — Risk Score & Decision Confidence
# ══════════════════════════════════════════════════════════════════════
elif page == "Risk Score & Decision Confidence":

    st.title("Risk Score & Decision Confidence Estimation")
    st.markdown("<p class='small-text'>Days 16–17 objectives — normalised risk score + confidence level + plain-language justification</p>", unsafe_allow_html=True)

    if not has_confidence:
        st.warning("Risk score columns not found.")
        st.info("Risk score columns not found. Re-run the pipeline with the extended mode to generate risk_score, confidence, and justification.")
        st.stop()

    st.markdown("""
    <div class="info-box">
    <b>Risk Score (0–1):</b> RUL proximity to failure (60% weight) + anomaly signal strength (40% weight).<br>
    <b>Confidence:</b> HIGH = both signals agree · MEDIUM = one signal is clear · LOW = contradictory signals.<br>
    <b>Justification:</b> Plain-language sentence explaining what drove the recommendation.
    </div>""", unsafe_allow_html=True)

    risk = float(latest["risk_score"])
    conf = str(latest["confidence"])
    just = str(latest.get("justification","—"))
    cfc  = f"conf-{conf.lower()}"

    dc1, dc2, dc3 = st.columns(3)
    with dc1:
        rc = "#ef4444" if risk>=0.7 else "#f59e0b" if risk>=0.4 else "#22c55e"
        fig = go.Figure(go.Indicator(mode="gauge+number", value=round(risk*100,1),
            title={"text":"Risk Score (%)"},
            gauge={"axis":{"range":[0,100]},"bar":{"color":rc},
                   "steps":[{"range":[0,40],"color":"#14532d"},{"range":[40,70],"color":"#78350f"},{"range":[70,100],"color":"#7f1d1d"}]}))
        fig.update_layout(height=240, margin=dict(l=20,r=20,t=50,b=10))
        st.plotly_chart(fig, use_container_width=True)
    with dc2:
        st.metric("Confidence Level", conf)
        st.metric("Predicted RUL", f"{pred_rul:.1f} cycles")
        st.metric("Recommended Action", latest_act[:28]+"…" if len(latest_act)>30 else latest_act)
    with dc3:
        st.markdown("**Decision Justification:**")
        st.markdown(f'<div class="info-box">{just}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Risk Score Over Time")
    fig_risk = go.Figure()
    fig_risk.add_trace(go.Scatter(x=unit_df_full["time_cycle"], y=unit_df_full["risk_score"],
        mode="lines", fill="tozeroy", fillcolor="rgba(239,68,68,0.2)",
        line=dict(color="#ef4444"), name="Risk Score"))
    fig_risk.add_hline(y=0.7, line_dash="dash", line_color="#ef4444",  annotation_text="Critical (0.7)")
    fig_risk.add_hline(y=0.4, line_dash="dash", line_color="#f59e0b",  annotation_text="Warning (0.4)")
    fig_risk.update_layout(template="plotly_dark", height=360,
                            xaxis_title="Cycle", yaxis_title="Risk Score", yaxis_range=[0,1])
    st.plotly_chart(fig_risk, use_container_width=True)

    p1, p2 = st.columns(2)
    with p1:
        st.subheader("Confidence Distribution")
        cc = unit_df_full["confidence"].value_counts()
        fig = px.pie(values=cc.values, names=cc.index, template="plotly_dark",
                     color=cc.index, color_discrete_map={"HIGH":"#22c55e","MEDIUM":"#f59e0b","LOW":"#ef4444"})
        fig.update_layout(height=320)
        st.plotly_chart(fig, use_container_width=True)
    with p2:
        st.subheader("Risk Score Distribution")
        fig = px.histogram(unit_df_full, x="risk_score", nbins=30, template="plotly_dark",
                           color_discrete_sequence=["#3b82f6"])
        fig.add_vline(x=0.4, line_dash="dash", line_color="#f59e0b")
        fig.add_vline(x=0.7, line_dash="dash", line_color="#ef4444")
        fig.update_layout(height=320, xaxis_title="Risk Score", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Recent Anomaly Justifications")
    anom_rows = unit_df_full[unit_df_full["anomaly_flag"] == -1].tail(10)
    if not anom_rows.empty:
        for _, row in anom_rows.iterrows():
            with st.expander(f"Cycle {int(row['time_cycle'])} — Risk: {row['risk_score']:.2f} | Confidence: {row['confidence']}"):
                st.markdown(f"**Action:** {row['recommended_action']}")
                st.markdown(f"**Justification:** {row.get('justification','—')}")
                st.markdown(f"Actual RUL: `{int(row['RUL'])}` · Predicted RUL: `{row['predicted_RUL']:.1f}` · Score: `{row['anomaly_score']:.4f}`")
    else:
        st.info("No anomaly rows for this unit.")


# ══════════════════════════════════════════════════════════════════════
# PAGE 4 — Autoencoder Analysis
# ══════════════════════════════════════════════════════════════════════
elif page == "Autoencoder Analysis":

    st.title("Autoencoder Anomaly Detection")
    st.markdown("<p class='small-text'>Day 10 objective — deep learning reconstruction error vs Isolation Forest</p>", unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    <b>Architecture:</b> Dense autoencoder (n_features → 32 → 16 → 8 → 16 → 32 → n_features).<br>
    <b>Training:</b> Trained only on healthy engine data (RUL &gt; 50). At inference, high reconstruction
    error = anomaly. Threshold = 95th percentile of reconstruction error on normal training data.
    </div>""", unsafe_allow_html=True)

    if not has_autoencoder:
        st.warning("Autoencoder columns not found.")
        st.code("Re-run the pipeline with extended mode to enable Autoencoder analysis.")
        st.stop()

    st.subheader(f"Reconstruction Error Over Time — Unit {selected_unit}")
    fig_re = go.Figure()
    fig_re.add_trace(go.Scatter(x=unit_df_full["time_cycle"], y=unit_df_full["reconstruction_error"],
        mode="lines", name="Reconstruction Error", line=dict(color="#a78bfa")))
    ae_anom = unit_df_full[unit_df_full["autoencoder_anomaly_flag"] == 1]
    fig_re.add_trace(go.Scatter(x=ae_anom["time_cycle"], y=ae_anom["reconstruction_error"],
        mode="markers", marker=dict(color="#ef4444", size=5), name="AE Anomaly"))
    fig_re.update_layout(template="plotly_dark", height=370,
                          xaxis_title="Cycle", yaxis_title="Reconstruction Error (MSE)")
    st.plotly_chart(fig_re, use_container_width=True)

    cmp1, cmp2 = st.columns(2)
    with cmp1:
        st.subheader("Isolation Forest")
        ifc = unit_df_full["anomaly_flag"].map({1:"Normal",-1:"Anomaly"}).value_counts()
        fig = px.pie(values=ifc.values, names=ifc.index, template="plotly_dark",
                     color=ifc.index, color_discrete_map={"Normal":"#22c55e","Anomaly":"#ef4444"})
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    with cmp2:
        st.subheader("Autoencoder")
        aec = unit_df_full["autoencoder_anomaly_flag"].map({0:"Normal",1:"Anomaly"}).value_counts()
        fig = px.pie(values=aec.values, names=aec.index, template="plotly_dark",
                     color=aec.index, color_discrete_map={"Normal":"#22c55e","Anomaly":"#ef4444"})
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Model Agreement Analysis")
    ag = unit_df_full.copy()
    ag["if_a"] = (ag["anomaly_flag"] == -1).astype(int)
    ag["ae_a"] = ag["autoencoder_anomaly_flag"].astype(int)
    ag["Agreement"] = ag.apply(lambda r:
        "Both detect"  if r["if_a"] and r["ae_a"]  else
        "IF only"      if r["if_a"]                 else
        "AE only"      if r["ae_a"]                 else "Both normal", axis=1)
    agc = ag["Agreement"].value_counts()
    cmap = {"Both detect":"#ef4444","IF only":"#f59e0b","AE only":"#a78bfa","Both normal":"#22c55e"}
    fig_ag = px.bar(x=agc.index, y=agc.values, template="plotly_dark",
                    color=agc.index, color_discrete_map=cmap)
    fig_ag.update_layout(height=320, showlegend=False,
                          xaxis_title="Agreement Status", yaxis_title="Count")
    st.plotly_chart(fig_ag, use_container_width=True)

    ae_rpt = BASE_DIR / "outputs" / "metrics" / "autoencoder_classification_report.txt"
    if ae_rpt.exists():
        st.markdown("---")
        st.subheader("Autoencoder Evaluation Report")
        st.code(ae_rpt.read_text(), language="text")


# ══════════════════════════════════════════════════════════════════════
# PAGE 5 — Reinforcement Learning Demo
# ══════════════════════════════════════════════════════════════════════
elif page == "Reinforcement Learning Demo":

    st.title("Reinforcement Learning — Spacecraft Decision Demo")
    st.markdown("<p class='small-text'>Day 14 objective — RL basics in a space context</p>", unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    <b>Why RL for spacecraft?</b> Communication delays prevent Earth from responding to every alert.
    An RL agent learns to choose corrective actions autonomously by maximising long-term reward.<br><br>
    This demo trains a <b>Q-table agent</b> on a toy environment:
    state = (anomaly_flag, RUL_band), actions = do_nothing / monitor / inspect / emergency_shutdown.
    In production, a Deep Q-Network (DQN) would operate on the full sensor feature vector.
    </div>""", unsafe_allow_html=True)

    st.subheader("Rule-Based vs RL Comparison")
    cdf = pd.DataFrame({
        "Criterion": ["Decision logic","Adaptability","Explainability","Edge cases","Training data needed"],
        "Rule-Based (Current)": ["Hand-crafted IF/ELSE","Fixed until manually updated","Fully transparent","Only anticipated cases","No"],
        "RL Agent (Extension)": ["Learned via reward feedback","Self-improves over episodes","Needs explainability layer","Can discover novel strategies","Yes — simulated environment"],
    })
    st.table(cdf)

    st.markdown("---")
    st.subheader("Interactive Q-Learning Demo")

    rl1, rl2 = st.columns([1,2])
    with rl1:
        episodes = st.slider("Training episodes", 100, 1000, 500, 50)
        alpha    = st.slider("Learning rate (α)", 0.01, 0.5, 0.1, 0.01)
        gamma    = st.slider("Discount factor (γ)", 0.5, 0.99, 0.9, 0.01)
        epsilon  = st.slider("Exploration (ε)", 0.1, 0.9, 0.3, 0.05)
        run_btn  = st.button("Train Q-Table Agent", type="primary")
    with rl2:
        st.markdown("""
        **State:** `anomaly_flag (0/1) × RUL_band (0–3)` = 8 states  
        **Actions:** `do_nothing | monitor | inspect | emergency_shutdown`  
        **Rewards:**  
        - +10 shutdown at critical RUL  
        - +5 inspection at high-risk RUL  
        - -10 no action when critical  
        - -5 unnecessary shutdown when safe  
        """)

    if run_btn:
        with st.spinner("Training..."):
            try:
                from src.decision_confidence import run_q_learning_demo, SpacecraftRLEnvironment
                Q   = run_q_learning_demo(episodes=episodes, alpha=alpha, gamma=gamma, epsilon=epsilon)
                env = SpacecraftRLEnvironment()
                ACTIONS    = env.ACTIONS
                RUL_BANDS  = ["Critical (<5)","High-risk (5–15)","Caution (15–30)","Safe (>30)"]
                ANOM_LBLS  = ["No anomaly","Anomaly detected"]

                rows = []
                for anom in range(2):
                    for band in range(4):
                        si = anom*4+band
                        best_a = int(np.argmax(Q[si]))
                        qv = Q[si]
                        rows.append({"Anomaly State":ANOM_LBLS[anom],"RUL Band":RUL_BANDS[band],
                            "Best Action":ACTIONS[best_a],
                            "Q(do_nothing)":round(qv[0],3),"Q(monitor)":round(qv[1],3),
                            "Q(inspect)":round(qv[2],3),"Q(shutdown)":round(qv[3],3)})

                st.success(f"Training complete — {episodes} episodes")
                st.subheader("Learned Policy")
                st.dataframe(pd.DataFrame(rows), use_container_width=True)

                st.subheader("Q-value Heatmap")
                q_df = pd.DataFrame(Q,
                    columns=["do_nothing","monitor","inspect","emergency_shutdown"],
                    index=[f"{ANOM_LBLS[i//4]}|{RUL_BANDS[i%4]}" for i in range(8)])
                fig_hm = px.imshow(q_df, text_auto=".2f", template="plotly_dark",
                                   color_continuous_scale="RdYlGn")
                fig_hm.update_layout(height=380, xaxis_title="Action", yaxis_title="State")
                st.plotly_chart(fig_hm, use_container_width=True)

            except Exception as e:
                st.error(f"Error: {e}")

    rl_q = BASE_DIR / "outputs" / "metrics" / "rl_q_table.txt"
    if rl_q.exists():
        st.markdown("---")
        st.subheader("Saved Q-table (from last pipeline run)")
        st.code(rl_q.read_text(), language="text")


# ══════════════════════════════════════════════════════════════════════
# PAGE 6 — LIVE STREAM
# Polls GET /live every 3 seconds and updates all visuals in real time.
# Each prediction made by the Kafka consumer → API appears here live.
# ══════════════════════════════════════════════════════════════════════
elif page == "🔴 Live Stream":

    st.title("🔴 Live Telemetry Stream")
    st.markdown(
        "<p style='color:#006622;font-weight:700;font-size:15px;'>"
        "This page reflects real-time predictions from the Kafka → API pipeline. "
        "Every row the Kafka consumer sends to <code>/predict</code> appears here automatically.</p>",
        unsafe_allow_html=True,
    )

    # ── Controls ──────────────────────────────────────────────────
    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1, 1, 2])
    with col_ctrl1:
        refresh_rate = st.selectbox("Refresh every", [2, 3, 5, 10], index=1)
    with col_ctrl2:
        max_rows = st.selectbox("Show last N predictions", [50, 100, 200, 500], index=1)
    with col_ctrl3:
        if st.button("🗑️ Clear Live Store", type="primary"):
            try:
                r = requests.delete(f"{API_BASE}/live/reset", timeout=3)
                if r.status_code == 200:
                    st.success("Live store cleared. Waiting for new predictions...")
                else:
                    st.warning(f"Reset failed: {r.status_code}")
            except Exception as e:
                st.error(f"Cannot reach API: {e}")

    st.markdown("---")

    # ── Placeholders — all updated in the polling loop ────────────
    status_bar   = st.empty()
    kpi_row      = st.empty()
    col_left, col_right = st.columns(2)
    with col_left:
        chart_anomaly = st.empty()
    with col_right:
        chart_rul     = st.empty()
    col_risk, col_conf = st.columns(2)
    with col_risk:
        chart_risk    = st.empty()
    with col_conf:
        chart_actions = st.empty()
    feed_header  = st.empty()
    live_table   = st.empty()

    # ── Polling loop ──────────────────────────────────────────────
    while True:
        try:
            resp = requests.get(
                f"{API_BASE}/live",
                params={"limit": max_rows},
                timeout=5,
            )
            data = resp.json() if resp.status_code == 200 else None
        except Exception:
            data = None

        # ── No data or API unreachable ────────────────────────────
        if not data or data["count"] == 0:
            status_bar.markdown(
                "<div style='background:#fff3cd;border:1px solid #ffc107;"
                "border-radius:6px;padding:10px 16px;font-weight:700;color:#7a4a00;'>"
                "⏳ Waiting for live predictions... "
                "Make sure <code>docker compose up --build</code> is running "
                "and the Kafka consumer is streaming.</div>",
                unsafe_allow_html=True,
            )
            time.sleep(refresh_rate)
            st.rerun()
            break

        # ── Build DataFrame ───────────────────────────────────────
        ldf = pd.DataFrame(data["predictions"])
        ldf["timestamp"]    = pd.to_datetime(ldf["timestamp"])
        ldf["anomaly_label"]= ldf["anomaly_flag"].map({1: "Normal", -1: "Anomaly"})
        ldf["seq"]          = range(1, len(ldf) + 1)

        total      = len(ldf)
        n_anomaly  = int((ldf["anomaly_flag"] == -1).sum())
        n_normal   = total - n_anomaly
        latest_row = ldf.iloc[-1]
        avg_rul    = ldf["predicted_rul"].mean()
        avg_risk   = ldf["risk_score"].mean()

        # ── Status bar ────────────────────────────────────────────
        act  = str(latest_row["recommended_action"])
        if "CRITICAL" in act:
            bar_color = "#fdd;border-color:#c00;color:#800"
        elif any(w in act for w in ["HIGH RISK", "WARNING", "CAUTION"]):
            bar_color = "#fff3cd;border-color:#e65100;color:#7a4a00"
        else:
            bar_color = "#d4edda;border-color:#155724;color:#155724"

        status_bar.markdown(
            f"<div style='background:{bar_color.split(';')[0].split(':')[1]};"
            f"border:2px solid {bar_color.split(';')[1].split(':')[1]};"
            f"border-radius:6px;padding:10px 18px;font-weight:800;font-size:14px;"
            f"color:{bar_color.split(';')[2].split(':')[1]};'>"
            f"🛰️ Latest:  Unit {int(latest_row['unit_id'])}  |  "
            f"Cycle {int(latest_row['time_cycle'])}  |  "
            f"RUL {latest_row['predicted_rul']:.1f}  |  "
            f"Risk {latest_row['risk_score']:.2f}  |  "
            f"Confidence: {latest_row['confidence']}  |  "
            f"<b>{act}</b>  |  "
            f"⏱ Latency: {latest_row.get('latency_ms', 0):.1f} ms"
            f"</div>",
            unsafe_allow_html=True,
        )

        # ── KPI cards ─────────────────────────────────────────────
        kpi_row.markdown(
            f"""
            <div style='display:flex;gap:12px;margin:10px 0;'>
              <div style='flex:1;background:#fff;border:1px solid #7bbfe0;border-radius:8px;
                   padding:14px;border-top:3px solid #0288d1;'>
                <div style='font-size:10px;font-weight:700;color:#607d8b;letter-spacing:1.5px;'>
                  TOTAL PREDICTIONS</div>
                <div style='font-size:28px;font-weight:900;color:#0288d1;'>{total:,}</div>
              </div>
              <div style='flex:1;background:#fff;border:1px solid #7bbfe0;border-radius:8px;
                   padding:14px;border-top:3px solid #c62828;'>
                <div style='font-size:10px;font-weight:700;color:#607d8b;letter-spacing:1.5px;'>
                  ANOMALIES DETECTED</div>
                <div style='font-size:28px;font-weight:900;color:#c62828;'>{n_anomaly:,}</div>
                <div style='font-size:10px;color:#607d8b;'>{n_anomaly/total*100:.1f}% of stream</div>
              </div>
              <div style='flex:1;background:#fff;border:1px solid #7bbfe0;border-radius:8px;
                   padding:14px;border-top:3px solid #2e7d32;'>
                <div style='font-size:10px;font-weight:700;color:#607d8b;letter-spacing:1.5px;'>
                  NORMAL READINGS</div>
                <div style='font-size:28px;font-weight:900;color:#2e7d32;'>{n_normal:,}</div>
              </div>
              <div style='flex:1;background:#fff;border:1px solid #7bbfe0;border-radius:8px;
                   padding:14px;border-top:3px solid #e65100;'>
                <div style='font-size:10px;font-weight:700;color:#607d8b;letter-spacing:1.5px;'>
                  AVG PREDICTED RUL</div>
                <div style='font-size:28px;font-weight:900;color:#e65100;'>{avg_rul:.1f}</div>
                <div style='font-size:10px;color:#607d8b;'>cycles remaining</div>
              </div>
              <div style='flex:1;background:#fff;border:1px solid #7bbfe0;border-radius:8px;
                   padding:14px;border-top:3px solid #6a1b9a;'>
                <div style='font-size:10px;font-weight:700;color:#607d8b;letter-spacing:1.5px;'>
                  AVG RISK SCORE</div>
                <div style='font-size:28px;font-weight:900;color:#6a1b9a;'>{avg_risk:.3f}</div>
                <div style='font-size:10px;color:#607d8b;'>0.0 safe → 1.0 critical</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Anomaly score timeline ────────────────────────────────
        fig_a = px.scatter(
            ldf, x="seq", y="anomaly_score", color="anomaly_label",
            color_discrete_map={"Normal": "#2e7d32", "Anomaly": "#c62828"},
            labels={"seq": "Prediction #", "anomaly_score": "Anomaly Score"},
            title="Anomaly Score — Live Stream",
        )
        fig_a.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
        fig_a.update_layout(
            height=300, margin=dict(l=40, r=10, t=40, b=30),
            legend_title_text="",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0.8)",
            xaxis=dict(gridcolor="#e0e0e0"),
            yaxis=dict(gridcolor="#e0e0e0"),
        )
        chart_anomaly.plotly_chart(fig_a, use_container_width=True)

        # ── Predicted RUL over stream ─────────────────────────────
        fig_r = px.line(
            ldf, x="seq", y="predicted_rul",
            labels={"seq": "Prediction #", "predicted_rul": "Predicted RUL"},
            title="Predicted RUL — Live Stream",
            color_discrete_sequence=["#1565c0"],
        )
        fig_r.update_layout(
            height=300, margin=dict(l=40, r=10, t=40, b=30),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0.8)",
            xaxis=dict(gridcolor="#e0e0e0"),
            yaxis=dict(gridcolor="#e0e0e0"),
        )
        chart_rul.plotly_chart(fig_r, use_container_width=True)

        # ── Risk score over stream ────────────────────────────────
        fig_risk = px.area(
            ldf, x="seq", y="risk_score",
            title="Risk Score — Live Stream",
            labels={"seq": "Prediction #", "risk_score": "Risk Score"},
            color_discrete_sequence=["#e65100"],
        )
        fig_risk.add_hline(y=0.7, line_dash="dash", line_color="#c62828",
                           line_width=1, annotation_text="Critical 0.7",
                           annotation_font_size=10)
        fig_risk.add_hline(y=0.4, line_dash="dash", line_color="#e65100",
                           line_width=1, annotation_text="Warning 0.4",
                           annotation_font_size=10)
        fig_risk.update_layout(
            height=280, yaxis_range=[0, 1],
            margin=dict(l=40, r=10, t=40, b=30),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0.8)",
            xaxis=dict(gridcolor="#e0e0e0"),
            yaxis=dict(gridcolor="#e0e0e0"),
        )
        chart_risk.plotly_chart(fig_risk, use_container_width=True)

        # ── Recommended action distribution ──────────────────────
        action_counts = ldf["recommended_action"].value_counts()
        colors = []
        for a in action_counts.index:
            if "CRITICAL" in a:   colors.append("#c62828")
            elif any(w in a for w in ["HIGH", "WARNING", "CAUTION"]): colors.append("#e65100")
            else:                  colors.append("#2e7d32")
        fig_act = px.bar(
            x=action_counts.index, y=action_counts.values,
            title="Recommended Actions — Live Stream",
            labels={"x": "Action", "y": "Count"},
            color=action_counts.index,
            color_discrete_sequence=colors,
        )
        fig_act.update_layout(
            height=280, showlegend=False,
            xaxis_tickangle=-20,
            margin=dict(l=40, r=10, t=40, b=60),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0.8)",
            xaxis=dict(gridcolor="#e0e0e0"),
            yaxis=dict(gridcolor="#e0e0e0"),
        )
        chart_actions.plotly_chart(fig_act, use_container_width=True)

        # ── Live prediction feed table ────────────────────────────
        feed_header.markdown(
            f"<div style='font-weight:800;font-size:14px;color:#0d1f35;"
            f"margin-top:8px;'>📋 Live Prediction Feed  "
            f"<span style='font-weight:500;font-size:12px;color:#607d8b;'>"
            f"(last {min(20, len(ldf))} predictions — newest first)</span></div>",
            unsafe_allow_html=True,
        )
        display_cols = [
            "timestamp", "unit_id", "time_cycle",
            "anomaly_flag", "anomaly_score",
            "predicted_rul", "risk_score",
            "confidence", "recommended_action",
        ]
        display_df = (
            ldf[display_cols]
            .tail(20)
            .iloc[::-1]
            .copy()
        )
        display_df["timestamp"] = display_df["timestamp"].dt.strftime("%H:%M:%S")
        display_df.columns = [
            "Time", "Unit", "Cycle",
            "Flag", "Score",
            "Pred RUL", "Risk",
            "Confidence", "Action",
        ]
        live_table.dataframe(display_df, use_container_width=True, height=380)

        # ── Wait then rerun ───────────────────────────────────────
        time.sleep(refresh_rate)
        st.rerun()