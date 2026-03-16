#  Space Telemetry Anomaly Detection & Predictive Maintenance System

## Project Overview
This project builds an **AI-based telemetry monitoring system** for detecting abnormal behaviour in jet engine data and predicting the **Remaining Useful Life (RUL)** of engines.

The system uses the **NASA C-MAPSS turbofan engine degradation dataset** to simulate real-world predictive maintenance scenarios in aerospace systems.

The pipeline performs:

- Telemetry data preprocessing
- Feature engineering
- Anomaly detection using Isolation Forest
- Remaining Useful Life prediction using Random Forest
- Maintenance decision recommendation
- Interactive monitoring dashboard using Streamlit
- Telemetry simulation for playback analysis

The goal is to demonstrate how **AI can support predictive maintenance and operational safety in aerospace systems.**

---

# 🛰 Dataset

**NASA Turbofan Engine Degradation Simulation Dataset (C-MAPSS)**

Dataset file used:
data/raw/train_FD001.txt

Telemetry data contains:

- Engine operational settings
- Multiple sensor measurements
- Engine cycle information
- Degradation behaviour over time

These signals are used to simulate telemetry streams and predict engine health.

---

# System Architecture
space-telemetry-anomaly-system
│
├── app
│ └── streamlit_app.py # Monitoring dashboard
│
├── data
│ ├── raw # Raw NASA dataset
│ ├── processed # Processed telemetry data
│ └── sample_stream # Sample simulated telemetry
│
├── docs
│ └── dashboard.png # Dashboard preview image
│
├── models # Saved ML models
│
├── outputs
│ ├── logs # Execution logs
│ ├── figures # Visualization outputs
│ └── telemetry_results.csv # Final processed telemetry output
│
├── src
│ ├── config.py # Central configuration
│ ├── data_loader.py # Dataset loader
│ ├── feature_engineering.py # Telemetry feature generation
│ ├── train_anomaly.py # Isolation Forest training
│ ├── train_rul.py # Random Forest RUL prediction
│ ├── decision_engine.py # Maintenance recommendation logic
│ ├── simulator.py # Telemetry simulation
│ └── utils.py # Utility functions
│
├── tests
│ └── test_simulator.py # Simulator test
│
├── main.py # End-to-end ML pipeline
└── README.md


---

#  Machine Learning Models

## Anomaly Detection Model

Algorithm used:

**Isolation Forest**

Purpose:

- Detect abnormal sensor behaviour
- Identify potential system faults
- Provide anomaly score and anomaly flag

Output:
anomaly_score
anomaly_flag


---

## Remaining Useful Life Prediction

Algorithm used:

**Random Forest Regressor**

Purpose:

- Predict how many cycles remain before engine failure
- Support maintenance planning

Output:
predicted_RUL


---

# Maintenance Decision Engine

The system includes a **rule-based decision engine** that combines:

- anomaly detection results
- anomaly score
- predicted RUL

to generate intelligent maintenance recommendations.

Example outputs:
NORMAL: System operating normally
WARNING: Monitor this unit closely
HIGH RISK: Schedule urgent inspection
CRITICAL: Immediate maintenance required


---

# Telemetry Simulation

The project includes a **telemetry simulator** that replays engine cycles to imitate real-time telemetry streaming.

Simulator functionality:

- Streams telemetry records sequentially
- Allows dashboard playback
- Demonstrates real-time monitoring scenarios

Simulator module: src/simulator.py


---

# Streamlit Monitoring Dashboard

The interactive dashboard allows users to:

- Select engine units
- Visualize sensor telemetry
- View anomaly detection results
- Track predicted RUL
- Observe maintenance recommendations
- Replay telemetry simulation

Launch dashboard: streamlit run app/streamlit_app.py

Dashboard preview: docs/dashboard.png

---

# Testing

Basic project tests are provided to validate simulator functionality.

Run tests:

pytest tests/

---

# Running the Project

### Install dependencies


pip install -r requirements.txt


or manually install:


pip install pandas numpy scikit-learn streamlit matplotlib joblib pytest


---

### Run ML pipeline


python main.py


This will:

- load dataset
- perform feature engineering
- train anomaly detection model
- train RUL prediction model
- generate maintenance recommendations
- save results

Output file: outputs/telemetry_results.csv


---

### Launch monitoring dashboard


streamlit run app/streamlit_app.py


The dashboard visualizes telemetry data and allows simulation playback.

---

# Example Output Fields


unit_id
time_cycle
RUL
predicted_RUL
anomaly_score
anomaly_flag
recommended_action


These values allow users to monitor engine health over time.

---

# Project Objectives

This project demonstrates how AI systems can support:

- predictive maintenance
- anomaly detection in telemetry systems
- operational safety in aerospace engineering
- AI-assisted maintenance decision making

The system simulates a **data-driven maintenance monitoring platform for jet engines.**

---

# Future Improvements

Potential enhancements include:

- real-time telemetry ingestion
- live inference API
- deep learning models for RUL prediction
- advanced anomaly detection models
- deployment using Docker or cloud infrastructure
- integration with IoT telemetry streams

---

# Author

**Purushothaman Shanmugam**

AI / Machine Learning Research Projects  
Telemetry Analytics & Predictive Maintenance Systems

---

# License

This project is for **educational and research purposes** demonstrating predictive maintenance techniques using open datasets.
