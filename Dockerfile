# Dockerfile — Streamlit Dashboard
# main.py is run by the pipeline service — this container only serves the dashboard

FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

RUN apt-get update && apt-get install -y --no-install-recommends bash curl \
    && rm -rf /var/lib/apt/lists/*

# Install only dashboard dependencies — no TensorFlow needed here
RUN pip install --no-cache-dir \
    pandas numpy scikit-learn joblib \
    streamlit plotly matplotlib \
    kafka-python requests

COPY . /app

RUN chmod +x /app/entrypoint.sh

EXPOSE 8501

ENTRYPOINT ["/app/entrypoint.sh"]