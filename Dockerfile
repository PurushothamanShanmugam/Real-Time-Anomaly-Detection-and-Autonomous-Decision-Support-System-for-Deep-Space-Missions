FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

COPY requirements.txt /app/

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

RUN chmod +x /app/entrypoint.sh

EXPOSE 8501

ENTRYPOINT ["/app/entrypoint.sh"]