# Vehicle Container — Dockerfile
# Single image handles both benign and attacker roles via ROLE env var.
# Phase 1 model artifacts are copied from /common/model/ at build time.

FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    dos2unix \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python dependencies ────────────────────────────────────────────────────
COPY vehicle/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application code ───────────────────────────────────────────────────────
COPY vehicle/vehicle.py          .
COPY vehicle/generator.py        .
COPY vehicle/ids_inference.py    .
COPY vehicle/network_controller.py .
COPY vehicle/crypto_switch.py    .
COPY vehicle/client_stub.py      .

# ── Shared common package ──────────────────────────────────────────────────
COPY common/ ./common/

# ── Phase 1 model artifacts ────────────────────────────────────────────────
# best_ids_model.keras, feature_scaler.pkl, label_encoder.pkl
COPY model/ ./model/

# ── Entrypoint ─────────────────────────────────────────────────────────────
COPY vehicle/entrypoint.sh .
RUN dos2unix entrypoint.sh && chmod +x entrypoint.sh

# ── Environment defaults (override in docker-compose.yml) ──────────────────
ENV VEHICLE_ID=""
ENV ROLE="benign"
ENV REDIS_HOST="redis"
ENV REDIS_PORT="6379"
ENV FL_SERVER_HOST="fl_server"
ENV FL_SERVER_PORT="8080"
ENV MODEL_DIR="/app/model"
ENV TRAIN_SAMPLE_THRESHOLD="500"
ENV TRAIN_TIME_INTERVAL_S="60"
ENV GENERATION_INTERVAL_S="0.5"
ENV ATTACKER_INTERVAL_S="0.1"
ENV PYTHONUNBUFFERED="1"
ENV TF_CPP_MIN_LOG_LEVEL="3"

ENTRYPOINT ["./entrypoint.sh"]
