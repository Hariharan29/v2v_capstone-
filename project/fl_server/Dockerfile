# FL Server Dockerfile
# Separate from the vehicle image — lighter, no Redis or generator deps.
# Loads Phase 1 model at startup as the initial global weights.

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python dependencies ────────────────────────────────────────────────────
COPY fl_server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Server code ────────────────────────────────────────────────────────────
COPY fl_server/server.py    .
COPY fl_server/krum.py      .
COPY fl_server/fedprox.py   .

# ── Phase 1 model artifacts (initial global weights) ──────────────────────
COPY model/ ./model/

# ── Environment defaults (override in docker-compose.yml) ─────────────────
ENV NUM_ROUNDS="10"
ENV MIN_FIT_CLIENTS="10"
ENV MIN_AVAIL_CLIENTS="10"
ENV MIN_EVAL_CLIENTS="5"
ENV KRUM_F="1"
ENV PROXIMAL_MU="0.01"
ENV LOCAL_EPOCHS="3"
ENV BATCH_SIZE="256"
ENV FL_SERVER_PORT="8080"
ENV MODEL_DIR="/app/model"
ENV MU_SCHEDULE="flat"
ENV PYTHONUNBUFFERED="1"
ENV TF_CPP_MIN_LOG_LEVEL="3"

EXPOSE 8080

CMD ["python", "server.py"]
