#!/bin/bash
# entrypoint.sh — Container startup script
# Derives VEHICLE_ID from hostname if not set via env var.
# This makes `docker compose up --scale vehicle=10` work without
# hardcoding 10 separate service blocks.

set -e

# Docker Compose sets hostname to <service>-<replica_index> (e.g. vehicle-3)
# Extract the numeric suffix and zero-pad it to 2 digits
if [ -z "$VEHICLE_ID" ]; then
    HOSTNAME_SUFFIX=$(hostname | grep -o '[0-9]*$' || echo "0")
    VEHICLE_ID="vehicle_$(printf '%02d' "${HOSTNAME_SUFFIX}")"
    export VEHICLE_ID
fi

echo "[entrypoint] Starting container:"
echo "  VEHICLE_ID = $VEHICLE_ID"
echo "  ROLE       = ${ROLE:-benign}"
echo "  REDIS_HOST = ${REDIS_HOST:-redis}"
echo "  FL_SERVER  = ${FL_SERVER_HOST:-fl_server}:${FL_SERVER_PORT:-8080}"
echo "  MODEL_DIR  = ${MODEL_DIR:-/app/model}"

exec python vehicle.py
