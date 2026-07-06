"""
vehicle.py — Core Vehicle Container Logic
==========================================
Parameterised by VEHICLE_ID and ROLE (benign/attacker) via environment variables.
A single Docker image handles both roles — no separate Dockerfiles.

Three concurrent asyncio loops:
  1. generation_loop()  — generate BSMs and publish to bsm:broadcast
  2. reception_loop()   — subscribe to bsm:broadcast, run IDS inference
  3. training_loop()    — periodic local FL training (every 500 samples or 60s)

LocalNetworkController handles:
  - Malicious detection response
  - Blacklisting
  - Crypto-mode switching (stub, Stage 3 fills)
  - Redis mode:alert broadcast

Member B integration point:
  - training_loop() calls start_fl_client() from client.py (or client_stub.py)
  - proximal_mu is read from FL server config per-round (not hardcoded)

Owner: Member A
"""

from __future__ import annotations
import asyncio
import json
import logging
import os
import sys
import time
from collections import deque
from typing import Optional

import numpy as np
import redis.asyncio as aioredis

# ── Path setup so container can find common/ ─────────────────────────────────
sys.path.insert(0, "/app")          # container working directory
sys.path.insert(0, "/app/common")   # shared schemas

from common.message_schema import (
    BSMMessage,
    ModeAlertEvent,
    BSM_CHANNEL,
    ALERT_CHANNEL,
    NORMAL_CHANNEL,
    FEATURE_COLS,
    WINDOW_SIZE,
    CLASS_NAMES,
    BENIGN_CLASS_IDX,
    MODE_NORMAL,
)

from generator       import BenignGenerator, AttackerGenerator
from ids_inference   import IDSInferenceEngine
from network_controller import LocalNetworkController
from crypto_switch   import sign_and_encrypt, verify_and_decrypt

# ── Try to import real Flower client, fall back to stub ──────────────────────
try:
    from client import IDSFlowerClient, start_fl_client  # Member B's implementation
    logger_client = logging.getLogger(__name__)
    logger_client.info("[vehicle] Using real client.py (Member B)")
except ImportError:
    from client_stub import IDSFlowerClient, start_fl_client  # fallback stub
    logger_client = logging.getLogger(__name__)
    logger_client.warning("[vehicle] client.py not found — using client_stub.py")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("vehicle")


# ── Configuration from environment variables ──────────────────────────────────
VEHICLE_ID       = os.environ.get("VEHICLE_ID", "vehicle_00")
ROLE             = os.environ.get("ROLE", "benign").lower()        # "benign" | "attacker"
REDIS_HOST       = os.environ.get("REDIS_HOST", "redis")
REDIS_PORT       = int(os.environ.get("REDIS_PORT", "6379"))
FL_SERVER_HOST   = os.environ.get("FL_SERVER_HOST", "fl_server")
FL_SERVER_PORT   = int(os.environ.get("FL_SERVER_PORT", "8080"))
MODEL_DIR        = os.environ.get("MODEL_DIR", "/app/model")

# Training trigger thresholds
TRAIN_SAMPLE_THRESHOLD = int(os.environ.get("TRAIN_SAMPLE_THRESHOLD", "500"))
TRAIN_TIME_INTERVAL_S  = float(os.environ.get("TRAIN_TIME_INTERVAL_S", "60"))

# Generation rate (seconds between BSM publishes)
GENERATION_INTERVAL_S  = float(os.environ.get("GENERATION_INTERVAL_S", "0.5"))
# Attacker floods faster
ATTACKER_INTERVAL_S    = float(os.environ.get("ATTACKER_INTERVAL_S", "0.1"))


# ─────────────────────────────────────────────────────────────────────────────
# Main Vehicle Class
# ─────────────────────────────────────────────────────────────────────────────
class Vehicle:
    def __init__(self):
        self.vehicle_id = VEHICLE_ID
        self.role       = ROLE
        self._redis: Optional[aioredis.Redis] = None
        self._controller: Optional[LocalNetworkController] = None

        # IDS engine (benign vehicles only)
        self._ids: Optional[IDSInferenceEngine] = None

        # Buffer for benign samples collected during reception (for local training)
        self._benign_buffer_X: list = []
        self._benign_buffer_y: list = []
        self._last_train_time: float = time.time()

        # Loaded Keras model (shared between IDS engine and FL client)
        self._model = None

        logger.info(
            f"[{self.vehicle_id}] Initialising as ROLE={self.role} | "
            f"Redis={REDIS_HOST}:{REDIS_PORT} | FL={FL_SERVER_HOST}:{FL_SERVER_PORT}"
        )

    # ── Lifecycle ─────────────────────────────────────────────────────────────
    async def start(self):
        """Connect to Redis, load model, then run all three loops concurrently."""
        # Redis connection
        self._redis = aioredis.Redis(
            host=REDIS_HOST, port=REDIS_PORT,
            decode_responses=True, socket_timeout=5,
        )
        await self._redis.ping()
        logger.info(f"[{self.vehicle_id}] Redis connected")

        # Network controller (both roles get it — attacker ignores its output)
        self._controller = LocalNetworkController(
            vehicle_id=self.vehicle_id,
            redis_conn=self._redis,
            timeout_s=30.0,
        )

        if self.role == "benign":
            # Load model for IDS inference and FL training
            import tensorflow as tf
            import joblib
            from pathlib import Path
            logger.info(f"[{self.vehicle_id}] Loading IDS model from {MODEL_DIR}")
            self._model = tf.keras.models.load_model(
                str(Path(MODEL_DIR) / "best_ids_model.keras")
            )
            self._ids = IDSInferenceEngine(
                model_dir=MODEL_DIR,
                window_size=WINDOW_SIZE,
                confidence_threshold=0.5,
            )
            logger.info(f"[{self.vehicle_id}] IDS model loaded")

        # Run all loops concurrently
        tasks = [
            asyncio.create_task(self._generation_loop(), name="generation"),
            asyncio.create_task(self._reception_loop(),  name="reception"),
        ]
        if self.role == "benign":
            tasks.append(asyncio.create_task(self._training_loop(), name="training"))

        logger.info(f"[{self.vehicle_id}] All loops started")
        await asyncio.gather(*tasks)

    # ── Loop 1: Generation ────────────────────────────────────────────────────
    async def _generation_loop(self):
        """Generate BSMs and publish to bsm:broadcast."""
        if self.role == "benign":
            gen      = BenignGenerator(self.vehicle_id)
            interval = GENERATION_INTERVAL_S
        else:
            gen      = AttackerGenerator(self.vehicle_id)
            interval = ATTACKER_INTERVAL_S

        logger.info(f"[{self.vehicle_id}] Generation loop started (interval={interval}s)")

        while True:
            try:
                msg = gen.generate()

                # Route through crypto_switch (Stage 3 hook)
                payload = sign_and_encrypt(
                    msg.__dict__,
                    mode=self._controller.current_mode,
                )

                await self._redis.publish(BSM_CHANNEL, json.dumps(payload))
                logger.debug(f"[{self.vehicle_id}] BSM published → {BSM_CHANNEL}")

            except Exception as e:
                logger.error(f"[{self.vehicle_id}] Generation error: {e}")

            await asyncio.sleep(interval)

    # ── Loop 2: Reception ─────────────────────────────────────────────────────
    async def _reception_loop(self):
        """Subscribe to bsm:broadcast and mode channels. Run IDS on incoming BSMs."""
        pubsub = self._redis.pubsub()
        await pubsub.subscribe(BSM_CHANNEL, ALERT_CHANNEL, NORMAL_CHANNEL)
        logger.info(f"[{self.vehicle_id}] Reception loop subscribed to channels")

        async for raw_msg in pubsub.listen():
            if raw_msg["type"] != "message":
                continue

            channel = raw_msg["channel"]
            data    = raw_msg["data"]

            try:
                if channel == BSM_CHANNEL:
                    await self._handle_bsm(data)
                elif channel == ALERT_CHANNEL:
                    await self._handle_alert(data)
                elif channel == NORMAL_CHANNEL:
                    await self._handle_normal(data)
            except Exception as e:
                logger.error(f"[{self.vehicle_id}] Reception error on {channel}: {e}")

    async def _handle_bsm(self, raw_data: str):
        """Process a received BSM from the broadcast channel."""
        payload  = json.loads(raw_data)
        # Decrypt / verify (Stage 3 hook)
        msg_dict = verify_and_decrypt(payload, mode=self._controller.current_mode)

        if msg_dict is None:
            logger.warning(f"[{self.vehicle_id}] Crypto verification failed — discarding")
            return

        # Ignore own messages
        if msg_dict.get("senderID") == self.vehicle_id:
            return

        sender = msg_dict.get("senderID", "unknown")

        # Skip blacklisted senders
        if self._controller.is_blacklisted(sender):
            logger.debug(f"[{self.vehicle_id}] Dropped message from blacklisted {sender}")
            return

        # Attacker vehicles don't run IDS
        if self.role != "benign" or self._ids is None:
            return

        # Parse into BSMMessage
        try:
            bsm = BSMMessage(**{k: msg_dict[k] for k in BSMMessage.__dataclass_fields__})
        except Exception as e:
            logger.warning(f"[{self.vehicle_id}] BSM parse error: {e}")
            return

        # IDS inference
        result = self._ids.ingest(bsm)
        if result is None:
            return

        if result.is_malicious:
            logger.warning(
                f"[{self.vehicle_id}] ⚠ MALICIOUS: sender={sender} "
                f"class={result.class_name} conf={result.confidence:.3f}"
            )
            # Discard, blacklist, trigger controller
            self._ids.reset_sender(sender)
            await self._controller.on_malicious_detection(
                sender_id=sender,
                detected_class=result.class_idx,
                confidence=result.confidence,
            )
        else:
            # Buffer benign sample for training
            self._buffer_benign_sample(bsm)
            logger.debug(
                f"[{self.vehicle_id}] ✓ Benign: sender={sender} "
                f"conf={result.confidence:.3f}"
            )

    async def _handle_alert(self, raw_data: str):
        """Received a mode:alert from a peer vehicle."""
        try:
            outer   = json.loads(raw_data)
            event   = ModeAlertEvent.from_json(outer.get("event_json", raw_data))
            if event.triggered_by != self.vehicle_id:  # don't react to own alerts
                await self._controller.on_peer_alert(event)
        except Exception as e:
            logger.error(f"[{self.vehicle_id}] Alert parse error: {e}")

    async def _handle_normal(self, raw_data: str):
        """Received a mode:normal revert from a peer vehicle (informational for now)."""
        logger.info(f"[{self.vehicle_id}] Received mode:normal from peer")

    # ── Loop 3: Training ──────────────────────────────────────────────────────
    async def _training_loop(self):
        """
        Trigger local FL training every TRAIN_SAMPLE_THRESHOLD benign samples
        OR every TRAIN_TIME_INTERVAL_S seconds, whichever comes first.

        FedProx mu is NOT hardcoded — it is read from the server's per-round
        config dict passed through Flower's fit() config. Member B's server.py
        sends proximal_mu in that dict.
        """
        logger.info(f"[{self.vehicle_id}] Training loop started")

        while True:
            await asyncio.sleep(5.0)   # check every 5s (low overhead)

            n_samples    = len(self._benign_buffer_X)
            elapsed      = time.time() - self._last_train_time
            time_trigger = elapsed >= TRAIN_TIME_INTERVAL_S
            size_trigger = n_samples >= TRAIN_SAMPLE_THRESHOLD

            if not (time_trigger or size_trigger):
                continue
            if n_samples == 0:
                logger.debug(f"[{self.vehicle_id}] No samples yet — skipping training")
                self._last_train_time = time.time()
                continue

            logger.info(
                f"[{self.vehicle_id}] Training triggered: "
                f"samples={n_samples} elapsed={elapsed:.1f}s "
                f"({'size' if size_trigger else 'time'})"
            )

            await self._run_fl_round()
            self._last_train_time = time.time()

    async def _run_fl_round(self):
        """Build local training arrays and call the Flower FL client."""
        if not self._benign_buffer_X:
            return

        # Grab snapshot and clear buffer
        X = np.array(self._benign_buffer_X, dtype=np.float32)
        y = np.array(self._benign_buffer_y, dtype=np.int32)
        self._benign_buffer_X.clear()
        self._benign_buffer_y.clear()

        # 80/20 train/val split from local buffer
        split        = max(1, int(len(X) * 0.8))
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        vehicle_id_int = int("".join(filter(str.isdigit, self.vehicle_id)) or "0")

        # Build Flower client (Member B's real implementation or stub)
        fl_client = IDSFlowerClient(
            model=self._model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            vehicle_id=vehicle_id_int,
        )

        server_address = f"{FL_SERVER_HOST}:{FL_SERVER_PORT}"

        # Run FL in a thread so it doesn't block the asyncio event loop
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(
                None,
                start_fl_client,
                server_address,
                fl_client,
            )
            logger.info(f"[{self.vehicle_id}] FL round completed")
        except Exception as e:
            logger.error(f"[{self.vehicle_id}] FL round failed: {e}")

    # ── Internal helpers ──────────────────────────────────────────────────────
    def _buffer_benign_sample(self, bsm: BSMMessage):
        """
        Buffer a benign BSM as a training sample (sequence not yet complete).
        For simplicity, we buffer raw feature vectors and build sequences at
        training time. The IDS engine already handles windowing for inference.
        """
        fv = bsm.to_feature_vector()
        self._benign_buffer_X.append(fv)
        self._benign_buffer_y.append(BENIGN_CLASS_IDX)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
async def main():
    vehicle = Vehicle()
    await vehicle.start()


if __name__ == "__main__":
    asyncio.run(main())
