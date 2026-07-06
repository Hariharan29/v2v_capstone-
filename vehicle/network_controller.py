"""
network_controller.py — Local Per-Vehicle Network Controller
============================================================
This module runs INSIDE each vehicle container (not a separate service).
Implements the decentralised V2V trust model: no single point of failure.

Current responsibilities (Phase 2):
  1. On malicious detection: log event, blacklist sender, publish mode:alert to Redis
  2. Start/reset timeout timer — after TIMEOUT_SECONDS with no new detections,
     publish mode:normal and log reversion
  3. Maintain current crypto mode string (passed to crypto_switch)

Stage 3 Hook:
  _trigger_crypto_switch(mode) is defined and called now but routes to
  crypto_switch stubs. Stage 3 fills the stub — nothing here changes.

Owner: Member A
"""

from __future__ import annotations
import asyncio
import logging
import time
from typing import Optional, Set

import redis.asyncio as aioredis

from common.message_schema import (
    ALERT_CHANNEL,
    NORMAL_CHANNEL,
    MODE_NORMAL,
    MODE_SECURE,
    ModeAlertEvent,
    make_alert_event,
    make_normal_event,
)
from crypto_switch import sign_and_encrypt   # Stage 3 hook — already wired in

logger = logging.getLogger(__name__)

# Seconds of no malicious traffic before reverting to normal mode
TIMEOUT_SECONDS: float = float(30)


class LocalNetworkController:
    """
    Per-vehicle network controller.

    Parameters
    ----------
    vehicle_id : str       — this vehicle's ID (e.g. "vehicle_03")
    redis_conn : aioredis.Redis  — shared async Redis connection
    timeout_s  : float     — revert-to-normal timeout in seconds (default 30s)
    """

    def __init__(
        self,
        vehicle_id: str,
        redis_conn: aioredis.Redis,
        timeout_s: float = TIMEOUT_SECONDS,
    ):
        self.vehicle_id  = vehicle_id
        self._redis      = redis_conn
        self._timeout_s  = timeout_s

        # Current crypto mode — passed to crypto_switch on every message
        self._current_mode: str = MODE_NORMAL

        # Blacklisted sender IDs (never process their messages again this session)
        self._blacklist: Set[str] = set()

        # Asyncio handle for the revert-to-normal timer
        self._revert_task: Optional[asyncio.Task] = None

        # Timestamp of last malicious detection (for metrics/logging)
        self._last_alert_time: Optional[float] = None

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def current_mode(self) -> str:
        return self._current_mode

    def is_blacklisted(self, sender_id: str) -> bool:
        return sender_id in self._blacklist

    async def on_malicious_detection(
        self,
        sender_id:      str,
        detected_class: int,
        confidence:     float,
    ):
        """
        Called by vehicle.py reception_loop when IDS flags a message as malicious.

        Actions taken:
        1. Blacklist the sender
        2. Switch to secure mode locally
        3. Broadcast mode:alert to all peers via Redis
        4. Reset/restart the revert-to-normal timer
        """
        # 1. Blacklist
        self._blacklist.add(sender_id)
        logger.warning(
            f"[Controller:{self.vehicle_id}] Malicious detection from '{sender_id}' "
            f"class={detected_class} conf={confidence:.3f} — blacklisted"
        )

        # 2. Switch mode locally
        await self._trigger_crypto_switch(MODE_SECURE)

        # 3. Broadcast alert
        event = make_alert_event(
            triggered_by=self.vehicle_id,
            detected_class=detected_class,
            confidence=confidence,
        )
        await self._publish_alert(event)
        self._last_alert_time = time.time()

        # 4. Reset revert timer
        self._reset_revert_timer()

    async def on_peer_alert(self, event: ModeAlertEvent):
        """
        Called when this vehicle receives a mode:alert from a peer vehicle.
        Switches own crypto mode to secure and resets its own revert timer.
        """
        logger.info(
            f"[Controller:{self.vehicle_id}] Received peer alert from "
            f"'{event.triggered_by}' — switching to secure mode"
        )
        await self._trigger_crypto_switch(MODE_SECURE)
        self._reset_revert_timer()

    def get_blacklist(self) -> frozenset:
        return frozenset(self._blacklist)

    # ── Internal ──────────────────────────────────────────────────────────────

    async def _trigger_crypto_switch(self, mode: str):
        """
        Switches the controller's active crypto mode.

        Stage 3 hook: this is where the real Falcon/Kyber handshake will be
        initiated. Right now it just updates the mode string.
        """
        if self._current_mode == mode:
            return  # already in this mode

        prev_mode = self._current_mode
        self._current_mode = mode
        logger.info(
            f"[Controller:{self.vehicle_id}] Crypto mode: {prev_mode} → {mode}"
            + (" [STUB — Stage 3 will activate real PQC here]" if mode == MODE_SECURE else "")
        )
        # Stage 3: insert liboqs Falcon/Kyber initialisation here

    async def _publish_alert(self, event: ModeAlertEvent):
        """Publish a mode-change event to the Redis alert channel."""
        try:
            payload = sign_and_encrypt(
                {"event_json": event.to_json()},
                mode=self._current_mode,
            )
            import json
            await self._redis.publish(ALERT_CHANNEL, json.dumps(payload))
            logger.debug(f"[Controller:{self.vehicle_id}] Published to {ALERT_CHANNEL}")
        except Exception as e:
            logger.error(f"[Controller:{self.vehicle_id}] Failed to publish alert: {e}")

    async def _publish_normal(self):
        """Publish a revert-to-normal event after timeout."""
        try:
            event = make_normal_event(triggered_by=self.vehicle_id)
            import json
            await self._redis.publish(NORMAL_CHANNEL, json.dumps({"event_json": event.to_json()}))
            logger.info(f"[Controller:{self.vehicle_id}] Published mode:normal (timeout revert)")
        except Exception as e:
            logger.error(f"[Controller:{self.vehicle_id}] Failed to publish normal event: {e}")

    def _reset_revert_timer(self):
        """Cancel any existing revert timer and start a fresh one."""
        if self._revert_task and not self._revert_task.done():
            self._revert_task.cancel()
        self._revert_task = asyncio.create_task(self._revert_after_timeout())

    async def _revert_after_timeout(self):
        """Wait TIMEOUT_SECONDS, then revert to normal mode if no further detections."""
        try:
            await asyncio.sleep(self._timeout_s)
            logger.info(
                f"[Controller:{self.vehicle_id}] No detections for {self._timeout_s}s "
                f"— reverting to normal mode"
            )
            await self._trigger_crypto_switch(MODE_NORMAL)
            await self._publish_normal()
        except asyncio.CancelledError:
            pass  # Timer was reset by a new detection — expected behaviour
