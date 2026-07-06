"""
generator.py — BSM Message Generator for Benign and Attacker Vehicles
======================================================================
Benign generator: samples realistic V2V messages from VeReMi-derived
                  statistical distributions (feature_stats.json).
Attacker generator: injects 4 attack types matching the agreed scope:
                    DoS, SpeedOffset, PositionOffset, Sybil.

All generated messages conform to the frozen BSMMessage schema from
common/message_schema.py. No other field names are used.

Owner: Member A
"""

from __future__ import annotations
import json
import logging
import math
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np

from common.message_schema import BSMMessage, FEATURE_COLS, CLASS_NAMES, BENIGN_CLASS_IDX

logger = logging.getLogger(__name__)

# ── Attack class indices (from CLASS_NAMES in message_schema.py) ──────────────
_ATTACK_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}

DOS_CLASS_IDX          = _ATTACK_IDX["DoS"]
SPEED_OFFSET_CLASS_IDX = _ATTACK_IDX["ConstSpeedOffset"]
POS_OFFSET_CLASS_IDX   = _ATTACK_IDX["ConstPosOffset"]
SYBIL_CLASS_IDX        = _ATTACK_IDX["DoSRandomSybil"]


# ─────────────────────────────────────────────────────────────────────────────
# Benign Generator
# ─────────────────────────────────────────────────────────────────────────────
class BenignGenerator:
    """
    Generates realistic BSM messages by sampling from VeReMi benign-class
    statistical distributions stored in feature_stats.json.

    Uses truncated Gaussian sampling so values stay within min/max bounds.
    Position evolves realistically over time (simple kinematic model).
    """

    def __init__(self, vehicle_id: str, stats_path: Optional[str] = None):
        self.vehicle_id = vehicle_id
        self._stats = self._load_stats(stats_path)
        self._rng   = np.random.default_rng(seed=abs(hash(vehicle_id)) % (2**31))

        # Track evolving state for realistic continuity
        self._pos   = np.array([
            self._sample("pos_0"),
            self._sample("pos_1"),
        ])
        # In VeReMi: spd_0 and spd_1 are both magnitude values (0-50 m/s)
        self._spd   = np.array([
            abs(self._sample("spd_0")),
            abs(self._sample("spd_1")),
        ])
        # hed_0/hed_1 are heading in DEGREES (0-360)
        self._hed_0 = self._rng.uniform(0, 360)
        self._hed_1 = self._rng.uniform(0, 360)
        self._rcv_time = time.time()

    # ── Public ────────────────────────────────────────────────────────────────
    def generate(self) -> BSMMessage:
        """Generate one benign BSM, advancing the vehicle's kinematic state."""
        dt = random.uniform(0.1, 0.5)   # 100–500ms between messages (V2V typical)
        self._advance_state(dt)
        self._rcv_time += dt

        acl = np.array([self._sample("acl_0"), self._sample("acl_1")])

        return BSMMessage(
            senderID     = self.vehicle_id,
            receiverID   = "broadcast",
            rcvTime      = round(self._rcv_time, 3),
            pos_0        = round(float(self._pos[0]), 4),
            pos_1        = round(float(self._pos[1]), 4),
            pos_noise_0  = round(self._sample("pos_noise_0"), 4),
            pos_noise_1  = round(self._sample("pos_noise_1"), 4),
            spd_0        = round(float(self._spd[0]), 4),
            spd_1        = round(float(self._spd[1]), 4),
            spd_noise_0  = round(self._sample("spd_noise_0"), 4),
            spd_noise_1  = round(self._sample("spd_noise_1"), 4),
            acl_0        = round(float(acl[0]), 4),
            acl_1        = round(float(acl[1]), 4),
            acl_noise_0  = round(self._sample("acl_noise_0"), 4),
            acl_noise_1  = round(self._sample("acl_noise_1"), 4),
            # Headings in degrees (0-360) matching VeReMi benign data
            hed_0        = round(self._hed_0, 4),
            hed_1        = round(self._hed_1, 4),
            hed_noise_0  = round(self._sample("hed_noise_0"), 4),
            hed_noise_1  = round(self._sample("hed_noise_1"), 4),
            label        = BENIGN_CLASS_IDX,
        )

    # ── Internal ──────────────────────────────────────────────────────────────
    def _load_stats(self, stats_path: Optional[str]) -> dict:
        if stats_path is None:
            # Default: look relative to this file
            here = Path(__file__).parent.parent / "common" / "feature_stats.json"
            stats_path = str(here)
        with open(stats_path, "r") as f:
            raw = json.load(f)
        # Strip metadata keys
        return {k: v for k, v in raw.items() if not k.startswith("_")}

    def _sample(self, feature: str) -> float:
        """Truncated Gaussian sample clipped to [min, max]."""
        s = self._stats[feature]
        val = self._rng.normal(s["mean"], max(s["std"], 1e-6))
        return float(np.clip(val, s["min"], s["max"]))

    def _advance_state(self, dt: float):
        """Simple kinematic update: pos advances by speed*dt, speed random-walks."""
        # Position advances proportional to speed magnitude and dt
        self._pos += np.array([
            self._spd[0] * math.cos(math.radians(self._hed_0)),
            self._spd[1] * math.sin(math.radians(self._hed_1)),
        ]) * dt
        # Clip position to dataset bounds
        pos_max = self._stats["pos_0"]["max"]
        self._pos = np.clip(self._pos, -pos_max, pos_max)
        # Small random walk on speed
        self._spd += self._rng.normal(0, 0.5, size=2) * dt
        spd_max = self._stats["spd_0"]["max"]
        self._spd = np.clip(self._spd, 0.0, spd_max)   # speed is non-negative in VeReMi
        # Occasionally drift heading (turns, degrees)
        if self._rng.random() < 0.05:
            self._hed_0 = (self._hed_0 + self._rng.uniform(-30, 30)) % 360
            self._hed_1 = (self._hed_1 + self._rng.uniform(-30, 30)) % 360


# ─────────────────────────────────────────────────────────────────────────────
# Attacker Generator
# ─────────────────────────────────────────────────────────────────────────────
class AttackerGenerator:
    """
    Generates malicious BSM messages using 4 agreed attack modes:
      - "dos"           : flood channel with repeated identical messages (DoS)
      - "speed_offset"  : constant large speed offset
      - "pos_offset"    : constant large position offset
      - "sybil"         : rotate through multiple fake sender IDs

    Attack mode is selected randomly per burst, or can be fixed via constructor.
    """

    # Agreed attack scope (from build plan — not the full 7-attack VeReMi taxonomy)
    ATTACK_MODES = ["dos", "speed_offset", "pos_offset", "sybil"]

    # Sybil identities pool (fake IDs the attacker cycles through)
    SYBIL_POOL_SIZE = 5

    def __init__(
        self,
        vehicle_id: str,
        attack_mode: Optional[str] = None,
        stats_path: Optional[str] = None,
    ):
        self.vehicle_id  = vehicle_id
        self.attack_mode = attack_mode  # None = randomly select per message
        self._rng        = np.random.default_rng(seed=abs(hash(vehicle_id)) % (2**31))
        self._stats      = self._load_stats(stats_path)
        self._rcv_time   = time.time()

        # Sybil pool — fixed fake IDs this attacker cycles through
        self._sybil_ids = [f"sybil_{vehicle_id}_{i}" for i in range(self.SYBIL_POOL_SIZE)]
        self._sybil_idx = 0

        # Reference position for offset attacks
        self._base_pos = np.array([
            self._rng.uniform(
                self._stats["pos_0"]["min"], self._stats["pos_0"]["max"]
            ),
            self._rng.uniform(
                self._stats["pos_1"]["min"], self._stats["pos_1"]["max"]
            ),
        ])

    # ── Public ────────────────────────────────────────────────────────────────
    def generate(self) -> BSMMessage:
        """Generate one malicious BSM."""
        mode = self.attack_mode or self._rng.choice(self.ATTACK_MODES)
        self._rcv_time += random.uniform(0.01, 0.1)

        if mode == "dos":
            return self._generate_dos()
        elif mode == "speed_offset":
            return self._generate_speed_offset()
        elif mode == "pos_offset":
            return self._generate_pos_offset()
        elif mode == "sybil":
            return self._generate_sybil()
        else:
            raise ValueError(f"Unknown attack mode: {mode}")

    # ── Attack implementations ────────────────────────────────────────────────
    def _generate_dos(self) -> BSMMessage:
        """DoS: flood with high-rate identical messages (near-zero dt)."""
        # Use same base values each time — creates unnatural repetition
        return BSMMessage(
            senderID     = self.vehicle_id,
            receiverID   = "broadcast",
            rcvTime      = round(self._rcv_time, 3),
            pos_0        = round(float(self._base_pos[0]), 4),
            pos_1        = round(float(self._base_pos[1]), 4),
            pos_noise_0  = 0.0, pos_noise_1 = 0.0,
            spd_0        = 0.0, spd_1       = 0.0,
            spd_noise_0  = 0.0, spd_noise_1 = 0.0,
            acl_0        = 0.0, acl_1       = 0.0,
            acl_noise_0  = 0.0, acl_noise_1 = 0.0,
            hed_0        = 1.0, hed_1       = 0.0,
            hed_noise_0  = 0.0, hed_noise_1 = 0.0,
            label        = DOS_CLASS_IDX,
        )

    def _generate_speed_offset(self) -> BSMMessage:
        """ConstSpeedOffset: claims unrealistically high constant speed."""
        spd_max = self._stats["spd_0"]["max"]
        # Report speed at ~3× the realistic maximum (clearly anomalous)
        fake_spd = spd_max * 3.0
        return BSMMessage(
            senderID     = self.vehicle_id,
            receiverID   = "broadcast",
            rcvTime      = round(self._rcv_time, 3),
            pos_0        = round(float(self._base_pos[0]), 4),
            pos_1        = round(float(self._base_pos[1]), 4),
            pos_noise_0  = 0.0, pos_noise_1 = 0.0,
            spd_0        = round(fake_spd, 4),
            spd_1        = round(fake_spd, 4),   # both components unrealistically high
            spd_noise_0  = 0.0, spd_noise_1 = 0.0,
            acl_0        = 0.0, acl_1       = 0.0,
            acl_noise_0  = 0.0, acl_noise_1 = 0.0,
            hed_0        = 90.0,  hed_1  = 90.0,   # constant heading in degrees
            hed_noise_0  = 0.0, hed_noise_1 = 0.0,
            label        = SPEED_OFFSET_CLASS_IDX,
        )

    def _generate_pos_offset(self) -> BSMMessage:
        """ConstPosOffset: claims a fixed position far from reality."""
        pos_max = self._stats["pos_0"]["max"]
        # Report a constant extreme position
        fake_pos = np.array([pos_max * 0.9, pos_max * 0.9])
        return BSMMessage(
            senderID     = self.vehicle_id,
            receiverID   = "broadcast",
            rcvTime      = round(self._rcv_time, 3),
            pos_0        = round(float(fake_pos[0]), 4),
            pos_1        = round(float(fake_pos[1]), 4),
            pos_noise_0  = 0.0, pos_noise_1 = 0.0,
            spd_0        = 0.0, spd_1       = 0.0,
            spd_noise_0  = 0.0, spd_noise_1 = 0.0,
            acl_0        = 0.0, acl_1       = 0.0,
            acl_noise_0  = 0.0, acl_noise_1 = 0.0,
            hed_0        = 90.0, hed_1      = 90.0,   # constant heading in degrees
            hed_noise_0  = 0.0, hed_noise_1 = 0.0,
            label        = POS_OFFSET_CLASS_IDX,
        )

    def _generate_sybil(self) -> BSMMessage:
        """DoSRandomSybil: rotate through fake sender IDs."""
        sender = self._sybil_ids[self._sybil_idx % self.SYBIL_POOL_SIZE]
        self._sybil_idx += 1
        return BSMMessage(
            senderID     = sender,
            receiverID   = "broadcast",
            rcvTime      = round(self._rcv_time, 3),
            pos_0        = round(float(self._base_pos[0]) + float(self._rng.uniform(-5, 5)), 4),
            pos_1        = round(float(self._base_pos[1]) + float(self._rng.uniform(-5, 5)), 4),
            pos_noise_0  = round(float(self._rng.uniform(-2, 2)), 4),
            pos_noise_1  = round(float(self._rng.uniform(-2, 2)), 4),
            spd_0        = round(float(self._rng.uniform(0, 15)), 4),
            spd_1        = round(float(self._rng.uniform(-5, 5)), 4),
            spd_noise_0  = 0.0, spd_noise_1 = 0.0,
            acl_0        = 0.0, acl_1       = 0.0,
            acl_noise_0  = 0.0, acl_noise_1 = 0.0,
            hed_0        = round(float(self._rng.uniform(0, 360)), 4),   # degrees
            hed_1        = round(float(self._rng.uniform(0, 360)), 4),
            hed_noise_0  = 0.0, hed_noise_1 = 0.0,
            label        = SYBIL_CLASS_IDX,
        )

    # ── Internal ──────────────────────────────────────────────────────────────
    def _load_stats(self, stats_path: Optional[str]) -> dict:
        if stats_path is None:
            here = Path(__file__).parent.parent / "common" / "feature_stats.json"
            stats_path = str(here)
        with open(stats_path, "r") as f:
            raw = json.load(f)
        return {k: v for k, v in raw.items() if not k.startswith("_")}
