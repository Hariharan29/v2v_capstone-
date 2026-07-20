"""
message_schema.py — Frozen shared contracts for V2V IDS system.

IMPORTANT: This file is a team-wide contract. Do NOT change field names
or types without agreement from both Member A and Member B, as both
sides of the codebase depend on this schema.

BSM Schema (Vehicle → Vehicle via Redis bsm:broadcast):
    senderID      : str   — unique vehicle identifier (e.g. "vehicle_03")
    receiverID    : str   — "broadcast" for V2V pub/sub
    rcvTime       : float — POSIX timestamp of message (from sender clock)
    pos_0         : float — x-position (metres)
    pos_1         : float — y-position (metres)
    pos_noise_0   : float — position measurement noise x
    pos_noise_1   : float — position measurement noise y
    spd_0         : float — speed x-component (m/s)
    spd_1         : float — speed y-component (m/s)
    spd_noise_0   : float — speed noise x
    spd_noise_1   : float — speed noise y
    acl_0         : float — acceleration x (m/s²)
    acl_1         : float — acceleration y (m/s²)
    acl_noise_0   : float — acceleration noise x
    acl_noise_1   : float — acceleration noise y
    hed_0         : float — heading x (unit vector component)
    hed_1         : float — heading y (unit vector component)
    hed_noise_0   : float — heading noise x
    hed_noise_1   : float — heading noise y
    label         : int   — ground truth class (0=Benign, 1-19=attack); -1 if unknown

FL Weight Update Schema (Vehicle → FL Server via Flower):
    vehicle_id    : int   — numeric vehicle ID
    weights       : list  — list of numpy arrays (model layer weights)
    num_samples   : int   — number of samples used for this local training step

Redis Channel Names (frozen):
    BSM_CHANNEL   = "bsm:broadcast"         — vehicle↔vehicle BSM messages
    ALERT_CHANNEL = "mode:alert"             — crypto-mode change trigger (Stage 3)
    NORMAL_CHANNEL= "mode:normal"            — revert-to-normal trigger (Stage 3)
"""

from __future__ import annotations
import time
from dataclasses import dataclass, asdict, field
from typing import Optional
import json

# ── Redis channel names ────────────────────────────────────────────────────────
BSM_CHANNEL    = "bsm:broadcast"   # vehicle ↔ vehicle BSM broadcast
ALERT_CHANNEL  = "mode:alert"      # on attack detection → trigger PQC switch (Stage 3)
NORMAL_CHANNEL = "mode:normal"     # after timeout → revert to normal crypto (Stage 3)

# ── Crypto mode strings ────────────────────────────────────────────────────────
MODE_NORMAL = "normal"             # ECC + AES-256-GCM  (default)
MODE_SECURE = "secure"            # Falcon + Kyber + XChaCha20 (Stage 3)

# ── Feature column order — MUST match Phase 1 training order exactly ──────────
FEATURE_COLS = [
    "rcvTime",
    "pos_0",       "pos_1",
    "pos_noise_0", "pos_noise_1",
    "spd_0",       "spd_1",
    "spd_noise_0", "spd_noise_1",
    "acl_0",       "acl_1",
    "acl_noise_0", "acl_noise_1",
    "hed_0",       "hed_1",
    "hed_noise_0", "hed_noise_1",
]

N_FEATURES  = len(FEATURE_COLS)  # 17
WINDOW_SIZE = 5                   # sequence length (from Phase 1 config)
N_CLASSES   = 20

CLASS_NAMES = [
    "Benign", "DoSRandomSybil", "DoSDisruptiveSybil", "DoSRandom",
    "DoSDisruptive", "DoS", "DataReplaySybil", "DataReplay",
    "GridSybil", "RandomPosOffset", "RandomPos", "ConstPosOffset",
    "ConstPos", "RandomSpeed", "RandomSpeedOffset", "ConstSpeedOffset",
    "ConstSpeed", "EventualStop", "DelayedMessages", "Disruptive",
]
BENIGN_CLASS_IDX = 0  # index of "Benign" in CLASS_NAMES


# ── BSM dataclass ─────────────────────────────────────────────────────────────
@dataclass
class BSMMessage:
    """Basic Safety Message — frozen schema, do not add fields without team consent."""
    senderID:     str
    receiverID:   str
    rcvTime:      float
    pos_0:        float
    pos_1:        float
    pos_noise_0:  float
    pos_noise_1:  float
    spd_0:        float
    spd_1:        float
    spd_noise_0:  float
    spd_noise_1:  float
    acl_0:        float
    acl_1:        float
    acl_noise_0:  float
    acl_noise_1:  float
    hed_0:        float
    hed_1:        float
    hed_noise_0:  float
    hed_noise_1:  float
    label:        int = -1   # -1 = unknown (receivers don't know the true label)



    def to_feature_vector(self) -> list[float]:
        """Returns features in FEATURE_COLS order for IDS inference."""
        return [
            self.rcvTime,
            self.pos_0,       self.pos_1,
            self.pos_noise_0, self.pos_noise_1,
            self.spd_0,       self.spd_1,
            self.spd_noise_0, self.spd_noise_1,
            self.acl_0,       self.acl_1,
            self.acl_noise_0, self.acl_noise_1,
            self.hed_0,       self.hed_1,
            self.hed_noise_0, self.hed_noise_1,
        ]


# ── FL weight update ───────────────────────────────────────────────────────────
@dataclass
class FLWeightUpdate:
    """Schema for weight updates sent from vehicle Flower client to FL server."""
    vehicle_id:  int
    num_samples: int
    # Note: actual weight arrays are passed through Flower's native wire format,
    # not serialised through this dataclass. This class is used for metadata only.

    def to_json(self) -> str:
        return json.dumps(asdict(self))


# ── Mode alert event ───────────────────────────────────────────────────────────
@dataclass
class ModeAlertEvent:
    """
    Published to ALERT_CHANNEL when a vehicle detects a malicious message.
    Stage 3: network_controller.py reads this to trigger the crypto switch.
    """
    event:        str    # "mode:alert" | "mode:normal"
    triggered_by: str    # vehicle_id that detected the attack
    timestamp:    float  # POSIX timestamp
    detected_class: Optional[int] = None   # predicted class index (if alert)
    confidence:   Optional[float] = None   # model confidence

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @staticmethod
    def from_json(data: str) -> "ModeAlertEvent":
        return ModeAlertEvent(**json.loads(data))


def make_alert_event(triggered_by: str, detected_class: int, confidence: float) -> ModeAlertEvent:
    return ModeAlertEvent(
        event="mode:alert",
        triggered_by=triggered_by,
        timestamp=time.time(),
        detected_class=detected_class,
        confidence=confidence,
    )


def make_normal_event(triggered_by: str) -> ModeAlertEvent:
    return ModeAlertEvent(
        event="mode:normal",
        triggered_by=triggered_by,
        timestamp=time.time(),
    )
