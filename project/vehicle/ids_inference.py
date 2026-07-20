"""
ids_inference.py — IDS Model Inference Engine
==============================================
Loads the pretrained CNN+GRU+MultiHeadAttention model ONCE at startup.
Maintains per-sender sliding windows (deque, maxlen=WINDOW_SIZE=5).
Windows are sorted by the message's rcvTime field (not local arrival time).
Short sequences are padded to WINDOW_SIZE before inference.

Returns a named tuple: InferenceResult(is_malicious, class_idx, class_name, confidence)

Owner: Member A
"""

from __future__ import annotations
import logging
import os
from collections import deque, namedtuple
from pathlib import Path
from typing import Optional

import joblib
import numpy as np

# Delay TF import to avoid slow startup at module load time — imported lazily
_tf_model = None
_scaler   = None
_encoder  = None

from common.message_schema import (
    BSMMessage,
    FEATURE_COLS,
    WINDOW_SIZE,
    N_CLASSES,
    CLASS_NAMES,
    BENIGN_CLASS_IDX,
)

logger = logging.getLogger(__name__)

InferenceResult = namedtuple(
    "InferenceResult",
    ["is_malicious", "class_idx", "class_name", "confidence"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Model loader (singleton, thread-safe via module-level globals)
# ─────────────────────────────────────────────────────────────────────────────
def _load_artifacts(model_dir: str):
    """Load model, scaler, and encoder once. Called from IDSInferenceEngine.__init__."""
    global _tf_model, _scaler, _encoder
    if _tf_model is not None:
        return  # Already loaded

    import tensorflow as tf  # lazy import
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # suppress TF noise

    model_path   = Path(model_dir) / "best_ids_model.keras"
    scaler_path  = Path(model_dir) / "feature_scaler.pkl"
    encoder_path = Path(model_dir) / "label_encoder.pkl"

    logger.info(f"[IDS] Loading model from {model_path}")
    _tf_model = tf.keras.models.load_model(str(model_path))

    logger.info(f"[IDS] Loading scaler from {scaler_path}")
    _scaler = joblib.load(str(scaler_path))

    logger.info(f"[IDS] Loading encoder from {encoder_path}")
    _encoder = joblib.load(str(encoder_path))

    logger.info(f"[IDS] Artifacts loaded. Model expects input shape: {_tf_model.input_shape}")


# ─────────────────────────────────────────────────────────────────────────────
# Inference Engine
# ─────────────────────────────────────────────────────────────────────────────
class IDSInferenceEngine:
    """
    Maintains per-sender sliding windows and runs IDS inference.

    Usage:
        engine = IDSInferenceEngine(model_dir="/app/model")
        result = engine.ingest(bsm_message)
        if result and result.is_malicious:
            ...
    """

    def __init__(
        self,
        model_dir: Optional[str] = None,
        window_size: int = WINDOW_SIZE,
        confidence_threshold: float = 0.5,
    ):
        if model_dir is None:
            # Default: /app/model inside the container
            model_dir = os.environ.get("MODEL_DIR", "/app/model")

        _load_artifacts(model_dir)

        self.window_size           = window_size
        self.confidence_threshold  = confidence_threshold

        # Per-sender sliding windows: {senderID: deque of feature vectors}
        self._windows: dict[str, deque] = {}

    # ── Public API ────────────────────────────────────────────────────────────
    def ingest(self, msg: BSMMessage) -> Optional[InferenceResult]:
        """
        Add a message to the sender's window. If the window is full (or padded),
        run inference and return an InferenceResult. Otherwise returns None.

        Note: windows are kept sorted by rcvTime to handle out-of-order delivery.
        """
        sender = msg.senderID
        if sender not in self._windows:
            self._windows[sender] = deque(maxlen=self.window_size)

        # Insert sorted by rcvTime (V2V messages may arrive out of order)
        window = self._windows[sender]
        window.append((msg.rcvTime, msg.to_feature_vector()))
        self._windows[sender] = deque(
            sorted(window, key=lambda x: x[0]),
            maxlen=self.window_size,
        )

        # Run inference on every new message (regardless of window fullness — pad if needed)
        return self._run_inference(sender)

    def reset_sender(self, sender_id: str):
        """Clear a sender's window (call when blacklisting)."""
        self._windows.pop(sender_id, None)

    # ── Internal ──────────────────────────────────────────────────────────────
    def _run_inference(self, sender: str) -> InferenceResult:
        """Build padded sequence and run model inference."""
        window      = list(self._windows[sender])   # list of (rcvTime, feature_vec)
        feature_seq = [fv for _, fv in window]      # strip timestamps

        # Pad short sequences with zeros (same strategy as Phase 1 preprocessing)
        if len(feature_seq) < self.window_size:
            pad_count  = self.window_size - len(feature_seq)
            pad_vector = [0.0] * len(FEATURE_COLS)
            feature_seq = [pad_vector] * pad_count + feature_seq

        # Scale: (window_size, n_features) → apply scaler row-wise
        seq_array = np.array(feature_seq, dtype=np.float32)       # (5, 17)
        seq_scaled = _scaler.transform(seq_array)                  # (5, 17)
        seq_input  = seq_scaled[np.newaxis, :, :]                  # (1, 5, 17)

        # Predict
        probs      = _tf_model.predict(seq_input, verbose=0)[0]   # (N_CLASSES,)
        class_idx  = int(np.argmax(probs))
        confidence = float(probs[class_idx])

        # Decode class name
        try:
            class_name = _encoder.inverse_transform([class_idx])[0]
        except Exception:
            class_name = CLASS_NAMES[class_idx] if class_idx < len(CLASS_NAMES) else f"class_{class_idx}"

        is_malicious = (class_idx != BENIGN_CLASS_IDX) and (confidence >= self.confidence_threshold)

        logger.debug(
            f"[IDS] sender={sender} class={class_name} "
            f"confidence={confidence:.3f} malicious={is_malicious}"
        )

        return InferenceResult(
            is_malicious=is_malicious,
            class_idx=class_idx,
            class_name=class_name,
            confidence=confidence,
        )
