"""
crypto_switch.py — PQC Crypto-Agility Stub (Stage 3 Hook)
==========================================================
CURRENT STATE: Pass-through stubs only. Messages are returned unchanged.
STAGE 3 TASK:  Fill in sign_and_encrypt() and verify_and_decrypt() with:
               - MODE_NORMAL: ECC (signing) + AES-256-GCM (encryption)
               - MODE_SECURE: Falcon (signing) + Kyber (key exchange) + XChaCha20-Poly1305

Every message published or received ALREADY routes through these two functions.
Stage 3 is purely: implement these two functions. Nothing else in the codebase changes.

Owner: Member A (stub) → Stage 3 team fills implementation
"""

from __future__ import annotations
import logging
from typing import Any

from common.message_schema import MODE_NORMAL, MODE_SECURE

logger = logging.getLogger(__name__)


# ── Public API (frozen — do not rename these functions) ───────────────────────

def sign_and_encrypt(message: dict, mode: str = MODE_NORMAL) -> dict:
    """
    Sign and encrypt an outgoing BSM message before publishing to Redis.

    Parameters
    ----------
    message : dict  — the raw BSM message dict (already serialised-ready)
    mode    : str   — MODE_NORMAL or MODE_SECURE

    Returns
    -------
    dict — the (possibly wrapped/encrypted) payload to publish to Redis.
           In stub mode, returns the message unchanged with a mode tag added.

    Stage 3 Implementation Notes:
    ------------------------------
    MODE_NORMAL:
        - Sign with ECDSA (P-256)
        - Encrypt payload with AES-256-GCM using shared session key
        - Return: {"payload": <encrypted_bytes_b64>, "sig": <sig_b64>, "mode": "normal"}

    MODE_SECURE:
        - Sign with Falcon-512 (via liboqs)
        - Key exchange via Kyber-768 (via liboqs)
        - Encrypt payload with XChaCha20-Poly1305
        - Return: {"payload": <encrypted_bytes_b64>, "sig": <sig_b64>,
                   "kem_ct": <kyber_ciphertext_b64>, "mode": "secure"}
    """
    # ── STUB: pass through unchanged ─────────────────────────────────────────
    if mode not in (MODE_NORMAL, MODE_SECURE):
        logger.warning(f"[crypto_switch] Unknown mode '{mode}', defaulting to normal")
        mode = MODE_NORMAL

    # Tag the message with its crypto mode so receivers know how to process it
    result = dict(message)
    result["_crypto_mode"] = mode
    # Stage 3: replace the body above with real crypto operations
    return result


def verify_and_decrypt(payload: dict, mode: str = MODE_NORMAL) -> dict | None:
    """
    Verify and decrypt an incoming BSM payload received from Redis.

    Parameters
    ----------
    payload : dict — the raw payload received from Redis
    mode    : str  — expected crypto mode (read from payload["_crypto_mode"] if present)

    Returns
    -------
    dict  — the decrypted + verified BSM message dict, or
    None  — if verification fails (caller should discard and blacklist sender)

    Stage 3 Implementation Notes:
    ------------------------------
    MODE_NORMAL:
        - Verify ECDSA signature; return None if invalid
        - Decrypt AES-256-GCM payload

    MODE_SECURE:
        - Verify Falcon-512 signature; return None if invalid
        - Decapsulate Kyber-768 ciphertext to recover session key
        - Decrypt XChaCha20-Poly1305 payload
    """
    # ── STUB: pass through unchanged ─────────────────────────────────────────
    # Infer mode from tagged payload if available
    effective_mode = payload.get("_crypto_mode", mode)

    if effective_mode not in (MODE_NORMAL, MODE_SECURE):
        logger.warning(f"[crypto_switch] Unknown mode '{effective_mode}' in payload, treating as normal")
        effective_mode = MODE_NORMAL

    # Stage 3: replace the body below with real crypto verification
    result = dict(payload)
    result.pop("_crypto_mode", None)   # strip internal tag before returning to caller
    return result


# ── Internal helpers (Stage 3 will implement these) ──────────────────────────

def _generate_keypair(mode: str) -> tuple[bytes, bytes]:
    """
    Generate (public_key, private_key) for the given mode.
    Stage 3: call liboqs.KeyEncapsulation for Kyber, liboqs.Signature for Falcon.
    """
    raise NotImplementedError("Stage 3: implement keypair generation")


def _get_current_mode() -> str:
    """
    Returns the currently active crypto mode for this vehicle.
    Stage 3: network_controller.py will manage this state and expose it here.
    Currently returns MODE_NORMAL always.
    """
    return MODE_NORMAL
