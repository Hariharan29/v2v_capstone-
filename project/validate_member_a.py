"""
validate_member_a.py — Standalone validation script for Member A's code
========================================================================
Run this OUTSIDE Docker to verify all components work before building containers.
Tests: schema imports, generator output, IDS inference engine, crypto stubs, controller logic.

Usage:
    cd project
    python validate_member_a.py

Requirements: pip install numpy redis (no TF needed for most tests; IDS test is skipped if TF absent)
"""

import json
import sys
import traceback
from pathlib import Path

# Force UTF-8 output on Windows
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "vehicle"))
sys.path.insert(0, str(PROJECT_ROOT / "common"))

PASS = "[PASS]"
FAIL = "[FAIL]"
SKIP = "[SKIP]"

results = []

def check(name, fn):
    try:
        fn()
        print(f"  {PASS} {name}")
        results.append((name, True, None))
    except Exception as e:
        print(f"  {FAIL} {name}: {e}")
        results.append((name, False, str(e)))

print("\n" + "="*60)
print("  Member A — Validation Suite")
print("="*60)

# ── 1. Schema imports ──────────────────────────────────────────────────────
print("\n[1] Schema & Constants")

def test_schema_import():
    from common.message_schema import (
        BSMMessage, FEATURE_COLS, WINDOW_SIZE, N_CLASSES,
        BSM_CHANNEL, ALERT_CHANNEL, CLASS_NAMES, BENIGN_CLASS_IDX,
        MODE_NORMAL, MODE_SECURE,
    )
    assert len(FEATURE_COLS) == 17, f"Expected 17 features, got {len(FEATURE_COLS)}"
    assert WINDOW_SIZE == 5
    assert N_CLASSES == 20
    assert BSM_CHANNEL == "bsm:broadcast"
    assert BENIGN_CLASS_IDX == 0

def test_bsm_roundtrip():
    from common.message_schema import BSMMessage
    msg = BSMMessage(
        senderID="test_01", receiverID="broadcast", rcvTime=12345.0,
        pos_0=10.0, pos_1=20.0, pos_noise_0=0.01, pos_noise_1=0.01,
        spd_0=25.0, spd_1=25.0, spd_noise_0=0.001, spd_noise_1=0.001,
        acl_0=0.5, acl_1=0.1, acl_noise_0=0.001, acl_noise_1=0.001,
        hed_0=90.0, hed_1=90.0, hed_noise_0=0.001, hed_noise_1=0.001,
        label=0,
    )
    j = msg.to_json()
    msg2 = BSMMessage.from_json(j)
    assert msg2.senderID == "test_01"
    fv = msg.to_feature_vector()
    assert len(fv) == 17

check("Schema imports & constants", test_schema_import)
check("BSMMessage JSON round-trip", test_bsm_roundtrip)

# ── 2. Feature stats ───────────────────────────────────────────────────────
print("\n[2] Feature Stats JSON")

def test_feature_stats():
    stats_path = PROJECT_ROOT / "common" / "feature_stats.json"
    assert stats_path.exists(), f"feature_stats.json not found at {stats_path}"
    with open(stats_path) as f:
        stats = json.load(f)
    for feat in ["rcvTime", "pos_0", "spd_0", "hed_0"]:
        assert feat in stats, f"Missing feature: {feat}"
        assert "mean" in stats[feat] and "std" in stats[feat]
    # Sanity check real values
    assert stats["spd_0"]["max"] > 40, "spd_0 max should be ~50 m/s"
    assert stats["hed_0"]["max"] > 350, "hed_0 max should be ~360 degrees"

check("feature_stats.json exists and has real values", test_feature_stats)

# ── 3. Crypto switch stub ──────────────────────────────────────────────────
print("\n[3] Crypto Switch Stub")

def test_crypto_passthrough():
    sys.path.insert(0, str(PROJECT_ROOT / "vehicle"))
    from crypto_switch import sign_and_encrypt, verify_and_decrypt
    test_msg = {"senderID": "v01", "rcvTime": 1000.0, "pos_0": 10.0}
    encrypted = sign_and_encrypt(test_msg, mode="normal")
    assert "_crypto_mode" in encrypted
    assert encrypted["_crypto_mode"] == "normal"
    decrypted = verify_and_decrypt(encrypted, mode="normal")
    assert decrypted is not None
    assert "_crypto_mode" not in decrypted  # tag stripped
    assert decrypted["senderID"] == "v01"

def test_crypto_secure_mode():
    from crypto_switch import sign_and_encrypt, verify_and_decrypt
    test_msg = {"data": "hello"}
    enc = sign_and_encrypt(test_msg, mode="secure")
    assert enc["_crypto_mode"] == "secure"
    dec = verify_and_decrypt(enc, mode="secure")
    assert dec["data"] == "hello"

check("sign_and_encrypt passthrough (normal)", test_crypto_passthrough)
check("sign_and_encrypt passthrough (secure)", test_crypto_secure_mode)

# ── 4. Generator ──────────────────────────────────────────────────────────
print("\n[4] BSM Generator")

def test_benign_generator():
    from generator import BenignGenerator
    gen = BenignGenerator("vehicle_01", stats_path=str(PROJECT_ROOT / "common" / "feature_stats.json"))
    msgs = [gen.generate() for _ in range(10)]
    assert len(msgs) == 10
    for m in msgs:
        assert m.senderID == "vehicle_01"
        assert m.label == 0
        fv = m.to_feature_vector()
        assert len(fv) == 17
        # Check headings are in degree range
        assert 0 <= m.hed_0 <= 360, f"hed_0 out of range: {m.hed_0}"
        assert 0 <= m.spd_0 <= 200, f"spd_0 out of range: {m.spd_0}"

def test_attacker_generator():
    from generator import AttackerGenerator
    for mode in ["dos", "speed_offset", "pos_offset", "sybil"]:
        gen = AttackerGenerator("attacker_01", attack_mode=mode,
                                stats_path=str(PROJECT_ROOT / "common" / "feature_stats.json"))
        msg = gen.generate()
        assert msg.label != 0, f"Attacker {mode} should not produce label=0"
        assert len(msg.to_feature_vector()) == 17

check("BenignGenerator: 10 messages, valid schema", test_benign_generator)
check("AttackerGenerator: all 4 modes", test_attacker_generator)

# ── 5. IDS Inference (skipped if TF not available) ────────────────────────
print("\n[5] IDS Inference Engine")

def test_ids_import():
    from ids_inference import IDSInferenceEngine, InferenceResult
    assert IDSInferenceEngine is not None

def test_ids_with_model():
    model_dir = PROJECT_ROOT.parent / "model"
    if not (model_dir / "best_ids_model.keras").exists():
        raise RuntimeError(f"Model not found at {model_dir} — copy Phase 1 artifacts first")
    from ids_inference import IDSInferenceEngine
    engine = IDSInferenceEngine(model_dir=str(model_dir), window_size=5)
    from generator import BenignGenerator
    gen = BenignGenerator("vehicle_test", stats_path=str(PROJECT_ROOT / "common" / "feature_stats.json"))
    from common.message_schema import BSMMessage
    result = None
    for _ in range(5):
        msg = gen.generate()
        result = engine.ingest(msg)
    assert result is not None
    assert hasattr(result, "is_malicious")
    assert hasattr(result, "class_name")
    assert hasattr(result, "confidence")
    assert 0.0 <= result.confidence <= 1.0
    print(f"       IDS result: class={result.class_name} conf={result.confidence:.3f} malicious={result.is_malicious}")

check("IDS module imports", test_ids_import)

try:
    import tensorflow  # noqa
    check("IDS inference with real model (requires TF + model)", test_ids_with_model)
except ImportError:
    print(f"  {SKIP} IDS inference skipped — TensorFlow not installed in this environment")
    results.append(("IDS inference (TF)", None, "skipped"))

# ── 6. Alert / Mode events ────────────────────────────────────────────────
print("\n[6] Mode Alert Events")

def test_alert_events():
    from common.message_schema import make_alert_event, make_normal_event, ModeAlertEvent
    alert = make_alert_event("vehicle_01", detected_class=3, confidence=0.87)
    j = alert.to_json()
    alert2 = ModeAlertEvent.from_json(j)
    assert alert2.triggered_by == "vehicle_01"
    assert alert2.event == "mode:alert"
    assert alert2.detected_class == 3

    normal = make_normal_event("vehicle_01")
    assert normal.event == "mode:normal"

check("Alert and normal event round-trip", test_alert_events)

# ── Summary ───────────────────────────────────────────────────────────────
print("\n" + "="*60)
passed = sum(1 for _, ok, _ in results if ok is True)
failed = sum(1 for _, ok, _ in results if ok is False)
skipped = sum(1 for _, ok, _ in results if ok is None)
print(f"  Results: {passed} passed | {failed} failed | {skipped} skipped")
print("="*60 + "\n")

if failed > 0:
    print("Failed tests:")
    for name, ok, err in results:
        if ok is False:
            print(f"  - {name}: {err}")
    sys.exit(1)
