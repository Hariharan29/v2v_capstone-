# V2V IDS + Federated Learning — Phase 2

## Project Structure

```
project/
├── Dockerfile                    ← Single image for benign + attacker vehicles
├── docker-compose.yml            ← 10 benign + 1 attacker + Redis; FL server stub
├── validate_member_a.py          ← Run this to verify Member A's code before Docker
│
├── common/                       ← Shared contracts (frozen — both members import from here)
│   ├── message_schema.py         ← BSMMessage, ModeAlertEvent, channel names, feature list
│   └── feature_stats.json        ← Real VeReMi benign-class statistics (113,477 rows)
│
├── vehicle/                      ← Member A's code
│   ├── vehicle.py                ← Core: 3 async loops (generate / receive / train)
│   ├── generator.py              ← BenignGenerator + AttackerGenerator (4 attack modes)
│   ├── ids_inference.py          ← IDSInferenceEngine: per-sender windows, padding, inference
│   ├── network_controller.py     ← LocalNetworkController: blacklist, mode alert, revert timer
│   ├── crypto_switch.py          ← [STUB] sign_and_encrypt / verify_and_decrypt (Stage 3 hook)
│   ├── client_stub.py            ← [STUB] IDSFlowerClient interface for Member B
│   ├── entrypoint.sh             ← Derives VEHICLE_ID from hostname at startup
│   └── requirements.txt
│
├── model/                        ← Phase 1 artifacts (copy here before docker build)
│   ├── best_ids_model.keras
│   ├── feature_scaler.pkl
│   └── label_encoder.pkl
│
└── fl_server/                    ← [Member B] — not yet created
    ├── Dockerfile
    ├── server.py
    ├── krum.py
    ├── fedprox.py
    └── client.py                 ← Drop here; vehicle.py auto-imports it over client_stub.py
```

---

## Quick Start

### Step 0 — Validate (no Docker needed)
```bash
cd project
pip install numpy redis
python validate_member_a.py
# Expected: all [PASS] except IDS (skipped if TF not installed locally)
```

### Step 1 — Copy Phase 1 model artifacts
```bash
# From Phase-2/model/ into project/model/
cp ../model/best_ids_model.keras  model/
cp ../model/feature_scaler.pkl    model/
cp ../model/label_encoder.pkl     model/
```

### Step 2 — Week 1: Single vehicle standalone test
```bash
docker compose up redis vehicle --scale vehicle=1 --build
docker logs <vehicle_container_id> -f
# Expected: "BSM published", "IDS inference", "Reception loop subscribed"
```

### Step 3 — Inject a fake attacker message (manual test)
```bash
# In another terminal:
docker exec -it v2v_redis redis-cli
> SUBSCRIBE mode:alert      # watch for alerts in one terminal
> PUBLISH bsm:broadcast '{"senderID":"evil_99","receiverID":"broadcast","rcvTime":9999.0,"pos_0":9000.0,"pos_1":9000.0,"pos_noise_0":0.0,"pos_noise_1":0.0,"spd_0":999.0,"spd_1":999.0,"spd_noise_0":0.0,"spd_noise_1":0.0,"acl_0":0.0,"acl_1":0.0,"acl_noise_0":0.0,"acl_noise_1":0.0,"hed_0":90.0,"hed_1":90.0,"hed_noise_0":0.0,"hed_noise_1":0.0,"label":5,"_crypto_mode":"normal"}'
# Expected vehicle logs: "MALICIOUS: sender=evil_99", "mode:alert published"
```

### Step 4 — Week 2: Full 10+1 deployment
```bash
docker compose up --scale vehicle=10 --build
# Verify all 10 vehicles receive each other's BSMs
# Verify attacker messages get flagged within first few windows
```

### Step 5 — Integration with Member B's FL server
```bash
# 1. Member B drops fl_server/ directory + client.py into vehicle/
# 2. Uncomment fl_server service block in docker-compose.yml
# 3. docker compose up
# 4. Confirm 10 FL rounds complete with Krum aggregation
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `VEHICLE_ID` | *(from hostname)* | Unique vehicle identifier |
| `ROLE` | `benign` | `benign` or `attacker` |
| `REDIS_HOST` | `redis` | Redis service hostname |
| `FL_SERVER_HOST` | `fl_server` | FL server hostname (Member B) |
| `MODEL_DIR` | `/app/model` | Path to Phase 1 model artifacts |
| `TRAIN_SAMPLE_THRESHOLD` | `500` | Benign samples before FL round triggers |
| `TRAIN_TIME_INTERVAL_S` | `60` | Max seconds between FL rounds |
| `GENERATION_INTERVAL_S` | `0.5` | Seconds between benign BSM publishes |
| `ATTACKER_INTERVAL_S` | `0.1` | Seconds between attacker BSM publishes |

---

## Stage 3 (PQC) — What Member A Leaves Ready

All of Stage 3 is **fill-in-the-blanks**, not a rewrite:

| File | What's Already There | Stage 3 Fills |
|---|---|---|
| `crypto_switch.py` | Function signatures, docstrings, mode routing | Falcon+Kyber+XChaCha20 implementation |
| `network_controller.py` | `_trigger_crypto_switch(mode)` called on detection | Real liboqs handshake inside that method |
| `vehicle.py` | Every publish/receive goes through `sign_and_encrypt`/`verify_and_decrypt` | Nothing — already wired |
| `requirements.txt` | liboqs, cryptography, PyNaCl listed but commented | Uncomment |

---

## Redis Channels (Frozen)

| Channel | Direction | Purpose |
|---|---|---|
| `bsm:broadcast` | Vehicle ↔ Vehicle | BSM message broadcast |
| `mode:alert` | Vehicle → All | Attack detected, switch to PQC |
| `mode:normal` | Vehicle → All | Revert to normal after timeout |

---

## Member B Integration Checklist

- [ ] Implement `fl_server/server.py` with Flower server (10 rounds, min 10 clients)
- [ ] Implement `fl_server/krum.py` (single-Krum, f=1)
- [ ] Implement `fl_server/fedprox.py` (server sends `proximal_mu` in fit config)
- [ ] Drop `client.py` into `vehicle/` — vehicle.py auto-imports it over the stub
- [ ] Uncomment `fl_server` block in `docker-compose.yml`
- [ ] Server sends `proximal_mu` in Flower fit config dict (Member A reads it in `training_loop`)
