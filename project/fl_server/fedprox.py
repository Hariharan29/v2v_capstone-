"""
fedprox.py — FedProx Boundary Documentation & Server-Side Config
=================================================================
FedProx reference: Li et al., "Federated Optimization in Heterogeneous
Networks", MLSys 2020. (https://arxiv.org/abs/1812.06127)

BOUNDARY DEFINITION — READ BEFORE TOUCHING EITHER SIDE:
=========================================================

Server side (THIS FILE / server.py):
  - Supplies proximal_mu as a float in the Flower fit_config dict each round.
  - Does NOT implement the proximal loss term itself.
  - Does NOT modify gradients or weights.

Client side (vehicle/client.py → vehicle.py training_loop):
  - Reads proximal_mu from the config dict received in fit().
  - Adds the proximal term  (mu/2) * ||w - w_global||^2  to the loss
    during local training steps.
  - Never redefines or ignores mu — always uses what the server sends.

This split is intentional and must not be violated:
  - If the server also applies a proximal correction, the regularisation
    is applied twice → model diverges faster.
  - If the client ignores mu, the proximal term is silently absent → system
    behaves as plain FedAvg, losing non-IID robustness.

Why FedProx for this project:
  - BSM data distribution across vehicles is non-IID: each vehicle sees
    only its local neighbourhood, so local data distributions differ.
  - FedProx prevents local models from drifting too far from the global
    model during local steps, stabilising convergence.
  - Krum selects one update per round; FedProx ensures that update is
    close enough to the global model to be a safe starting point.

Owner: Member B (server config) + Member A (client loss term)
"""

from __future__ import annotations
import os
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Default proximal mu — overridden by PROXIMAL_MU env var or per-round config
DEFAULT_PROXIMAL_MU: float = 0.01

# Mu schedule: can optionally increase mu over rounds to tighten regularisation
# as the model converges. Set MU_SCHEDULE=linear to enable, or flat (default).
MU_SCHEDULE: str = os.environ.get("MU_SCHEDULE", "flat")   # "flat" | "linear"
MU_FINAL: float  = float(os.environ.get("MU_FINAL", "0.1"))


def get_proximal_mu(
    server_round: int,
    total_rounds: int,
    base_mu: float = DEFAULT_PROXIMAL_MU,
) -> float:
    """
    Return the proximal mu value to send to clients for a given FL round.

    Parameters
    ----------
    server_round  : int   — current FL round number (1-indexed)
    total_rounds  : int   — total number of planned FL rounds
    base_mu       : float — starting mu value (default from env / DEFAULT_PROXIMAL_MU)

    Returns
    -------
    float — the proximal_mu value to include in the fit_config dict

    Schedules
    ---------
    "flat"   (default) : mu is constant at base_mu throughout all rounds.
    "linear"           : mu increases linearly from base_mu → MU_FINAL over
                         total_rounds. Tighter regularisation as model matures.
    """
    if MU_SCHEDULE == "linear" and total_rounds > 1:
        # Linear ramp: mu increases from base_mu to MU_FINAL
        progress = (server_round - 1) / (total_rounds - 1)
        mu = base_mu + progress * (MU_FINAL - base_mu)
        mu = round(float(mu), 6)
        logger.debug(
            f"[FedProx] Round {server_round}/{total_rounds} "
            f"mu={mu:.4f} (linear schedule)"
        )
        return mu
    else:
        logger.debug(f"[FedProx] Round {server_round}/{total_rounds} mu={base_mu:.4f} (flat)")
        return float(base_mu)


def make_fit_config(
    server_round:  int,
    total_rounds:  int,
    local_epochs:  int   = 3,
    batch_size:    int   = 256,
    base_mu:       float = DEFAULT_PROXIMAL_MU,
) -> Dict[str, Any]:
    """
    Build the fit_config dict that the Flower server sends to each client
    at the start of a training round.

    This dict is passed as `config` to IDSFlowerClient.fit() on the vehicle side.
    Member A's client reads:
        local_epochs  — number of local training epochs per FL round
        batch_size    — local training batch size
        proximal_mu   — FedProx proximal term coefficient (mu/2 * ||w - w_global||^2)

    Parameters
    ----------
    server_round  : int   — current round (1-indexed)
    total_rounds  : int   — total planned rounds
    local_epochs  : int   — local epochs per round (default 3)
    batch_size    : int   — local batch size (default 256)
    base_mu       : float — proximal mu starting value

    Returns
    -------
    Dict[str, Any]  — the config dict for Flower's fit_config_fn
    """
    mu = get_proximal_mu(server_round, total_rounds, base_mu)

    config = {
        "local_epochs":  local_epochs,
        "batch_size":    batch_size,
        "proximal_mu":   mu,
        "server_round":  server_round,   # informational — clients may log this
        "total_rounds":  total_rounds,
    }

    logger.info(
        f"[FedProx] fit_config round={server_round}: "
        f"epochs={local_epochs} batch={batch_size} mu={mu:.4f}"
    )

    return config


def make_evaluate_config(server_round: int) -> Dict[str, Any]:
    """
    Build the evaluate_config dict for Flower's evaluate_config_fn.
    Currently minimal — just passes the round number for logging.
    """
    return {"server_round": server_round}
