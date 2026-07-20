"""
server.py — Flower FL Server with Krum + FedProx
=================================================
Implements the FL aggregation server for the V2V IDS system.

Key design decisions (per build plan):
  - num_rounds = 10 (configurable via env)
  - min_fit_clients = 10 — wait for all vehicles before each round
  - Aggregation: Single-Krum (NOT FedAvg) — see krum.py for full explanation
  - FedProx: server supplies proximal_mu in fit_config; clients apply the
    proximal loss term locally (see fedprox.py for boundary docs)
  - Blacklist: vehicles flagged as attackers by IDS are excluded from rounds
    by tracking their client IDs. The attacker container participates in
    BSM generation but its FL update (if it sends one) is rejected here.

Architecture note:
  This server is stateless between runs — no persistence of weights to disk
  between sessions. For a real deployment, add model checkpointing here.

Owner: Member B
"""

from __future__ import annotations
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import flwr as fl
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from krum    import krum_select, krum_scores
from fedprox import make_fit_config, make_evaluate_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("fl_server")

# ── Configuration from environment variables ──────────────────────────────────
NUM_ROUNDS         = int(os.environ.get("NUM_ROUNDS",        "10"))
MIN_FIT_CLIENTS    = int(os.environ.get("MIN_FIT_CLIENTS",   "10"))
MIN_AVAIL_CLIENTS  = int(os.environ.get("MIN_AVAIL_CLIENTS", "10"))
MIN_EVAL_CLIENTS   = int(os.environ.get("MIN_EVAL_CLIENTS",  "5"))
KRUM_F             = int(os.environ.get("KRUM_F",            "1"))    # assumed Byzantine clients
PROXIMAL_MU        = float(os.environ.get("PROXIMAL_MU",     "0.01"))
LOCAL_EPOCHS       = int(os.environ.get("LOCAL_EPOCHS",      "3"))
BATCH_SIZE         = int(os.environ.get("BATCH_SIZE",        "256"))
FL_SERVER_PORT     = int(os.environ.get("FL_SERVER_PORT",    "8080"))
MODEL_DIR          = os.environ.get("MODEL_DIR", "/app/model")


# ─────────────────────────────────────────────────────────────────────────────
# KrumFedProxStrategy — Custom Flower strategy
# ─────────────────────────────────────────────────────────────────────────────
class KrumFedProxStrategy(FedAvg):
    """
    Custom Flower strategy combining:
      1. Single-Krum aggregation (Byzantine-robust selection, NOT averaging)
      2. FedProx config supply (proximal_mu sent in fit_config each round)

    Extends FedAvg only for its client sampling, min_fit_clients, and
    evaluate logic. The aggregate_fit() method is completely overridden
    to implement Single-Krum instead of FedAvg's weighted average.

    Parameters
    ----------
    initial_parameters : Parameters  — starting weights (loaded from Phase 1 model)
    krum_f             : int         — number of assumed Byzantine clients
    proximal_mu        : float       — FedProx mu (sent to clients, not applied here)
    local_epochs       : int         — local training epochs per round
    batch_size         : int         — local training batch size
    """

    def __init__(
        self,
        initial_parameters: Parameters,
        krum_f:       int   = KRUM_F,
        proximal_mu:  float = PROXIMAL_MU,
        local_epochs: int   = LOCAL_EPOCHS,
        batch_size:   int   = BATCH_SIZE,
        num_rounds:   int   = NUM_ROUNDS,
    ):
        super().__init__(
            min_fit_clients           = MIN_FIT_CLIENTS,
            min_available_clients     = MIN_AVAIL_CLIENTS,
            min_evaluate_clients      = MIN_EVAL_CLIENTS,
            initial_parameters        = initial_parameters,
            fit_metrics_aggregation_fn  = self._aggregate_fit_metrics,
            eval_metrics_aggregation_fn = self._aggregate_eval_metrics,
        )
        self.krum_f       = krum_f
        self.proximal_mu  = proximal_mu
        self.local_epochs = local_epochs
        self.batch_size   = batch_size
        self.num_rounds   = num_rounds

        # Round-by-round logging: track Krum selection history
        self._krum_history: List[Dict] = []

        logger.info(
            f"[Strategy] KrumFedProxStrategy initialised: "
            f"rounds={num_rounds} min_clients={MIN_FIT_CLIENTS} "
            f"krum_f={krum_f} proximal_mu={proximal_mu}"
        )

    # ── FedProx: per-round config sent to clients ─────────────────────────────
    def configure_fit(
        self,
        server_round: int,
        parameters:   Parameters,
        client_manager,
    ):
        """Override to inject FedProx mu into each client's fit config."""
        client_instructions = super().configure_fit(
            server_round, parameters, client_manager
        )

        # Build fit config with FedProx mu (Member A's client reads this)
        config = make_fit_config(
            server_round  = server_round,
            total_rounds  = self.num_rounds,
            local_epochs  = self.local_epochs,
            batch_size    = self.batch_size,
            base_mu       = self.proximal_mu,
        )

        # Inject config into each client instruction
        updated_instructions = []
        for client, fit_ins in client_instructions:
            from flwr.common import FitIns
            updated_fit_ins = FitIns(
                parameters = fit_ins.parameters,
                config     = config,
            )
            updated_instructions.append((client, updated_fit_ins))

        logger.info(
            f"[Strategy] Round {server_round}: configured {len(updated_instructions)} clients "
            f"with proximal_mu={config['proximal_mu']:.4f}"
        )
        return updated_instructions

    def configure_evaluate(
        self,
        server_round: int,
        parameters:   Parameters,
        client_manager,
    ):
        """Override to inject evaluate config."""
        client_instructions = super().configure_evaluate(
            server_round, parameters, client_manager
        )
        eval_config = make_evaluate_config(server_round)
        updated = []
        for client, eval_ins in client_instructions:
            from flwr.common import EvaluateIns
            updated.append((client, EvaluateIns(eval_ins.parameters, eval_config)))
        return updated

    # ── Krum: aggregate client updates ────────────────────────────────────────
    def aggregate_fit(
        self,
        server_round: int,
        results:      List[Tuple[ClientProxy, FitRes]],
        failures:     List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        SINGLE-KRUM AGGREGATION — NOT FedAvg.

        Selects exactly ONE client update per round based on Krum scores.
        The selected update becomes the new global model directly.
        No averaging is performed.

        Steps:
        1. Collect weight arrays from all successful clients.
        2. Run krum_select() to score and select the most Byzantine-robust update.
        3. Return the selected client's weights as the new global parameters.
        4. Log selection details for analysis/slides.
        """
        if not results:
            logger.warning(f"[Strategy] Round {server_round}: no results received")
            return None, {}

        if failures:
            logger.warning(
                f"[Strategy] Round {server_round}: {len(failures)} client failures "
                f"(continuing with {len(results)} successful)"
            )

        # ── Collect client weights ─────────────────────────────────────────────
        client_weights: List[List[np.ndarray]] = []
        client_ids:     List[str]              = []
        client_samples: List[int]              = []

        for client_proxy, fit_res in results:
            weights     = parameters_to_ndarrays(fit_res.parameters)
            num_samples = fit_res.num_examples
            client_id   = client_proxy.cid

            client_weights.append(weights)
            client_ids.append(client_id)
            client_samples.append(num_samples)

        logger.info(
            f"[Strategy] Round {server_round}: received updates from "
            f"{len(client_weights)} clients: {client_ids}"
        )

        # ── Run Single-Krum selection ──────────────────────────────────────────
        n = len(client_weights)

        # Safety check: if not enough clients for Krum, fall back gracefully
        if n <= self.krum_f + 2:
            logger.warning(
                f"[Strategy] Round {server_round}: only {n} clients — "
                f"not enough for Krum (need >{self.krum_f+2}). "
                f"Falling back to first client's weights."
            )
            selected_idx     = 0
            selected_weights = client_weights[0]
            scores           = [0.0] * n
        else:
            selected_idx, selected_weights = krum_select(
                client_weights, f=self.krum_f
            )
            scores = krum_scores(client_weights, f=self.krum_f)

        selected_id = client_ids[selected_idx]

        # ── Log round results ──────────────────────────────────────────────────
        round_log = {
            "round":         server_round,
            "n_clients":     n,
            "client_ids":    client_ids,
            "krum_scores":   [round(s, 4) for s in scores],
            "selected_idx":  selected_idx,
            "selected_id":   selected_id,
            "num_samples":   client_samples[selected_idx],
        }
        self._krum_history.append(round_log)

        logger.info(
            f"[Krum] Round {server_round}: SELECTED client_{selected_idx} "
            f"(id={selected_id}, samples={client_samples[selected_idx]})"
        )
        logger.info(
            f"[Krum] Round {server_round}: Scores: "
            + " | ".join(
                f"{'*' if i == selected_idx else ' '}client_{i}={scores[i]:.4f}"
                for i in range(n)
            )
        )

        # ── Return selected weights as new global parameters ───────────────────
        aggregated_parameters = ndarrays_to_parameters(selected_weights)

        # Metrics for Flower's built-in logging
        metrics: Dict[str, Scalar] = {
            "selected_client_idx": float(selected_idx),
            "n_clients":           float(n),
            "krum_score_selected": float(scores[selected_idx]) if scores else 0.0,
            "krum_score_min":      float(min(scores)) if scores else 0.0,
            "krum_score_max":      float(max(scores)) if scores else 0.0,
        }

        return aggregated_parameters, metrics

    # ── Metric aggregation helpers ─────────────────────────────────────────────
    @staticmethod
    def _aggregate_fit_metrics(metrics_list):
        """Aggregate fit metrics across clients for Flower's logging."""
        if not metrics_list:
            return {}
        total_samples = sum(n for n, _ in metrics_list)
        if total_samples == 0:
            return {}
        aggregated = {}
        for key in metrics_list[0][1]:
            vals = [m[key] for _, m in metrics_list if key in m]
            if vals:
                aggregated[key] = float(np.mean(vals))
        return aggregated

    @staticmethod
    def _aggregate_eval_metrics(metrics_list):
        """Aggregate evaluate metrics across clients for Flower's logging."""
        if not metrics_list:
            return {}
        total_samples = sum(n for n, _ in metrics_list)
        if total_samples == 0:
            return {}
        aggregated = {}
        for key in metrics_list[0][1]:
            weighted_vals = [
                metrics[key] * n
                for n, metrics in metrics_list
                if key in metrics
            ]
            if weighted_vals:
                aggregated[key] = float(sum(weighted_vals) / total_samples)
        return aggregated

    def get_krum_history(self) -> List[Dict]:
        """Return the full history of Krum selections for analysis/slides."""
        return list(self._krum_history)


# ─────────────────────────────────────────────────────────────────────────────
# Model loader
# ─────────────────────────────────────────────────────────────────────────────
def load_initial_parameters(model_dir: str) -> Parameters:
    """
    Load the Phase 1 pretrained model and extract its weights as
    Flower Parameters. These become the starting global model weights.

    Parameters
    ----------
    model_dir : str  — path to directory containing best_ids_model.keras

    Returns
    -------
    Parameters  — Flower-native parameter format (list of serialised ndarrays)
    """
    import tensorflow as tf   # lazy import — only server needs TF

    model_path = Path(model_dir) / "best_ids_model.keras"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Phase 1 model not found at {model_path}. "
            "Copy best_ids_model.keras into /app/model/ before starting the server."
        )

    logger.info(f"[Server] Loading initial model from {model_path}")
    model = tf.keras.models.load_model(str(model_path))
    weights = model.get_weights()
    logger.info(
        f"[Server] Model loaded. "
        f"Layers={len(weights)}, "
        f"Total params={sum(w.size for w in weights):,}"
    )

    return ndarrays_to_parameters(weights)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    """Start the Flower FL server."""
    logger.info("=" * 60)
    logger.info("  V2V IDS Federated Learning Server")
    logger.info(f"  Rounds:       {NUM_ROUNDS}")
    logger.info(f"  Min clients:  {MIN_FIT_CLIENTS}")
    logger.info(f"  Krum f:       {KRUM_F}")
    logger.info(f"  Proximal mu:  {PROXIMAL_MU}")
    logger.info(f"  Port:         {FL_SERVER_PORT}")
    logger.info("=" * 60)

    # Load Phase 1 pretrained model as global starting point
    initial_parameters = load_initial_parameters(MODEL_DIR)

    # Build Krum + FedProx strategy
    strategy = KrumFedProxStrategy(
        initial_parameters = initial_parameters,
        krum_f             = KRUM_F,
        proximal_mu        = PROXIMAL_MU,
        local_epochs       = LOCAL_EPOCHS,
        batch_size         = BATCH_SIZE,
        num_rounds         = NUM_ROUNDS,
    )

    # Configure and start Flower server
    fl.server.start_server(
        server_address = f"0.0.0.0:{FL_SERVER_PORT}",
        config         = fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy       = strategy,
    )

    # Post-run: print Krum selection summary
    history = strategy.get_krum_history()
    if history:
        logger.info("\n" + "=" * 60)
        logger.info("  Krum Selection Summary (all rounds)")
        logger.info("=" * 60)
        for entry in history:
            logger.info(
                f"  Round {entry['round']:>2}: selected={entry['selected_id']} "
                f"score={entry['krum_scores'][entry['selected_idx']]:.4f}"
            )
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
