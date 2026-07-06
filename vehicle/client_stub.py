"""
client_stub.py — Member B Interface Provision (FL Client)
=========================================================
THIS FILE IS A STUB / INTERFACE SPECIFICATION.

Member B owns and implements the real `client.py` which vehicle.py imports.
This stub documents the exact interface contract so both members can work
in parallel without breaking each other.

When Member B delivers client.py, simply drop it in /vehicle/ alongside
this stub. vehicle.py already imports `from client import IDSFlowerClient`
and will use the real implementation automatically.

Expected interface (based on the Phase 1 IDSFlowerClient class):
"""

from __future__ import annotations
import logging
import numpy as np
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class IDSFlowerClient:
    """
    STUB — Member B implements the real version.

    Interface contract (frozen — do not change method signatures):

    Parameters
    ----------
    model      : compiled Keras model
    X_train    : (N, WINDOW_SIZE, N_FEATURES) float32
    y_train    : (N,) int32
    X_val      : (M, WINDOW_SIZE, N_FEATURES) float32
    y_val      : (M,) int32
    vehicle_id : int  — numeric vehicle ID for FL round identification
    """

    def __init__(
        self,
        model,
        X_train:    np.ndarray,
        y_train:    np.ndarray,
        X_val:      np.ndarray,
        y_val:      np.ndarray,
        vehicle_id: int = 0,
    ):
        self.model      = model
        self.X_train    = X_train
        self.y_train    = y_train
        self.X_val      = X_val
        self.y_val      = y_val
        self.vehicle_id = vehicle_id
        logger.warning(
            "[client_stub] Using STUB IDSFlowerClient — weights will only be logged locally. "
            "Member B must supply the real client.py."
        )

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """Return current model weights as a list of NumPy arrays."""
        return self.model.get_weights()

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Update local model weights from aggregated global weights."""
        self.model.set_weights(parameters)

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict,
    ):
        """
        Receive global weights, run local training, return updated weights.

        config keys Member B's server should send:
            - local_epochs  : int   (default 3)
            - batch_size    : int   (default 256)
            - proximal_mu   : float (default 0.01) — FedProx mu
        """
        self.set_parameters(parameters)

        local_epochs = int(config.get("local_epochs", 3))
        batch_size   = int(config.get("batch_size", 256))
        proximal_mu  = float(config.get("proximal_mu", 0.01))

        logger.warning(
            f"[client_stub] Stub fit() — logging weights locally instead of training. "
            f"epochs={local_epochs} batch={batch_size} mu={proximal_mu}"
        )

        # STUB: return current weights unchanged (no real training)
        weights     = self.model.get_weights()
        num_samples = len(self.X_train)
        metrics     = {"train_loss": 0.0, "train_accuracy": 0.0}

        return weights, num_samples, metrics

    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict,
    ):
        """
        Evaluate the model on local validation set.
        Returns (loss, num_val_samples, metrics_dict).
        """
        self.set_parameters(parameters)
        logger.warning("[client_stub] Stub evaluate() — returning dummy metrics")
        return 0.0, len(self.X_val), {"accuracy": 0.0}


def start_fl_client(
    server_address: str,
    flower_client: IDSFlowerClient,
):
    """
    Start the Flower client and connect to the FL server.

    Member B implements this. Member A calls it from vehicle.py's training_loop.
    The real implementation should call:
        fl.client.start_numpy_client(server_address=server_address, client=flower_client)

    Parameters
    ----------
    server_address : str  — "fl_server:8080" (from docker-compose env var)
    flower_client  : IDSFlowerClient — the local client instance
    """
    logger.warning(
        f"[client_stub] start_fl_client() stub called (server={server_address}). "
        "Member B must supply the real client.py with actual Flower connection."
    )
    # STUB: just log the weights count
    weights = flower_client.get_parameters({})
    total_params = sum(w.size for w in weights)
    logger.info(f"[client_stub] Would send {total_params:,} parameters to {server_address}")
