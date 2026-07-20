"""
client.py — Real Flower FL Client (Member B's implementation)
=============================================================
This file REPLACES client_stub.py — it is the live implementation.
vehicle.py imports `from client import IDSFlowerClient, start_fl_client`
and will automatically use this file over the stub once it's present.

INTERFACE CONTRACT (frozen — matches client_stub.py exactly):
  IDSFlowerClient(model, X_train, y_train, X_val, y_val, vehicle_id)
    .get_parameters(config)  → List[np.ndarray]
    .set_parameters(params)  → None
    .fit(parameters, config) → (weights, num_samples, metrics)
    .evaluate(parameters, config) → (loss, num_samples, metrics)
  start_fl_client(server_address, flower_client) → None

FedProx Proximal Term (MEMBER A'S BOUNDARY):
  The proximal_mu value is sent by the server in config["proximal_mu"].
  The loss modification is: loss += (mu/2) * ||w - w_global||^2
  This is applied HERE in fit() using a custom training loop.
  The server does NOT apply any proximal correction — see fedprox.py.

Owner: Member B
"""

from __future__ import annotations
import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
import flwr as fl
from flwr.common import NDArrays, Scalar

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# IDSFlowerClient — Real Flower numpy client
# ─────────────────────────────────────────────────────────────────────────────
class IDSFlowerClient(fl.client.NumPyClient):
    """
    Flower federated learning client wrapping the V2V IDS Keras model.

    Implements FedProx proximal regularisation client-side:
        loss = cross_entropy(logits, y) + (mu/2) * ||w - w_global||^2

    The proximal term keeps local weights close to the last received global
    weights, preventing excessive drift on non-IID vehicle data.

    Parameters
    ----------
    model      : tf.keras.Model  — compiled CNN+GRU+Attention model
    X_train    : (N, WINDOW_SIZE, N_FEATURES) float32
    y_train    : (N,) int32
    X_val      : (M, WINDOW_SIZE, N_FEATURES) float32
    y_val      : (M,) int32
    vehicle_id : int  — numeric vehicle ID (for logging)
    """

    def __init__(
        self,
        model:      tf.keras.Model,
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

        # Store reference to global weights for proximal term
        # Updated in set_parameters() at the start of each fit() call
        self._global_weights: Optional[List[np.ndarray]] = None

        logger.info(f"[Client:{vehicle_id}] IDSFlowerClient ready — "
                    f"train={len(X_train)} val={len(X_val)} samples")

    # ── Flower NumPyClient API ────────────────────────────────────────────────

    def get_parameters(self, config: Dict) -> NDArrays:
        """Return current model weights as a list of NumPy arrays."""
        return self.model.get_weights()

    def set_parameters(self, parameters: NDArrays) -> None:
        """
        Update local model weights with the received global weights.
        Also saves a copy as the reference for the proximal term.
        """
        self.model.set_weights(parameters)
        # Save reference copy for FedProx proximal term
        self._global_weights = [w.copy() for w in parameters]

    def fit(
        self,
        parameters: NDArrays,
        config:     Dict[str, Scalar],
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """
        Receive global weights, run local training with FedProx, return updated weights.

        FedProx proximal term is applied here using a custom training step:
            total_loss = cross_entropy_loss + (mu/2) * ||w - w_global||^2

        Config keys received from server (see fedprox.make_fit_config()):
            local_epochs  : int   — local training epochs
            batch_size    : int   — batch size
            proximal_mu   : float — FedProx mu coefficient
            server_round  : int   — current FL round (for logging)
        """
        # 1. Pull in global weights (also sets self._global_weights reference)
        self.set_parameters(parameters)

        # 2. Read FL-round hyperparameters
        local_epochs = int(config.get("local_epochs", 3))
        batch_size   = int(config.get("batch_size", 256))
        proximal_mu  = float(config.get("proximal_mu", 0.01))
        server_round = int(config.get("server_round", -1))

        logger.info(
            f"[Client:{self.vehicle_id}] fit() round={server_round} "
            f"epochs={local_epochs} batch={batch_size} mu={proximal_mu:.4f} "
            f"samples={len(self.X_train)}"
        )

        if len(self.X_train) == 0:
            logger.warning(f"[Client:{self.vehicle_id}] No training data — returning unchanged weights")
            return self.model.get_weights(), 0, {"train_loss": 0.0, "train_accuracy": 0.0}

        # 3. Build TensorFlow dataset
        dataset = (
            tf.data.Dataset
            .from_tensor_slices((self.X_train, self.y_train))
            .shuffle(buffer_size=min(len(self.X_train), 10_000))
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )

        # 4. FedProx training loop
        optimizer    = tf.keras.optimizers.Adam(learning_rate=1e-3)
        loss_fn      = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        train_losses = []
        train_accs   = []

        global_weights_tf = [
            tf.constant(w, dtype=tf.float32)
            for w in self._global_weights
        ] if self._global_weights else None

        for epoch in range(local_epochs):
            epoch_losses = []
            epoch_accs   = []

            for X_batch, y_batch in dataset:
                with tf.GradientTape() as tape:
                    logits = self.model(X_batch, training=True)
                    ce_loss = loss_fn(y_batch, logits)

                    # FedProx proximal term: (mu/2) * ||w - w_global||^2
                    prox_term = tf.constant(0.0)
                    if proximal_mu > 0.0 and global_weights_tf is not None:
                        prox_parts = []
                        for w_local, w_global in zip(
                            self.model.trainable_variables, global_weights_tf
                        ):
                            # Only apply to matching-shape variables
                            if w_local.shape == w_global.shape:
                                prox_parts.append(
                                    tf.reduce_sum(tf.square(w_local - w_global))
                                )
                        if prox_parts:
                            prox_term = (proximal_mu / 2.0) * tf.add_n(prox_parts)

                    total_loss = ce_loss + prox_term

                grads = tape.gradient(total_loss, self.model.trainable_variables)
                optimizer.apply_gradients(
                    zip(grads, self.model.trainable_variables)
                )

                # Accuracy for logging
                preds = tf.argmax(logits, axis=1)
                acc   = tf.reduce_mean(
                    tf.cast(tf.equal(preds, tf.cast(y_batch, tf.int64)), tf.float32)
                )
                epoch_losses.append(float(total_loss.numpy()))
                epoch_accs.append(float(acc.numpy()))

            mean_loss = float(np.mean(epoch_losses))
            mean_acc  = float(np.mean(epoch_accs))
            train_losses.append(mean_loss)
            train_accs.append(mean_acc)

            logger.info(
                f"[Client:{self.vehicle_id}] Round {server_round} "
                f"epoch {epoch+1}/{local_epochs}: "
                f"loss={mean_loss:.4f} acc={mean_acc:.4f}"
            )

        final_loss = float(np.mean(train_losses))
        final_acc  = float(np.mean(train_accs))

        metrics: Dict[str, Scalar] = {
            "train_loss":     final_loss,
            "train_accuracy": final_acc,
            "vehicle_id":     float(self.vehicle_id),
        }

        logger.info(
            f"[Client:{self.vehicle_id}] fit() done: "
            f"loss={final_loss:.4f} acc={final_acc:.4f}"
        )

        return self.model.get_weights(), len(self.X_train), metrics

    def evaluate(
        self,
        parameters: NDArrays,
        config:     Dict[str, Scalar],
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        """
        Evaluate the global model on local validation set.

        Returns (loss, num_val_samples, metrics_dict).
        Called by the server after each round to assess global model quality.
        """
        self.set_parameters(parameters)
        server_round = int(config.get("server_round", -1))

        if len(self.X_val) == 0:
            return 0.0, 0, {"accuracy": 0.0, "vehicle_id": float(self.vehicle_id)}

        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        logits  = self.model(self.X_val, training=False)
        loss    = float(loss_fn(self.y_val, logits).numpy())

        preds   = tf.argmax(logits, axis=1).numpy()
        acc     = float(np.mean(preds == self.y_val.astype(np.int64)))

        logger.info(
            f"[Client:{self.vehicle_id}] evaluate() round={server_round}: "
            f"loss={loss:.4f} acc={acc:.4f} "
            f"(val_samples={len(self.X_val)})"
        )

        metrics: Dict[str, Scalar] = {
            "accuracy":   acc,
            "vehicle_id": float(self.vehicle_id),
        }

        return loss, len(self.X_val), metrics


# ─────────────────────────────────────────────────────────────────────────────
# start_fl_client — called from vehicle.py's training_loop
# ─────────────────────────────────────────────────────────────────────────────
def start_fl_client(
    server_address: str,
    flower_client:  IDSFlowerClient,
) -> None:
    """
    Connect to the Flower FL server and participate in one FL session.

    Called from vehicle.py's _run_fl_round() in a thread executor so it
    does not block the asyncio event loop.

    Parameters
    ----------
    server_address : str              — e.g. "fl_server:8080"
    flower_client  : IDSFlowerClient  — the local client instance

    Notes
    -----
    - Uses insecure gRPC for the simulation environment.
    - For production, replace with TLS certificates.
    - The function returns when the server completes all rounds,
      or raises an exception if the connection fails.
    """
    logger.info(
        f"[Client:{flower_client.vehicle_id}] "
        f"Connecting to FL server at {server_address}"
    )

    try:
        fl.client.start_numpy_client(
            server_address = server_address,
            client         = flower_client,
        )
        logger.info(
            f"[Client:{flower_client.vehicle_id}] "
            f"FL session completed successfully"
        )
    except Exception as e:
        logger.error(
            f"[Client:{flower_client.vehicle_id}] "
            f"FL session failed: {e}"
        )
        raise
