"""
krum.py — Single-Krum Byzantine-Robust Aggregation
====================================================
Implements Single-Krum as described in:
  Blanchard et al., "Machine Learning with Adversaries: Byzantine Tolerant
  Gradient Descent", NeurIPS 2017.

IMPORTANT — This is SINGLE-KRUM, not multi-Krum or FedAvg-with-extra-steps:
  - Single-Krum selects exactly ONE client update per round.
  - The selected update is applied directly as the new global model.
  - No averaging is performed across clients.
  - This is by design: averaging would dilute Byzantine-resistance.

Why Single-Krum for this project:
  - With 10 vehicles and f=1 assumed Byzantine client, n-f-2 = 7 nearest
    neighbours are used to score each client.
  - The client whose update is most consistent with the majority of other
    clients (smallest sum of distances to its 7 nearest neighbours) wins.
  - The attacker vehicle's poisoned update will be far from the 9 benign
    clients' updates, so it reliably loses the selection.

Algorithm:
  Given n client weight vectors w_1, ..., w_n:
  1. For each client i, compute pairwise L2 distance to all other clients.
  2. Sort distances for client i; take the sum of the n-f-2 smallest.
     (n-f-2 = 10 - 1 - 2 = 7 for this phase)
  3. The client i* with the smallest such sum is selected.
  4. Return w_i* as the new global weights.

Owner: Member B
"""

from __future__ import annotations
import logging
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def flatten_weights(weights: List[np.ndarray]) -> np.ndarray:
    """
    Flatten a list of weight arrays into a single 1-D vector.
    Used to compute L2 distances between full model parameter sets.

    Parameters
    ----------
    weights : List[np.ndarray]  — model weight arrays (one per layer)

    Returns
    -------
    np.ndarray  — 1-D float64 vector of all parameters concatenated
    """
    return np.concatenate([w.flatten().astype(np.float64) for w in weights])


def krum_select(
    client_weights: List[List[np.ndarray]],
    f: int = 1,
) -> Tuple[int, List[np.ndarray]]:
    """
    Single-Krum selection: choose the client update closest to the majority.

    Parameters
    ----------
    client_weights : List[List[np.ndarray]]
        Weight updates from each client. Each entry is a list of numpy arrays
        (one per model layer), matching the Flower get_parameters() format.
    f : int
        Number of assumed Byzantine (malicious) clients. Default 1.
        The Krum score uses the n-f-2 smallest pairwise distances.

    Returns
    -------
    (selected_idx, selected_weights) : Tuple[int, List[np.ndarray]]
        Index of the selected client and their weight arrays.

    Raises
    ------
    ValueError  — if n <= f+2 (not enough clients for Krum to be meaningful)
    """
    n = len(client_weights)
    if n <= f + 2:
        raise ValueError(
            f"Krum requires n > f+2 clients. Got n={n}, f={f}. "
            f"Need at least {f+3} clients."
        )

    # ── Step 1: Flatten all weight vectors ────────────────────────────────────
    flat = [flatten_weights(w) for w in client_weights]

    # ── Step 2: Compute symmetric pairwise L2 distance matrix ─────────────────
    # distances[i][j] = ||flat[i] - flat[j]||_2
    distances = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            dist = float(np.linalg.norm(flat[i] - flat[j]))
            distances[i][j] = dist
            distances[j][i] = dist

    logger.debug(f"[Krum] Distance matrix (n={n}):\n{np.round(distances, 4)}")

    # ── Step 3: Compute Krum score for each client ────────────────────────────
    # For client i: sort distances to all other clients, take sum of smallest n-f-2
    k = n - f - 2   # number of nearest neighbours to use (= 7 for n=10, f=1)
    scores = np.zeros(n, dtype=np.float64)

    for i in range(n):
        # Distances from client i to all others (excluding self)
        dists_from_i = sorted([distances[i][j] for j in range(n) if j != i])
        scores[i] = sum(dists_from_i[:k])

    logger.info(
        f"[Krum] Scores (n={n}, f={f}, k={k}): "
        + " | ".join(f"client_{i}={scores[i]:.4f}" for i in range(n))
    )

    # ── Step 4: Select client with minimum score ───────────────────────────────
    selected_idx = int(np.argmin(scores))
    selected_weights = client_weights[selected_idx]

    logger.info(
        f"[Krum] Selected client_{selected_idx} "
        f"(score={scores[selected_idx]:.4f}, "
        f"runner-up={sorted(scores)[1]:.4f})"
    )

    return selected_idx, selected_weights


def krum_scores(
    client_weights: List[List[np.ndarray]],
    f: int = 1,
) -> List[float]:
    """
    Return Krum scores for all clients without selecting one.
    Useful for logging, monitoring, or multi-Krum extension.

    Parameters
    ----------
    client_weights : List[List[np.ndarray]]  — weight updates per client
    f              : int                      — assumed Byzantine clients

    Returns
    -------
    List[float]  — Krum score per client (lower = more trusted)
    """
    n = len(client_weights)
    if n <= f + 2:
        return [float("inf")] * n

    flat = [flatten_weights(w) for w in client_weights]
    k    = n - f - 2
    distances = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(flat[i] - flat[j]))
            distances[i][j] = d
            distances[j][i] = d

    scores = []
    for i in range(n):
        dists = sorted([distances[i][j] for j in range(n) if j != i])
        scores.append(sum(dists[:k]))

    return scores
