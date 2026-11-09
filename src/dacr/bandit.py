"""Contextual bandit components used by the online DACR router."""

from __future__ import annotations

import numpy as np


class ResidualLinUCB:
    """Per-arm ridge regression with UCB exploration, learning a *residual* over a prior.

    For each arm ``a`` it models ``quality - prior(x, a) ~ theta_a . z``. With a zero prior it is
    plain LinUCB. The inverse design matrix is maintained with Sherman-Morrison updates, so an
    update costs O(d^2) instead of a matrix inversion.
    """

    def __init__(self, n_arms: int, dim: int, alpha: float = 0.3, ridge: float = 1.0):
        self.n_arms, self.dim, self.alpha, self.ridge = n_arms, dim, alpha, ridge
        self.A_inv = np.repeat((np.eye(dim) / ridge)[None], n_arms, axis=0)
        self.b = np.zeros((n_arms, dim))
        self.theta = np.zeros((n_arms, dim))
        self.counts = np.zeros(n_arms, dtype=int)

    def mean_width(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mu = self.theta @ z
        Az = self.A_inv @ z  # (arms, d)
        width = np.sqrt(np.clip(Az @ z, 0.0, None))
        return mu, width

    def ucb(self, z: np.ndarray) -> np.ndarray:
        mu, width = self.mean_width(z)
        return mu + self.alpha * width

    def update(self, arm: int, z: np.ndarray, target: float) -> None:
        Az = self.A_inv[arm] @ z
        self.A_inv[arm] -= np.outer(Az, Az) / (1.0 + z @ Az)
        self.b[arm] += target * z
        self.theta[arm] = self.A_inv[arm] @ self.b[arm]
        self.counts[arm] += 1


class DualVariable:
    """Online projected dual ascent for a long-run average constraint ``E[x] <= budget``.

    The selection penalty is ``lam * x_hat / budget`` (dimensionless). After each observation,
    ``lam <- clip(lam + eta * (x / budget - 1), 0, lam_max)``: it rises while we overspend and
    decays toward zero while we are under budget.
    """

    def __init__(self, budget: float, eta: float = 0.01, lam0: float = 0.0, lam_max: float = 50.0):
        if budget <= 0:
            raise ValueError("budget must be positive")
        self.budget, self.eta, self.lam, self.lam_max = budget, eta, float(lam0), lam_max

    def penalty(self, x_hat: np.ndarray) -> np.ndarray:
        return self.lam * np.asarray(x_hat) / self.budget

    def update(self, observed: float) -> None:
        self.lam = float(np.clip(self.lam + self.eta * (observed / self.budget - 1.0), 0.0, self.lam_max))
