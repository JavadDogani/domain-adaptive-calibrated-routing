"""Offline routing policies.

Every policy maps estimates for ``n`` queries x ``m`` models to one model index per query.
Policies only see *estimates* (predicted quality, predicted cost, predicted latency); the oracle
is the only exception and is reported as an upper bound.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Estimates:
    quality: np.ndarray  # (n, m) calibrated expected quality
    cost: np.ndarray  # (n, m) predicted USD
    latency: np.ndarray  # (n, m) predicted seconds

    def subset(self, idx: np.ndarray) -> Estimates:
        return Estimates(self.quality[idx], self.cost[idx], self.latency[idx])


def route_random(n: int, m: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, m, size=n)


def route_single(n: int, j: int) -> np.ndarray:
    return np.full(n, j, dtype=int)


def route_cheapest(est: Estimates) -> np.ndarray:
    return np.argmin(est.cost, axis=1)


def route_oracle(quality: np.ndarray, cost: np.ndarray) -> np.ndarray:
    """Best true quality per query, ties broken by lowest true cost. Upper bound, not a router."""
    best = quality.max(axis=1, keepdims=True)
    return np.argmin(np.where(quality >= best - 1e-12, cost, np.inf), axis=1)


def route_threshold(est: Estimates, threshold: float, cost_order: np.ndarray) -> np.ndarray:
    """Cheapest model (in ``cost_order``) whose predicted quality >= threshold, else the best one."""
    q = est.quality[:, cost_order]
    ok = q >= threshold
    first = np.argmax(ok, axis=1)
    chosen = cost_order[first]
    none = ~ok.any(axis=1)
    chosen[none] = np.argmax(est.quality[none], axis=1)
    return chosen


def utility(est: Estimates, lam_cost: float, lam_latency: float = 0.0) -> np.ndarray:
    return est.quality - lam_cost * est.cost - lam_latency * est.latency


def route_utility(
    est: Estimates, lam_cost: float, lam_latency: float = 0.0, latency_slo: float | None = None
) -> np.ndarray:
    """Predict-then-optimize (MixLLM-style): argmax quality - lam_cost*cost - lam_latency*latency.

    ``lam_cost`` is in quality points per USD. With ``latency_slo``, models whose predicted
    latency exceeds the SLO are excluded unless no model meets it.
    """
    u = utility(est, lam_cost, lam_latency)
    if latency_slo is not None:
        bad = est.latency > latency_slo
        bad[bad.all(axis=1)] = False
        u = np.where(bad, -np.inf, u)
    return np.argmax(u, axis=1)


def route_binary(p_strong_wins: np.ndarray, weak: int, strong: int, threshold: float) -> np.ndarray:
    """RouteLLM: send to the strong model when P(strong beats weak) >= threshold."""
    return np.where(p_strong_wins >= threshold, strong, weak).astype(int)


def fit_lambda_for_budget(
    est: Estimates,
    true_cost: np.ndarray,
    budget: float,
    lam_latency: float = 0.0,
    latency_slo: float | None = None,
    lo: float = 1e-2,
    hi: float = 1e7,
    iters: int = 60,
) -> float:
    """Smallest cost multiplier whose routing meets the average-cost ``budget`` on held-out data."""
    rows = np.arange(len(true_cost))

    def spend(lam: float) -> float:
        return float(true_cost[rows, route_utility(est, lam, lam_latency, latency_slo)].mean())

    if spend(0.0) <= budget:
        return 0.0
    if spend(hi) > budget:
        return hi
    for _ in range(iters):
        mid = np.sqrt(lo * hi)
        if spend(mid) <= budget:
            hi = mid
        else:
            lo = mid
    return hi
