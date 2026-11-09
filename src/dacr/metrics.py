"""Evaluation metrics for routing decisions."""

from __future__ import annotations

import numpy as np
import pandas as pd


def realized(choice: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return matrix[np.arange(len(choice)), choice]


def summarize(
    choice: np.ndarray, quality: np.ndarray, cost: np.ndarray, latency: np.ndarray, slo_s: float
) -> dict[str, float]:
    q, c, lat = realized(choice, quality), realized(choice, cost), realized(choice, latency)
    return summarize_outcomes(q, c, lat, slo_s)


def summarize_outcomes(q: np.ndarray, c: np.ndarray, lat: np.ndarray, slo_s: float) -> dict[str, float]:
    return {
        "n": int(len(q)),
        "quality": float(np.mean(q)),
        "cost_usd": float(np.mean(c)),
        "latency_mean_s": float(np.mean(lat)),
        "latency_p50_s": float(np.quantile(lat, 0.50)),
        "latency_p95_s": float(np.quantile(lat, 0.95)),
        "latency_p99_s": float(np.quantile(lat, 0.99)),
        "slo_violation": float(np.mean(lat > slo_s)),
    }


def model_shares(choice: np.ndarray, models: list[str]) -> dict[str, float]:
    counts = np.bincount(choice, minlength=len(models)) / max(len(choice), 1)
    return {m: float(s) for m, s in zip(models, counts, strict=True)}


def per_domain(choice: np.ndarray, quality: np.ndarray, cost: np.ndarray, domains: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame({"domain": domains, "quality": realized(choice, quality), "cost_usd": realized(choice, cost)})
    return df.groupby("domain").agg(n=("quality", "size"), quality=("quality", "mean"), cost_usd=("cost_usd", "mean"))


def pareto_mask(cost: np.ndarray, quality: np.ndarray) -> np.ndarray:
    """True for points not dominated (lower-or-equal cost and higher-or-equal quality) by another."""
    cost, quality = np.asarray(cost), np.asarray(quality)
    order = np.lexsort((-quality, cost))
    mask = np.zeros(len(cost), dtype=bool)
    best = -np.inf
    for i in order:
        if quality[i] > best:
            mask[i] = True
            best = quality[i]
    return mask
