"""Probability calibration: Platt scaling, domain-adaptive (grouped) calibration, and metrics.

All functions accept *soft* labels ``y`` in [0, 1] (RouterBench has graded scores such as 0.25 or
0.75). Calibration then means: among queries with predicted value ``p``, the average observed
quality is ``p``.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

_EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


def soft_label_expand(
    X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Turn soft targets into weighted binary rows so any classifier minimizes soft log-loss.

    A row with target 0.75 becomes a positive row with weight 0.75 and a negative row with
    weight 0.25. Rows with hard 0/1 targets stay single rows.
    """
    y = np.clip(np.asarray(y, dtype=float), 0.0, 1.0)
    w = np.ones_like(y) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    pos, neg = y > 0, y < 1
    X2 = np.concatenate([X[pos], X[neg]])
    y2 = np.concatenate([np.ones(pos.sum(), dtype=int), np.zeros(neg.sum(), dtype=int)])
    w2 = np.concatenate([(w * y)[pos], (w * (1 - y))[neg]])
    return X2, y2, w2


class PlattCalibrator:
    """Logistic regression on the logit of the score (Platt scaling)."""

    def __init__(self, C: float = 100.0):
        self.C = C
        self.lr: LogisticRegression | None = None
        self.const_: float | None = None

    def fit(self, p: np.ndarray, y: np.ndarray) -> PlattCalibrator:
        X2, y2, w2 = soft_label_expand(_logit(p).reshape(-1, 1), y)
        if len(np.unique(y2)) < 2:
            self.const_, self.lr = float(np.mean(y)), None
            return self
        self.lr = LogisticRegression(C=self.C, max_iter=1000).fit(X2, y2, sample_weight=w2)
        self.const_ = None
        return self

    def predict(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=float).reshape(-1)
        if self.lr is None:
            return np.full_like(p, self.const_ if self.const_ is not None else 0.5)
        return self.lr.predict_proba(_logit(p).reshape(-1, 1))[:, 1]


class GroupCalibrator:
    """Domain-adaptive calibration: a global Platt calibrator plus per-group calibrators.

    The output for a query in group ``g`` is ``w_g * cal_g(p) + (1 - w_g) * cal_global(p)`` with
    ``w_g = n_g / (n_g + prior_strength)``: groups with little calibration data are shrunk toward
    the global calibrator, and groups never seen during fitting use the global one.
    """

    def __init__(self, min_group_size: int = 30, prior_strength: float = 100.0):
        self.min_group_size = min_group_size
        self.prior_strength = prior_strength
        self.global_cal = PlattCalibrator()
        self.group_cal: dict[str, PlattCalibrator] = {}
        self.group_weight: dict[str, float] = {}

    def fit(self, p: np.ndarray, y: np.ndarray, groups: np.ndarray | None = None) -> GroupCalibrator:
        p, y = np.asarray(p, dtype=float), np.asarray(y, dtype=float)
        self.global_cal.fit(p, y)
        self.group_cal, self.group_weight = {}, {}
        if groups is None:
            return self
        groups = np.asarray(groups).astype(str)
        for g in np.unique(groups):
            mask = groups == g
            n = int(mask.sum())
            if n >= self.min_group_size:
                self.group_cal[g] = PlattCalibrator().fit(p[mask], y[mask])
                self.group_weight[g] = n / (n + self.prior_strength)
        return self

    def predict(self, p: np.ndarray, groups: np.ndarray | None = None) -> np.ndarray:
        p = np.asarray(p, dtype=float).reshape(-1)
        out = self.global_cal.predict(p)
        if groups is None or not self.group_cal:
            return out
        groups = np.asarray(groups).astype(str)
        for g in np.unique(groups):
            if g in self.group_cal:
                mask = groups == g
                w = self.group_weight[g]
                out[mask] = w * self.group_cal[g].predict(p[mask]) + (1 - w) * out[mask]
        return out


class MatrixCalibrator:
    """Calibrates an ``(n, m)`` matrix of per-model scores, one calibrator per model column.

    ``mode``: ``none`` (identity), ``global`` (Platt) or ``group`` (domain-adaptive).
    """

    def __init__(self, mode: str = "group", min_group_size: int = 30, prior_strength: float = 100.0):
        if mode not in ("none", "global", "group"):
            raise ValueError(f"unknown calibration mode {mode!r}")
        self.mode = mode
        self.min_group_size = min_group_size
        self.prior_strength = prior_strength
        self.cals: list[GroupCalibrator] = []

    def fit(self, P: np.ndarray, Y: np.ndarray, groups: np.ndarray | None = None) -> MatrixCalibrator:
        if self.mode == "none":
            return self
        g = groups if self.mode == "group" else None
        self.cals = [
            GroupCalibrator(self.min_group_size, self.prior_strength).fit(P[:, j], Y[:, j], g)
            for j in range(P.shape[1])
        ]
        return self

    def predict(self, P: np.ndarray, groups: np.ndarray | None = None) -> np.ndarray:
        if self.mode == "none":
            return np.asarray(P, dtype=float)
        g = groups if self.mode == "group" else None
        return np.column_stack([cal.predict(P[:, j], g) for j, cal in enumerate(self.cals)])


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------


def reliability_bins(p: np.ndarray, y: np.ndarray, n_bins: int = 15):
    """Equal-width bins. Returns (mean predicted, mean observed, count) per non-empty bin."""
    p = np.clip(np.asarray(p, dtype=float).reshape(-1), 0.0, 1.0)
    y = np.asarray(y, dtype=float).reshape(-1)
    idx = np.minimum((p * n_bins).astype(int), n_bins - 1)
    count = np.bincount(idx, minlength=n_bins)
    conf = np.bincount(idx, weights=p, minlength=n_bins)
    obs = np.bincount(idx, weights=y, minlength=n_bins)
    nz = count > 0
    return conf[nz] / count[nz], obs[nz] / count[nz], count[nz]


def expected_calibration_error(p: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    if np.size(p) == 0:
        return float("nan")
    conf, obs, count = reliability_bins(p, y, n_bins)
    return float(np.sum(count * np.abs(conf - obs)) / np.sum(count))


def brier_score(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2))
