"""A fitted, savable router that scores new prompts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .calibration import MatrixCalibrator
from .domains import DomainAssigner
from .features import FeatureBuilder, clean_prompt
from .latency import LatencyModel
from .policies import Estimates, route_utility
from .predictors import PriceSheet, QualityPredictor, ResponseLengthPredictor


@dataclass
class Router:
    models: list[str]
    features: FeatureBuilder
    quality: QualityPredictor
    domains: DomainAssigner
    calibrator: MatrixCalibrator
    lengths: ResponseLengthPredictor
    prices: PriceSheet
    latency: LatencyModel
    lam_cost: float  # quality points per USD; meets the cost budget on calibration data
    lam_latency: float = 0.0

    def estimate(self, prompts: list[str]) -> Estimates:
        texts = [clean_prompt(p) for p in prompts]
        emb = self.features.embed(texts)
        X = self.features.transform(texts, emb)
        groups = self.domains.assign(emb, np.array(["unknown"] * len(texts)))
        P = self.calibrator.predict(self.quality.predict(X), groups)
        R = self.lengths.predict(X)
        prompt_chars = np.array([len(str(p)) for p in prompts], dtype=float)
        return Estimates(P, self.prices.predict(prompt_chars, R), self.latency.expected(prompt_chars, R))

    def route(self, prompts: list[str], lam_cost: float | None = None) -> list[str]:
        est = self.estimate(prompts)
        lam = self.lam_cost if lam_cost is None else lam_cost
        return [self.models[j] for j in route_utility(est, lam, self.lam_latency)]

    def explain(self, prompt: str, lam_cost: float | None = None) -> pd.DataFrame:
        est = self.estimate([prompt])
        lam = self.lam_cost if lam_cost is None else lam_cost
        df = pd.DataFrame(
            {
                "model": self.models,
                "quality_hat": est.quality[0],
                "cost_hat_usd": est.cost[0],
                "latency_hat_s": est.latency[0],
            }
        )
        df["utility"] = df["quality_hat"] - lam * df["cost_hat_usd"] - self.lam_latency * df["latency_hat_s"]
        return df.sort_values("utility", ascending=False).reset_index(drop=True)

    def save(self, path: str | Path) -> None:
        self.features.cache_dir = None  # don't depend on the training machine's cache
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> Router:
        obj = joblib.load(path)
        if not isinstance(obj, Router):
            raise TypeError(f"{path} does not contain a dacr Router")
        return obj
