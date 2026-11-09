"""Per-model predictors of quality, response length and cost.

Only information available *before* calling a model is used at routing time: the prompt
features, the prompt length and each model's price sheet. Response length (and therefore cost
and latency) must be predicted.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .calibration import soft_label_expand


class _Constant:
    def __init__(self, value: float):
        self.value = float(value)

    def predict_proba(self, X):
        return np.column_stack([np.full(len(X), 1 - self.value), np.full(len(X), self.value)])

    def predict(self, X):
        return np.full(len(X), self.value)


def _quality_classifier(kind: str, seed: int):
    if kind == "hgb":
        return HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.08,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=seed,
        )
    if kind == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=3000))
    raise ValueError(f"unknown quality model {kind!r} (use 'hgb' or 'logreg')")


class QualityPredictor:
    """One classifier per model predicting expected quality, trained on soft labels."""

    def __init__(self, kind: str = "hgb", seed: int = 0):
        self.kind = kind
        self.seed = seed
        self.models_: list = []

    def fit(self, X: np.ndarray, Q: np.ndarray) -> QualityPredictor:
        self.models_ = []
        for j in range(Q.shape[1]):
            X2, y2, w2 = soft_label_expand(X, Q[:, j])
            if len(np.unique(y2)) < 2:
                self.models_.append(_Constant(Q[:, j].mean()))
                continue
            clf = _quality_classifier(self.kind, self.seed)
            if self.kind == "logreg":
                clf.fit(X2, y2, logisticregression__sample_weight=w2)
            else:
                clf.fit(X2, y2, sample_weight=w2)
            self.models_.append(clf)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.column_stack([m.predict_proba(X)[:, 1] for m in self.models_])


class ResponseLengthPredictor:
    """Per-model regressor of response length (chars), fitted in log space.

    Returns the *mean* (not median) via Duan's smearing estimate, because cost and latency are
    linear in response length.
    """

    def __init__(self, kind: str = "hgb", seed: int = 0):
        self.kind = kind
        self.seed = seed
        self.models_: list = []
        self.smear_: np.ndarray | None = None

    def _make(self):
        if self.kind == "hgb":
            return HistGradientBoostingRegressor(
                max_iter=200,
                learning_rate=0.08,
                early_stopping=True,
                n_iter_no_change=20,
                random_state=self.seed,
            )
        if self.kind == "ridge":
            return make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        raise ValueError(f"unknown length model {self.kind!r}")

    def fit(self, X: np.ndarray, R: np.ndarray) -> ResponseLengthPredictor:
        self.models_, smear = [], []
        for j in range(R.shape[1]):
            y = np.log1p(np.maximum(R[:, j], 0.0))
            reg = self._make().fit(X, y)
            resid = y - reg.predict(X)
            self.models_.append(reg)
            smear.append(float(np.mean(np.exp(resid))))
        self.smear_ = np.array(smear)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        logs = np.column_stack([m.predict(X) for m in self.models_])
        return np.maximum(np.expm1(logs) * self.smear_, 0.0)


class PriceSheet:
    """Per-model linear cost model ``cost = fixed + a * prompt_chars + b * response_chars``.

    Fitted on training data; in deployment this is simply the provider's public price list
    (tokens ~ chars / 4).
    """

    def __init__(self):
        self.coef_: np.ndarray | None = None  # (m, 3): fixed, per prompt char, per response char

    def fit(self, prompt_chars: np.ndarray, R: np.ndarray, C: np.ndarray) -> PriceSheet:
        coefs = []
        for j in range(C.shape[1]):
            X = np.column_stack([prompt_chars, R[:, j]])
            lr = LinearRegression(positive=True).fit(X, C[:, j])
            coefs.append([max(lr.intercept_, 0.0), *lr.coef_])
        self.coef_ = np.array(coefs)
        return self

    def predict(self, prompt_chars: np.ndarray, R_hat: np.ndarray) -> np.ndarray:
        c = self.coef_
        return c[:, 0][None, :] + np.outer(prompt_chars, c[:, 1]) + R_hat * c[:, 2][None, :]


class WinPredictor:
    """RouteLLM-style classifier: probability that the strong model beats the weak one."""

    def __init__(self):
        self.clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=3000))
        self.const_: float | None = None

    def fit(self, X: np.ndarray, q_strong: np.ndarray, q_weak: np.ndarray) -> WinPredictor:
        y = (q_strong > q_weak).astype(int)
        if len(np.unique(y)) < 2:
            self.const_ = float(y.mean())
        else:
            self.clf.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.const_ is not None:
            return np.full(len(X), self.const_)
        return self.clf.predict_proba(X)[:, 1]


def pick_weak_strong(Q: np.ndarray, C: np.ndarray, models: list[str]) -> tuple[int, int]:
    """Strong = best mean quality. Weak = cheapest model whose mean quality is at least the median.

    On RouterBench this picks GPT-4 vs. Mixtral-8x7B, the pair used in the RouteLLM paper.
    """
    mq, mc = Q.mean(0), C.mean(0)
    strong = int(np.argmax(mq))
    others = [j for j in range(len(models)) if j != strong]
    good = [j for j in others if mq[j] >= np.median(mq)] or others
    weak = int(min(good, key=lambda j: mc[j]))
    return weak, strong
