"""Assigning queries to calibration groups ("domains").

* ``label``: use the dataset's domain label. Domains unseen at fit time fall back to the global
  calibrator, so this mode cannot adapt to genuinely new domains.
* ``cluster``: k-means on prompt embeddings ("pseudo-domains"). Needs no labels and assigns every
  new query, including ones from unseen domains, to its nearest cluster.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans


class DomainAssigner:
    def __init__(self, mode: str = "cluster", n_clusters: int = 24, seed: int = 0):
        if mode not in ("label", "cluster"):
            raise ValueError(f"unknown domain mode {mode!r}")
        self.mode = mode
        self.n_clusters = n_clusters
        self.seed = seed
        self.kmeans: KMeans | None = None

    def fit(self, embeddings: np.ndarray, labels: np.ndarray | None = None) -> DomainAssigner:
        if self.mode == "cluster":
            k = max(1, min(self.n_clusters, len(embeddings)))
            self.kmeans = KMeans(n_clusters=k, n_init=4, random_state=self.seed).fit(embeddings)
        return self

    def assign(self, embeddings: np.ndarray, labels: np.ndarray | None = None) -> np.ndarray:
        if self.mode == "label":
            if labels is None:
                raise ValueError("label mode needs domain labels")
            return np.asarray(labels).astype(str)
        assert self.kmeans is not None, "call fit() first"
        return np.char.add("c", self.kmeans.predict(embeddings).astype(str))
