"""A small synthetic routing benchmark with known structure.

Used by the test-suite and by ``dacr run --data synthetic`` so the full pipeline can be tried in
seconds without downloading RouterBench. Model skill depends on the domain and on a per-query
difficulty that is visible in the prompt text, so a good router can beat every single model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_VOCAB = {
    "math-arithmetic": "add subtract multiply divide sum total number apples price remainder",
    "math-algebra": "solve equation variable polynomial root factor linear quadratic expression",
    "code-python": "python function list dict loop return implement class string parse",
    "code-sql": "sql query table join select where group index database schema",
    "trivia": "capital country river author year invented famous city mountain planet",
    "chat": "hello advice friend weekend recipe movie feel recommend plan travel",
}
_HARD_WORDS = "prove rigorous edge-case optimal multi-step formally derive complexity"

# (price $/Mchar in, price $/Mchar out, fixed $ per request)
_MODELS = {
    "tiny-chat": (0.03, 0.05, 1e-5),
    "small-coder": (0.15, 0.20, 5e-5),
    "medium-general": (0.40, 1.20, 1e-4),
    "large-frontier": (2.00, 6.00, 8e-4),
}
# base log-odds of success per (domain family, model)
_SKILL = {
    "math": [-1.0, -0.2, 0.6, 1.8],
    "code": [-1.5, 1.4, 0.4, 1.6],
    "trivia": [0.3, -0.3, 1.0, 1.6],
    "chat": [1.5, 0.2, 1.2, 1.6],
}
# how much difficulty hurts each model (bigger models degrade less)
_DIFFICULTY_SLOPE = [1.6, 1.3, 1.0, 0.5]


def make_synthetic_long(n_queries: int = 2400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    domains = list(_VOCAB)
    models = list(_MODELS)
    hard_words = _HARD_WORDS.split()
    rows = []
    for i in range(n_queries):
        dom = domains[rng.integers(len(domains))]
        words = _VOCAB[dom].split()
        difficulty = rng.uniform(0, 1)
        n_words = int(rng.integers(8, 40) * (1 + difficulty))
        toks = list(rng.choice(words, size=n_words))
        n_hard = rng.binomial(6, difficulty)
        toks += list(rng.choice(hard_words, size=n_hard))
        rng.shuffle(toks)
        prompt = " ".join(toks) + "?"
        fam = dom.split("-")[0]
        for j, m in enumerate(models):
            logit = _SKILL[fam][j] - _DIFFICULTY_SLOPE[j] * (2 * difficulty)
            p = 1 / (1 + np.exp(-logit))
            if fam == "chat":  # graded rubric score in [0, 1]
                quality = float(np.clip(np.round(rng.beta(1 + 6 * p, 1 + 6 * (1 - p)) * 10) / 10, 0, 1))
            else:
                quality = float(rng.random() < p)
            resp_chars = float(rng.lognormal(np.log(80 + 300 * (fam in ("code", "math")) * (j + 1) / 2), 0.6))
            p_in, p_out, fixed = _MODELS[m]
            cost = fixed + (p_in * len(prompt) + p_out * resp_chars) / 1e6
            rows.append((f"syn.{i}", prompt, dom, m, "", cost, quality, len(prompt), resp_chars))
    return pd.DataFrame(
        rows,
        columns=[
            "qid",
            "prompt",
            "domain",
            "model",
            "response",
            "cost",
            "quality",
            "prompt_len_chars",
            "resp_len_chars",
        ],
    )
