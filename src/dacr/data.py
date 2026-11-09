"""Loading RouterBench-style data and turning it into dense (query x model) matrices.

RouterBench evaluates every query on every model, so the natural in-memory form is a set of
``(n_queries, n_models)`` matrices (quality, cost, response length) plus a query table.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

RESPONSE_SUFFIXES = [
    "model_response",
    "response",
    "output",
    "output_text",
    "completion",
    "completion_text",
    "text",
    "generation",
]
COST_SUFFIXES = ["total_cost", "cost", "cost_usd", "usd_cost"]
QUALITY_SUFFIXES = [
    "performance",
    "perf",
    "is_correct",
    "correct",
    "accuracy",
    "score",
    "final_score",
    "metric",
    "quality",
]
LONG_COLUMNS = ["qid", "prompt", "domain", "model", "response", "cost", "quality"]


# --------------------------------------------------------------------------------------
# Raw RouterBench -> long table
# --------------------------------------------------------------------------------------


def _ci_lookup(df: pd.DataFrame, name: str) -> str | None:
    lower = {str(c).lower(): c for c in df.columns}
    return lower.get(name.lower())


def _first_col(df: pd.DataFrame, names: list[str]) -> str | None:
    for n in names:
        c = _ci_lookup(df, n)
        if c is not None:
            return c
    return None


def _model_col(df: pd.DataFrame, model: str, suffixes: list[str]) -> str | None:
    return _first_col(df, [f"{model}|{s}" for s in suffixes])


def load_pickle_frame(path: str | Path) -> pd.DataFrame:
    """Load a RouterBench pickle. If it holds several frames, pick the most RouterBench-like one.

    Note: unpickling executes code from the file; only load files from a source you trust
    (the official ``withmartian/routerbench`` dataset on Hugging Face).
    """
    with Path(path).open("rb") as f:
        obj = pickle.load(f)  # noqa: S301 - RouterBench is distributed as pickles
    if isinstance(obj, pd.DataFrame):
        return obj
    if isinstance(obj, dict):
        frames = [v for v in obj.values() if isinstance(v, pd.DataFrame)]
        if not frames:
            raise TypeError(f"Pickle dict has no DataFrame values (keys={list(obj)[:20]})")

        def score(df: pd.DataFrame) -> int:
            cols = {str(c).lower() for c in df.columns}
            s = sum(5 for n in ("model", "model_name", "prompt", "performance", "total_cost") if n in cols)
            s += 10 * any(c.endswith("|model_response") for c in cols)
            return s

        return max(frames, key=score)
    raise TypeError(f"Unsupported pickle content: {type(obj)}")


def to_long_table(
    df: pd.DataFrame,
    domain_col: str = "eval_name",
    quality_col: str | None = None,
    skip_missing_quality: bool = True,
) -> pd.DataFrame:
    """Convert RouterBench (long or wide schema) to one row per (query, model).

    Output columns: ``qid, prompt, domain, model, response, cost, quality,
    prompt_len_chars, resp_len_chars``.
    """
    prompt_col = _first_col(df, ["prompt", "question", "query", "input", "instruction"])
    id_col = _first_col(df, ["sample_id", "id", "qid"])
    dom_col = _first_col(df, [domain_col, "domain"])
    if prompt_col is None or id_col is None:
        raise ValueError(f"Need an id and a prompt column; got columns {list(df.columns)[:30]}")

    model_col = _first_col(df, ["model", "model_name"])
    if model_col is not None:  # ---- long schema
        resp_col = _first_col(df, RESPONSE_SUFFIXES)
        cost_col = _first_col(df, COST_SUFFIXES)
        q_col = _first_col(df, [quality_col] if quality_col else []) or _first_col(df, QUALITY_SUFFIXES)
        missing = [n for n, c in [("response", resp_col), ("cost", cost_col), ("quality", q_col)] if c is None]
        if missing:
            raise ValueError(f"Long schema detected but missing columns: {missing}")
        out = pd.DataFrame(
            {
                "qid": df[id_col].astype(str),
                "prompt": df[prompt_col].astype(str),
                "domain": df[dom_col].astype(str) if dom_col else "unknown",
                "model": df[model_col].astype(str),
                "response": df[resp_col].astype(str),
                "cost": pd.to_numeric(df[cost_col], errors="coerce"),
                "quality": pd.to_numeric(df[q_col], errors="coerce"),
            }
        )
    else:  # ---- wide schema: "<model>|<suffix>" columns
        resp_suffixes = {s.lower() for s in RESPONSE_SUFFIXES}
        models = sorted(
            {
                str(c).split("|", 1)[0]
                for c in df.columns
                if "|" in str(c) and str(c).split("|", 1)[1].lower() in resp_suffixes
            }
        )
        if not models:
            raise ValueError("No '<model>|model_response' columns found in wide schema.")
        parts = []
        for m in models:
            resp_col = _model_col(df, m, RESPONSE_SUFFIXES)
            cost_col = _model_col(df, m, COST_SUFFIXES)
            q_col = _model_col(df, m, [quality_col]) if quality_col else None
            q_col = q_col or _model_col(df, m, QUALITY_SUFFIXES)
            if q_col is None and m in df.columns and pd.api.types.is_numeric_dtype(df[m]):
                q_col = m  # some RouterBench files store the score under the bare model name
            if resp_col is None or cost_col is None or q_col is None:
                if skip_missing_quality:
                    print(f"[data] skipping model {m!r}: missing response/cost/quality column")
                    continue
                raise ValueError(f"Model {m!r} is missing a response, cost or quality column")
            parts.append(
                pd.DataFrame(
                    {
                        "qid": df[id_col].astype(str),
                        "prompt": df[prompt_col].astype(str),
                        "domain": df[dom_col].astype(str) if dom_col else "unknown",
                        "model": m,
                        "response": df[resp_col].astype(str),
                        "cost": pd.to_numeric(df[cost_col], errors="coerce"),
                        "quality": pd.to_numeric(df[q_col], errors="coerce"),
                    }
                )
            )
        if not parts:
            raise ValueError("No model had response, cost and quality columns.")
        out = pd.concat(parts, ignore_index=True)

    out["prompt_len_chars"] = out["prompt"].str.len()
    out["resp_len_chars"] = out["response"].str.len()
    return out


def download_routerbench(hf_repo: str = "withmartian/routerbench", hf_file: str = "routerbench_raw.pkl") -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:  # pragma: no cover - optional dependency
        raise ImportError("Install the data extra: pip install 'dacr-router[data]'") from e
    return Path(hf_hub_download(repo_id=hf_repo, filename=hf_file, repo_type="dataset"))


# --------------------------------------------------------------------------------------
# Long table -> dense matrices
# --------------------------------------------------------------------------------------


@dataclass
class RoutingData:
    """Dense routing benchmark: ``n`` queries evaluated on ``m`` models."""

    queries: pd.DataFrame  # columns: qid, prompt, domain, prompt_len_chars (index 0..n-1)
    models: list[str]
    quality: np.ndarray  # (n, m) in [0, 1]
    cost: np.ndarray  # (n, m) USD
    response_chars: np.ndarray  # (n, m)

    @property
    def n_queries(self) -> int:
        return len(self.queries)

    @property
    def n_models(self) -> int:
        return len(self.models)

    @property
    def domains(self) -> np.ndarray:
        return self.queries["domain"].astype(str).to_numpy()

    @property
    def prompt_chars(self) -> np.ndarray:
        return self.queries["prompt_len_chars"].to_numpy(dtype=float)

    def subset(self, idx: np.ndarray) -> RoutingData:
        idx = np.asarray(idx)
        return RoutingData(
            queries=self.queries.iloc[idx].reset_index(drop=True),
            models=list(self.models),
            quality=self.quality[idx],
            cost=self.cost[idx],
            response_chars=self.response_chars[idx],
        )

    @classmethod
    def from_long(cls, long_df: pd.DataFrame, models: list[str] | None = None) -> RoutingData:
        df = long_df.copy()
        df["qid"] = df["qid"].astype(str)
        if "resp_len_chars" not in df.columns:
            df["resp_len_chars"] = df["response"].astype(str).str.len() if "response" in df else np.nan
        if "prompt_len_chars" not in df.columns:
            df["prompt_len_chars"] = df["prompt"].astype(str).str.len()
        models = sorted(df["model"].astype(str).unique()) if models is None else list(models)
        df = df[df["model"].isin(models)]

        def pivot(col: str) -> pd.DataFrame:
            return df.pivot_table(index="qid", columns="model", values=col, aggfunc="first").reindex(columns=models)

        q, c, r = pivot("quality"), pivot("cost"), pivot("resp_len_chars")
        complete = q.notna().all(axis=1) & c.notna().all(axis=1)
        dropped = int((~complete).sum())
        if dropped:
            print(f"[data] dropping {dropped} queries without quality/cost for every model")
        keep = q.index[complete]
        r = r.loc[keep]
        r = r.fillna(r.median()).fillna(0.0)

        queries = (
            df.drop_duplicates("qid").set_index("qid").loc[keep, ["prompt", "domain", "prompt_len_chars"]].reset_index()
        )
        queries["domain"] = queries["domain"].astype(str)
        return cls(
            queries=queries,
            models=models,
            quality=np.clip(q.loc[keep].to_numpy(float), 0.0, 1.0),
            cost=c.loc[keep].to_numpy(float),
            response_chars=r.to_numpy(float),
        )


def load_routing_data(path: str | Path, max_queries: int | None = None, seed: int = 0) -> RoutingData:
    """Load a long table (parquet/csv) produced by ``dacr prepare`` (or the ``synthetic`` demo)."""
    if str(path) == "synthetic":
        from .synthetic import make_synthetic_long

        long_df = make_synthetic_long(seed=seed)
    else:
        path = Path(path)
        if path.suffix == ".parquet":
            import pyarrow.parquet as pq

            available = set(pq.read_schema(path).names)
            wanted = ["qid", "prompt", "domain", "model", "cost", "quality", "prompt_len_chars", "resp_len_chars"]
            cols = [c for c in wanted if c in available]
            if "resp_len_chars" not in available:
                cols.append("response")  # needed to derive response lengths
            long_df = pd.read_parquet(path, columns=cols)
        else:
            long_df = pd.read_csv(path)
    data = RoutingData.from_long(long_df)
    if max_queries is not None and max_queries < data.n_queries:
        idx = np.sort(np.random.default_rng(seed).choice(data.n_queries, size=max_queries, replace=False))
        data = data.subset(idx)
    return data


# --------------------------------------------------------------------------------------
# Splits
# --------------------------------------------------------------------------------------


@dataclass
class Split:
    train: np.ndarray
    calib: np.ndarray
    test: np.ndarray
    kind: str
    holdout: tuple[str, ...] = ()

    def describe(self) -> str:
        extra = f" holdout={','.join(self.holdout)}" if self.holdout else ""
        return f"{self.kind}: train={len(self.train)} calib={len(self.calib)} test={len(self.test)}{extra}"


def make_split(
    data: RoutingData,
    kind: str = "random",
    test_frac: float = 0.2,
    calib_frac: float = 0.15,
    holdout: tuple[str, ...] | list[str] = (),
    seed: int = 0,
) -> Split:
    """Split queries into train / calibration / test.

    * ``random``: i.i.d. split of queries.
    * ``ood``: every query whose domain starts with one of ``holdout`` goes to test; the rest is
      split into train and calibration. Calibration data therefore never sees the test domains.
    """
    rng = np.random.default_rng(seed)
    n = data.n_queries
    if kind == "random":
        perm = rng.permutation(n)
        n_test = int(round(test_frac * n))
        test, pool = perm[:n_test], perm[n_test:]
    elif kind == "ood":
        holdout = tuple(h.strip() for h in holdout if h.strip())
        if not holdout:
            raise ValueError("OOD split needs at least one holdout domain prefix")
        domains = data.domains
        unmatched = [h for h in holdout if not any(d.startswith(h) for d in domains)]
        if unmatched:
            examples = ", ".join(sorted(set(domains))[:15])
            raise ValueError(f"Holdout prefixes {unmatched} match no domain. Examples: {examples} ...")
        is_test = np.array([any(d.startswith(h) for h in holdout) for d in domains])
        test = np.flatnonzero(is_test)
        pool = rng.permutation(np.flatnonzero(~is_test))
    else:
        raise ValueError(f"Unknown split kind {kind!r} (use 'random' or 'ood')")
    n_calib = int(round(calib_frac * len(pool)))
    calib, train = pool[:n_calib], pool[n_calib:]
    return Split(np.sort(train), np.sort(calib), np.sort(test), kind, tuple(holdout))
