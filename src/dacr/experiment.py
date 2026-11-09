"""End-to-end experiment: train predictors, calibrate, evaluate offline routers, replay online."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from . import __version__
from .calibration import MatrixCalibrator, brier_score, expected_calibration_error, reliability_bins
from .data import RoutingData, Split, load_routing_data, make_split
from .domains import DomainAssigner
from .features import DEFAULT_EMBEDDER, FeatureBuilder, clean_prompt, make_embedder
from .latency import LatencyModel, load_profiles
from .metrics import model_shares, per_domain, summarize, summarize_outcomes
from .online import DACRPolicy, OnlinePolicy, StaticUtilityPolicy, StreamConfig, build_stream, parse_bursts, replay
from .plots import plot_frontier, plot_online, plot_reliability
from .policies import (
    Estimates,
    fit_lambda_for_budget,
    route_binary,
    route_cheapest,
    route_oracle,
    route_random,
    route_single,
    route_threshold,
    route_utility,
)
from .predictors import (
    PriceSheet,
    QualityPredictor,
    ResponseLengthPredictor,
    WinPredictor,
    pick_weak_strong,
)
from .router import Router

CALIBRATION_NAMES = {
    "none": "uncalibrated",
    "global": "global Platt",
    "label": "domain (labels)",
    "cluster": "domain (clusters)",
}


@dataclass
class ExperimentConfig:
    data: str = "synthetic"
    out: str = "results/run"
    split: str = "random"
    holdout: tuple[str, ...] = ()
    test_frac: float = 0.2
    calib_frac: float = 0.15
    max_queries: int | None = None
    seed: int = 0
    embedder: str | None = None  # default: MiniLM, or tfidf for the synthetic demo
    cache_dir: str = ".cache/dacr"
    quality_model: str = "hgb"
    length_model: str = "hgb"
    calibration: str = "cluster"  # none | global | label | cluster
    n_clusters: int = 24
    prior_strength: float = 100.0
    latency_profiles: str | None = None
    slo_s: float = 10.0
    budget_frac: float = 0.5
    cost_budget: float | None = None
    latency_budget: float | None = None
    online: bool = True
    online_seeds: tuple[int, ...] = (0, 1, 2)
    rate_qps: float = 2.0
    bursts: str = ""
    delay_mean_s: float = 60.0
    missing_prob: float = 0.2
    bandit_features: str = "cluster"  # pca | cluster | cluster+pca (for the residual-bandit variant)
    bandit_dim: int = 16
    alpha: float = 0.0
    ridge: float = 50.0
    eta: float = 0.01

    def resolved_embedder(self) -> str:
        if self.embedder:
            return self.embedder
        return "tfidf" if self.data == "synthetic" else DEFAULT_EMBEDDER


class _Log:
    def __init__(self, quiet: bool = False):
        self.t0 = time.time()
        self.quiet = quiet

    def __call__(self, msg: str) -> None:
        if not self.quiet:
            print(f"[{time.time() - self.t0:7.1f}s] {msg}", flush=True)


LAMBDA_GRID = np.concatenate([[0.0], np.logspace(0, 5, 26)])
THRESHOLD_GRID = np.round(np.linspace(0.05, 0.95, 19), 3)
WIN_GRID = np.round(np.linspace(0.0, 1.0, 21), 3)


@dataclass
class Prepared:
    """Everything fitted on train/calibration data, plus estimates for every query."""

    cfg: ExperimentConfig
    data: RoutingData
    split: Split
    features: FeatureBuilder
    E: np.ndarray  # embeddings
    X: np.ndarray  # predictor features
    quality_model: QualityPredictor
    P_raw: np.ndarray
    P_variants: dict[str, np.ndarray]
    calibrators: dict[str, tuple[MatrixCalibrator, DomainAssigner | None]]
    clusters: np.ndarray  # cluster id per query (str)
    length_model: ResponseLengthPredictor
    prices: PriceSheet
    latency_model: LatencyModel
    C_hat: np.ndarray
    L_hat: np.ndarray
    L: np.ndarray  # realized (simulated) latency
    budget: float
    best_model: int
    lam_budget: dict[str, float]  # per calibration variant

    @property
    def P(self) -> np.ndarray:
        return self.P_variants[CALIBRATION_NAMES[self.cfg.calibration]]


def prepare(cfg: ExperimentConfig, log: _Log | None = None) -> Prepared:
    log = log or _Log()
    if cfg.calibration not in CALIBRATION_NAMES:
        raise ValueError(f"calibration must be one of {list(CALIBRATION_NAMES)}")
    data = load_routing_data(cfg.data, cfg.max_queries, cfg.seed)
    split = make_split(data, cfg.split, cfg.test_frac, cfg.calib_frac, cfg.holdout, cfg.seed)
    tr, ca = split.train, split.calib
    Q, C, R = data.quality, data.cost, data.response_chars
    log(f"data: {data.n_queries} queries x {data.n_models} models | split {split.describe()}")

    texts = [clean_prompt(p) for p in data.queries["prompt"]]
    fb = FeatureBuilder(make_embedder(cfg.resolved_embedder()), cache_dir=cfg.cache_dir)
    fb.fit([texts[i] for i in tr])
    E = fb.embed(texts)
    X = fb.transform(texts, E)
    log(f"features: {X.shape[1]} dims ({fb.embedder.name})")

    qp = QualityPredictor(cfg.quality_model, cfg.seed).fit(X[tr], Q[tr])
    P_raw = qp.predict(X)
    log("quality predictor fitted")

    label_assigner = DomainAssigner("label").fit(E[tr], data.domains[tr])
    cluster_assigner = DomainAssigner("cluster", cfg.n_clusters, cfg.seed).fit(E[tr])
    groups = {"label": label_assigner.assign(E, data.domains), "cluster": cluster_assigner.assign(E)}
    calibrators: dict[str, tuple[MatrixCalibrator, DomainAssigner | None]] = {
        "none": (MatrixCalibrator("none"), None),
        "global": (MatrixCalibrator("global"), None),
        "label": (MatrixCalibrator("group", prior_strength=cfg.prior_strength), label_assigner),
        "cluster": (MatrixCalibrator("group", prior_strength=cfg.prior_strength), cluster_assigner),
    }
    P_variants = {}
    for key, (cal, _) in calibrators.items():
        g = groups.get(key)
        cal.fit(P_raw[ca], Q[ca], None if g is None else g[ca])
        P_variants[CALIBRATION_NAMES[key]] = cal.predict(P_raw, g)

    lp = ResponseLengthPredictor(cfg.length_model, cfg.seed).fit(X[tr], R[tr])
    R_hat = lp.predict(X)
    prices = PriceSheet().fit(data.prompt_chars[tr], R[tr], C[tr])
    C_hat = prices.predict(data.prompt_chars, R_hat)
    lat_model = LatencyModel(data.models, load_profiles(data.models, cfg.latency_profiles))
    L_hat = lat_model.expected(data.prompt_chars, R_hat)
    L = lat_model.sample(data.prompt_chars, R, np.random.default_rng(cfg.seed + 1))
    log("cost/latency estimators fitted")

    best_model = int(np.argmax(Q[tr].mean(0)))
    budget = cfg.cost_budget if cfg.cost_budget else cfg.budget_frac * float(C[ca, best_model].mean())
    lam_budget = {
        name: fit_lambda_for_budget(Estimates(Pv, C_hat, L_hat).subset(ca), C[ca], budget)
        for name, Pv in P_variants.items()
    }
    return Prepared(
        cfg,
        data,
        split,
        fb,
        E,
        X,
        qp,
        P_raw,
        P_variants,
        calibrators,
        groups["cluster"],
        lp,
        prices,
        lat_model,
        C_hat,
        L_hat,
        L,
        budget,
        best_model,
        lam_budget,
    )


# --------------------------------------------------------------------------------------
# Online replay
# --------------------------------------------------------------------------------------


def bandit_context(prep: Prepared, kind: str, dim: int) -> np.ndarray:
    """Context vectors for the residual bandit: intercept + cluster one-hot and/or whitened PCA."""
    parts = [np.ones((prep.data.n_queries, 1))]
    if "cluster" in kind:
        ids = np.unique(prep.clusters[prep.split.train])
        parts.append((prep.clusters[:, None] == ids[None, :]).astype(float))
    if "pca" in kind:
        tr = prep.split.train
        pca = PCA(n_components=min(dim, prep.E.shape[1], len(tr)), whiten=True, random_state=prep.cfg.seed)
        pca.fit(prep.E[tr])
        parts.append(pca.transform(prep.E))
    if kind not in ("pca", "cluster", "cluster+pca"):
        raise ValueError(f"unknown bandit features {kind!r}")
    return np.hstack(parts)


def online_arrays(prep: Prepared, Z: np.ndarray) -> dict[str, np.ndarray]:
    te = prep.split.test
    return {
        "Z": Z[te],
        "prior": prep.P[te],
        "prior_raw": prep.P_raw[te],
        "cost_hat": prep.C_hat[te],
        "latency_hat": prep.L_hat[te],
        "quality": prep.data.quality[te],
        "cost": prep.data.cost[te],
        "latency": prep.L[te],
    }


def default_online_policies(prep: Prepared, dim: int) -> list[OnlinePolicy]:
    cfg, B = prep.cfg, prep.budget
    lam = prep.lam_budget[CALIBRATION_NAMES[cfg.calibration]]
    lam_raw = prep.lam_budget["uncalibrated"]
    common = {
        "n_arms": prep.data.n_models,
        "dim": dim,
        "cost_budget": B,
        "latency_budget": cfg.latency_budget,
        "alpha": cfg.alpha,
        "ridge": cfg.ridge,
        "eta": cfg.eta,
    }
    cold = {**common, "alpha": 0.2, "ridge": 1.0}  # a cold-start bandit needs exploration to learn anything
    return [
        DACRPolicy("dacr", prior="calibrated", learn=False, lam_cost0=lam * B, **common),
        StaticUtilityPolicy("static@budget", lam),
        DACRPolicy("dacr + residual bandit", prior="calibrated", learn=True, lam_cost0=lam * B, **common),
        DACRPolicy("dacr w/o calibration", prior="raw", learn=False, lam_cost0=lam_raw * B, **common),
        DACRPolicy("linucb (cold start)", prior="none", learn=True, lam_cost0=lam * B, **cold),
    ]


def run_online(prep: Prepared, make_policies=None, seeds=None) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Replay the test stream for every policy and seed. Returns per-run metrics and seed-0 traces."""
    cfg = prep.cfg
    Z = bandit_context(prep, cfg.bandit_features, cfg.bandit_dim)
    arrays = online_arrays(prep, Z)
    make_policies = make_policies or (lambda: default_online_policies(prep, Z.shape[1]))
    seeds = cfg.online_seeds if seeds is None else seeds
    rows, traces = [], {}
    for s in seeds:
        stream_cfg = StreamConfig(cfg.rate_qps, parse_bursts(cfg.bursts), cfg.delay_mean_s, cfg.missing_prob, seed=s)
        stream = build_stream(arrays, stream_cfg)
        for pol in make_policies():
            trace = replay(pol, stream)
            res = summarize_outcomes(
                trace["quality"].to_numpy(), trace["cost_usd"].to_numpy(), trace["latency_s"].to_numpy(), cfg.slo_s
            )
            rows.append({"policy": pol.name, "seed": s, **res, "cost_vs_budget": res["cost_usd"] / prep.budget})
            if s == seeds[0]:
                traces[pol.name] = trace
    return pd.DataFrame(rows), traces


# --------------------------------------------------------------------------------------
# Full experiment with reports
# --------------------------------------------------------------------------------------


def run_experiment(cfg: ExperimentConfig) -> dict:
    log = _Log()
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    prep = prepare(cfg, log)
    data, split = prep.data, prep.split
    tr, te = split.train, split.test
    models = data.models
    Q, C = data.quality, data.cost
    Qt, Ct, Lt = Q[te], C[te], prep.L[te]
    budget, best_model = prep.budget, prep.best_model
    main_cal = CALIBRATION_NAMES[cfg.calibration]
    lam_b = prep.lam_budget[main_cal]

    # ------------------------------------------------------------------ calibration report
    cal_rows, curves = [], {}
    for name in [main_cal] + [k for k in prep.P_variants if k != main_cal]:
        Pv = prep.P_variants[name]
        for j, m in enumerate(models):
            cal_rows.append(
                {
                    "calibration": name,
                    "model": m,
                    "ece": expected_calibration_error(Pv[te, j], Q[te, j]),
                    "brier": brier_score(Pv[te, j], Q[te, j]),
                }
            )
        pooled_p, pooled_y = Pv[te].ravel(), Q[te].ravel()
        conf, obs, _ = reliability_bins(pooled_p, pooled_y)
        ece = expected_calibration_error(pooled_p, pooled_y)
        curves[name] = (conf, obs, ece)
        cal_rows.append({"calibration": name, "model": "ALL", "ece": ece, "brier": brier_score(pooled_p, pooled_y)})
    pd.DataFrame(cal_rows).to_csv(out / "calibration.csv", index=False)
    plot_reliability(curves, out / "reliability.png")
    log("calibration (test ECE): " + ", ".join(f"{k} {v[2]:.3f}" for k, v in curves.items()))

    pred_rows = [
        {
            "model": m,
            "cost_ape_median": float(np.median(np.abs(prep.C_hat[te, j] - Ct[:, j]) / np.maximum(Ct[:, j], 1e-12))),
            "latency_ape_median": float(np.median(np.abs(prep.L_hat[te, j] - Lt[:, j]) / Lt[:, j])),
            "mean_cost_true": float(Ct[:, j].mean()),
            "mean_cost_pred": float(prep.C_hat[te, j].mean()),
            "mean_latency_true": float(Lt[:, j].mean()),
            "mean_latency_pred": float(prep.L_hat[te, j].mean()),
        }
        for j, m in enumerate(models)
    ]
    pd.DataFrame(pred_rows).to_csv(out / "predictors.csv", index=False)
    log(f"cost budget ${budget:.6f}/query ({cfg.budget_frac:.0%} of {models[best_model]}), lambda={lam_b:.1f}")

    # ------------------------------------------------------------------ offline evaluation
    rng = np.random.default_rng(cfg.seed)
    est_t = Estimates(prep.P, prep.C_hat, prep.L_hat).subset(te)
    est_raw_t = Estimates(prep.P_raw, prep.C_hat, prep.L_hat).subset(te)
    n_t, m = len(te), len(models)
    rows, choices = [], {}

    def add(policy: str, family: str, knob: float | None, choice: np.ndarray) -> None:
        rows.append({"policy": policy, "family": family, "knob": knob, **summarize(choice, Qt, Ct, Lt, cfg.slo_s)})
        choices[policy] = choice

    for j, name in enumerate(models):
        add(f"single:{name}", "single", None, route_single(n_t, j))
    add("random", "baseline", None, route_random(n_t, m, rng))
    add("cheapest (predicted)", "baseline", None, route_cheapest(est_t))
    add("oracle", "upper-bound", None, route_oracle(Qt, Ct))
    cost_order = np.argsort(C[tr].mean(0))
    for t in THRESHOLD_GRID:
        add(f"threshold@{t}", "threshold", float(t), route_threshold(est_t, t, cost_order))
    weak, strong = pick_weak_strong(Q[tr], C[tr], models)
    p_win = WinPredictor().fit(prep.X[tr], Q[tr, strong], Q[tr, weak]).predict(prep.X[te])
    for t in WIN_GRID:
        add(f"routellm@{t}", "routellm", float(t), route_binary(p_win, weak, strong, t))
    for lam in LAMBDA_GRID:
        add(f"utility@{lam:.3g}", "utility", float(lam), route_utility(est_t, lam))
    add("utility@budget", "budget", lam_b, route_utility(est_t, lam_b))
    lam_raw = prep.lam_budget["uncalibrated"]
    add("utility@budget (uncalibrated)", "budget", lam_raw, route_utility(est_raw_t, lam_raw))
    offline = pd.DataFrame(rows)
    offline.to_csv(out / "offline_metrics.csv", index=False)
    log(f"offline: evaluated {len(offline)} policy settings (RouteLLM pair: {models[weak]} vs {models[strong]})")

    # ------------------------------------------------------------------ online replay
    online_summary = None
    if cfg.online:
        on_df, traces = run_online(prep)
        on_df.to_csv(out / "online_runs.csv", index=False)
        metric_cols = [c for c in on_df.columns if c not in ("policy", "seed", "n")]
        g = on_df.groupby("policy", sort=False)[metric_cols]
        online_summary = g.mean().join(g.std().add_suffix("_std")).reset_index()
        online_summary.to_csv(out / "online_metrics.csv", index=False)
        pd.concat([t.assign(policy=k) for k, t in traces.items()]).to_parquet(out / "online_trace.parquet")
        plot_online(traces, budget, out / "online.png")
        dacr_choice = np.empty(n_t, dtype=int)  # seed-0 decisions mapped back to test-set order
        dacr_choice[traces["dacr"]["query"].to_numpy()] = traces["dacr"]["arm"].to_numpy()
        choices["dacr (online, seed 0)"] = dacr_choice
        log(f"online replay done ({len(cfg.online_seeds)} seeds)")

    plot_frontier(offline, online_summary, budget, out / "frontier.png")

    # ------------------------------------------------------------------ per-domain + shares
    keep = [f"single:{models[best_model]}", "oracle", "utility@budget"]
    keep += ["dacr (online, seed 0)"] if cfg.online else []
    dom_t = data.domains[te]
    per_dom = pd.concat([per_domain(choices[k], Qt, Ct, dom_t).assign(policy=k) for k in keep]).reset_index()
    per_dom.to_csv(out / "per_domain.csv", index=False)
    pd.DataFrame({k: model_shares(choices[k], models) for k in keep}).to_csv(out / "model_shares.csv")

    # ------------------------------------------------------------------ summary + router
    headline = _headline(offline, online_summary, models, best_model)
    summary = {
        "dacr_version": __version__,
        "config": {**asdict(cfg), "embedder": cfg.resolved_embedder()},
        "data": {
            "n_queries": data.n_queries,
            "models": models,
            "split": split.describe(),
            "test_domains": int(len(np.unique(dom_t))),
        },
        "budget_usd_per_query": budget,
        "lambda_budget": lam_b,
        "routellm_pair": {"weak": models[weak], "strong": models[strong]},
        "calibration_ece_test": {k: v[2] for k, v in curves.items()},
        "headline": headline.to_dict(orient="records"),
        "runtime_s": round(time.time() - log.t0, 1),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    (out / "summary.md").write_text(_markdown(summary, headline))

    cal, assigner = prep.calibrators[cfg.calibration]
    router = Router(
        models,
        prep.features,
        prep.quality_model,
        assigner or prep.calibrators["cluster"][1],
        cal,
        prep.length_model,
        prep.prices,
        prep.latency_model,
        lam_cost=lam_b,
    )
    router.save(out / "router.joblib")
    log(f"done -> {out}")
    print(headline.to_string(index=False))
    return summary


def _headline(offline: pd.DataFrame, online: pd.DataFrame | None, models: list[str], best: int) -> pd.DataFrame:
    cols = ["quality", "cost_usd", "latency_p95_s", "slo_violation"]
    pick = {
        f"best single model ({models[best].split('/')[-1]})": f"single:{models[best]}",
        "random": "random",
        "cheapest (predicted)": "cheapest (predicted)",
        "oracle (upper bound)": "oracle",
        "RouteLLM-style @0.5": "routellm@0.5",
        "static router @ budget (uncalibrated)": "utility@budget (uncalibrated)",
        "static router @ budget": "utility@budget",
    }
    rows = []
    for label, key in pick.items():
        r = offline[offline["policy"] == key]
        if len(r):
            rows.append({"policy": label, "mode": "offline", **r.iloc[0][cols].to_dict()})
    if online is not None:
        for _, r in online.iterrows():
            rows.append({"policy": r["policy"], "mode": "online", **{c: r[c] for c in cols}})
    return pd.DataFrame(rows)


def _markdown(summary: dict, headline: pd.DataFrame) -> str:
    lines = [
        f"Data: {summary['data']['n_queries']} queries x {len(summary['data']['models'])} models; "
        f"split {summary['data']['split']}",
        f"Cost budget: ${summary['budget_usd_per_query']:.5f} per query",
        "",
        "| policy | mode | quality | cost (USD/query) | p95 latency (s) | SLO violations |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for _, r in headline.iterrows():
        lines.append(
            f"| {r['policy']} | {r['mode']} | {r['quality']:.3f} | {r['cost_usd']:.5f} | "
            f"{r['latency_p95_s']:.1f} | {r['slo_violation']:.1%} |"
        )
    lines += [
        "",
        "Calibration (pooled ECE on test): "
        + ", ".join(f"{k} {v:.3f}" for k, v in summary["calibration_ece_test"].items()),
    ]
    return "\n".join(lines) + "\n"
