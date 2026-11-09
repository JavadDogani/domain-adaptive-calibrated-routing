"""Command-line interface: ``dacr prepare | run | route | profile-latency``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _csv(s: str) -> tuple[str, ...]:
    return tuple(x.strip() for x in s.split(",") if x.strip())


def _ints(s: str) -> tuple[int, ...]:
    return tuple(int(x) for x in _csv(s))


def cmd_prepare(a: argparse.Namespace) -> None:
    from .data import download_routerbench, load_pickle_frame, to_long_table

    pkl = Path(a.pkl) if a.pkl else download_routerbench(a.hf_repo, a.hf_file)
    print(f"[prepare] loading {pkl}")
    long_df = to_long_table(load_pickle_frame(pkl), domain_col=a.domain_col, quality_col=a.quality_col)
    if not a.keep_responses:
        long_df = long_df.drop(columns=["response"])
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    long_df.to_parquet(out, index=False)
    print(f"[prepare] wrote {out}: {long_df['qid'].nunique()} queries x {long_df['model'].nunique()} models")


def cmd_run(a: argparse.Namespace) -> None:
    from .experiment import ExperimentConfig, run_experiment

    cfg = ExperimentConfig(
        data=a.data,
        out=a.out,
        split=a.split,
        holdout=_csv(a.holdout),
        test_frac=a.test_frac,
        calib_frac=a.calib_frac,
        max_queries=a.max_queries,
        seed=a.seed,
        embedder=a.embedder,
        cache_dir=a.cache_dir,
        quality_model=a.quality_model,
        length_model=a.length_model,
        calibration=a.calibration,
        n_clusters=a.n_clusters,
        prior_strength=a.prior_strength,
        latency_profiles=a.latency_profiles,
        slo_s=a.slo,
        budget_frac=a.budget_frac,
        cost_budget=a.cost_budget,
        latency_budget=a.latency_budget,
        online=not a.no_online,
        online_seeds=_ints(a.online_seeds),
        rate_qps=a.rate_qps,
        bursts=a.bursts,
        delay_mean_s=a.delay_mean,
        missing_prob=a.missing_prob,
        bandit_features=a.bandit_features,
        bandit_dim=a.bandit_dim,
        alpha=a.alpha,
        ridge=a.ridge,
        eta=a.eta,
    )
    run_experiment(cfg)


def cmd_route(a: argparse.Namespace) -> None:
    import pandas as pd

    from .router import Router

    router = Router.load(a.router)
    prompts = a.prompt or [line for line in sys.stdin.read().splitlines() if line.strip()]
    with pd.option_context("display.width", 140, "display.float_format", "{:.5f}".format):
        for p in prompts:
            table = router.explain(p, a.lam_cost)
            print(f"\nprompt: {p[:100]!r}\n-> {table.iloc[0]['model']}\n{table.to_string(index=False)}")


def cmd_profile(a: argparse.Namespace) -> None:
    from .profiling import run_profiling

    run_profiling(a.endpoints, a.prompts, a.out, a.n_prompts, a.max_tokens, a.concurrency, a.seed)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="dacr", description="Domain-Adaptive Calibrated Routing (DACR) for LLMs.")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare", help="download RouterBench and convert it to a long parquet table")
    p.add_argument("--out", default="data/routerbench.parquet")
    p.add_argument("--hf-repo", default="withmartian/routerbench")
    p.add_argument("--hf-file", default="routerbench_raw.pkl")
    p.add_argument("--pkl", help="use a local RouterBench pickle instead of downloading")
    p.add_argument("--domain-col", default="eval_name")
    p.add_argument("--quality-col", default=None)
    p.add_argument("--keep-responses", action="store_true", help="keep full response text (large file)")
    p.set_defaults(func=cmd_prepare)

    r = sub.add_parser("run", help="train, calibrate and evaluate routers (offline + online replay)")
    r.add_argument("--data", default="synthetic", help="long table (.parquet/.csv) or 'synthetic'")
    r.add_argument("--out", default="results/run")
    g = r.add_argument_group("split")
    g.add_argument("--split", choices=["random", "ood"], default="random")
    g.add_argument("--holdout", default="", help="comma-separated domain prefixes held out for --split ood")
    g.add_argument("--test-frac", type=float, default=0.2)
    g.add_argument("--calib-frac", type=float, default=0.15)
    g.add_argument("--max-queries", type=int, default=None, help="subsample queries for a quick run")
    g.add_argument("--seed", type=int, default=0)
    g = r.add_argument_group("models")
    g.add_argument("--embedder", default=None, help="sentence-transformers model name or 'tfidf[:dims]'")
    g.add_argument("--cache-dir", default=".cache/dacr")
    g.add_argument("--quality-model", choices=["hgb", "logreg"], default="hgb")
    g.add_argument("--length-model", choices=["hgb", "ridge"], default="hgb")
    g.add_argument(
        "--calibration",
        choices=["none", "global", "label", "cluster"],
        default="cluster",
        help="offline calibration: global Platt, or per domain label / embedding cluster",
    )
    g.add_argument("--n-clusters", type=int, default=24)
    g.add_argument("--prior-strength", type=float, default=100.0)
    g.add_argument("--latency-profiles", default=None, help="YAML of per-model latency profiles")
    g = r.add_argument_group("constraints")
    g.add_argument("--slo", type=float, default=10.0, help="latency SLO in seconds (reporting)")
    g.add_argument(
        "--budget-frac", type=float, default=0.5, help="cost budget as a fraction of the best single model's mean cost"
    )
    g.add_argument("--cost-budget", type=float, default=None, help="absolute cost budget (USD/query)")
    g.add_argument("--latency-budget", type=float, default=None, help="mean-latency budget for DACR (s)")
    g = r.add_argument_group("online replay")
    g.add_argument("--no-online", action="store_true")
    g.add_argument("--online-seeds", default="0,1,2")
    g.add_argument("--rate-qps", type=float, default=2.0)
    g.add_argument("--bursts", default="", help='e.g. "10@30,2@120" (qps@seconds)')
    g.add_argument("--delay-mean", type=float, default=60.0, help="mean feedback delay (s)")
    g.add_argument("--missing-prob", type=float, default=0.2, help="probability feedback never arrives")
    g.add_argument("--bandit-features", choices=["pca", "cluster", "cluster+pca"], default="cluster")
    g.add_argument("--bandit-dim", type=int, default=16, help="PCA components in the bandit context")
    g.add_argument("--alpha", type=float, default=0.0, help="UCB exploration weight (residual-bandit variant)")
    g.add_argument("--ridge", type=float, default=50.0)
    g.add_argument("--eta", type=float, default=0.01, help="dual step size")
    r.set_defaults(func=cmd_run)

    q = sub.add_parser("route", help="route prompts with a saved router (results/<run>/router.joblib)")
    q.add_argument("--router", required=True)
    q.add_argument("--lam-cost", type=float, default=None, help="override the cost multiplier")
    q.add_argument("prompt", nargs="*", help="prompts (or read one per line from stdin)")
    q.set_defaults(func=cmd_route)

    f = sub.add_parser("profile-latency", help="measure OpenAI-compatible endpoints and fit latency profiles")
    f.add_argument("--endpoints", required=True, help="YAML list of endpoints (see configs/)")
    f.add_argument("--prompts", required=True, help=".parquet/.jsonl with a 'prompt' column, or .txt")
    f.add_argument("--out", default="configs/latency_profiles.measured.yaml")
    f.add_argument("--n-prompts", type=int, default=30)
    f.add_argument("--max-tokens", type=int, default=256)
    f.add_argument("--concurrency", type=int, default=1)
    f.add_argument("--seed", type=int, default=0)
    f.set_defaults(func=cmd_profile)
    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
