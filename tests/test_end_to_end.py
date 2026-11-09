import json

import pandas as pd

from dacr.cli import main
from dacr.experiment import ExperimentConfig, run_experiment
from dacr.router import Router


def test_full_pipeline_on_synthetic(tmp_path):
    cfg = ExperimentConfig(data="synthetic", out=str(tmp_path / "run"), max_queries=800, online_seeds=(0,))
    summary = run_experiment(cfg)
    out = tmp_path / "run"
    for name in [
        "offline_metrics.csv",
        "online_metrics.csv",
        "calibration.csv",
        "predictors.csv",
        "per_domain.csv",
        "model_shares.csv",
        "summary.json",
        "summary.md",
        "frontier.png",
        "reliability.png",
        "online.png",
        "router.joblib",
    ]:
        assert (out / name).exists(), name

    offline = pd.read_csv(out / "offline_metrics.csv")
    singles = offline[offline.family == "single"]
    oracle = offline[offline.policy == "oracle"].iloc[0]
    assert oracle.quality >= singles.quality.max()

    online = pd.read_csv(out / "online_metrics.csv")
    dacr = online[online.policy == "dacr"].iloc[0]
    assert 0.8 < dacr.cost_vs_budget < 1.2
    assert json.loads((out / "summary.json").read_text())["budget_usd_per_query"] == summary["budget_usd_per_query"]

    router = Router.load(out / "router.joblib")
    picks = router.route(["solve the quadratic equation and prove it", "hello, recommend a movie"])
    assert all(p in router.models for p in picks)
    assert len(router.explain("sql join query")) == len(router.models)


def test_cli_ood_without_online(tmp_path):
    main(
        [
            "run",
            "--data",
            "synthetic",
            "--out",
            str(tmp_path / "ood"),
            "--split",
            "ood",
            "--holdout",
            "code",
            "--max-queries",
            "600",
            "--no-online",
            "--calibration",
            "label",
        ]
    )
    assert (tmp_path / "ood" / "offline_metrics.csv").exists()
    assert not (tmp_path / "ood" / "online_metrics.csv").exists()
