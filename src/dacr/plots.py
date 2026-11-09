"""Static figures (matplotlib) for experiment reports."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Validated categorical palette (fixed order) and chart chrome, light surface.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "h"]

# One fixed color + marker per policy, shared by every figure (color follows the entity).
POLICY_STYLE = {
    "utility": (SERIES[0], "o"),
    "static@budget": (SERIES[0], "s"),
    "threshold": (SERIES[1], "s"),
    "routellm": (SERIES[2], "^"),
    "dacr w/o calibration": (SERIES[3], "v"),
    "dacr": (SERIES[4], "D"),
    "dacr + residual bandit": (SERIES[6], "P"),
    "linucb (cold start)": (SERIES[7], "X"),
}


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(1)
    ax.tick_params(colors=MUTED, labelcolor=INK2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)
    ax.title.set_color(INK)


def _legend(ax, **kw):
    leg = ax.legend(frameon=False, fontsize=9, labelcolor=INK2, **kw)
    return leg


def _short(model: str) -> str:
    return model.split("/")[-1]


def plot_frontier(offline: pd.DataFrame, online: pd.DataFrame | None, budget: float, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.2), facecolor=SURFACE)
    _style(ax)

    singles = offline[offline["family"] == "single"]
    ax.scatter(
        singles["cost_usd"],
        singles["quality"],
        s=36,
        color=MUTED,
        edgecolor=SURFACE,
        linewidth=1.5,
        zorder=3,
        label="single model",
    )
    best = singles.sort_values("quality").iloc[-1]
    ax.scatter(
        [best["cost_usd"]],
        [best["quality"]],
        s=70,
        facecolor="none",
        edgecolor=INK,
        linewidth=1.5,
        zorder=5,
        label=f"best single model ({_short(best['policy'].removeprefix('single:'))})",
    )
    cheap = singles.sort_values("cost_usd").iloc[0]
    ax.annotate(
        _short(cheap["policy"].removeprefix("single:")),
        (cheap["cost_usd"], cheap["quality"]),
        textcoords="offset points",
        xytext=(6, -12),
        fontsize=8,
        color=INK2,
    )

    sweeps = [
        ("utility", "predict-then-optimize (λ sweep)"),
        ("threshold", "threshold (sweep)"),
        ("routellm", "RouteLLM-style (sweep)"),
    ]
    for fam, label in sweeps:
        d = offline[offline["family"] == fam].sort_values("cost_usd")
        if len(d):
            color, marker = POLICY_STYLE[fam]
            ax.plot(
                d["cost_usd"],
                d["quality"],
                color=color,
                linewidth=2,
                marker=marker,
                markersize=4,
                solid_capstyle="round",
                label=label,
                zorder=4,
            )

    ora = offline[offline["policy"] == "oracle"]
    if len(ora):
        ax.scatter(
            ora["cost_usd"],
            ora["quality"],
            marker="*",
            s=180,
            color=INK,
            edgecolor=SURFACE,
            linewidth=1.5,
            zorder=6,
            label="oracle (upper bound)",
        )

    at_budget = offline[offline["policy"] == "utility@budget"]
    if len(at_budget):
        ax.scatter(
            at_budget["cost_usd"],
            at_budget["quality"],
            marker=POLICY_STYLE["static@budget"][1],
            s=70,
            color=POLICY_STYLE["static@budget"][0],
            edgecolor=SURFACE,
            linewidth=2,
            zorder=6,
            label="static router @ budget",
        )
    online_points = [("dacr", "DACR (online)"), ("dacr + residual bandit", "DACR + residual bandit (online)")]
    for name, label in online_points if online is not None else []:
        if name in online["policy"].values:
            r = online[online["policy"] == name].iloc[0]
            color, marker = POLICY_STYLE[name]
            ax.scatter(
                [r["cost_usd"]],
                [r["quality"]],
                marker=marker,
                s=90,
                color=color,
                edgecolor=SURFACE,
                linewidth=2,
                zorder=8 if name == "dacr" else 7,
                label=label,
            )

    ax.axvline(budget, color=AXIS, linewidth=1, zorder=1)
    ax.annotate(
        "cost budget", (budget, ax.get_ylim()[0]), textcoords="offset points", xytext=(4, 6), fontsize=8, color=MUTED
    )
    ax.set_xscale("log")
    ax.set_xlabel("Mean cost per query (USD, log scale)")
    ax.set_ylabel("Mean quality")
    ax.set_title("Quality vs. cost on the test set", loc="left", fontsize=12)
    _legend(ax, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def plot_reliability(curves: dict[str, tuple[np.ndarray, np.ndarray, float]], path: Path) -> None:
    """``curves``: name -> (mean predicted per bin, mean observed per bin, ECE)."""
    fig, ax = plt.subplots(figsize=(5.4, 5.2), facecolor=SURFACE)
    _style(ax)
    ax.plot([0, 1], [0, 1], color=AXIS, linewidth=1, zorder=1)
    for k, (name, (conf, obs, ece)) in enumerate(curves.items()):
        ax.plot(
            conf,
            obs,
            color=SERIES[k],
            linewidth=2,
            marker=MARKERS[k],
            markersize=5,
            markeredgecolor=SURFACE,
            label=f"{name}  (ECE {ece:.3f})",
            zorder=3,
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted quality")
    ax.set_ylabel("Observed quality")
    ax.set_title("Calibration on the test set (all models pooled)", loc="left", fontsize=11)
    _legend(ax, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def plot_online(traces: dict[str, pd.DataFrame], budget: float, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE)
    for ax in axes:
        _style(ax)
    n = max(len(t) for t in traces.values())
    start = min(100, n // 10)  # the first few running means are too noisy to read
    for k, (name, t) in enumerate(traces.items()):
        x = np.arange(1, len(t) + 1)
        run_q = (t["quality"].cumsum() / x)[start:]
        run_cost = (t["cost_usd"].cumsum() / x / budget)[start:]
        color = POLICY_STYLE.get(name, (SERIES[k % len(SERIES)], "o"))[0]
        style = {"color": color, "linewidth": 2 if name == "dacr" else 1.5, "label": name}
        axes[0].plot(x[start:], run_q, **style)
        axes[1].plot(x[start:], run_cost, **style)
    axes[1].axhline(1.0, color=AXIS, linewidth=1)
    axes[1].annotate("budget", (n, 1.0), textcoords="offset points", xytext=(-40, 4), fontsize=8, color=MUTED)
    axes[0].set_title("Running mean quality", loc="left", fontsize=11)
    axes[1].set_title("Running mean cost / budget", loc="left", fontsize=11)
    for ax in axes:
        ax.set_xlabel("Queries routed")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, fontsize=9, labelcolor=INK2)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
