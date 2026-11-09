"""Online replay: route a stream of queries with delayed and missing quality feedback.

Event model (discrete-event simulation over the test queries):

* queries arrive as a Poisson process (optionally with bursts);
* when a query is routed to model ``a``, its cost and latency become known when the response
  completes (``t + latency``);
* a quality signal (user rating / judge score) arrives ``Exp(delay_mean_s)`` seconds after
  completion, or never (with probability ``missing_prob``).

All policies see exactly the same arrivals, latencies, delays and missing-feedback draws
(common random numbers), so differences between them are due to the policies alone.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .bandit import DualVariable, ResidualLinUCB


@dataclass
class StreamConfig:
    rate_qps: float = 2.0
    bursts: tuple[tuple[float, float], ...] = ()  # (qps, duration_s) phases before the base rate
    delay_mean_s: float = 60.0
    missing_prob: float = 0.2
    seed: int = 0


def parse_bursts(spec: str) -> tuple[tuple[float, float], ...]:
    """``"10@30,2@120"`` -> 10 qps for 30 s, then 2 qps for 120 s."""
    out = []
    for part in filter(None, (p.strip() for p in (spec or "").split(","))):
        qps, dur = part.split("@")
        out.append((float(qps), float(dur)))
    return tuple(out)


def arrival_times(n: int, cfg: StreamConfig, rng: np.random.Generator) -> np.ndarray:
    phases = list(cfg.bursts)
    times = np.empty(n)
    t, phase, phase_end = 0.0, 0, (phases[0][1] if phases else np.inf)
    for i in range(n):
        while phase < len(phases) and t >= phase_end:
            phase += 1
            phase_end = phase_end + phases[phase][1] if phase < len(phases) else np.inf
        rate = phases[phase][0] if phase < len(phases) else cfg.rate_qps
        t += rng.exponential(1.0 / rate)
        times[i] = t
    return times


@dataclass
class Context:
    z: np.ndarray  # bandit features
    prior: np.ndarray  # (m,) calibrated expected quality
    prior_raw: np.ndarray  # (m,) uncalibrated expected quality
    cost: np.ndarray  # (m,) predicted cost
    latency: np.ndarray  # (m,) predicted latency


class OnlinePolicy:
    name = "policy"

    def select(self, ctx: Context) -> int:
        raise NotImplementedError

    def observe_completion(self, arm: int, cost: float, latency: float) -> None:
        pass

    def observe_feedback(self, ctx: Context, arm: int, quality: float) -> None:
        pass

    def state(self) -> dict[str, float]:
        return {}


class StaticUtilityPolicy(OnlinePolicy):
    """Offline predict-then-optimize router with a fixed cost multiplier. Never learns."""

    def __init__(self, name: str, lam_cost: float, lam_latency: float = 0.0):
        self.name, self.lam_cost, self.lam_latency = name, lam_cost, lam_latency

    def select(self, ctx: Context) -> int:
        return int(np.argmax(ctx.prior - self.lam_cost * ctx.cost - self.lam_latency * ctx.latency))


class DACRPolicy(OnlinePolicy):
    """Domain-adaptive calibrated router.

    score(a) = prior(a) + residual_ucb(a) - lam_c * cost_hat(a) / B_c - lam_l * latency_hat(a) / B_l

    * ``prior``: ``"calibrated"`` (domain-adaptive calibrated predictor), ``"raw"`` (uncalibrated)
      or ``"none"`` (cold-start LinUCB).
    * ``learn``: whether the residual bandit learns from delayed feedback.
    * the dual variables ``lam_c`` / ``lam_l`` enforce average cost / latency budgets online.
    """

    def __init__(
        self,
        name: str,
        n_arms: int,
        dim: int,
        cost_budget: float,
        latency_budget: float | None = None,
        prior: str = "calibrated",
        learn: bool = True,
        alpha: float = 0.3,
        ridge: float = 1.0,
        eta: float = 0.01,
        lam_cost0: float = 0.0,
        lam_latency0: float = 0.0,
    ):
        if prior not in ("calibrated", "raw", "none"):
            raise ValueError(f"unknown prior {prior!r}")
        self.name, self.prior, self.learn = name, prior, learn
        self.bandit = ResidualLinUCB(n_arms, dim, alpha=alpha, ridge=ridge)
        self.dual_cost = DualVariable(cost_budget, eta=eta, lam0=lam_cost0)
        self.dual_lat = DualVariable(latency_budget, eta=eta, lam0=lam_latency0) if latency_budget else None

    def _base(self, ctx: Context) -> np.ndarray:
        if self.prior == "calibrated":
            return ctx.prior
        if self.prior == "raw":
            return ctx.prior_raw
        return np.zeros_like(ctx.prior)

    def select(self, ctx: Context) -> int:
        score = self._base(ctx).copy()
        if self.learn:
            score += self.bandit.ucb(ctx.z)
        score -= self.dual_cost.penalty(ctx.cost)
        if self.dual_lat is not None:
            score -= self.dual_lat.penalty(ctx.latency)
        return int(np.argmax(score))

    def observe_completion(self, arm: int, cost: float, latency: float) -> None:
        self.dual_cost.update(cost)
        if self.dual_lat is not None:
            self.dual_lat.update(latency)

    def observe_feedback(self, ctx: Context, arm: int, quality: float) -> None:
        if self.learn:
            self.bandit.update(arm, ctx.z, quality - self._base(ctx)[arm])

    def state(self) -> dict[str, float]:
        out = {"lam_cost": self.dual_cost.lam}
        if self.dual_lat is not None:
            out["lam_latency"] = self.dual_lat.lam
        return out


@dataclass
class Stream:
    """Everything the replay needs for ``n`` test queries, in arrival order."""

    Z: np.ndarray
    prior: np.ndarray
    prior_raw: np.ndarray
    cost_hat: np.ndarray
    latency_hat: np.ndarray
    quality: np.ndarray  # true
    cost: np.ndarray  # true
    latency: np.ndarray  # realized (simulated)
    times: np.ndarray
    feedback_delay: np.ndarray
    feedback_missing: np.ndarray
    order: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))  # index into the test set

    def context(self, i: int) -> Context:
        return Context(self.Z[i], self.prior[i], self.prior_raw[i], self.cost_hat[i], self.latency_hat[i])


def build_stream(arrays: dict[str, np.ndarray], cfg: StreamConfig) -> Stream:
    """Shuffle the test queries into a random arrival order and draw arrival/feedback randomness."""
    rng = np.random.default_rng(cfg.seed)
    n = len(arrays["quality"])
    order = rng.permutation(n)
    return Stream(
        **{k: v[order] for k, v in arrays.items()},
        times=arrival_times(n, cfg, rng),
        feedback_delay=rng.exponential(cfg.delay_mean_s, size=n) if cfg.delay_mean_s > 0 else np.zeros(n),
        feedback_missing=rng.random(n) < cfg.missing_prob,
        order=order,
    )


_COMPLETE, _FEEDBACK = 0, 1


def replay(policy: OnlinePolicy, stream: Stream) -> pd.DataFrame:
    """Run ``policy`` over the stream. Returns one row per query (in arrival order)."""
    n = len(stream.times)
    heap: list = []
    seq = 0
    arms = np.empty(n, dtype=int)
    lam_cost = np.full(n, np.nan)
    lam_lat = np.full(n, np.nan)
    feedback_used = 0
    for i in range(n):
        t = stream.times[i]
        while heap and heap[0][0] <= t:
            _, _, kind, payload = heapq.heappop(heap)
            if kind == _COMPLETE:
                policy.observe_completion(*payload)
            else:
                j, arm, q = payload
                policy.observe_feedback(stream.context(j), arm, q)
                feedback_used += 1
        ctx = stream.context(i)
        a = policy.select(ctx)
        arms[i] = a
        st = policy.state()
        lam_cost[i] = st.get("lam_cost", np.nan)
        lam_lat[i] = st.get("lam_latency", np.nan)
        lat = stream.latency[i, a]
        heapq.heappush(heap, (t + lat, seq, _COMPLETE, (a, stream.cost[i, a], lat)))
        seq += 1
        if not stream.feedback_missing[i]:
            done = t + lat + stream.feedback_delay[i]
            heapq.heappush(heap, (done, seq, _FEEDBACK, (i, a, stream.quality[i, a])))
            seq += 1
    rows = np.arange(n)
    return pd.DataFrame(
        {
            "t": stream.times,
            "query": stream.order,
            "arm": arms,
            "quality": stream.quality[rows, arms],
            "cost_usd": stream.cost[rows, arms],
            "latency_s": stream.latency[rows, arms],
            "lam_cost": lam_cost,
            "lam_latency": lam_lat,
        }
    ).assign(feedback_seen_by_end=feedback_used)
