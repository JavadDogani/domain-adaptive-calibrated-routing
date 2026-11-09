import numpy as np
import pytest

from dacr.online import (
    DACRPolicy,
    OnlinePolicy,
    StaticUtilityPolicy,
    StreamConfig,
    arrival_times,
    build_stream,
    parse_bursts,
    replay,
)


def _arrays(rng, n=1500, m=3, d=4):
    q_true = np.array([0.4, 0.6, 0.9])
    return {
        "Z": np.hstack([np.ones((n, 1)), rng.normal(size=(n, d - 1))]),
        "prior": np.tile(q_true - 0.1, (n, 1)),
        "prior_raw": np.tile(q_true, (n, 1)),
        "cost_hat": np.tile([0.001, 0.005, 0.02], (n, 1)),
        "latency_hat": np.tile([1.0, 2.0, 4.0], (n, 1)),
        "quality": (rng.random((n, m)) < q_true).astype(float),
        "cost": np.tile([0.001, 0.005, 0.02], (n, 1)) * rng.uniform(0.5, 1.5, (n, m)),
        "latency": np.tile([1.0, 2.0, 4.0], (n, 1)),
    }


class _Recorder(OnlinePolicy):
    def __init__(self):
        self.feedback = 0
        self.completions = 0

    def select(self, ctx):
        return 0

    def observe_completion(self, arm, cost, latency):
        self.completions += 1

    def observe_feedback(self, ctx, arm, quality):
        self.feedback += 1


def test_parse_bursts_and_arrivals(rng):
    assert parse_bursts("10@30, 2@120") == ((10.0, 30.0), (2.0, 120.0))
    t = arrival_times(2000, StreamConfig(rate_qps=5.0, bursts=((50.0, 10.0),)), rng)
    assert np.all(np.diff(t) > 0)
    assert 350 < np.sum(t < 10.0) < 650  # ~500 arrivals during the 50 qps burst


def test_missing_feedback_is_never_delivered(rng):
    stream = build_stream(_arrays(rng, n=300), StreamConfig(missing_prob=1.0, delay_mean_s=0.0))
    pol = _Recorder()
    replay(pol, stream)
    assert pol.feedback == 0 and pol.completions > 250


def test_feedback_is_delayed(rng):
    arrays = _arrays(rng, n=300)
    fast, slow = _Recorder(), _Recorder()
    replay(fast, build_stream(arrays, StreamConfig(rate_qps=1.0, delay_mean_s=0.0, missing_prob=0.0)))
    replay(slow, build_stream(arrays, StreamConfig(rate_qps=1.0, delay_mean_s=1e4, missing_prob=0.0)))
    assert fast.feedback > 250 and slow.feedback < 30


def test_replay_is_deterministic(rng):
    arrays = _arrays(rng)
    cfg = StreamConfig(seed=7)
    a = replay(StaticUtilityPolicy("s", 10.0), build_stream(arrays, cfg))
    b = replay(StaticUtilityPolicy("s", 10.0), build_stream(arrays, cfg))
    assert a.equals(b)


@pytest.mark.parametrize("budget", [0.004, 0.01])
def test_dacr_spends_close_to_budget(rng, budget):
    arrays = _arrays(rng, n=3000)
    pol = DACRPolicy("dacr", n_arms=3, dim=4, cost_budget=budget, eta=0.02, alpha=0.1, ridge=10.0)
    trace = replay(pol, build_stream(arrays, StreamConfig(rate_qps=5.0, delay_mean_s=5.0, missing_prob=0.2)))
    assert 0.85 < trace["cost_usd"].mean() / budget < 1.15
    assert np.isfinite(trace["lam_cost"]).all()


def test_residual_bandit_corrects_a_biased_prior(rng):
    arrays = _arrays(rng, n=4000)
    # prior claims arm 2 is the worst; truth says it is the best. Loose budget, so quality decides.
    arrays["prior"] = np.tile([0.8, 0.6, 0.2], (4000, 1))
    cfg = StreamConfig(rate_qps=5.0, delay_mean_s=2.0, missing_prob=0.0)
    learn = DACRPolicy("l", 3, 4, cost_budget=1.0, learn=True, alpha=0.2, ridge=1.0)
    fixed = DACRPolicy("f", 3, 4, cost_budget=1.0, learn=False)
    q_learn = replay(learn, build_stream(arrays, cfg))["quality"].iloc[-1000:].mean()
    q_fixed = replay(fixed, build_stream(arrays, cfg))["quality"].iloc[-1000:].mean()
    assert q_learn > q_fixed + 0.3
