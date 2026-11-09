import numpy as np

from dacr.metrics import pareto_mask, summarize
from dacr.policies import (
    Estimates,
    fit_lambda_for_budget,
    route_binary,
    route_cheapest,
    route_oracle,
    route_threshold,
    route_utility,
)


def _toy(rng, n=500):
    cost = np.tile([0.001, 0.01, 0.1], (n, 1)) * rng.uniform(0.5, 1.5, (n, 3))
    quality = (rng.random((n, 3)) < np.array([0.3, 0.6, 0.9])).astype(float)
    est = Estimates(np.clip(quality * 0.6 + 0.2, 0, 1), cost, np.ones((n, 3)))
    return quality, cost, est


def test_oracle_dominates_single_models(rng):
    Q, C, _ = _toy(rng)
    ch = route_oracle(Q, C)
    q = Q[np.arange(len(ch)), ch].mean()
    assert all(q >= Q[:, j].mean() for j in range(3))


def test_utility_extremes(rng):
    _, _, est = _toy(rng)
    np.testing.assert_array_equal(route_utility(est, 0.0), np.argmax(est.quality, axis=1))
    np.testing.assert_array_equal(route_utility(est, 1e9), route_cheapest(est))


def test_utility_latency_slo_mask():
    est = Estimates(np.array([[0.2, 0.9]]), np.zeros((1, 2)), np.array([[1.0, 20.0]]))
    assert route_utility(est, 0.0)[0] == 1
    assert route_utility(est, 0.0, latency_slo=5.0)[0] == 0


def test_threshold_picks_cheapest_passing():
    est = Estimates(np.array([[0.5, 0.8, 0.9], [0.1, 0.2, 0.3]]), np.zeros((2, 3)), np.zeros((2, 3)))
    ch = route_threshold(est, 0.7, cost_order=np.array([0, 1, 2]))
    assert list(ch) == [1, 2]  # second row: nothing passes -> best predicted


def test_binary_router():
    assert list(route_binary(np.array([0.2, 0.8]), weak=0, strong=2, threshold=0.5)) == [0, 2]


def test_lambda_for_budget_meets_budget(rng):
    Q, C, est = _toy(rng)
    budget = 0.02
    lam = fit_lambda_for_budget(est, C, budget)
    spent = C[np.arange(len(C)), route_utility(est, lam)].mean()
    assert spent <= budget * 1.001
    assert fit_lambda_for_budget(est, C, budget=10.0) == 0.0


def test_summarize_and_pareto(rng):
    Q, C, _ = _toy(rng)
    s = summarize(np.zeros(len(Q), dtype=int), Q, C, np.full(Q.shape, 3.0), slo_s=2.0)
    assert s["slo_violation"] == 1.0 and s["cost_usd"] < 0.002
    mask = pareto_mask(np.array([1, 2, 3, 2.5]), np.array([0.1, 0.5, 0.9, 0.4]))
    assert list(mask) == [True, True, True, False]
