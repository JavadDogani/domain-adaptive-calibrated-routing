import numpy as np

from dacr.bandit import DualVariable, ResidualLinUCB


def test_sherman_morrison_matches_explicit_inverse(rng):
    b = ResidualLinUCB(n_arms=2, dim=4, ridge=2.0)
    A = 2.0 * np.eye(4)
    for _ in range(25):
        z = rng.normal(size=4)
        b.update(1, z, rng.normal())
        A += np.outer(z, z)
    np.testing.assert_allclose(b.A_inv[1], np.linalg.inv(A), atol=1e-8)
    np.testing.assert_allclose(b.A_inv[0], np.eye(4) / 2.0)


def test_linucb_learns_best_arm(rng):
    d, arms = 5, 3
    theta = rng.normal(size=(arms, d))
    bandit = ResidualLinUCB(arms, d, alpha=0.5, ridge=1.0)
    correct = []
    for t in range(3000):
        z = rng.normal(size=d)
        a = int(np.argmax(bandit.ucb(z)))
        bandit.update(a, z, float(theta[a] @ z + 0.1 * rng.normal()))
        if t >= 2500:
            correct.append(a == int(np.argmax(theta @ z)))
    assert np.mean(correct) > 0.9


def test_dual_variable_tracks_budget():
    dv = DualVariable(budget=1.0, eta=0.1)
    for _ in range(10):
        dv.update(3.0)  # overspending
    assert dv.lam > 1.0
    high = dv.lam
    for _ in range(10):
        dv.update(0.0)  # underspending
    assert dv.lam < high
    for _ in range(100):
        dv.update(0.0)
    assert dv.lam == 0.0  # projected onto lam >= 0
