import numpy as np

from dacr.calibration import (
    GroupCalibrator,
    MatrixCalibrator,
    PlattCalibrator,
    brier_score,
    expected_calibration_error,
    soft_label_expand,
)


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _miscalibrated(rng, n, temp):
    true_logit = rng.normal(0, 1.5, n)
    y = (rng.random(n) < _sigmoid(true_logit)).astype(float)
    return _sigmoid(temp * true_logit), y  # temp > 1: overconfident


def test_soft_label_expand():
    X = np.arange(3).reshape(-1, 1)
    X2, y2, w2 = soft_label_expand(X, np.array([1.0, 0.0, 0.25]))
    assert len(X2) == 4  # one row each for hard labels, two for the soft one
    np.testing.assert_allclose(sorted(w2[X2[:, 0] == 2]), [0.25, 0.75])
    assert w2.sum() == 3


def test_ece_extremes():
    assert expected_calibration_error(np.ones(10), np.zeros(10)) == 1.0
    assert expected_calibration_error(np.full(10, 0.5), np.array([0, 1] * 5)) == 0.0


def test_platt_fixes_overconfidence(rng):
    p, y = _miscalibrated(rng, 6000, temp=3.0)
    cal = PlattCalibrator().fit(p[:3000], y[:3000])
    before = expected_calibration_error(p[3000:], y[3000:])
    after = expected_calibration_error(cal.predict(p[3000:]), y[3000:])
    assert after < before / 3
    assert brier_score(cal.predict(p[3000:]), y[3000:]) < brier_score(p[3000:], y[3000:])


def test_platt_single_class_is_constant():
    cal = PlattCalibrator().fit(np.array([0.2, 0.9]), np.array([1.0, 1.0]))
    np.testing.assert_allclose(cal.predict(np.array([0.1, 0.5])), [1.0, 1.0])


def test_group_calibration_beats_global_when_groups_differ(rng):
    p_a, y_a = _miscalibrated(rng, 4000, temp=3.0)  # overconfident group
    p_b, y_b = _miscalibrated(rng, 4000, temp=0.4)  # underconfident group
    p, y = np.concatenate([p_a, p_b]), np.concatenate([y_a, y_b])
    g = np.array(["a"] * 4000 + ["b"] * 4000)
    tr = rng.random(8000) < 0.5
    glob = GroupCalibrator().fit(p[tr], y[tr])
    grp = GroupCalibrator(prior_strength=10).fit(p[tr], y[tr], g[tr])
    for name in ("a", "b"):
        te = (~tr) & (g == name)
        e_glob = expected_calibration_error(glob.predict(p[te]), y[te])
        e_grp = expected_calibration_error(grp.predict(p[te], g[te]), y[te])
        assert e_grp < e_glob


def test_unseen_group_falls_back_to_global(rng):
    p, y = _miscalibrated(rng, 1000, temp=2.0)
    cal = GroupCalibrator().fit(p, y, np.array(["a"] * 1000))
    np.testing.assert_allclose(cal.predict(p[:5], np.array(["new"] * 5)), cal.global_cal.predict(p[:5]))


def test_matrix_calibrator_modes(rng):
    P = rng.random((200, 3))
    Y = (rng.random((200, 3)) < P).astype(float)
    np.testing.assert_array_equal(MatrixCalibrator("none").fit(P, Y).predict(P), P)
    out = MatrixCalibrator("group").fit(P, Y, np.array(["x"] * 200)).predict(P, np.array(["x"] * 200))
    assert out.shape == P.shape and np.all((out >= 0) & (out <= 1))
