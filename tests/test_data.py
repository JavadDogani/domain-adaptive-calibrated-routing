import numpy as np
import pandas as pd
import pytest

from dacr.data import RoutingData, make_split, to_long_table


def _wide(bare_quality: bool = False) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "sample_id": ["q1", "q2", "q3"],
            "prompt": ["['a']", "['b']", "['c']"],
            "eval_name": ["math", "code", "math"],
        }
    )
    for m, q in [("m1", [1, 0, 1]), ("m2", [0.5, 1, 0])]:
        df[f"{m}|model_response"] = ["x", "yy", "zzz"]
        df[f"{m}|total_cost"] = [0.1, 0.2, 0.3]
        df[m if bare_quality else f"{m}|performance"] = q
    return df


@pytest.mark.parametrize("bare", [False, True])
def test_wide_schema_conversion(bare):
    long = to_long_table(_wide(bare))
    assert len(long) == 6
    assert set(long["model"]) == {"m1", "m2"}
    assert long.loc[(long.qid == "q1") & (long.model == "m2"), "quality"].item() == 0.5
    assert list(long.loc[long.model == "m1", "resp_len_chars"]) == [1, 2, 3]


def test_long_schema_conversion():
    df = pd.DataFrame(
        {
            "sample_id": ["q1", "q1"],
            "model_name": ["a", "b"],
            "prompt": ["p", "p"],
            "eval_name": ["d", "d"],
            "model_response": ["r1", "r22"],
            "total_cost": [1.0, 2.0],
            "performance": [1.0, 0.0],
        }
    )
    long = to_long_table(df)
    assert list(long["model"]) == ["a", "b"]
    assert list(long["quality"]) == [1.0, 0.0]


def test_from_long_builds_matrices_and_drops_incomplete():
    long = to_long_table(_wide())
    long = long[~((long.qid == "q3") & (long.model == "m2"))]  # q3 missing for m2
    data = RoutingData.from_long(long)
    assert data.n_queries == 2 and data.models == ["m1", "m2"]
    assert data.quality.shape == data.cost.shape == data.response_chars.shape == (2, 2)
    assert data.subset(np.array([1])).n_queries == 1


def test_random_split_is_disjoint_and_complete(synthetic_data):
    s = make_split(synthetic_data, "random", test_frac=0.2, calib_frac=0.25, seed=3)
    allidx = np.concatenate([s.train, s.calib, s.test])
    assert len(allidx) == len(set(allidx)) == synthetic_data.n_queries
    assert abs(len(s.test) - 0.2 * synthetic_data.n_queries) <= 1


def test_ood_split_holds_out_domains(synthetic_data):
    s = make_split(synthetic_data, "ood", holdout=["math"], seed=0)
    dom = synthetic_data.domains
    assert all(d.startswith("math") for d in dom[s.test])
    assert not any(d.startswith("math") for d in dom[np.concatenate([s.train, s.calib])])


def test_ood_split_rejects_unknown_prefix(synthetic_data):
    with pytest.raises(ValueError, match="match no domain"):
        make_split(synthetic_data, "ood", holdout=["gsm8k"])


def test_prepare_cli_on_local_pickle(tmp_path):
    from dacr.cli import main
    from dacr.data import load_routing_data

    pkl = tmp_path / "rb.pkl"
    _wide().to_pickle(pkl)
    out = tmp_path / "long.parquet"
    main(["prepare", "--pkl", str(pkl), "--out", str(out)])
    long = pd.read_parquet(out)
    assert "response" not in long.columns and len(long) == 6
    data = load_routing_data(out)
    assert data.n_queries == 3 and data.models == ["m1", "m2"]
