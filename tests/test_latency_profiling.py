import asyncio
from types import SimpleNamespace

import numpy as np
import pandas as pd

from dacr.latency import LatencyModel, LatencyProfile, load_profiles
from dacr.profiling import Endpoint, fit_profile, measure_one, profile_endpoints


def test_expected_matches_sample_mean(rng):
    prof = LatencyProfile(ttft_s=0.5, prefill_tps=2000, decode_tps=40, jitter=0.3, tail_prob=0.05, tail_mult=4)
    pc, rc = np.full(200_000, 4000.0), np.full(200_000, 800.0)
    emp = prof.sample(pc, rc, rng).mean()
    assert abs(emp / prof.expected(4000.0, 800.0) - 1) < 0.02


def test_load_profiles_overrides(tmp_path):
    path = tmp_path / "p.yaml"
    path.write_text(
        "default: {decode_tps: 10}\nmodels:\n  custom: {ttft_s: 2.0}\n  gpt-3.5-turbo-1106: {decode_tps: 99}\n"
    )
    prof = load_profiles(["custom", "other", "gpt-3.5-turbo-1106"], path)
    assert prof["custom"].ttft_s == 2.0 and prof["custom"].decode_tps == 10
    assert prof["other"].decode_tps == 10
    assert prof["gpt-3.5-turbo-1106"].decode_tps == 99 and prof["gpt-3.5-turbo-1106"].ttft_s == 0.35
    lm = LatencyModel(["custom", "other"], prof)
    assert lm.expected(np.array([100.0]), np.array([[40.0, 40.0]])).shape == (1, 2)


def test_fit_profile_recovers_parameters(rng):
    true = LatencyProfile(ttft_s=0.4, prefill_tps=3000, decode_tps=60, jitter=0.0, tail_prob=0.0)
    pt = rng.uniform(100, 4000, 80)
    ct = rng.uniform(50, 400, 80)
    ttft = true.ttft_s + pt / true.prefill_tps
    df = pd.DataFrame(
        {"ok": True, "prompt_tokens": pt, "completion_tokens": ct, "ttft_s": ttft, "total_s": ttft + (ct - 1) / 60}
    )
    fit = fit_profile(df)
    assert abs(fit.ttft_s - 0.4) < 0.02
    assert abs(fit.prefill_tps / 3000 - 1) < 0.05
    assert abs(fit.decode_tps / 60 - 1) < 0.02


class _FakeStream:
    def __init__(self, pieces):
        self.chunks = [
            SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=p))], usage=None) for p in pieces
        ]
        self.chunks.append(SimpleNamespace(choices=[], usage=SimpleNamespace(prompt_tokens=12, completion_tokens=3)))

    def __aiter__(self):
        self._it = iter(self.chunks)
        return self

    async def __anext__(self):
        try:
            return next(self._it)
        except StopIteration:
            raise StopAsyncIteration from None


class _FakeClient:
    def __init__(self, fail=False):
        self.fail = fail
        self.kwargs = None
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    async def create(self, **kwargs):
        self.kwargs = kwargs
        if self.fail:
            raise RuntimeError("401 bad key")
        return _FakeStream(["Hel", "lo", "!"])


def test_measure_one_with_fake_client():
    client = _FakeClient()
    row = asyncio.run(measure_one(client, Endpoint("m", token_param="max_tokens"), "hi there", 16))
    assert row["ok"] and row["completion_tokens"] == 3 and row["prompt_tokens"] == 12
    assert row["ttft_s"] <= row["total_s"]
    assert client.kwargs["max_tokens"] == 16 and client.kwargs["stream"] is True


def test_profile_endpoints_records_errors():
    df = asyncio.run(profile_endpoints([Endpoint("bad")], ["a", "b"], client_factory=lambda ep: _FakeClient(fail=True)))
    assert (~df["ok"]).all() and df["error"].str.contains("401").all()
    assert fit_profile(df) is None
