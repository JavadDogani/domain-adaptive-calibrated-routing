"""Latency simulation.

RouterBench has no latency measurements (and most of its models are retired), so latency is
*simulated* from token counts with a per-model profile::

    latency = (ttft + prompt_tokens / prefill_tps + output_tokens / decode_tps) * noise * tail

with log-normal ``noise`` and an occasional slow "tail" request (retries, cold starts, queueing).
The default profiles below are illustrative assumptions, not measurements. Measure your own
endpoints with ``dacr profile-latency`` and pass the resulting YAML via ``--latency-profiles``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

import numpy as np
import yaml

CHARS_PER_TOKEN = 4.0


@dataclass(frozen=True)
class LatencyProfile:
    ttft_s: float = 0.5  # fixed time to first token (network, scheduling)
    prefill_tps: float = 3000.0  # prompt tokens processed per second
    decode_tps: float = 50.0  # output tokens generated per second
    jitter: float = 0.25  # sigma of multiplicative log-normal noise
    tail_prob: float = 0.02  # probability that a request is slow
    tail_mult: float = 4.0  # mean slowdown of a slow request

    def base(self, prompt_chars, resp_chars):
        return (
            self.ttft_s
            + np.asarray(prompt_chars) / CHARS_PER_TOKEN / self.prefill_tps
            + np.asarray(resp_chars) / CHARS_PER_TOKEN / self.decode_tps
        )

    def expected(self, prompt_chars, resp_chars):
        noise_mean = np.exp(self.jitter**2 / 2)
        tail_mean = 1 + self.tail_prob * (self.tail_mult - 1)
        return self.base(prompt_chars, resp_chars) * noise_mean * tail_mean

    def sample(self, prompt_chars, resp_chars, rng: np.random.Generator):
        base = self.base(prompt_chars, resp_chars)
        noise = rng.lognormal(0.0, self.jitter, size=np.shape(base))
        slow = rng.random(np.shape(base)) < self.tail_prob
        tail = np.where(slow, rng.uniform(1.0, 2 * self.tail_mult - 1.0, size=np.shape(base)), 1.0)
        return base * noise * tail


# Illustrative profiles for RouterBench's models (late-2023 era APIs). Assumptions, not data.
DEFAULT_PROFILES: dict[str, LatencyProfile] = {
    "gpt-4-1106-preview": LatencyProfile(ttft_s=0.70, prefill_tps=2500, decode_tps=25),
    "gpt-3.5-turbo-1106": LatencyProfile(ttft_s=0.35, prefill_tps=5000, decode_tps=70),
    "claude-v2": LatencyProfile(ttft_s=1.00, prefill_tps=2000, decode_tps=30),
    "claude-v1": LatencyProfile(ttft_s=0.90, prefill_tps=2000, decode_tps=35),
    "claude-instant-v1": LatencyProfile(ttft_s=0.45, prefill_tps=4000, decode_tps=75),
    "meta/llama-2-70b-chat": LatencyProfile(ttft_s=0.60, prefill_tps=2000, decode_tps=30),
    "meta/code-llama-instruct-34b-chat": LatencyProfile(ttft_s=0.45, prefill_tps=3000, decode_tps=40),
    "mistralai/mixtral-8x7b-chat": LatencyProfile(ttft_s=0.35, prefill_tps=4000, decode_tps=70),
    "mistralai/mistral-7b-chat": LatencyProfile(ttft_s=0.20, prefill_tps=6000, decode_tps=110),
    "zero-one-ai/Yi-34B-Chat": LatencyProfile(ttft_s=0.45, prefill_tps=3000, decode_tps=40),
    "WizardLM/WizardLM-13B-V1.2": LatencyProfile(ttft_s=0.30, prefill_tps=5000, decode_tps=65),
    # synthetic demo models
    "tiny-chat": LatencyProfile(ttft_s=0.15, prefill_tps=8000, decode_tps=150),
    "small-coder": LatencyProfile(ttft_s=0.25, prefill_tps=6000, decode_tps=90),
    "medium-general": LatencyProfile(ttft_s=0.40, prefill_tps=4000, decode_tps=55),
    "large-frontier": LatencyProfile(ttft_s=0.80, prefill_tps=2500, decode_tps=25),
}


def load_profiles(models: list[str], path: str | Path | None = None) -> dict[str, LatencyProfile]:
    """Profiles for ``models``. YAML overrides: ``{default: {...}, models: {name: {...}}}``."""
    default = LatencyProfile()
    overrides: dict[str, dict] = {}
    if path is not None:
        cfg = yaml.safe_load(Path(path).read_text()) or {}
        valid = {f.name for f in fields(LatencyProfile)}
        default = replace(default, **{k: float(v) for k, v in (cfg.get("default") or {}).items() if k in valid})
        overrides = {
            m: {k: float(v) for k, v in (d or {}).items() if k in valid} for m, d in (cfg.get("models") or {}).items()
        }
    out = {}
    for m in models:
        base = DEFAULT_PROFILES.get(m, default)
        out[m] = replace(base, **overrides.get(m, {}))
    return out


def profiles_to_yaml(profiles: dict[str, LatencyProfile]) -> str:
    return yaml.safe_dump({"models": {m: asdict(p) for m, p in profiles.items()}}, sort_keys=True)


class LatencyModel:
    def __init__(self, models: list[str], profiles: dict[str, LatencyProfile]):
        self.models = list(models)
        self.profiles = [profiles[m] for m in self.models]

    def expected(self, prompt_chars: np.ndarray, resp_chars: np.ndarray) -> np.ndarray:
        return np.column_stack([p.expected(prompt_chars, resp_chars[:, j]) for j, p in enumerate(self.profiles)])

    def sample(self, prompt_chars: np.ndarray, resp_chars: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return np.column_stack([p.sample(prompt_chars, resp_chars[:, j], rng) for j, p in enumerate(self.profiles)])
