"""Measure latency profiles of OpenAI-compatible chat endpoints (OpenAI, vLLM, Ollama, ...).

For each endpoint we stream a set of prompts, record time-to-first-token and total time, and fit
a :class:`~dacr.latency.LatencyProfile` that ``dacr run --latency-profiles`` can use instead of
the illustrative defaults.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .features import clean_prompt
from .latency import CHARS_PER_TOKEN, LatencyProfile, profiles_to_yaml


@dataclass
class Endpoint:
    model: str
    base_url: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    token_param: str = "max_completion_tokens"  # some servers only accept "max_tokens"
    name: str | None = None  # name to use in the profile file (defaults to model)

    @property
    def label(self) -> str:
        return self.name or self.model


def load_endpoints(path: str | Path) -> list[Endpoint]:
    cfg = yaml.safe_load(Path(path).read_text()) or {}
    return [Endpoint(**e) for e in cfg.get("endpoints", [])]


def load_prompts(path: str | Path, n: int, seed: int = 0, max_chars: int = 8000) -> list[str]:
    path = Path(path)
    if path.suffix == ".parquet":
        prompts = pd.read_parquet(path, columns=["prompt"])["prompt"].drop_duplicates().tolist()
    elif path.suffix == ".jsonl":
        prompts = pd.read_json(path, lines=True)["prompt"].tolist()
    else:
        prompts = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(prompts), size=min(n, len(prompts)), replace=False)
    return [clean_prompt(prompts[i])[:max_chars] for i in pick]


async def measure_one(client, ep: Endpoint, prompt: str, max_tokens: int) -> dict:
    t0 = time.perf_counter()
    ttft, out_chars, usage = None, 0, None
    try:
        stream = await client.chat.completions.create(
            model=ep.model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            stream_options={"include_usage": True},
            **{ep.token_param: max_tokens},
        )
        async for chunk in stream:
            if chunk.choices:
                delta = chunk.choices[0].delta
                text = getattr(delta, "content", None) if delta is not None else None
                if text:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    out_chars += len(text)
            if getattr(chunk, "usage", None) is not None:
                usage = chunk.usage
        total = time.perf_counter() - t0
    except Exception as e:  # noqa: BLE001 - record any API error and keep going
        return {
            "model": ep.label,
            "ok": False,
            "error": str(e)[:300],
            "prompt_chars": len(prompt),
            "prompt_tokens": None,
            "completion_tokens": None,
            "ttft_s": None,
            "total_s": None,
        }
    prompt_tokens = getattr(usage, "prompt_tokens", None) or len(prompt) / CHARS_PER_TOKEN
    completion_tokens = getattr(usage, "completion_tokens", None) or out_chars / CHARS_PER_TOKEN
    return {
        "model": ep.label,
        "ok": ttft is not None,
        "error": None if ttft else "no content streamed",
        "prompt_chars": len(prompt),
        "prompt_tokens": float(prompt_tokens),
        "completion_tokens": float(completion_tokens),
        "ttft_s": ttft,
        "total_s": total,
    }


async def profile_endpoints(
    endpoints: list[Endpoint], prompts: list[str], max_tokens: int = 256, concurrency: int = 1, client_factory=None
) -> pd.DataFrame:
    if client_factory is None:
        try:
            from openai import AsyncOpenAI
        except ImportError as e:  # pragma: no cover - optional dependency
            raise ImportError("Install the profile extra: pip install 'dacr-router[profile]'") from e

        def client_factory(ep: Endpoint):
            key = os.environ.get(ep.api_key_env)
            if not key:
                raise RuntimeError(f"environment variable {ep.api_key_env} is not set for {ep.model}")
            return AsyncOpenAI(api_key=key, base_url=ep.base_url)

    sem = asyncio.Semaphore(concurrency)
    rows = []
    for ep in endpoints:
        client = client_factory(ep)

        async def run(p: str, ep=ep, client=client):
            async with sem:
                return await measure_one(client, ep, p, max_tokens)

        rows += await asyncio.gather(*(run(p) for p in prompts))
    return pd.DataFrame(rows)


def fit_profile(df: pd.DataFrame) -> LatencyProfile | None:
    """Fit a latency profile from successful measurements of one endpoint."""
    if not {"ttft_s", "total_s"} <= set(df.columns):
        return None
    d = df[df["ok"].astype(bool)].dropna(subset=["ttft_s", "total_s"])
    if len(d) < 3:
        return None
    ttft, total = d["ttft_s"].to_numpy(), d["total_s"].to_numpy()
    pt, ct = d["prompt_tokens"].to_numpy(), d["completion_tokens"].to_numpy()

    ttft0, prefill = float(np.median(ttft)), 1e5
    if np.ptp(pt) > 0:
        slope, intercept = np.polyfit(pt, ttft, 1)
        if slope > 0:
            prefill, ttft0 = 1.0 / slope, max(float(intercept), 0.01)
    gen = (total - ttft) > 1e-3
    decode = float(np.median(np.maximum(ct[gen] - 1, 1) / (total - ttft)[gen])) if gen.any() else 50.0

    base = LatencyProfile(ttft_s=ttft0, prefill_tps=prefill, decode_tps=decode, jitter=0.0, tail_prob=0.0)
    ratio = total / base.base(pt * CHARS_PER_TOKEN, ct * CHARS_PER_TOKEN)
    log_r = np.log(ratio)
    jitter = float(1.4826 * np.median(np.abs(log_r - np.median(log_r))))
    slow = ratio > 2.5 * np.median(ratio)
    tail_prob = float(slow.mean())
    tail_mult = float(ratio[slow].mean() / np.median(ratio)) if slow.any() else 4.0
    return LatencyProfile(
        ttft_s=ttft0,
        prefill_tps=prefill,
        decode_tps=decode,
        jitter=max(jitter, 0.01),
        tail_prob=tail_prob,
        tail_mult=max(tail_mult, 1.0),
    )


def run_profiling(
    endpoints_path: str,
    prompts_path: str,
    out_yaml: str,
    n_prompts: int = 30,
    max_tokens: int = 256,
    concurrency: int = 1,
    seed: int = 0,
) -> dict[str, LatencyProfile]:
    endpoints = load_endpoints(endpoints_path)
    prompts = load_prompts(prompts_path, n_prompts, seed)
    df = asyncio.run(profile_endpoints(endpoints, prompts, max_tokens, concurrency))
    out = Path(out_yaml)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out.with_suffix(".measurements.csv"), index=False)
    profiles = {}
    for model, g in df.groupby("model"):
        prof = fit_profile(g)
        n_ok = int(g["ok"].sum())
        if prof is None:
            errs = g["error"].dropna().unique()[:2]
            print(f"[profile] {model}: only {n_ok}/{len(g)} successful requests, skipped. Errors: {list(errs)}")
            continue
        profiles[model] = prof
        print(f"[profile] {model}: {n_ok}/{len(g)} ok -> {prof}")
    out.write_text(profiles_to_yaml(profiles))
    print(f"[profile] wrote {out}")
    return profiles
