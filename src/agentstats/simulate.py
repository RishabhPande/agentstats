"""A generative model of agent-benchmark outcomes, for power analysis and testing.

logit P(success | agent a, task t, run r) =
      scale * (theta_a - b_t + u_at)            # ability, difficulty, agent x task fit
    + gamma[a, api_date(r)]                     # provider-side drift per API snapshot
    + eta[a, env(r)]                            # environment / harness version effect
then, independently, a run fails by timeout with prob tau[latency(r)] * h_a * long_t,
and in non-baseline env versions a fraction of tasks is broken for everyone.
An optional per-(agent, run) logit shock models anything that moves a whole
run at once (provider load, a transient tool outage, a bad container image).

All parameters are explicit; nothing here is fitted to real agent logs unless
you fit it yourself. Use :func:`calibrate_scale` to hit a target single-run SD.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import optimize, special


@dataclass
class SimConfig:
    n_tasks: int = 500
    true_scores: Sequence[float] = (0.70, 0.68, 0.65, 0.60, 0.55)
    scale: float = 2.5              # larger -> more deterministic tasks
    sd_difficulty: float = 1.5
    sd_interaction: float = 0.6
    api_dates: Sequence[str] = ("d0",)
    sd_api: float = 0.0            # per-agent logit shift SD across API snapshots
    env_versions: Sequence[str] = ("e0",)
    sd_env: float = 0.0
    env_break_frac: float = 0.0    # fraction of tasks broken in non-baseline env versions
    latencies: Sequence[str] = ("p50",)
    timeout_rate: dict = field(default_factory=lambda: {"p50": 0.0})
    long_task_frac: float = 0.3
    sd_run_shock: float = 0.0      # per (agent, run) logit shock shared by all tasks in the run
    seed: int = 0


def _solve_thetas(cfg, b, u):
    thetas = []
    for i, target in enumerate(cfg.true_scores):
        f = lambda th: special.expit(cfg.scale * (th - b + u[i])).mean() - target
        thetas.append(optimize.brentq(f, -20, 20))
    return np.array(thetas)


def simulate(cfg: SimConfig, n_runs: int, *, design: str = "crossed", run_seed: int | None = None) -> pd.DataFrame:
    """Simulate a run table. ``design="crossed"`` cycles every run through all
    (api_date, env, latency) combinations so factors are crossed with seeds."""
    rng = np.random.default_rng(cfg.seed)
    A, T = len(cfg.true_scores), cfg.n_tasks
    b = rng.normal(0, cfg.sd_difficulty, T)
    u = rng.normal(0, cfg.sd_interaction, (A, T))
    theta = _solve_thetas(cfg, b, u)
    gamma = rng.normal(0, cfg.sd_api, (A, len(cfg.api_dates)))
    gamma[:, 0] = 0.0
    eta = rng.normal(0, cfg.sd_env, (A, len(cfg.env_versions)))
    eta[:, 0] = 0.0
    broken = rng.random((len(cfg.env_versions), T)) < cfg.env_break_frac
    broken[0] = False
    h = rng.uniform(0.5, 1.5, A)          # agent sensitivity to slow tools
    long_t = rng.random(T) < cfg.long_task_frac

    r_rng = np.random.default_rng(run_seed if run_seed is not None else cfg.seed + 1)
    combos = [(d, e, l) for d in range(len(cfg.api_dates)) for e in range(len(cfg.env_versions))
              for l in range(len(cfg.latencies))]
    logits_base = cfg.scale * (theta[:, None] - b[None, :] + u)
    frames = []
    for r in range(n_runs):
        d, e, l = combos[r % len(combos)] if design == "crossed" else combos[0]
        lat = cfg.latencies[l]
        shock = r_rng.normal(0, cfg.sd_run_shock, (A, 1)) if cfg.sd_run_shock > 0 else 0.0
        p = special.expit(logits_base + gamma[:, [d]] + eta[:, [e]] + shock)
        p = p * (1 - cfg.timeout_rate.get(lat, 0.0) * h[:, None] * long_t[None, :])
        p = np.where(broken[e][None, :], 0.0, p)
        y = (r_rng.random((A, T)) < p).astype(float)
        frames.append(pd.DataFrame({
            "agent": np.repeat([f"A{i + 1:02d}" for i in range(A)], T),
            "task": np.tile([f"t{j:04d}" for j in range(T)], A),
            "run": r,
            "success": y.ravel(),
            "api_date": cfg.api_dates[d],
            "env_version": cfg.env_versions[e],
            "latency": lat,
        }))
    return pd.concat(frames, ignore_index=True)


def true_probabilities(cfg: SimConfig) -> np.ndarray:
    """(A, T) success probabilities at the baseline condition."""
    rng = np.random.default_rng(cfg.seed)
    A, T = len(cfg.true_scores), cfg.n_tasks
    b = rng.normal(0, cfg.sd_difficulty, T)
    u = rng.normal(0, cfg.sd_interaction, (A, T))
    theta = _solve_thetas(cfg, b, u)
    return special.expit(cfg.scale * (theta[:, None] - b[None, :] + u))


def calibrate_scale(cfg: SimConfig, target_single_run_sd: float) -> float:
    """Find ``scale`` so that the median agent's single-run SD (fraction units) hits the target."""
    def gap(log_s):
        c = SimConfig(**{**cfg.__dict__, "scale": float(np.exp(log_s))})
        p = true_probabilities(c)
        sd = np.sqrt((p * (1 - p)).sum(1)) / c.n_tasks
        return np.median(sd) - target_single_run_sd
    return float(np.exp(optimize.brentq(gap, np.log(0.1), np.log(200))))
