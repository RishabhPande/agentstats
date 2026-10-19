"""Point estimates and confidence intervals for agent success rates.

Two inferential targets are supported, and the distinction matters:

``inference="benchmark"``
    The claim is about *this* benchmark (the fixed set of T tasks). The only
    randomness is run-to-run: sampling temperature, provider-side
    non-determinism, API snapshot, tool latency, environment flakiness.
    Requires >= 2 runs per task. This is what a leaderboard implicitly claims.

``inference="population"``
    The claim is about the population of tasks the benchmark samples from.
    Task sampling variance is added on top of run variance. Valid with a
    single run, but then run noise and task noise cannot be separated.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats

from .data import TaskStats, task_stats

INFERENCE = ("benchmark", "population")


def _check_inference(inference: str) -> None:
    if inference not in INFERENCE:
        raise ValueError(f"inference must be one of {INFERENCE}, got {inference!r}")


def score_se(mean: np.ndarray, var: np.ndarray, k: np.ndarray, inference: str) -> float:
    """Standard error of the benchmark score for one agent (arrays over tasks)."""
    T = mean.shape[0]
    if inference == "population":
        return float(np.std(mean, ddof=1) / np.sqrt(T))
    if np.any(k < 2):
        return float("nan")
    return float(np.sqrt(np.nansum(var / k)) / T)


def agent_scores(df: pd.DataFrame, inference: str = "benchmark", alpha: float = 0.05,
                 common_tasks: bool = True) -> pd.DataFrame:
    """Per-agent score, standard error and (1 - alpha) CI.

    Returns a DataFrame sorted by score with columns
    ``agent, score, se, lo, hi, n_tasks, runs_min, runs_max``.
    """
    _check_inference(inference)
    ts = task_stats(df, common_tasks=common_tasks)
    z = stats.norm.ppf(1 - alpha / 2)
    rows = []
    for i, a in enumerate(ts.agents):
        m, v, k = ts.mean[i], ts.var[i], ts.k[i]
        s = float(np.mean(m))
        se = score_se(m, v, k, inference)
        rows.append(dict(agent=a, score=s, se=se, lo=s - z * se, hi=s + z * se,
                         n_tasks=len(ts.tasks), runs_min=int(k.min()), runs_max=int(k.max())))
    out = pd.DataFrame(rows).sort_values("score", ascending=False, ignore_index=True)
    if inference == "benchmark" and out["se"].isna().any():
        warnings.warn("benchmark-level SEs need >= 2 runs per task; some agents have 1 run. "
                      "Use inference='population' or collect more runs.", stacklevel=2)
    return out


def flakiness(ts: TaskStats) -> pd.Series:
    """Mean within-task run variance per agent (phi = avg_t p_t (1 - p_t) for binary).

    phi = 0 means every task is solved deterministically (always / never);
    phi = 0.25 is the maximum (every task is a coin flip). The single-run SD of
    the benchmark score is sqrt(phi / T).
    """
    return pd.Series(np.nanmean(ts.var, axis=1), index=ts.agents, name="flakiness")


def single_run_sd(df: pd.DataFrame) -> pd.Series:
    """Estimated SD of a single full-benchmark run's score, per agent."""
    ts = task_stats(df)
    phi = flakiness(ts)
    return np.sqrt(phi / len(ts.tasks)).rename("single_run_sd")


def design_effect(df: pd.DataFrame) -> pd.DataFrame:
    """Is run-to-run noise independent across tasks? (needs >= 3 complete runs)

    Compares the observed variance of complete-run scores with the variance
    implied by independent per-task noise, sum_t v_t / T^2. Under
    independence the ratio is ~1. Values well above 1 mean something shifts
    whole runs together (API snapshot drift, a broken environment version,
    slow tools causing correlated timeouts). Task-level tests and the
    task-level planner then understate noise: use ``level="run"``.

    Returns per agent: ``var_run`` (observed), ``var_indep`` (implied),
    ``deff`` (ratio) and a one-sided chi-square p-value for deff > 1.
    """
    ts = task_stats(df)
    rows = []
    for i, a in enumerate(ts.agents):
        sub = df[(df.agent == a) & df.task.isin(ts.tasks)]
        rs = sub.groupby("run")["success"].mean().to_numpy()
        n = len(rs)
        v_run = float(np.var(rs, ddof=1)) if n > 1 else np.nan
        k = ts.k[i]
        # per-task variance of ONE run's outcome is v_t; one full run's score variance:
        v_ind = float(np.nansum(ts.var[i]) / len(ts.tasks) ** 2)
        deff = v_run / v_ind if v_ind > 0 else np.nan
        p = float(stats.chi2.sf((n - 1) * deff, n - 1)) if n > 1 and np.isfinite(deff) else np.nan
        rows.append(dict(agent=a, n_runs=n, var_run=v_run, var_indep=v_ind, deff=deff, p_value=p))
    return pd.DataFrame(rows)
