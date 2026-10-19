"""The minimum-runs protocol: how many runs per agent before a gap means anything.

Model. Agent j solves task t with probability p_jt. With k runs per task, the
benchmark-score gap between agents A and B has variance

    Var(D_hat) = w / k                                   (benchmark inference)
    Var(D_hat) = w / k + sigma2_delta / T                (population inference)

where  w = (1/T^2) * sum_t [p_At(1-p_At) + p_Bt(1-p_Bt)]   is per-run noise and
sigma2_delta = Var_t(p_At - p_Bt) is genuine task heterogeneity of the gap.
For a two-sided level-alpha test with power 1-beta against a true gap delta,

    k_min = ceil( (z_{1-alpha/2} + z_{1-beta})^2 * w / delta^2 )

(population inference: divide by delta^2 - z^2 sigma2_delta / T instead, and if
that is <= 0 no number of runs suffices -- you need more *tasks*).

Both w and sigma2_delta are estimated from a small pilot (>= 2, ideally 3-5,
runs per task) by unbiased method-of-moments.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from itertools import combinations
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .data import task_stats


@dataclass
class RunPlan:
    a: str
    b: str
    delta: float
    k_min: float           # math.inf if unattainable by runs alone
    w: float               # per-run variance of the gap (x k)
    sigma2_delta: float    # task heterogeneity of the gap
    n_tasks: int
    inference: str
    note: str = ""
    w_task: float = float("nan")
    w_run: float = float("nan")
    method: str = "max"

    def to_dict(self):
        return asdict(self)


def _z(alpha, power):
    return stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)


def k_min_from_components(w: float, delta: float, *, sigma2_delta: float = 0.0, n_tasks: int | None = None,
                          alpha: float = 0.05, power: float = 0.8, inference: str = "benchmark",
                          k_floor: int = 3) -> float:
    """Closed-form minimum runs given variance components."""
    if delta <= 0:
        raise ValueError("delta must be > 0")
    z2 = _z(alpha, power) ** 2
    denom = delta ** 2
    if inference == "population":
        if n_tasks is None:
            raise ValueError("population inference needs n_tasks")
        denom -= z2 * sigma2_delta / n_tasks
        if denom <= 0:
            return math.inf
    return float(max(k_floor, math.ceil(z2 * w / denom)))


def k_min_generic(n_tasks: int, delta: float, flakiness: float = 0.05, *, alpha: float = 0.05,
                  power: float = 0.8, k_floor: int = 3) -> float:
    """Planning without a pilot. ``flakiness`` = mean_t p_t(1-p_t) per agent.

    Published repeated-run studies on SWE-bench Verified imply single-run SDs of
    roughly 1-2 pp on 500 tasks, i.e. flakiness ~0.05-0.15 (phi = T * sd^2).
    """
    w = 2 * flakiness / n_tasks
    return k_min_from_components(w, delta, alpha=alpha, power=power, k_floor=k_floor)


def min_detectable_gap(n_tasks: int, k: int, flakiness: float = 0.05, *, alpha: float = 0.05,
                       power: float = 0.8) -> float:
    """Smallest true gap detectable with k runs (benchmark inference)."""
    return float(_z(alpha, power) * math.sqrt(2 * flakiness / (n_tasks * k)))


def min_runs_for_exact_run_test(alpha: float) -> int:
    """Smallest balanced k for which a two-sided exact run-permutation test can reach p < alpha."""
    k = 2
    while 2 / math.comb(2 * k, k) > alpha:
        k += 1
    return k


def plan_pair(pilot: pd.DataFrame, a: str, b: str, delta: float, *, alpha: float = 0.05,
              power: float = 0.8, inference: str = "benchmark", k_floor: int = 3,
              method: str = "max") -> RunPlan:
    """Minimum runs for one pair.

    ``method``: "task" assumes independent noise across tasks (sum of per-task
    variances); "run" uses the observed variance of complete-run scores, which
    also captures noise that shifts whole runs (API drift, env versions);
    "max" (default) takes the larger -- conservative under correlated noise.
    Run-level variance is estimated from few runs and is itself noisy; with a
    3-run pilot treat k_min as an order of magnitude, re-plan after 5 runs.
    """
    if method not in ("task", "run", "max"):
        raise ValueError("method must be 'task', 'run' or 'max'")
    ts = task_stats(pilot, agents=[a, b])
    if np.any(ts.k < 2):
        raise ValueError("pilot needs >= 2 runs per task for both agents (3-5 recommended)")
    T = len(ts.tasks)
    vA, vB = ts.var[0], ts.var[1]
    w_task = float((vA.sum() + vB.sum()) / T ** 2)
    sub = pilot[pilot.task.isin(ts.tasks)]
    rv = [sub[sub.agent == x].groupby("run")["success"].mean().var(ddof=1) for x in (a, b)]
    w_run = float(sum(rv)) if all(np.isfinite(rv)) else float("nan")
    if method == "task" or not np.isfinite(w_run):
        w = w_task
    elif method == "run":
        w = w_run
    else:
        w = max(w_task, w_run)
    d = ts.mean[0] - ts.mean[1]
    noise = vA / ts.k[0] + vB / ts.k[1]
    sigma2 = float(max(0.0, d.var(ddof=1) - noise.mean()))
    k = k_min_from_components(w, delta, sigma2_delta=sigma2, n_tasks=T, alpha=alpha, power=power,
                              inference=inference, k_floor=k_floor)
    note = "unattainable by runs: add tasks" if math.isinf(k) else ""
    return RunPlan(a, b, delta, k, w, sigma2, T, inference, note, w_task, w_run, method)


def plan(pilot: pd.DataFrame, delta: float, agents: Sequence[str] | None = None, *,
         adjacent_only: bool = True, alpha: float = 0.05, power: float = 0.8,
         inference: str = "benchmark", k_floor: int = 3, correct_multiple: bool = True,
         method: str = "max") -> pd.DataFrame:
    """Minimum runs for every (adjacent) pair on a leaderboard.

    With ``correct_multiple`` the per-test alpha is Bonferroni-divided by the
    number of pairs, matching the Holm-corrected tests in :func:`compare.pairwise`
    (Holm is never less powerful than Bonferroni, so this is conservative).
    """
    if agents is None:
        from .estimate import agent_scores
        agents = list(agent_scores(pilot, inference="population")["agent"])
    pairs = list(zip(agents[:-1], agents[1:])) if adjacent_only else list(combinations(agents, 2))
    a_eff = alpha / len(pairs) if correct_multiple else alpha
    rows = [plan_pair(pilot, a, b, delta, alpha=a_eff, power=power, inference=inference,
                      k_floor=k_floor, method=method).to_dict() for a, b in pairs]
    return pd.DataFrame(rows)
