"""Decompose run-to-run variance of benchmark scores into nuisance factors.

For each agent we compute the benchmark score of every complete run and,
for every nuisance factor (API snapshot date, tool-latency profile,
environment version, ...), the one-way random-effects variance component
(ANOVA method of moments, Searle et al. 1992):

    sigma2_f = max(0, (MS_between - MS_within) / n0)

The residual of an additive main-effects fit is reported as "pure seed"
noise (interactions between factors are absorbed into it). Components are reported in percentage points^2 and as
SDs in pp. Factors should be crossed with seeds for this to be meaningful;
for unbalanced or nested designs fit a mixed model instead (see docs).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .data import factor_columns


def _one_way(scores: np.ndarray, groups: np.ndarray) -> float:
    levels, inv = np.unique(groups, return_inverse=True)
    g = len(levels)
    N = len(scores)
    if g < 2 or N <= g:
        return float("nan")
    n_i = np.bincount(inv)
    means = np.bincount(inv, weights=scores) / n_i
    grand = scores.mean()
    ssb = np.sum(n_i * (means - grand) ** 2)
    ssw = np.sum((scores - means[inv]) ** 2)
    msb, msw = ssb / (g - 1), ssw / (N - g)
    n0 = (N - np.sum(n_i ** 2) / N) / (g - 1)
    return float(max(0.0, (msb - msw) / n0))


def decompose(df: pd.DataFrame, factors: Sequence[str] | None = None) -> pd.DataFrame:
    """Variance components (pp^2) of a single run's benchmark score, per agent.

    Requires that factor values are constant within a run (each ``run`` is one
    full pass under one condition).
    """
    factors = list(factors) if factors is not None else factor_columns(df)
    for f in factors:
        bad = df.groupby(["agent", "run"])[f].nunique() > 1
        if bad.any():
            raise ValueError(f"factor {f!r} varies within a run; decompose expects run-level factors")
    runs = df.groupby(["agent", "run", *factors], as_index=False)["success"].mean()
    runs["score"] = runs["success"] * 100.0
    rows = []
    for a, g in runs.groupby("agent"):
        s = g["score"].to_numpy()
        row = {"agent": a, "n_runs": len(s), "total_var": float(np.var(s, ddof=1)) if len(s) > 1 else np.nan}
        for f in factors:
            row[f"var_{f}"] = _one_way(s, g[f].astype(str).to_numpy())
        if factors:
            # residual of an additive main-effects model = "pure seed" noise
            X = [np.ones(len(s))]
            for f in factors:
                lv = g[f].astype(str).to_numpy()
                for level in np.unique(lv)[1:]:
                    X.append((lv == level).astype(float))
            X = np.column_stack(X)
            dof = len(s) - np.linalg.matrix_rank(X)
            if dof > 0:
                beta, *_ = np.linalg.lstsq(X, s, rcond=None)
                row["var_seed"] = float(np.sum((s - X @ beta) ** 2) / dof)
            else:
                row["var_seed"] = np.nan
        else:
            row["var_seed"] = row["total_var"]
        rows.append(row)
    out = pd.DataFrame(rows)
    for c in [c for c in out.columns if c.startswith("var_")]:
        out[c.replace("var_", "sd_")] = np.sqrt(out[c])
    return out


def audit_leaderboard(scores: dict[str, float] | pd.Series, single_run_sd: float | dict, *,
                      alpha: float = 0.05) -> pd.DataFrame:
    """Audit a published single-run leaderboard without re-running anything.

    ``scores`` in percentage points; ``single_run_sd`` (pp) either one value or
    a per-agent dict, e.g. estimated from a pilot with
    :func:`agentstats.estimate.single_run_sd` or taken from the literature.
    Flags adjacent gaps smaller than the noise floor
    z_{1-alpha/2} * sqrt(sd_a^2 + sd_b^2).
    """
    from scipy import stats
    s = pd.Series(scores, dtype=float).sort_values(ascending=False)
    sd = (pd.Series(single_run_sd) if isinstance(single_run_sd, dict)
          else pd.Series(single_run_sd, index=s.index)).reindex(s.index)
    z = stats.norm.ppf(1 - alpha / 2)
    rows = []
    names = list(s.index)
    for i in range(len(names) - 1):
        a, b = names[i], names[i + 1]
        floor = z * float(np.hypot(sd[a], sd[b]))
        gap = float(s[a] - s[b])
        p = float(2 * stats.norm.sf(gap / np.hypot(sd[a], sd[b])))
        rows.append({"a": a, "b": b, "gap_pp": gap, "noise_floor_pp": floor,
                     "within_noise": gap < floor, "p_value": p,
                     "p_flip_single_rerun": float(stats.norm.sf(gap / np.hypot(sd[a], sd[b])))})
    return pd.DataFrame(rows)


def rerun_stability(scores: dict[str, float] | pd.Series, single_run_sd: float, *, method: str = "plug-in",
                    n_draws: int = 4000, seed: int | None = 0) -> dict:
    """How stable is a published single-run leaderboard under re-runs?

    ``scores`` and ``single_run_sd`` in percentage points. Simulates two
    independent re-runs of every entry and reports P(#1 changes), P(the truly
    best entry is not ranked first) and E[Kendall tau]. ``method="plug-in"``
    treats published scores as true; ``method="eb"`` shrinks them toward the
    board mean by B = 1 - sd^2 / Var(scores) and draws true scores from the
    resulting normal posterior (accounts for the published scores being noisy).
    """
    from scipy.stats import kendalltau
    rng = np.random.default_rng(seed)
    s = pd.Series(scores, dtype=float).to_numpy() / 100
    sd = single_run_sd / 100
    if method == "plug-in":
        theta = np.tile(s, (n_draws, 1))
    elif method == "eb":
        v = s.var(ddof=1)
        B = max(0.0, (v - sd ** 2) / v) if v > 0 else 0.0
        theta = s.mean() + B * (s - s.mean()) + rng.normal(0, np.sqrt(B) * sd, (n_draws, len(s)))
    else:
        raise ValueError("method must be 'plug-in' or 'eb'")
    a = theta + rng.normal(0, sd, theta.shape)
    b = theta + rng.normal(0, sd, theta.shape)
    m = min(n_draws, 1500)
    return {"p_top1_changes": float(np.mean(a.argmax(1) != b.argmax(1))),
            "p_true_best_not_first": float(np.mean(a.argmax(1) != theta.argmax(1))),
            "kendall_tau": float(np.mean([kendalltau(x, y).statistic for x, y in zip(a[:m], b[:m])]))}
