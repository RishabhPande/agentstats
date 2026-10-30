"""Paired significance tests between agents.

Three tests, chosen by ``level`` / ``inference``:

``level="task"`` (default), ``inference="benchmark"``
    *Stratified, studentized run-permutation test.* Run labels are permuted
    between the two agents within each task (x nuisance stratum) cell. The
    statistic is the score gap divided by its plug-in SE (Chung & Romano 2013),
    which makes the test asymptotically valid for the *weak* null "equal
    benchmark score", not only the sharp null "identical on every task".
    Assumes runs are independent across tasks. Check with
    :func:`agentstats.estimate.design_effect`; if it is well above 1,
    variance is correlated across tasks within a run and you should use
    ``level="run"``.

``level="run"``
    *Run-level Welch test.* Each complete benchmark pass is one observation.
    Robust to anything that shifts a whole run (API snapshot drift, a broken
    env version, a slow tool backend), at the cost of power when runs are few.
    A Welch t reference (default) or a label-permutation reference.

``inference="population"``
    Paired sign-flip test over per-task differences; valid with one run,
    targets the task population rather than the fixed benchmark.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, asdict
from itertools import combinations
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .data import task_stats
from .estimate import _check_inference

_EPS = 1e-12


@dataclass
class Comparison:
    a: str
    b: str
    gap: float          # score(a) - score(b), fraction units
    se: float
    lo: float
    hi: float
    p_value: float
    n_tasks: int
    n_runs_a: int
    n_runs_b: int
    inference: str
    level: str
    test: str

    def significant(self, alpha: float = 0.05) -> bool:
        return self.p_value < alpha

    def to_dict(self):
        return asdict(self)


def _perm_p(obs, null):
    """Two-sided Monte Carlo p-value with a normal tail approximation.

    A Monte Carlo p-value cannot go below 1/(n_perm+1), which makes it useless
    after Holm correction over many pairs (e.g. 91 pairs x 1/1001 > 0.05).
    When fewer than 10 null draws are as extreme as the observed statistic,
    we return the normal-approximation tail probability of the (studentized,
    hence ~N(0,1)) null instead, and flag it.
    """
    n = len(null)
    hits = int(np.sum(np.abs(null) >= abs(obs) - _EPS))
    if hits >= 10:
        return (1 + hits) / (n + 1), False
    mu, sd = float(np.mean(null)), float(np.std(null)) + _EPS
    return float(2 * stats.norm.sf(abs(obs - mu) / sd)), True


# ---------------------------------------------------------------- task level
def _task_level(df, a, b, strata, n_perm, rng, studentize):
    sub = df[df["agent"].isin([a, b])]
    tasks = sorted(set(sub.loc[sub.agent == a, "task"]) & set(sub.loc[sub.agent == b, "task"]))
    sub = sub[sub["task"].isin(tasks)].copy()
    sub["_y2"] = sub["success"] ** 2
    keys = ["task", *strata]
    g = sub.groupby(keys + ["agent"]).agg(s=("success", "sum"), s2=("_y2", "sum"),
                                          n=("success", "size")).unstack("agent", fill_value=0)
    ka, kb = g[("n", a)].to_numpy(), g[("n", b)].to_numpy()
    keep = (ka > 0) & (kb > 0)
    if keep.mean() < 0.9:
        warnings.warn(f"only {keep.mean():.0%} of task x stratum cells contain both agents; "
                      "runs may be confounded with nuisance factors", stacklevel=3)
    g = g[keep]
    ka, kb = ka[keep], kb[keep]
    T = len(tasks)
    t_idx = pd.Index(tasks).get_indexer(g.index.get_level_values("task"))
    KA = np.bincount(t_idx, weights=ka, minlength=T)
    KB = np.bincount(t_idx, weights=kb, minlength=T)
    ok = (KA > 0) & (KB > 0)
    can_stud = studentize and np.all(KA[ok] >= 2) and np.all(KB[ok] >= 2)

    def statistic(SA, SA2, SB, SB2):
        mA, mB = SA[..., ok] / KA[ok], SB[..., ok] / KB[ok]
        gap = (mA - mB).mean(-1)
        if not can_stud:
            return gap, gap
        vA = np.maximum(SA2[..., ok] - KA[ok] * mA ** 2, 0) / (KA[ok] - 1) / KA[ok]
        vB = np.maximum(SB2[..., ok] - KB[ok] * mB ** 2, 0) / (KB[ok] - 1) / KB[ok]
        se = np.sqrt((vA + vB).sum(-1)) / ok.sum()
        return gap, gap / (se + _EPS)

    xa, xa2 = g[("s", a)].to_numpy(), g[("s2", a)].to_numpy()
    xb, xb2 = g[("s", b)].to_numpy(), g[("s2", b)].to_numpy()
    agg = lambda w: np.bincount(t_idx, weights=w, minlength=T)
    obs_gap, obs_t = statistic(agg(xa), agg(xa2), agg(xb), agg(xb2))

    binary = np.isin(sub["success"].unique(), [0.0, 1.0]).all()
    SA = np.zeros((n_perm, T)); SA2 = np.zeros((n_perm, T))
    if binary:  # hypergeometric fast path
        s, n = (xa + xb).astype(np.int64), (ka + kb).astype(np.int64)
        draws = rng.hypergeometric(s, n - s, ka.astype(np.int64), size=(n_perm, len(s)))
        np.add.at(SA.T, t_idx, draws.T)
        SA2 = SA
        tot = agg(xa + xb)
        SB = tot - SA
        SB2 = SB
    else:
        tot, tot2 = agg(xa + xb), agg(xa2 + xb2)
        sub["_cell"] = sub.groupby(keys).ngroup()
        kept_cells = set(np.flatnonzero(keep))
        for cid, cell in sub.groupby("_cell"):
            if cid not in kept_cells:
                continue
            vals = cell["success"].to_numpy()
            na = int((cell["agent"] == a).sum())
            i = tasks.index(cell["task"].iloc[0])
            perm = np.argsort(rng.random((n_perm, len(vals))), axis=1)
            pv = vals[perm][:, :na]
            SA[:, i] += pv.sum(1); SA2[:, i] += (pv ** 2).sum(1)
        SB, SB2 = tot - SA, tot2 - SA2
    _, null_t = statistic(SA, SA2, SB, SB2)
    p, tail = _perm_p(obs_t, null_t)
    name = ("studentized " if can_stud else "") + "stratified run-permutation (task level)"
    if tail:
        name += "; normal tail approx."
    if strata:
        name += f"; strata: {', '.join(strata)}"
    return float(obs_gap), float(p), name


# ----------------------------------------------------------------- run level
def _welch(x, y):
    vx = x.var(ddof=1) / len(x) if len(x) > 1 else 0.0
    vy = y.var(ddof=1) / len(y) if len(y) > 1 else 0.0
    return (x.mean() - y.mean()) / math.sqrt(vx + vy + _EPS)


def _welch_t(x, y):
    vx, vy = x.var(ddof=1) / len(x), y.var(ddof=1) / len(y)
    t = (x.mean() - y.mean()) / math.sqrt(vx + vy + _EPS)
    dof = (vx + vy) ** 2 / ((vx ** 2) / (len(x) - 1) + (vy ** 2) / (len(y) - 1) + _EPS)
    return t, dof


def _run_level(df, a, b, strata, n_perm, rng, run_test="welch"):
    tasks = sorted(set(df.loc[df.agent == a, "task"]) & set(df.loc[df.agent == b, "task"]))
    sub = df[df["agent"].isin([a, b]) & df["task"].isin(tasks)]
    runs = sub.groupby(["agent", "run", *strata], as_index=False)["success"].mean()
    na = int((runs.agent == a).sum()); nb = int((runs.agent == b).sum())
    if na < 2 or nb < 2:
        raise ValueError("run-level test needs >= 2 complete runs per agent (>= 5 recommended)")
    y = runs["success"].to_numpy()
    isA = (runs["agent"] == a).to_numpy()
    groups = (runs[list(strata)].astype(str).agg("|".join, axis=1).to_numpy() if strata
              else np.zeros(len(runs), dtype=int))
    if run_test == "welch":
        # block on strata by centring every run on its stratum's pooled mean
        yc = y.copy()
        if strata:
            for gv in np.unique(groups):
                m = groups == gv
                yc[m] = y[m] - y[m].mean()
        t, dof = _welch_t(yc[isA], yc[~isA])
        p = float(2 * stats.t.sf(abs(t), max(dof, 1.0)))
        gap = float(y[isA].mean() - y[~isA].mean())
        se = math.sqrt(yc[isA].var(ddof=1) / na + yc[~isA].var(ddof=1) / nb)
        name = f"run-level Welch t (df={dof:.1f})" + (f"; strata: {', '.join(strata)}" if strata else "")
        return gap, se, p, name, na, nb
    obs = _welch(y[isA], y[~isA])
    n_total = 1
    for gv in np.unique(groups):
        m = groups == gv
        n_total *= math.comb(int(m.sum()), int(isA[m].sum()))
    null = []
    if n_total <= n_perm and not strata:
        from itertools import combinations as comb
        for idx in comb(range(len(y)), na):
            lab = np.zeros(len(y), bool); lab[list(idx)] = True
            null.append(_welch(y[lab], y[~lab]))
        null = np.array(null)
        p = float(np.mean(np.abs(null) >= abs(obs) - _EPS))
        kind = "exact"
    else:
        for _ in range(n_perm):
            lab = isA.copy()
            for gv in np.unique(groups):
                m = np.flatnonzero(groups == gv)
                lab[m] = rng.permutation(lab[m])
            null.append(_welch(y[lab], y[~lab]))
        null = np.array(null)
        p, tail = _perm_p(obs, null)
        kind = "Monte Carlo" + (", normal tail approx." if tail else "")
    gap = float(y[isA].mean() - y[~isA].mean())
    se = math.sqrt(y[isA].var(ddof=1) / na + y[~isA].var(ddof=1) / nb)
    name = f"run-level Welch permutation ({kind})" + (f"; strata: {', '.join(strata)}" if strata else "")
    if na < 5 or nb < 5:
        min_p = 2 / math.comb(na + nb, na) if na == nb else 1 / math.comb(na + nb, na)
        name += f"; few runs, smallest attainable p = {min_p:.3g}"
    return gap, se, p, name, na, nb


# --------------------------------------------------------------- public API
def compare(df: pd.DataFrame, a: str, b: str, *, inference: str = "benchmark", level: str = "task",
            strata: Sequence[str] = (), n_perm: int = 10_000, alpha: float = 0.05,
            studentize: bool = True, run_test: str = "welch", seed: int | None = 0) -> Comparison:
    """Test whether agents ``a`` and ``b`` differ in benchmark score.

    ``strata``: nuisance-factor columns (e.g. ``["api_date", "env_version"]``)
    to permute *within*, when both agents were run under the same levels.
    Blocking removes the factor's shared variance from the test.

    ``run_test`` (level="run" only): "welch" (default) is a Welch t-test on
    complete-run scores; each run score averages hundreds of task outcomes, so
    it is close to normal by the CLT and the t reference is accurate even with
    3-5 runs. "permutation" is distribution-free but its p-values are coarse
    for small k (smallest attainable two-sided p is 2 / C(2k, k)).
    """
    _check_inference(inference)
    if level not in ("task", "run"):
        raise ValueError("level must be 'task' or 'run'")
    rng = np.random.default_rng(seed)
    z = stats.norm.ppf(1 - alpha / 2)
    ts = task_stats(df, agents=[a, b])
    T = len(ts.tasks)
    d = ts.mean[0] - ts.mean[1]
    ra = df.loc[df.agent == a, "run"].nunique(); rb = df.loc[df.agent == b, "run"].nunique()

    if inference == "population":
        gap = float(d.mean())
        se = float(d.std(ddof=1) / np.sqrt(T))
        signs = rng.choice([-1.0, 1.0], size=(n_perm, T))
        null = (signs * d).mean(1)
        p, tail = _perm_p(gap / (se + _EPS), null / (se + _EPS))
        return Comparison(a, b, gap, se, gap - z * se, gap + z * se, p, T, ra, rb, inference, "task",
                          "paired sign-flip over tasks" + ("; normal tail approx." if tail else ""))

    if level == "run":
        gap, se, p, name, ra, rb = _run_level(df, a, b, list(strata), n_perm, rng, run_test)
        tq = stats.t.ppf(1 - alpha / 2, max(1, min(ra, rb) - 1))
        return Comparison(a, b, gap, se, gap - tq * se, gap + tq * se, p, T, ra, rb,
                          inference, "run", name)

    gap, p, name = _task_level(df, a, b, list(strata), n_perm, rng, studentize)
    v = ts.var[0] / ts.k[0] + ts.var[1] / ts.k[1]
    se = float(np.sqrt(np.nansum(v)) / T) if np.all(ts.k >= 2) else float("nan")
    return Comparison(a, b, gap, se, gap - z * se, gap + z * se, p, T, ra, rb, inference, "task", name)


def holm(pvals: Sequence[float]) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (FWER control)."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def pairwise(df: pd.DataFrame, agents: Sequence[str] | None = None, *, adjacent_only: bool = False,
             alpha: float = 0.05, seed: int | None = 0, **kw) -> pd.DataFrame:
    """All pairwise (or leaderboard-adjacent) comparisons, Holm-corrected.

    Extra keyword arguments are passed to :func:`compare`.
    """
    if agents is None:
        from .estimate import agent_scores
        agents = list(agent_scores(df, inference="population")["agent"])
    pairs = list(zip(agents[:-1], agents[1:])) if adjacent_only else list(combinations(agents, 2))
    res = [compare(df, a, b, alpha=alpha, seed=None if seed is None else seed + i, **kw).to_dict()
           for i, (a, b) in enumerate(pairs)]
    out = pd.DataFrame(res)
    out["p_holm"] = holm(out["p_value"])
    out["significant"] = out["p_holm"] < alpha
    return out
