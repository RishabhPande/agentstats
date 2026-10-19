"""Rank (in)stability: rank intervals, statistical tiers, and single-run flips."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .data import task_stats


def _resample_scores(ts, n_boot, inference, rng, run_sd=None):
    """(n_boot, A) bootstrap replicates of benchmark scores, tasks aligned."""
    A, T = ts.mean.shape
    if run_sd is not None:  # run-level: SE of the mean of k complete runs
        return ts.mean.mean(axis=1) + rng.standard_normal((n_boot, A)) * run_sd
    if inference == "population":
        idx = rng.integers(0, T, size=(n_boot, T))
        return ts.mean[:, idx].mean(axis=2).T
    # benchmark: parametric-normal resampling of each task mean from its run variance;
    # with binary outcomes this matches a within-task run bootstrap to O(1/k).
    se_t = np.sqrt(np.nan_to_num(ts.var) / np.maximum(ts.k, 1))
    sd = np.sqrt((se_t ** 2).sum(axis=1)) / T  # (A,)
    base = ts.mean.mean(axis=1)
    return base + rng.standard_normal((n_boot, A)) * sd


def rank_distribution(df: pd.DataFrame, *, n_boot: int = 5000, inference: str = "benchmark",
                      level: str = "task", alpha: float = 0.05, seed: int | None = 0) -> pd.DataFrame:
    """Bootstrap rank distribution: mean rank, (1-alpha) rank interval, P(rank 1).

    ``level="run"`` uses the observed spread of complete-run scores (robust to
    run-correlated noise); ``level="task"`` assumes independent task noise.
    """
    rng = np.random.default_rng(seed)
    ts = task_stats(df)
    run_sd = None
    if level == "run":
        sub = df[df.task.isin(ts.tasks)]
        g = sub.groupby(["agent", "run"])["success"].mean().groupby("agent")
        run_sd = (g.std(ddof=1) / np.sqrt(g.size())).reindex(ts.agents).to_numpy()
    S = _resample_scores(ts, n_boot, inference, rng, run_sd)
    R = (-S).argsort(axis=1).argsort(axis=1) + 1  # 1 = best
    out = pd.DataFrame({
        "agent": ts.agents,
        "score": ts.mean.mean(axis=1),
        "mean_rank": R.mean(0),
        "rank_lo": np.quantile(R, alpha / 2, axis=0).astype(int),
        "rank_hi": np.quantile(R, 1 - alpha / 2, axis=0).astype(int),
        "p_top1": (R == 1).mean(0),
    })
    return out.sort_values("score", ascending=False, ignore_index=True)


def tiers(df: pd.DataFrame, *, alpha: float = 0.05, n_perm: int = 5000, seed: int | None = 0,
          **compare_kw) -> pd.DataFrame:
    """Group agents into statistically indistinguishable tiers.

    All pairwise tests are run once and Holm-corrected *jointly*, so every
    "A is better than B" statement implied by the tiers holds with
    family-wise error <= alpha. Tiers are then formed greedily top-down: a
    tier starts at the best remaining agent and absorbs every remaining
    agent not significantly worse than that leader. Tiers may overlap in the
    sense that a lower-tier agent can be tied with a non-leader of the tier
    above; we report ``tied_with`` so this is visible.
    """
    from .compare import pairwise
    from .estimate import agent_scores
    order = list(agent_scores(df, inference="population")["agent"])
    pw = pairwise(df, order, alpha=alpha, n_perm=n_perm, seed=seed, **compare_kw)
    sig = {(r.a, r.b): r.significant for r in pw.itertuples()}
    sig.update({(b, a): v for (a, b), v in list(sig.items())})
    tier_of, t, remaining = {}, 1, order[:]
    while remaining:
        lead = remaining[0]
        members = [lead] + [b for b in remaining[1:] if not sig[(lead, b)]]
        for m in members:
            tier_of[m] = t
        remaining = [b for b in remaining if b not in members]
        t += 1
    tied = {a: sorted(b for b in order if b != a and not sig[(a, b)]) for a in order}
    return pd.DataFrame({"agent": order, "tier": [tier_of[a] for a in order],
                         "tied_with": [", ".join(tied[a]) for a in order]})


def single_run_leaderboards(df: pd.DataFrame, n_draws: int = 2000, seed: int | None = 0) -> np.ndarray:
    """Simulate leaderboards built from ONE complete run per agent.

    Each draw picks, independently per agent, one observed ``run`` index (a full
    benchmark pass). Returns (n_draws, A) scores, agents in sorted order.
    """
    rng = np.random.default_rng(seed)
    per = df.groupby(["agent", "run"])["success"].mean()
    agents = sorted(df["agent"].unique())
    cols = []
    for a in agents:
        s = per.loc[a].to_numpy()
        cols.append(s[rng.integers(0, len(s), size=n_draws)])
    return np.stack(cols, axis=1)


def flip_statistics(df: pd.DataFrame, n_draws: int = 2000, seed: int | None = 0) -> dict:
    """How much does a single-run leaderboard disagree with another single-run one?

    Returns expected Kendall tau between two independent single-run leaderboards,
    P(the #1 agent changes), and the expected fraction of agent pairs whose
    order flips.
    """
    a = single_run_leaderboards(df, n_draws, seed)
    b = single_run_leaderboards(df, n_draws, None if seed is None else seed + 1)
    taus = np.array([stats.kendalltau(x, y).statistic for x, y in zip(a, b)])
    top_change = np.mean(a.argmax(1) != b.argmax(1))
    A = a.shape[1]
    iu = np.triu_indices(A, 1)
    sa = np.sign(a[:, :, None] - a[:, None, :])[:, iu[0], iu[1]]
    sb = np.sign(b[:, :, None] - b[:, None, :])[:, iu[0], iu[1]]
    flips = np.mean(sa * sb < 0)
    return {"kendall_tau_mean": float(np.nanmean(taus)),
            "kendall_tau_q05": float(np.nanquantile(taus, 0.05)),
            "p_top1_changes": float(top_change),
            "pair_flip_rate": float(flips)}


def factor_rank_agreement(df: pd.DataFrame, factor: str) -> pd.DataFrame:
    """Kendall tau between the leaderboards obtained at each pair of factor levels."""
    lb = df.groupby([factor, "agent"])["success"].mean().unstack("agent")
    levels = list(lb.index)
    rows = []
    for i in range(len(levels)):
        for j in range(i + 1, len(levels)):
            x, y = lb.loc[levels[i]], lb.loc[levels[j]]
            ok = x.notna() & y.notna()
            tau = stats.kendalltau(x[ok], y[ok]).statistic
            rows.append({"level_a": levels[i], "level_b": levels[j], "kendall_tau": tau,
                         "top1_a": x[ok].idxmax(), "top1_b": y[ok].idxmax()})
    return pd.DataFrame(rows)
