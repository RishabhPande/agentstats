"""One-call leaderboard report and protocol compliance check."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .data import factor_columns, task_stats
from .estimate import agent_scores, design_effect
from .compare import pairwise
from .rank import tiers, flip_statistics
from .variance import decompose

#: The minimum-runs protocol, as machine-checkable rules (see paper, Sec. "Protocol").
PROTOCOL = {
    "min_runs": 5,               # complete benchmark passes per agent (P1)
    "min_factor_levels": 2,      # every declared nuisance factor spans >= 2 levels (P2)
    "max_deff_for_task_tests": 1.5,  # above this, use run-level tests (P3)
}


def check_protocol(df: pd.DataFrame, factors: Sequence[str] | None = None) -> pd.DataFrame:
    factors = list(factors) if factors is not None else factor_columns(df)
    rows = []
    runs = df.groupby("agent")["run"].nunique()
    for a, k in runs.items():
        rows.append({"agent": a, "rule": "P1 runs >= %d" % PROTOCOL["min_runs"], "value": int(k),
                     "ok": k >= PROTOCOL["min_runs"]})
        for f in factors:
            n = df.loc[df.agent == a, f].nunique()
            rows.append({"agent": a, "rule": f"P2 {f} levels >= {PROTOCOL['min_factor_levels']}",
                         "value": int(n), "ok": n >= PROTOCOL["min_factor_levels"]})
    return pd.DataFrame(rows)


def _fmt(df: pd.DataFrame, pct=()) -> str:
    d = df.copy()
    for c in pct:
        if c in d:
            d[c] = (100 * d[c]).map(lambda x: f"{x:.1f}" if np.isfinite(x) else "nan")
    return d.to_markdown(index=False, floatfmt=".3g")


def report(df: pd.DataFrame, *, alpha: float = 0.05, factors: Sequence[str] | None = None,
           n_perm: int = 5000, seed: int = 0) -> str:
    """Markdown report: scores, noise diagnostics, adjacent tests, tiers, flips, variance."""
    factors = list(factors) if factors is not None else factor_columns(df)
    k_min = int(df.groupby("agent")["run"].nunique().min())
    out = ["# agentstats report", ""]
    sc = agent_scores(df, inference="population", alpha=alpha)
    if k_min >= 2:
        bm = agent_scores(df, inference="benchmark", alpha=alpha).set_index("agent")
        sc["se_run_noise"] = sc.agent.map(bm["se"])
    sc = sc.rename(columns={"se": "se_tasks+runs", "lo": "lo_pop", "hi": "hi_pop"})
    out += ["## Scores (pp)", "",
            "`se_tasks+runs`: task-population SE (CI `lo_pop`..`hi_pop`); "
            "`se_run_noise`: run-to-run SE for claims about this benchmark.", "",
            _fmt(sc, pct=("score", "se_tasks+runs", "lo_pop", "hi_pop", "se_run_noise")), ""]

    level = "task"
    if k_min >= 3:
        de = design_effect(df)
        out += ["## Is noise correlated within runs? (design effect)", "", _fmt(de), ""]
        if np.nanmedian(de["deff"]) > PROTOCOL["max_deff_for_task_tests"]:
            level = "run"
            out += [f"Median design effect > {PROTOCOL['max_deff_for_task_tests']}: using **run-level** tests.", ""]
    inference = "benchmark" if k_min >= 2 else "population"
    if inference == "population":
        out += ["> Only one run per agent: run noise cannot be separated from task noise. "
                "Tests below are task-population tests. Collect >= 5 runs.", ""]
    adj = pairwise(df, list(sc.agent), adjacent_only=True, alpha=alpha, n_perm=n_perm, seed=seed,
                   inference=inference, level=level)
    adj_show = adj[["a", "b", "gap", "lo", "hi", "p_value", "p_holm", "significant", "test"]]
    n_sig = int(adj["significant"].sum())
    out += [f"## Adjacent leaderboard gaps ({n_sig}/{len(adj)} significant after Holm)", "",
            _fmt(adj_show, pct=("gap", "lo", "hi")), ""]
    tr = tiers(df, alpha=alpha, n_perm=n_perm, seed=seed, inference=inference, level=level)
    out += [f"## Statistical tiers ({tr.tier.nunique()} tiers for {len(tr)} agents)", "", _fmt(tr), ""]
    if k_min >= 2:
        fs = flip_statistics(df, seed=seed)
        out += ["## Single-run leaderboard stability", "",
                f"- Expected Kendall tau between two single-run leaderboards: {fs['kendall_tau_mean']:.3f} "
                f"(5th pct {fs['kendall_tau_q05']:.3f})",
                f"- P(the #1 agent changes on a re-run): {fs['p_top1_changes']:.2f}",
                f"- Fraction of agent pairs whose order flips: {fs['pair_flip_rate']:.3f}", ""]
    if factors and k_min >= 3:
        try:
            dec = decompose(df, factors)
            cols = ["agent", "n_runs"] + [c for c in dec.columns if c.startswith("sd_")]
            out += ["## Variance decomposition (SD of one run's score, pp)", "", _fmt(dec[cols]), ""]
        except ValueError as e:
            out += [f"_Variance decomposition skipped: {e}_", ""]
    if factors:
        cp = check_protocol(df, factors)
        bad = cp[~cp.ok]
        out += ["## Protocol compliance", "",
                "All rules satisfied." if bad.empty else _fmt(bad), ""]
    return "\n".join(out)
