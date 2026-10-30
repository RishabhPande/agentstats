"""Simulation study for the paper. EVERYTHING HERE IS SYNTHETIC.

Parameters are assumptions, stated in SETTINGS and in the paper's Table 2,
anchored only to published single-run SDs on SWE-bench Verified
(Bjarnason et al., 2026: up to ~1.8 pp). Outputs:
  results.json, fig_flips.pdf, fig_kmin.pdf, fig_tiers.pdf
"""
import json
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

import agentstats as ast
from agentstats.simulate import SimConfig, simulate, calibrate_scale

OUT = Path(__file__).parent
rng = np.random.default_rng(2026)

BENCHES = {"SWE-like (T=500)": 500, "Web-like (T=812)": 812, "OS-like (T=369)": 369}
# single-run SD on a 500-task benchmark, pp -> flakiness phi = 500 * sd^2 (held fixed across benchmarks)
REGIMES = {"low": 0.8, "medium": 1.4, "high": 2.0}
N_AGENTS = 14


def hypothetical_leaderboard(top=0.72, mean_gap=0.02, n=N_AGENTS, seed=0):
    r = np.random.default_rng(seed)
    gaps = r.exponential(mean_gap, n - 1)
    return tuple(np.clip(top - np.concatenate([[0], np.cumsum(gaps)]), 0.05, 0.95))


TRUE = hypothetical_leaderboard()
FACTORS = dict(api_dates=("d0", "d1", "d2"), sd_api=0.12, env_versions=("e0", "e1"), sd_env=0.08,
               env_break_frac=0.01, latencies=("p50", "p95"), timeout_rate={"p50": 0.0, "p95": 0.04},
               sd_run_shock=0.08)

C1, C2, C3, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.5,
                     "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b",
                     "xtick.color": "#52514e", "ytick.color": "#52514e", "lines.linewidth": 1.6,
                     "legend.frameon": False, "pdf.fonttype": 42})


def cfg_for(T, sd500, factors=False, seed=0, true=TRUE):
    base = SimConfig(n_tasks=T, true_scores=true, seed=seed, **(FACTORS if factors else {}))
    target = np.sqrt(500 * (sd500 / 100) ** 2 / T)
    base.scale = calibrate_scale(base, target)
    return base


results = {"true_scores": [round(100 * x, 2) for x in TRUE],
           "true_adjacent_gaps_pp": [round(100 * (a - b), 2) for a, b in zip(TRUE[:-1], TRUE[1:])],
           "factors": {k: (list(v) if isinstance(v, tuple) else v) for k, v in FACTORS.items()}}
t0 = time.time()

# ---------------------------------------------------------------- E1: flips
e1 = []
for bname, T in BENCHES.items():
    for reg, sd in REGIMES.items():
        for fac in (False, True):
            cfg = cfg_for(T, sd, fac, seed=11)
            df = simulate(cfg, 24, run_seed=99)
            sdr = ast.design_effect(df)
            single_sd = float(np.sqrt(sdr["var_run"]).median() * 100)
            fs = ast.flip_statistics(df, n_draws=3000, seed=1)
            gaps = np.array(results["true_adjacent_gaps_pp"])
            floor = 1.96 * np.sqrt(2) * single_sd
            e1.append(dict(bench=bname, T=T, regime=reg, factors=fac, single_run_sd_pp=single_sd,
                           deff_median=float(sdr["deff"].median()),
                           frac_gaps_within_noise=float(np.mean(gaps < floor)), noise_floor_pp=floor, **fs))
            print(e1[-1]["bench"], reg, fac, f"sd={single_sd:.2f} tau={fs['kendall_tau_mean']:.3f} "
                  f"top1={fs['p_top1_changes']:.2f} within={e1[-1]['frac_gaps_within_noise']:.2f}", flush=True)
results["E1"] = e1

# ------------------------------------------------------- E2: test calibration
def naive_z(df, a, b):
    """What many papers do: two-proportion z-test pooling all task-runs as iid."""
    xa = df.loc[df.agent == a, "success"]; xb = df.loc[df.agent == b, "success"]
    p = np.r_[xa, xb].mean()
    se = np.sqrt(p * (1 - p) * (1 / len(xa) + 1 / len(xb)))
    return 2 * stats.norm.sf(abs(xa.mean() - xb.mean()) / se)


def single_run_z(df, a, b):
    """Single-run leaderboard comparison: run 0 only, unpaired binomial z-test."""
    return naive_z(df[df.run == 0], a, b)


CONDS = {
    "sharp null": dict(true_scores=(0.5, 0.5), sd_interaction=0.0),
    "weak null": dict(true_scores=(0.5, 0.5), sd_interaction=1.0),
    "weak null, shock 0.05": dict(true_scores=(0.5, 0.5), sd_interaction=1.0, sd_run_shock=0.05),
    "weak null, shock 0.10": dict(true_scores=(0.5, 0.5), sd_interaction=1.0, sd_run_shock=0.10),
    "weak null, shock 0.20": dict(true_scores=(0.5, 0.5), sd_interaction=1.0, sd_run_shock=0.20),
    "3pp gap": dict(true_scores=(0.515, 0.485), sd_interaction=1.0),
    "3pp gap, shock 0.10": dict(true_scores=(0.515, 0.485), sd_interaction=1.0, sd_run_shock=0.10),
}
N_SIM, K, T2 = 300, 5, 500
e2 = []
for cname, kw in CONDS.items():
    rej = {"single-run z": 0, "pooled z (iid)": 0, "population sign-flip": 0,
           "task-level perm (ours)": 0, "run-level Welch (ours)": 0}
    for s in range(N_SIM):
        cfg = SimConfig(n_tasks=T2, seed=1000 + s, scale=1.0, **kw)
        df = simulate(cfg, K, run_seed=5000 + s)
        rej["single-run z"] += single_run_z(df, "A01", "A02") < .05
        rej["pooled z (iid)"] += naive_z(df, "A01", "A02") < .05
        rej["population sign-flip"] += ast.compare(df, "A01", "A02", inference="population", n_perm=500, seed=s).p_value < .05
        rej["task-level perm (ours)"] += ast.compare(df, "A01", "A02", n_perm=500, seed=s).p_value < .05
        rej["run-level Welch (ours)"] += ast.compare(df, "A01", "A02", level="run", n_perm=500, seed=s).p_value < .05
    probe = simulate(SimConfig(n_tasks=T2, seed=1, scale=1.0, **kw), 40, run_seed=3)
    de = ast.design_effect(probe)
    row = {"condition": cname, "deff": float(de.deff.mean()),
           "single_run_sd_pp": float(100 * np.sqrt(de.var_run).mean()),
           **{k: v / N_SIM for k, v in rej.items()}}
    e2.append(row); print(row, flush=True)
results["E2"] = {"n_sim": N_SIM, "k": K, "T": T2, "rows": e2}

# ------------------------------------------------------------ E3: k_min curves
deltas = np.linspace(0.005, 0.05, 46)
e3 = {}
fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3), sharey=True)
for ax, (bname, T) in zip(axes, BENCHES.items()):
    for (reg, sd), col in zip(REGIMES.items(), (C1, C2, C3)):
        phi = 500 * (sd / 100) ** 2
        ks = [ast.k_min_generic(T, d, phi, k_floor=1) for d in deltas]
        e3[f"{bname}|{reg}"] = {f"{d:.3f}": k for d, k in zip(deltas, ks) if round(d * 1000) % 5 == 0}
        ax.plot(100 * deltas, ks, color=col, label=f"{reg} noise (SD$_{{500}}$={sd} pp)")
    ax.axhline(1, color=GRAY, lw=0.8, ls=":")
    ax.set_yscale("log"); ax.set_title(bname, fontsize=8); ax.set_xlabel("true gap $\\delta$ (pp)")
axes[0].set_ylabel("runs per agent, $k_{\\min}$")
axes[0].legend(fontsize=6.5, loc="upper right")
fig.tight_layout(); fig.savefig(OUT / "fig_kmin.pdf"); plt.close(fig)
results["E3"] = e3

# --------------------------------------------- E4: tiers & resolved gaps vs k
e4 = []
for bname, T in BENCHES.items():
    cfg = cfg_for(T, REGIMES["medium"], True, seed=21)
    big = simulate(cfg, 20, run_seed=7)
    for k in (1, 3, 5, 10, 20):
        df = big[big.run < k]
        inf = "population" if k == 1 else "benchmark"
        lvl = "run" if k >= 5 else "task"
        adj = ast.pairwise(df, [f"A{i + 1:02d}" for i in range(N_AGENTS)], adjacent_only=True,
                           inference=inf, level=lvl, n_perm=1000)
        tr = ast.tiers(df, n_perm=1000, inference=inf, level=lvl)
        e4.append(dict(bench=bname, k=k, test=f"{inf}/{lvl}", adjacent_resolved=int(adj.significant.sum()),
                       n_adjacent=len(adj), n_tiers=int(tr.tier.nunique())))
        print(e4[-1], flush=True)
results["E4"] = e4

# ---------------------------------------------------------------- figures
import pandas as pd
d1 = pd.DataFrame(e1)
fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3))
x = np.arange(len(REGIMES))
for j, (metric, lab) in enumerate([("kendall_tau_mean", "E[Kendall $\\tau$], two single-run boards"),
                                    ("p_top1_changes", "P(#1 changes on re-run)"),
                                    ("frac_gaps_within_noise", "adjacent gaps < noise floor")]):
    ax = axes[j]
    for i, ((bname, _), col) in enumerate(zip(BENCHES.items(), (C1, C2, C3))):
        sub = d1[(d1.bench == bname) & d1.factors]
        base = d1[(d1.bench == bname) & ~d1.factors]
        ax.plot(x, sub[metric].to_numpy(), "-o", color=col, ms=4, label=bname)
        ax.plot(x, base[metric].to_numpy(), ":", color=col, lw=1)
    ax.set_xticks(x, list(REGIMES)); ax.set_xlabel("seed-noise regime"); ax.set_title(lab, fontsize=8)
axes[0].legend(fontsize=6.5, loc="lower left")
fig.tight_layout(); fig.savefig(OUT / "fig_flips.pdf"); plt.close(fig)

d4 = pd.DataFrame(e4)
fig, ax = plt.subplots(figsize=(3.3, 2.3))
for (bname, _), col in zip(BENCHES.items(), (C1, C2, C3)):
    sub = d4[d4.bench == bname]
    ax.plot(sub.k, sub.adjacent_resolved, "-o", color=col, ms=4, label=bname)
ax.axhline(N_AGENTS - 1, color=GRAY, lw=0.8, ls=":")
ax.text(20, N_AGENTS - 1.5, "all 13 gaps", ha="right", va="top", fontsize=6.5, color="#52514e")
ax.set_xscale("log"); ax.set_xticks([1, 3, 5, 10, 20], ["1", "3", "5", "10", "20"])
ax.set_xlabel("runs per agent $k$"); ax.set_ylabel("adjacent gaps significant (Holm)")
ax.legend(fontsize=6.5, loc="upper left"); fig.tight_layout(); fig.savefig(OUT / "fig_tiers.pdf"); plt.close(fig)

results["runtime_s"] = round(time.time() - t0, 1)
json.dump(results, open(OUT / "results.json", "w"), indent=2, default=float)
print("done", results["runtime_s"])
