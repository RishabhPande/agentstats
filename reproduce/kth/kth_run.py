"""Section 7 on the real KTH artifacts: 12 configurations x 10 runs x 500 SWE-bench Verified tasks."""
import json, warnings
from itertools import combinations
import numpy as np, pandas as pd
from scipy import stats
import agentstats as ast
warnings.filterwarnings("ignore")
rng = np.random.default_rng(0)
df = ast.load("kth_runs.csv")
ts = ast.task_stats(df)
T = len(ts.tasks)
out = {"n_configs": len(ts.agents), "n_tasks_common": T, "runs": int(df.run.nunique())}

# 1. per-config noise
rs = df[df.task.isin(ts.tasks)].groupby(["agent", "run"]).success.mean().unstack() * 100
de = ast.design_effect(df).set_index("agent")
cfg = pd.DataFrame({"mean": rs.mean(1), "sd_run": rs.std(1, ddof=1), "range": rs.max(1) - rs.min(1),
                    "sd_indep": 100 * np.sqrt(de.var_indep), "deff": de.deff, "deff_p": de.p_value})
# resubmission-style estimator from just two runs (validates Sec. 5): sd = sqrt(mean_t 1{y1!=y2}/2 / T)
W = df[df.task.isin(ts.tasks)].pivot_table(index=["agent", "task"], columns="run", values="success")
est = []
for a in ts.agents:
    w = W.loc[a]
    pairs = list(combinations(w.columns, 2))
    est.append(np.mean([100 * np.sqrt((w[i] != w[j]).mean() / 2 / T) for i, j in pairs]))
cfg["sd_two_run_est"] = est
cfg = cfg.sort_values("mean", ascending=False)
print(cfg.round(2).to_string())
out["configs"] = cfg.reset_index().to_dict("records")

# 2. held-out type-I: each config vs itself (5 vs 5 runs), true gap 0
rej = {"pooled z (iid)": [], "task-level perm": [], "run-level Welch": []}
for a in ts.agents:
    sub = df[(df.agent == a) & df.task.isin(ts.tasks)]
    splits = list(combinations(range(10), 5))
    splits = [s for s in splits if 0 in s]  # each split once (complement symmetric)
    for s in [splits[i] for i in rng.choice(len(splits), 40, replace=False)]:
        x = sub.copy(); x["agent"] = np.where(x.run.isin(s), "A", "B")
        x["run"] = x.groupby(["agent", "task"]).cumcount()
        ya, yb = x[x.agent == "A"].success, x[x.agent == "B"].success
        p = np.r_[ya, yb].mean(); se = np.sqrt(p * (1 - p) * (1 / len(ya) + 1 / len(yb)))
        rej["pooled z (iid)"].append(2 * stats.norm.sf(abs(ya.mean() - yb.mean()) / se) < .05)
        rej["task-level perm"].append(ast.compare(x, "A", "B", n_perm=400).p_value < .05)
        rej["run-level Welch"].append(ast.compare(x, "A", "B", level="run").p_value < .05)
out["heldout"] = {k: dict(rate=float(np.mean(v)), n=len(v)) for k, v in rej.items()}
print(out["heldout"])

# 3. leaderboard of the 12 configurations
order = list(cfg.index)
for lvl in ("task", "run"):
    adj = ast.pairwise(df, order, adjacent_only=True, level=lvl, n_perm=5000)
    tr = ast.tiers(df, level=lvl, n_perm=5000)
    out[f"lb_{lvl}"] = dict(adjacent_sig=int(adj.significant.sum()), n_adj=len(adj), tiers=int(tr.tier.nunique()),
                            top_tier=int((tr.tier == 1).sum()), gaps=[round(100 * g, 2) for g in adj.gap])
    print(lvl, out[f"lb_{lvl}"])
single = ast.pairwise(df[df.run == 0], order, adjacent_only=True, inference="population", n_perm=5000)
out["lb_single_run0_population_sig"] = int(single.significant.sum())
out["flip"] = ast.flip_statistics(df, n_draws=5000)
# observed: rank of each config in each of the 10 single-run leaderboards
ranks = rs.rank(axis=0, ascending=False)
out["rank_range_per_config"] = (ranks.max(1) - ranks.min(1)).to_dict()
out["n_distinct_leaders_over_10_runs"] = int(rs.idxmax(axis=0).nunique())
out["leaders_over_runs"] = rs.idxmax(axis=0).value_counts().to_dict()
print(out["flip"], out["n_distinct_leaders_over_10_runs"], out["leaders_over_runs"])

# 4. planner from a 3-run pilot vs truth from all 10 runs
pilot = df[df.run < 3]
pl = ast.plan(pilot, 0.02, order, adjacent_only=True, method="max")
out["plan_pilot3_delta2"] = pl[["a", "b", "k_min", "w_task", "w_run"]].to_dict("records")
out["plan_generic"] = {f"{d}": ast.k_min_generic(T, d, float(np.mean(cfg.sd_run.values) ** 2 * T / 1e4), k_floor=1)
                       for d in (0.01, 0.02, 0.03)}
print(pl[["a", "b", "k_min"]].to_string(), out["plan_generic"])
json.dump(out, open("kth_results.json", "w"), indent=1, default=float)
